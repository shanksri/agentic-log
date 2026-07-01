"""Strategy-Aware Confidence Normalization Framework (Phase 18C).

Phase 18B's integration exposed a real architectural gap: Dense retrieval
scores are cosine distances (roughly ``[0, 2]``, usually near ``[0, 1]``),
BM25 scores (Phase 17A) are unbounded, non-negative, idf-weighted sums with
no fixed ceiling, and Hybrid/RRF scores (Phase 17B) are sums of
``1/(k+rank)`` terms — tiny numbers, typically well under ``0.04`` for
``k=60``. Feeding any of these directly into
``app.services.confidence.classify_confidence`` (thresholds ``0.40``/
``0.55``, calibrated against dense similarity scores only — see that
module's own docstring) produces a meaningless classification for BM25/
Hybrid: a raw BM25 score of ``3.5`` is nowhere near the ``[0, 1]`` range
those thresholds assume, so it would always classify ``HIGH`` regardless of
actual retrieval quality. This phase fixes that by inserting one new layer
— a normalizer, chosen per strategy — between "the strategy's native
score" and "the shared LOW/MEDIUM/HIGH classifier", so every strategy
produces a comparable ``[0.0, 1.0]`` value before classification ever runs.

This phase builds the normalization ARCHITECTURE only. Per explicit
instruction, it does NOT implement:

- Platt scaling, isotonic regression, temperature scaling, or any other
  statistically fit calibration technique (those require labeled data this
  project does not yet have at scale).
- Any ML model.
- Any change to ``app.services.confidence`` (the existing
  ``classify_confidence``/thresholds are reused completely unmodified —
  see "Backward compatibility" below) or to
  ``app.services.routing``/``app.services.routed_search`` (Phase 18A/18B,
  untouched; this module is freestanding, ready to be wired in by a future
  phase, the same pattern Phase 18A/18B's frameworks themselves followed
  before their own integration phase existed).
- Any tuning of the thresholds, routing policy, or evaluation metrics.

The normalizers below are deliberately simple, deterministic, closed-form
functions — see "Why statistical calibration is intentionally deferred".

# Updated architecture

```
                Query → Routing (18A) → Retrieval Strategy (Dense/BM25/Hybrid)
                                              │
                                              ▼
                                      Candidate Score
                              (strategy's OWN native score —
                               see "Normalization strategy" for what this
                               means concretely per strategy)
                                              │
                                              ▼
                     get_confidence_normalizer(strategy)   [registry —
                                              │              no if/else
                                              │              chain at any
                                              │              call site]
                                              ▼
                              ConfidenceNormalizer.normalize(raw_score)
                                              │
                                              ▼
                                  NormalizedConfidence
                              (value in [0.0, 1.0], strategy,
                               raw_score — for traceability only)
                                              │
                                              ▼
                    classify_confidence(value)    [app.services.confidence,
                                              │      UNMODIFIED — same
                                              │      thresholds, same
                                              │      function, for every
                                              │      strategy]
                                              ▼
                                  LOW / MEDIUM / HIGH
                                              │
                                              ▼
                                    Investigation Agent
                          (receives ONLY value + level — never which
                           strategy or normalizer produced them)
```

# Confidence lifecycle

```
normalize_confidence(strategy, raw_score)
  1. normalizer = get_confidence_normalizer(strategy)     [registry lookup]
  2. normalized = normalizer.normalize(raw_score)
       a. raw_score is None -> value = 0.0 (no candidates; same convention
          as classify_confidence(None) -> LOW)
       b. else -> strategy-specific deterministic transform -> value,
          ALWAYS clamped to [0.0, 1.0] regardless of how extreme raw_score is
       c. level = classify_confidence(value)   [shared, unmodified]
       d. -> NormalizedConfidence(value, level, strategy, raw_score)
  3. return normalized
```

# Normalization strategy for each retrieval engine

- **``DenseConfidenceNormalizer``** — input: a cosine *distance* (the same
  quantity ``IncidentSearchResult.distance`` already carries for dense
  results). ``value = clamp(1 - distance, 0, 1)`` — this is the *exact*
  transform ``IncidentSearchResult.similarity_score`` already performs
  today (pre-16). Dense's normalized value is therefore numerically
  identical to what ``classify_confidence`` already receives in
  production, and classification is unchanged — see "Backward
  compatibility".
- **``BM25ConfidenceNormalizer``** — input: a raw BM25 score (the same
  quantity ``BM25SearchResult.score`` carries — unbounded, ``>= 0``).
  ``value = score / (score + BM25_MIDPOINT)`` (``BM25_MIDPOINT = 4.0``,
  see "Design decisions" for why). This is a *saturating* function: ``0``
  at ``score = 0``, ``0.5`` at ``score = BM25_MIDPOINT``, monotonically
  approaching but never reaching ``1.0`` as ``score`` grows — squashing an
  unbounded scale into ``[0, 1)`` without any fitted parameter beyond a
  single, documented, order-of-magnitude midpoint constant.
- **``HybridConfidenceNormalizer``** — input: a raw RRF score (the same
  quantity ``HybridSearchResult.rrf_score`` carries — typically well under
  ``0.04`` for the default ``rrf_k=60``, Phase 17B). Same saturating
  formula as BM25, with ``HYBRID_MIDPOINT = 0.016`` — an order of magnitude
  appropriate to RRF's much smaller native scale, not a separately
  invented formula. Reusing one functional form (just re-scaled) for both
  BM25 and Hybrid is itself a design choice: it keeps the framework's
  total surface area small and makes "we changed the midpoint constant"
  cleanly separable from "we changed the kind of function," which matters
  for whoever tunes these later.

None of these midpoint constants were fit to any dataset — they are
illustrative defaults chosen to be the right order of magnitude for each
strategy's known native scale (BM25's score magnitudes observed in Phase
17C/17D's benchmark runs; RRF's mathematically-bounded maximum given
``rrf_k=60``), exactly matching this phase's "heuristic mappings, not
statistically calibrated probabilities" instruction.

# Factory/registry design

``get_confidence_normalizer(strategy: RoutingStrategy) -> ConfidenceNormalizer``
is a single dict-backed lookup (``_NORMALIZERS``), not an if/elif chain
repeated at every call site that needs strategy-aware confidence. This
mirrors Phase 17C's ``build_strategy()`` (a name -> object lookup, not a
branch duplicated across callers) and Phase 18A's ``RoutingPolicy``
interface (a single swap point, not scattered conditionals).
``register_confidence_normalizer(strategy, normalizer)`` lets a future
phase add or replace a normalizer for a given strategy (e.g. swapping in
a Platt-scaled ``DenseConfidenceNormalizer`` once enough labeled data
exists) without touching this module's call sites or
``normalize_confidence()`` itself — the registry, not the call sites, is
the extension point.

# Why statistical calibration is intentionally deferred

Platt scaling, isotonic regression, and temperature scaling all require a
sizable labeled dataset (query, retrieved-document, true-relevance triples
across the operating range) to fit a curve that is actually trustworthy —
fitting one on a handful of examples produces a curve that looks precise
but generalizes worse than the simple heuristic it replaced (overfitting a
calibration curve is a well-documented failure mode in the calibration
literature). This project's current labeled data is Phase 17C/17D's 36
hand-authored gold queries — nowhere near enough to fit three independent
per-strategy calibration curves with any confidence that the result
reflects reality rather than the noise in 36 samples. The right amount of
work for *this* phase is therefore the part that doesn't depend on dataset
size at all: a stable interface (``ConfidenceNormalizer``) and registry
that a future, properly-resourced calibration phase can implement against,
without anything downstream (the classifier, the investigation agent)
needing to change when a heuristic normalizer is swapped for a fitted one.

# Strategy independence

``NormalizedConfidence`` carries ``strategy``/``raw_score`` purely for
traceability/debugging (e.g. a log line, or a future calibration phase
inspecting what raw scores produced what normalized values) — every
*consumer* of confidence (the eventual Investigation Agent integration,
``RoutingEngine``, or any other downstream component) is expected to read
only ``.value`` and ``.level``. Neither field requires knowing which
strategy or normalizer produced it to be interpreted correctly — that is
the entire point of normalizing in the first place. See
``tests/unit/test_confidence_normalization.py``'s strategy-independence
tests, which verify a downstream function using only ``.value``/``.level``
behaves identically given equivalent-confidence raw scores from different
strategies.

# Design decisions

- **Why classification is not reimplemented here.** Centralizing LOW/
  MEDIUM/HIGH classification in one already-existing, already-tested
  function (``app.services.confidence.classify_confidence``) guarantees
  every strategy's normalized value is judged by literally the same
  thresholds — there is no way for BM25's classifier and Dense's
  classifier to silently drift apart, because there is only one
  classifier, called from a single place
  (``normalize_confidence``/each normalizer's ``.normalize()``).
- **Why ``NormalizedConfidence`` is a frozen dataclass.** Same convention
  as every other report/result type across this project (Phases 16-18) —
  immutable, easy to compare in tests, safe to hold past the originating
  retrieval call.
- **Why ``raw_score`` is the strategy's OWN native score, not
  ``IncidentSearchResult.distance``.** Phase 18B's
  ``RoutedSearchService`` stores ``distance = -score`` for BM25/Hybrid
  candidates as an internal sorting convenience (so "lower distance is
  better" holds uniformly across strategies for merge/sort purposes) — that
  is an implementation detail of Phase 18B's candidate pipeline, not a
  semantically meaningful distance. A future integration wiring this
  module into ``RoutedSearchService`` must pass the strategy's *original*
  score (``BM25SearchResult.score``, ``HybridSearchResult.rrf_score``) to
  ``normalize_confidence`` — re-negating ``IncidentSearchResult.distance``
  back to the raw score, not using it directly. This is called out again
  in "Risks discovered" because it is an easy mistake for a future
  integration to make silently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.services.confidence import classify_confidence
from app.services.routing import RoutingStrategy

BM25_MIDPOINT = 4.0
HYBRID_MIDPOINT = 0.016


@dataclass(frozen=True)
class NormalizedConfidence:
    """The common confidence representation every strategy produces.
    ``strategy``/``raw_score`` are for traceability only — see module
    docstring's "Strategy independence".
    """

    value: float
    level: str
    strategy: RoutingStrategy
    raw_score: float | None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _saturating_normalize(raw_score: float, *, midpoint: float) -> float:
    """``score / (score + midpoint)``: 0 at score=0, 0.5 at score=midpoint,
    monotonically approaching (never reaching) 1.0 as score grows. A
    deterministic squashing function for an unbounded, non-negative score
    — see module docstring's "Normalization strategy" for the midpoint
    values used per strategy.
    """
    if raw_score <= 0:
        return 0.0
    return raw_score / (raw_score + midpoint)


class ConfidenceNormalizer(ABC):
    """The swappable extension point. Every concrete normalizer maps one
    strategy's native score onto ``[0.0, 1.0]`` and classifies it via the
    shared, unmodified ``classify_confidence`` — see module docstring's
    "Why classification is not reimplemented here".
    """

    strategy: RoutingStrategy

    @abstractmethod
    def normalize(self, raw_score: float | None) -> NormalizedConfidence:
        """Return a ``NormalizedConfidence`` for ``raw_score`` (this
        strategy's own native score type — see module docstring for what
        that means per strategy), or for ``raw_score=None`` (no
        candidates retrieved at all).
        """


class DenseConfidenceNormalizer(ConfidenceNormalizer):
    """``raw_score`` is a cosine distance — see module docstring's
    "Backward compatibility": this reproduces
    ``IncidentSearchResult.similarity_score`` exactly.
    """

    strategy = RoutingStrategy.DENSE

    def normalize(self, raw_score: float | None) -> NormalizedConfidence:
        value = 0.0 if raw_score is None else _clamp(1.0 - raw_score)
        return NormalizedConfidence(
            value=value, level=classify_confidence(value), strategy=self.strategy,
            raw_score=raw_score,
        )


class BM25ConfidenceNormalizer(ConfidenceNormalizer):
    """``raw_score`` is a raw, unbounded BM25 score — see module
    docstring's "Normalization strategy".
    """

    strategy = RoutingStrategy.BM25

    def normalize(self, raw_score: float | None) -> NormalizedConfidence:
        value = (
            0.0 if raw_score is None
            else _clamp(_saturating_normalize(raw_score, midpoint=BM25_MIDPOINT))
        )
        return NormalizedConfidence(
            value=value, level=classify_confidence(value), strategy=self.strategy,
            raw_score=raw_score,
        )


class HybridConfidenceNormalizer(ConfidenceNormalizer):
    """``raw_score`` is a raw RRF score — see module docstring's
    "Normalization strategy".
    """

    strategy = RoutingStrategy.HYBRID

    def normalize(self, raw_score: float | None) -> NormalizedConfidence:
        value = (
            0.0 if raw_score is None
            else _clamp(_saturating_normalize(raw_score, midpoint=HYBRID_MIDPOINT))
        )
        return NormalizedConfidence(
            value=value, level=classify_confidence(value), strategy=self.strategy,
            raw_score=raw_score,
        )


_NORMALIZERS: dict[RoutingStrategy, ConfidenceNormalizer] = {
    RoutingStrategy.DENSE: DenseConfidenceNormalizer(),
    RoutingStrategy.BM25: BM25ConfidenceNormalizer(),
    RoutingStrategy.HYBRID: HybridConfidenceNormalizer(),
}


def register_confidence_normalizer(
    strategy: RoutingStrategy, normalizer: ConfidenceNormalizer
) -> None:
    """Register (or replace) the normalizer used for ``strategy``. The
    extension point a future calibration phase uses to swap in a
    statistically fit normalizer — see module docstring's "Factory/registry
    design" — without changing ``normalize_confidence`` or any call site.
    """
    _NORMALIZERS[strategy] = normalizer


def get_confidence_normalizer(strategy: RoutingStrategy) -> ConfidenceNormalizer:
    """Return the registered normalizer for ``strategy``. Raises
    ``ValueError`` if none is registered — never silently falls back to a
    different strategy's normalizer, which would defeat the entire purpose
    of strategy-aware normalization.
    """
    try:
        return _NORMALIZERS[strategy]
    except KeyError:
        raise ValueError(
            f"no confidence normalizer registered for strategy {strategy!r}"
        ) from None


def normalize_confidence(
    strategy: RoutingStrategy, raw_score: float | None
) -> NormalizedConfidence:
    """The single public entry point: look up the right normalizer for
    ``strategy`` and normalize ``raw_score`` through it.
    """
    return get_confidence_normalizer(strategy).normalize(raw_score)
