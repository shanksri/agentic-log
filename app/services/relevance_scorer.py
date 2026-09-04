"""Cross-encoder relevance scoring for evidence evaluation.

# Why this exists

``HypothesisEvaluator`` originally split retrieved incidents into
"supporting" and "contradicting" by thresholding *cosine similarity to the
hypothesis's own keywords* at ``LOW_CONFIDENCE_THRESHOLD`` (0.40). Two
things are wrong with that, and they compound:

1. **It asks the wrong question.** Searching with the hypothesis's own
   keywords and then counting what comes back as "support" is confirmation
   bias mechanized -- a hypothesis always resembles its own keywords, so it
   always finds agreement.
2. **Cosine similarity is not relevance.** Bi-encoder scores over a
   topically coherent corpus bunch tightly together, so genuinely relevant
   and merely topical incidents land in the same narrow band and the
   threshold cannot separate them.

Measured on the production corpus, for the problem "ZookeeperConsumerConnector\
MBean needs to close SimpleConsumer when done", the five retrieved incidents
scored 0.525-0.626 by cosine -- all five above 0.40, all five counted as
supporting evidence, including a ``SocketTimeoutException`` incident that has
nothing to do with the question. Across a 34-investigation probe the whole
pipeline produced 78 supporting items against 2 contradicting, and the
critic's ``rejected`` verdict never fired once.

A cross-encoder scores the (query, passage) pair jointly rather than
comparing two independently-computed vectors, which is what makes it able to
say "topically close but does not answer this". On the same five incidents it
returned 9.81, 0.73, -2.95, -8.11 and -11.44.

# Scoring contract

``score(query, passages)`` returns one float per passage, aligned by index.
Scores are the model's **raw logits, not probabilities and not 0-1** --
``ms-marco`` cross-encoders emit roughly -11..+11. ``0.0`` is the natural
decision boundary and is the default threshold; do NOT reuse
``LOW_CONFIDENCE_THRESHOLD`` here, it is a cosine threshold and means nothing
in this space.

# Deliberate non-goals

- **No LLM call.** This runs locally next to the existing
  ``SentenceTransformer``, so evidence evaluation stays free, offline,
  reproducible, and rate-limit-free. Evaluation runs must be deterministic
  for the same input; an LLM stage here would break that.
- **The two thresholds are different numbers for different jobs.**
  ``relevance_threshold`` (0.0, the model's own natural boundary) splits
  supporting from contradicting evidence and is NOT calibrated against any
  dataset. ``retrieval_gate_threshold`` (2.0) decides whether retrieval found
  anything relevant at all, and WAS read off a measured 56-query curve --
  see ``evaluate_retrieval_gate``. Do not reuse one for the other.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Protocol

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RelevanceScorerError(RuntimeError):
    """Raised when the cross-encoder fails to load or score (model download
    failure, out-of-memory, tokenizer error). Typed so callers can tell
    "relevance backend is broken" from an unrelated bug, matching
    ``EmbeddingServiceError``.
    """


class RelevanceScorer(Protocol):
    """The minimal abstraction ``HypothesisEvaluator`` depends on, so a fake
    can be injected in tests without loading a model.
    """

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderRelevanceScorer:
    """Cross-encoder backed ``RelevanceScorer``. The model is loaded lazily
    on first ``score()`` call, never at import or construction.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.relevance_model_name

    @cached_property
    def model(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder

        try:
            return CrossEncoder(self.model_name)
        except Exception as exc:  # noqa: BLE001 — model loading fails in many SDK-internal ways
            raise RelevanceScorerError(
                f"Failed to load relevance model {self.model_name!r}: {exc}"
            ) from exc

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one raw logit per passage, aligned by index. An empty
        ``passages`` short-circuits without touching the model.
        """
        if not passages:
            return []
        try:
            raw = self.model.predict([(query, passage) for passage in passages])
        except Exception as exc:  # noqa: BLE001 — encode failures are SDK-internal
            raise RelevanceScorerError(
                f"Failed to score {len(passages)} passage(s) with {self.model_name!r}: {exc}"
            ) from exc
        return [float(value) for value in raw]


_relevance_scorer_cache: CrossEncoderRelevanceScorer | None = None
_relevance_scorer_lock = threading.Lock()


def get_relevance_scorer() -> CrossEncoderRelevanceScorer:
    """Return the process-local scorer, constructing it once (thread-safe,
    double-checked locking) and reusing it thereafter. Mirrors
    ``get_embedding_service`` exactly, including lazy model loading.
    """
    global _relevance_scorer_cache
    if _relevance_scorer_cache is None:
        with _relevance_scorer_lock:
            if _relevance_scorer_cache is None:
                logger.info("relevance_scorer.cache_initialized")
                _relevance_scorer_cache = CrossEncoderRelevanceScorer()
    return _relevance_scorer_cache


def reset_relevance_scorer_cache() -> None:
    """Drop the cached instance so the next ``get_relevance_scorer`` builds a
    fresh one. Exposed for tests; production has no invalidation trigger
    (same contract as ``reset_embedding_service_cache``).
    """
    global _relevance_scorer_cache
    with _relevance_scorer_lock:
        _relevance_scorer_cache = None


# ── Retrieval confidence gate ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalGateDecision:
    """Immutable record of one gate evaluation. ``score`` is ``None`` when the
    gate could not run (disabled, no results, scorer failure), in which case
    ``accepted`` is always True -- the gate fails OPEN so a broken relevance
    backend degrades to today's behavior instead of refusing every query.
    """

    accepted: bool
    score: float | None
    threshold: float
    reason: str


def evaluate_retrieval_gate(
    query: str,
    passages: Sequence[str],
    *,
    scorer: RelevanceScorer | None = None,
    threshold: float | None = None,
) -> RetrievalGateDecision:
    """Decide whether retrieval surfaced anything genuinely relevant to
    ``query``, using the best cross-encoder score across ``passages``.

    Measured on 56 gold queries, 28 positives and 28 negatives, retrieval via
    the production ``retrieve(expand=True, rerank=True)`` path:

    ```
    gate                        recall   FPR(hard neg)   precision
    cosine >= 0.40 (current)     1.000       0.900          0.553
    cosine >= 0.60               0.846       0.450          0.710
    cosine >= 0.70               0.654       0.200          0.810
    cross-encoder >= 2.0         0.885       0.150          0.885
    ```

    The cross-encoder dominates rather than trades: at higher recall than
    cosine at 0.60 it has a third of the false-accept rate, and cosine only
    reaches a comparable FPR by collapsing recall to 0.654. ``2.0`` is the
    knee of that curve -- above it recall falls sharply for no FPR gain.

    Caveats carried deliberately: one corpus, one run, 20 hard negatives, and
    the threshold is read off this curve rather than fitted on a held-out
    split. Treat 0.885/0.150 as "measured once here", not as a guarantee.
    """
    if not settings.retrieval_gate_enabled:
        return RetrievalGateDecision(True, None, 0.0, "retrieval gate disabled")
    cut = settings.retrieval_gate_threshold if threshold is None else threshold
    if not passages:
        return RetrievalGateDecision(True, None, cut, "no retrieved passages to score")

    try:
        scores = (scorer or get_relevance_scorer()).score(query, passages)
    except RelevanceScorerError:
        logger.warning("retrieval_gate.scoring_failed", exc_info=True)
        return RetrievalGateDecision(True, None, cut, "relevance scoring failed; gate failed open")

    best = max(scores, default=None)
    if best is None:
        return RetrievalGateDecision(True, None, cut, "scorer returned no scores")
    accepted = best >= cut
    verdict = "at or above" if accepted else "below"
    return RetrievalGateDecision(
        accepted, best, cut,
        f"best relevance {best:.2f} is {verdict} threshold {cut:.2f}",
    )
