"""Adaptive Retrieval Routing Framework (Phase 18A).

Introduces the routing *architecture* — the extension point Phase 17D's
findings argued for (different query types benefit from different
retrieval strategies) — without committing to any particular routing
algorithm. This phase implements one simple, deterministic, fully
explainable policy (``DefaultRuleBasedRoutingPolicy``); it does NOT
implement an ML classifier or an LLM-based router, and the rule
thresholds below are illustrative v1 defaults, not tuned values — see
"Design decisions" for why optimizing them is explicitly out of scope
here.

This module implements NONE of the following — it only decides which
already-built retrieval primitive a query should use, never how that
primitive itself works:

- **No retrieval.** ``RoutingEngine``/``RoutingPolicy`` never call
  ``IncidentSearchService``, ``BM25Retriever``, or ``HybridRetriever``
  (Phases pre-16, 17A, 17B). They return a ``RoutingStrategy`` label —
  ``DENSE``/``BM25``/``HYBRID`` — and nothing more.
- **No expansion/reranking decision.** Per this phase's explicit
  "Integration" constraint, the engine decides only the retrieval
  strategy. ``expand``/``rerank`` remain controlled by existing
  configuration (``EvaluationConfig``, or a future production config),
  completely outside this module's concerns.
- **No production wiring.** Nothing here is imported by
  ``IncidentSearchService``, the API routes, or the investigation agent.
  Wiring a ``RoutingDecision`` to an actual retriever instance is
  integration glue a future phase or caller performs (e.g. via Phase 17C's
  ``build_strategy(decision.strategy.value, ...)`` — see
  ``tests/unit/test_routing.py``'s integration tests for exactly that,
  demonstrated without this module importing
  ``app.evaluation.retrieval_strategies`` at all).

# Updated architecture

```
                                  Query
                                    │
                                    ▼
                          RoutingEngine.route(query)
                              │         │
                    extract_routing_signals()
                              │         │
                              ▼         ▼
                      RoutingSignals  (query, token_count,
                                        has_stack_trace,
                                        has_exact_error_signature,
                                        has_quoted_identifier,
                                        lexical_density)
                                    │
                                    ▼
                    policy.decide(query, signals)     [RoutingPolicy —
                                    │                   THIS PHASE: only
                                    │                   DefaultRuleBasedRoutingPolicy
                                    │                   is implemented]
                                    ▼
                            RoutingDecision
                          (strategy, reason, signals)
                                    │
                                    ▼
                  Dense   or   BM25   or   Hybrid        (NOT decided here:
                                    │                      which retriever
                                    ▼                      instance to use —
                            Existing Pipeline               that's the
                                    │                       caller's job)
                                    ▼
                       Expansion (if enabled)
                                    │
                                    ▼
                        Reranking (if enabled)
                                    │
                                    ▼
                              Confidence
```

# Routing lifecycle

```
RoutingEngine(policy=DefaultRuleBasedRoutingPolicy())
  .route(query)
    1. extract_routing_signals(query) -> RoutingSignals
       (pure, deterministic, regex/tokenization only — no I/O, no LLM,
        no database, no embedding call; computed once, shared by whatever
        policy is plugged in, so every future policy receives the same
        signal vocabulary rather than re-deriving its own)
    2. policy.decide(query, signals) -> RoutingDecision
    3. return RoutingDecision unchanged
```

# Routing decision flow (``DefaultRuleBasedRoutingPolicy``)

Rules are evaluated in a fixed priority order; the first rule that matches
wins, so the policy is always deterministic — there is never a tie to
break arbitrarily:

```
1. has_stack_trace            -> BM25    (exact frame text; embeddings
                                            blur stack traces, lexical
                                            matching does not)
2. has_exact_error_signature  -> BM25    (precise tokens like
                                            "ValidationError" or a ticket
                                            id like "KAFKA-17" are exactly
                                            what BM25's idf-weighted exact
                                            match rewards)
3. has_quoted_identifier      -> BM25    (a quoted/backtick'd identifier is
                                            an explicit signal the user
                                            wants an exact string match)
4. token_count <= 3           -> BM25    (very short queries read as
                                            keyword lookups, not natural-
                                            language descriptions)
5. token_count >= 12          -> HYBRID  (long, multi-clause queries read
                                            as multi-concept; Phase 17C/17D
                                            found Hybrid's fused candidates
                                            specifically help this category)
6. (none of the above)        -> DENSE   (default: a natural-language,
                                            single-concept query — dense
                                            semantic similarity is the
                                            right default for paraphrase-
                                            style queries per Phase 17C)
```

# Policy interface design

``RoutingPolicy`` is an ``ABC`` with a single method,
``decide(query: str, signals: RoutingSignals) -> RoutingDecision``. Both
the raw query text *and* the precomputed signals are passed — not just
one or the other — because no single representation serves every future
policy type this phase is required to keep swappable without changing
``RoutingEngine``:

- A rule-based policy (this phase) only needs ``signals`` and can ignore
  ``query`` entirely.
- A future ``MLRoutingPolicy`` would likely treat ``signals`` (or a
  superset of it) as a feature vector and also ignore raw ``query`` text.
- A future ``LLMRoutingPolicy`` would want the raw ``query`` text (to
  prompt a model) and could ignore ``signals``, or use both.
- A future ``EnsembleRoutingPolicy`` can hold several ``RoutingPolicy``
  instances and combine their individual ``RoutingDecision``s — it is
  itself just another ``RoutingPolicy``, requiring no change to
  ``RoutingEngine``.

``RoutingEngine`` depends only on this interface (constructor injection,
not a hardcoded policy), and computes ``RoutingSignals`` itself rather
than delegating extraction to the policy — this guarantees every policy
sees the same signal vocabulary computed the same way, so swapping
policies never changes what "stack trace" or "token count" means, only
what decision is made from them.

# Why ``RoutingEngine`` never touches a retriever

Per this phase's explicit integration constraint, the engine's output is a
label (``RoutingStrategy.DENSE``/``BM25``/``HYBRID``), not a retriever
instance. Constructing or dispatching to an actual
``IncidentSearchService``/``BM25Retriever``/``HybridRetriever`` would
couple this module to Phase 17A/17B/17C's concrete implementations and to
however a caller wants those instances built (e.g. with which
``EmbeddingService``, which database session) — none of which routing
needs to know. ``RoutingStrategy``'s string values (``"dense"``/
``"bm25"``/``"hybrid"``) deliberately match Phase 17C's
``StrategyName``/``build_strategy()`` vocabulary exactly, so a caller that
*does* want to dispatch can do
``build_strategy(decision.strategy.value, ...)`` directly — but that
wiring lives in the caller (demonstrated only in this phase's integration
tests), never inside this module.

# Design decisions

- **Why these five signals.** Stack traces, exact error signatures, and
  quoted identifiers are all "this query contains a literal string that
  must match exactly" signals — the conceptual opposite of what dense
  embeddings are good at and exactly what BM25's exact-token matching
  rewards (the same reasoning Phase 17A's own docstring gives for why BM25
  exists at all). Token count is the simplest available proxy for query
  complexity, used by both Phase 0's gold-set planning and Phase 16B's
  category taxonomy ("multi-concept" queries are definitionally longer/
  more compound than "lexical-overlap" ones). Lexical density is computed
  and exposed on every ``RoutingSignals`` even though the default policy
  does not use it — a deliberate choice so a future policy can use it
  without this module needing a signal-extraction change.
- **Why the thresholds (3, 12 tokens) are not "tuned."** This phase's
  explicit instruction is "do not optimize the routing rules" — these
  values are reasonable, documented defaults chosen for explainability
  (a human reviewing a routing decision can immediately see why a 2-word
  query routed to BM25), not values fit to Phase 17C/17D's gold dataset or
  any other empirical tuning process. A future phase that wants to tune
  them can do so by replacing ``DefaultRuleBasedRoutingPolicy`` (or
  parameterizing it) without touching ``RoutingEngine`` — exactly the
  swap this architecture is built to support.
- **Why rules are priority-ordered rather than a scoring/voting scheme.**
  A priority order is the simplest structure that is still fully
  deterministic and fully explainable — "the first matching rule wins" is
  a one-sentence description of the entire policy's behavior. A
  scoring/voting scheme would be a step toward an ML-style policy, which
  this phase explicitly defers.
- **Why ``RoutingDecision`` carries the originating ``signals``.** A
  decision that cannot be traced back to the inputs that produced it is
  not explainable — keeping ``signals`` on the decision lets a caller (or
  a test) verify *why* a particular strategy was chosen without
  re-extracting signals itself.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# ── Strategy label ─────────────────────────────────────────────────────────────


class RoutingStrategy(str, Enum):
    """The only three things ``RoutingEngine`` may decide. Values
    deliberately match Phase 17C's ``StrategyName``/``build_strategy()``
    vocabulary (``"dense"``/``"bm25"``/``"hybrid"``) so a caller can
    dispatch a decision directly without a translation step — see module
    docstring's "Why RoutingEngine never touches a retriever".
    """

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


# ── Signals ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoutingSignals:
    """Observable, precomputed characteristics of one query. Computed once
    by ``RoutingEngine.route()`` (via ``extract_routing_signals``) and
    handed to every policy, so policies never re-derive their own
    tokenization/regex logic — see module docstring's "Policy interface
    design".
    """

    query: str
    token_count: int
    has_exact_error_signature: bool
    has_stack_trace: bool
    has_quoted_identifier: bool
    lexical_density: float


_WORD_PATTERN = re.compile(r"\w+")

_STACK_TRACE_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"\bat\s+[\w.$]+\([\w.<>$]*(?:\.java)?(?::\d+)?\)"),
    re.compile(r'File "[^"]+", line \d+'),
)

_ERROR_SIGNATURE_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception|Fault|Timeout)\b")
_TICKET_LIKE_PATTERN = re.compile(r"\b[A-Z]{2,}-\d+\b")

_QUOTED_IDENTIFIER_PATTERN = re.compile(r"`[^`\s]+`|\"[\w./_:-]+\"|'[\w./_:-]+'")


def _has_stack_trace(query: str) -> bool:
    return any(pattern.search(query) for pattern in _STACK_TRACE_PATTERNS)


def _has_exact_error_signature(query: str) -> bool:
    return bool(_ERROR_SIGNATURE_PATTERN.search(query) or _TICKET_LIKE_PATTERN.search(query))


def _has_quoted_identifier(query: str) -> bool:
    return bool(_QUOTED_IDENTIFIER_PATTERN.search(query))


def extract_routing_signals(query: str) -> RoutingSignals:
    """Pure, deterministic signal extraction: tokenization + regex
    matching only. No I/O, no LLM call, no embedding call, no database
    access — every signal here is computable from the query string alone.
    """
    tokens = _WORD_PATTERN.findall(query.lower())
    token_count = len(tokens)
    lexical_density = (len(set(tokens)) / token_count) if token_count else 0.0
    return RoutingSignals(
        query=query,
        token_count=token_count,
        has_exact_error_signature=_has_exact_error_signature(query),
        has_stack_trace=_has_stack_trace(query),
        has_quoted_identifier=_has_quoted_identifier(query),
        lexical_density=lexical_density,
    )


# ── Decision ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoutingDecision:
    """The result of routing one query: which strategy to use, a short
    human-readable reason (so a decision is always explainable, never a
    black box), and the ``RoutingSignals`` that produced it.
    """

    strategy: RoutingStrategy
    reason: str
    signals: RoutingSignals


# ── Policy interface ───────────────────────────────────────────────────────────


class RoutingPolicy(ABC):
    """The swappable extension point. Any future policy (rule-based, ML,
    LLM, ensemble) implements this single method — see module docstring's
    "Policy interface design" for why both ``query`` and ``signals`` are
    passed.
    """

    @abstractmethod
    def decide(self, query: str, signals: RoutingSignals) -> RoutingDecision:
        """Return a ``RoutingDecision`` for ``query``, given its
        precomputed ``signals``.
        """


# ── Default rule-based policy ───────────────────────────────────────────────────


class DefaultRuleBasedRoutingPolicy(RoutingPolicy):
    """Simple, deterministic, fully explainable v1 policy. See module
    docstring's "Routing decision flow" for the exact rule order and
    "Design decisions" for why the thresholds below are illustrative
    defaults, not tuned values.
    """

    SHORT_QUERY_TOKEN_THRESHOLD = 3
    LONG_QUERY_TOKEN_THRESHOLD = 12

    def decide(self, query: str, signals: RoutingSignals) -> RoutingDecision:
        if signals.has_stack_trace:
            return RoutingDecision(
                strategy=RoutingStrategy.BM25,
                reason="stack trace detected — exact frame text favors lexical matching",
                signals=signals,
            )
        if signals.has_exact_error_signature:
            return RoutingDecision(
                strategy=RoutingStrategy.BM25,
                reason="exact error signature detected — lexical matching rewards precise terms",
                signals=signals,
            )
        if signals.has_quoted_identifier:
            return RoutingDecision(
                strategy=RoutingStrategy.BM25,
                reason="quoted identifier detected — favors exact lexical matching",
                signals=signals,
            )
        if signals.token_count <= self.SHORT_QUERY_TOKEN_THRESHOLD:
            return RoutingDecision(
                strategy=RoutingStrategy.BM25,
                reason=(
                    f"short query (<= {self.SHORT_QUERY_TOKEN_THRESHOLD} tokens) — "
                    "reads as a keyword lookup"
                ),
                signals=signals,
            )
        if signals.token_count >= self.LONG_QUERY_TOKEN_THRESHOLD:
            return RoutingDecision(
                strategy=RoutingStrategy.HYBRID,
                reason=(
                    f"long query (>= {self.LONG_QUERY_TOKEN_THRESHOLD} tokens) — "
                    "reads as multi-concept, benefits from combining lexical and semantic signals"
                ),
                signals=signals,
            )
        return RoutingDecision(
            strategy=RoutingStrategy.DENSE,
            reason="no strong lexical signal — default to dense semantic retrieval",
            signals=signals,
        )


# ── Engine ─────────────────────────────────────────────────────────────────────


class RoutingEngine:
    """Depends only on ``RoutingPolicy`` (constructor injection) — never
    on a concrete policy implementation. See module docstring's "Updated
    architecture" for where this sits in the pipeline.
    """

    def __init__(self, policy: RoutingPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    def route(self, query: str) -> RoutingDecision:
        signals = extract_routing_signals(query)
        return self._policy.decide(query, signals)
