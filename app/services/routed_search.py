"""Adaptive Routing Integration (Phase 18B).

Activates Phase 18A's ``RoutingEngine`` as the mechanism that chooses
between Dense (pre-16), BM25 (Phase 17A), and Hybrid (Phase 17B) retrieval
for every incoming query — without modifying any of those three, without
modifying ``IncidentSearchService`` (so today's production callers are
completely unaffected unless they explicitly opt in), and without changing
Phase 18A's routing policy/rules, the evaluation framework, or the
benchmark framework.

``RoutedSearchService`` is the new production-facing orchestrator. It is
not wired into any API route or the investigation agent by this phase —
per the Stop Condition, this phase produces the integration point, ready
to be adopted, not the adoption itself.

# Updated architecture

```
                                   Query
                                     │
                                     ▼
                    RoutingEngine.route(query)     [Phase 18A, unmodified
                                     │               — executes exactly
                                     │               once per request]
                                     ▼
                            RoutingDecision
                                     │
                    ┌────────────────┼────────────────────┐
                    ▼ routing        ▼ routing enabled,    ▼ routing enabled,
                 disabled              no filters            filters present
                    │                    │                     │
                    ▼                    ▼                     ▼
                 DENSE              decision.strategy         DENSE
              (today's exact                                (only dense
               production                                   supports
               behavior)                                    filters)
                    │                    │                     │
                    └────────────────────┴─────────────────────┘
                                     ▼
                  effective_strategy: Dense | BM25 | Hybrid
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
   IncidentSearchService    BM25Retriever.retrieve()  HybridRetriever.retrieve()
   .retrieve() — entire      (Phase 17A) wrapped in    (Phase 17B) wrapped in
   expand/merge/rerank       the SAME generic           the SAME generic
   pipeline reused           expand/merge/rerank        expand/merge/rerank
   UNCHANGED                 pipeline dense already     pipeline dense already
                             has, applied generically    has, applied generically
                │                    │                    │
                └────────────────────┼────────────────────┘
                                     ▼
                        list[IncidentSearchResult]
                     (identical shape regardless of
                      which strategy produced it —
                      real Incident, real .distance)
                                     │
                                     ▼
                          Confidence (existing,
                       IncidentSearchService.confidence_for,
                              unmodified, reused as-is)
```

# End-to-end request lifecycle

```
RoutedSearchService.retrieve(query, *, limit, expand, rerank, call_site, ...)
  1. decision = routing_engine.route(query)              [always once,
                                                            even when routing
                                                            is disabled — see
                                                            "Observability"]
  2. effective_strategy = DENSE                            if routing disabled
                         = DENSE                            if filters present
                                                             (BM25/Hybrid don't
                                                              support filters)
                         = decision.strategy                otherwise
  3. record + log a RoutingObservation (decision, override reason, signals)
  4. if effective_strategy == DENSE:
        delegate to IncidentSearchService.retrieve(...) UNCHANGED — dense's
        own expand/merge/rerank algorithm is reused verbatim, not
        reimplemented
     else:
        run the SAME expand/merge/rerank algorithm (see
        ``_ProductionCandidatePipeline``) against BM25/Hybrid candidates
  5. -> list[IncidentSearchResult] (same shape regardless of branch taken)
```

# Routing integration flow

The routing decision affects step 4 ONLY — which candidate-generation
primitive produces the initial pool. Expansion, candidate merging,
reranking, and confidence classification are the exact same algorithm
regardless of which branch was taken in step 4:

- **Dense branch**: delegates entirely to
  ``IncidentSearchService.retrieve()`` (pre-16, unmodified) — that method
  already implements expand/merge/rerank for dense, so there is nothing to
  re-derive.
- **BM25/Hybrid branch**: ``_ProductionCandidatePipeline`` (new, generic,
  parameterized by a ``generate(phrase, limit) -> list[IncidentSearchResult]``
  callable) implements the identical expand/merge/rerank algorithm — same
  candidate-pool sizing rule (25 when expanding/reranking, else ``limit``),
  same "keep the lowest distance on a repeat" merge rule, same reranker
  payload shape, same reranker-failure fallback to distance order. This is
  one implementation shared by BM25 and Hybrid, not two copies — written
  once here rather than duplicating Phase 17D's
  ``HybridProductionAdapter`` (which remains untouched; it is evaluation-
  only infrastructure and is not reused by this module, since reusing it
  would have meant either modifying it to add BM25 support or accepting
  two near-duplicate pipelines — writing one new, strategy-agnostic
  pipeline avoids both).
- **Real ``Incident`` objects, not stand-ins.** Unlike Phase 17C/17D's
  evaluation-only adapters (which only ever needed a candidate's id),
  BM25/Hybrid candidates here are converted to real, DB-fetched
  ``Incident`` objects (``_fetch_incident_result``) before merging — so the
  reranker payload (title, owner, symptoms, ...) and
  ``IncidentSearchService.confidence_for`` see the exact same
  ``IncidentSearchResult`` shape a dense-only caller already gets. This is
  what "the downstream pipeline must receive the same candidate interface
  regardless of retrieval strategy" means concretely: not just the same
  Python type, but the same populated fields.

# Configuration behavior

``RoutedSearchConfig.routing_enabled`` (default ``False``) is the single
opt-in switch. When ``False``, ``RoutedSearchService.retrieve()`` always
takes the dense branch with the caller's exact arguments passed through
unchanged — today's production behavior, verified unchanged by this
phase's regression tests (same results as calling
``IncidentSearchService.retrieve()`` directly). Routing is therefore
purely additive: nothing about existing behavior changes until a caller
explicitly constructs a ``RoutedSearchConfig(routing_enabled=True)``.

**Filters force dense even when routing is enabled.** ``BM25Retriever``
(Phase 17A) and ``HybridRetriever`` (Phase 17B) were both built without
filter support (``source_type``/``tags``/``owner``/``repo``/``source``/
``state``) — Phase 17A/17B's own scope decisions, not something this
phase can retroactively add without touching those modules. Silently
dropping a caller's filter when routed to BM25/Hybrid would be a
correctness bug (the caller asked for a scoped search and would
transparently get an unscoped one), not a "routing optimization" question
— so this is a structural integration rule, decided before consulting the
policy's decision at all, not a change to the routing policy/rules
themselves (which remain exactly as Phase 18A built them).

# Observability

Every call to ``retrieve()`` produces a ``RoutingObservation`` — chosen
(effective) strategy, the policy's raw decision and reason, any override
reason, and the full ``RoutingSignals`` — available two ways:

1. ``RoutedSearchService.last_observation`` — the most recent
   observation, a plain frozen dataclass a caller or test can inspect
   directly, with no log-scraping required.
2. A structured ``logger.info("retrieval.routed_search.routing_decision",
   extra={...})`` call, the same observability convention
   ``IncidentSearchService`` already uses for its own
   ``retrieval.search``/``retrieval.retrieve`` log events.

No dashboard is built (explicitly out of scope) — this is the "make the
information available" half of the requirement, not the "display it"
half.

**Why signals are recorded even when routing is disabled.** This is a
deliberate "shadow observability" choice: computing
``RoutingSignals``/``RoutingDecision`` is pure, side-effect-free
regex/tokenization (no I/O, no LLM, no embedding call — see Phase 18A's
own module docstring), so doing it unconditionally costs negligible
latency and never changes what gets *returned* to the caller. It lets a
team evaluate "what would routing have chosen" against real production
traffic before ever flipping ``routing_enabled`` on — without that, the
first time routing affects anything would be the first time anyone could
observe what it does.

# File-by-file summary

- ``app/services/routed_search.py`` (this file) — ``RoutedSearchConfig``,
  ``RoutingObservation``, ``RoutedSearchService``,
  ``_ProductionCandidatePipeline`` (private). The only new file.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from app.db.models import Incident
from app.services.bm25_search import BM25Retriever
from app.services.hybrid_search import HybridRetriever, HybridSearchResult
from app.services.llm_service import LLMService
from app.services.routing import (
    DefaultRuleBasedRoutingPolicy,
    RoutingEngine,
    RoutingSignals,
    RoutingStrategy,
)
from app.services.search import IncidentSearchResult, IncidentSearchService

logger = logging.getLogger(__name__)

_EXPAND_CANDIDATE_LIMIT = 25


@dataclass(frozen=True)
class RoutedSearchConfig:
    """The single opt-in switch — see module docstring's "Configuration
    behavior". Default ``routing_enabled=False`` preserves today's
    production behavior exactly until a caller opts in.
    """

    routing_enabled: bool = False


@dataclass(frozen=True)
class RoutingObservation:
    """Per-request observability record — see module docstring's
    "Observability". ``policy_strategy`` is what ``RoutingPolicy`` decided;
    ``effective_strategy`` is what was actually used (they differ exactly
    when ``override_reason`` is set: routing disabled, or filters present).
    """

    query: str
    call_site: str | None
    routing_enabled: bool
    policy_strategy: RoutingStrategy
    effective_strategy: RoutingStrategy
    reason: str
    override_reason: str | None
    signals: RoutingSignals


def _fetch_incident_result(db, document_id: str, distance: float) -> IncidentSearchResult | None:
    incident = db.get(Incident, uuid.UUID(document_id))
    if incident is None:
        return None
    return IncidentSearchResult(incident=incident, distance=distance)


def _hybrid_to_incident_result(db, result: HybridSearchResult) -> IncidentSearchResult | None:
    if result.dense_result is not None:
        return IncidentSearchResult(
            incident=result.dense_result.incident, distance=-result.rrf_score
        )
    return _fetch_incident_result(db, result.document_id, -result.rrf_score)


class _ProductionCandidatePipeline:
    """Strategy-agnostic expand/merge/rerank pipeline, identical in
    behavior to ``IncidentSearchService.retrieve()``'s own algorithm (same
    candidate-pool sizing, same merge rule, same reranker payload shape,
    same reranker-failure fallback) but parameterized by a
    ``generate(phrase, limit) -> list[IncidentSearchResult]`` callable
    instead of being hardcoded to dense's own ``.search()``. Used for the
    BM25 and Hybrid branches only — see module docstring's "Routing
    integration flow".
    """

    def __init__(self, *, llm_service: LLMService | None) -> None:
        self._llm_service = llm_service

    def run(
        self,
        query: str,
        *,
        generate: Callable[[str, int], list[IncidentSearchResult]],
        limit: int,
        expand: bool,
        rerank: bool,
        call_site: str | None,
        strategy_label: str,
    ) -> list[IncidentSearchResult]:
        candidate_limit = _EXPAND_CANDIDATE_LIMIT if (expand or rerank) else limit

        phrases = [query]
        if expand:
            phrases += self._expand_query(query)

        candidate_map: dict[str, IncidentSearchResult] = {}
        for phrase in phrases:
            for result in generate(phrase, candidate_limit):
                self._merge(candidate_map, result)

        candidates = sorted(candidate_map.values(), key=lambda r: r.distance)

        if rerank:
            try:
                results = self._rerank(query=query, candidates=candidates, limit=limit)
            except Exception:
                logger.exception(
                    "retrieval.routed_search.rerank_failed",
                    extra={"call_site": call_site or "unknown", "strategy": strategy_label},
                )
                results = candidates[:limit]
        else:
            results = candidates[:limit]
        return results

    def _expand_query(self, query: str) -> list[str]:
        if self._llm_service is None:
            return []
        return self._llm_service.expand_search_query(query)

    def _merge(
        self, candidate_map: dict[str, IncidentSearchResult], result: IncidentSearchResult
    ) -> None:
        document_id = str(result.incident.id)
        existing = candidate_map.get(document_id)
        if existing is None or result.distance < existing.distance:
            candidate_map[document_id] = result

    def _rerank(
        self, *, query: str, candidates: list[IncidentSearchResult], limit: int
    ) -> list[IncidentSearchResult]:
        candidates = sorted(candidates, key=lambda r: r.distance)
        if not candidates or self._llm_service is None:
            return candidates[:limit]

        payloads = [
            self._payload(index=index, result=result)
            for index, result in enumerate(candidates, start=1)
        ]
        selected_ids = self._llm_service.rerank_incident_search_results(
            query=query, candidates=payloads, limit=limit
        )
        result_by_id = {str(index): result for index, result in enumerate(candidates, start=1)}
        reranked = [
            result_by_id[candidate_id]
            for candidate_id in selected_ids
            if candidate_id in result_by_id
        ]
        if len(reranked) >= limit:
            return reranked[:limit]

        selected_set = {id(result) for result in reranked}
        for candidate in candidates:
            if id(candidate) not in selected_set:
                reranked.append(candidate)
            if len(reranked) >= limit:
                break
        return reranked

    def _payload(self, *, index: int, result: IncidentSearchResult) -> dict[str, object]:
        incident = result.incident
        symptoms = [symptom.text for symptom in incident.symptoms]
        return {
            "candidate_id": str(index),
            "title": incident.title,
            "owner": incident.owner,
            "repo": incident.repo,
            "source": incident.source,
            "state": incident.state,
            "symptoms": symptoms,
            "severity": incident.severity,
            "status": incident.status,
            "resolution_summary": incident.resolution_summary,
            "similarity_score": result.similarity_score,
        }


class RoutedSearchService:
    """Production-facing orchestrator: routes each query to Dense, BM25, or
    Hybrid retrieval via a ``RoutingEngine`` (Phase 18A, unmodified), then
    runs the same expand/merge/rerank algorithm regardless of which
    strategy was chosen. See module docstring for the full architecture.
    """

    def __init__(
        self,
        dense: IncidentSearchService,
        *,
        bm25: BM25Retriever | None = None,
        hybrid: HybridRetriever | None = None,
        routing_engine: RoutingEngine | None = None,
        config: RoutedSearchConfig | None = None,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._hybrid = hybrid
        self._routing_engine = routing_engine or RoutingEngine(DefaultRuleBasedRoutingPolicy())
        self._config = config or RoutedSearchConfig()
        self._pipeline = _ProductionCandidatePipeline(llm_service=dense.llm_service)
        self._last_observation: RoutingObservation | None = None

    @property
    def db(self):
        return self._dense.db

    @property
    def config(self) -> RoutedSearchConfig:
        return self._config

    @property
    def last_observation(self) -> RoutingObservation | None:
        return self._last_observation

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        source_type: str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        repo: str | None = None,
        source: str | None = None,
        state: str | None = None,
        expand: bool = False,
        rerank: bool = False,
        call_site: str | None = None,
    ) -> list[IncidentSearchResult]:
        decision = self._routing_engine.route(query)  # exactly once per request
        has_filters = any([source_type, tags, owner, repo, source, state])

        if not self._config.routing_enabled:
            effective_strategy = RoutingStrategy.DENSE
            override_reason = "routing disabled — preserving existing production behavior"
        elif has_filters:
            effective_strategy = RoutingStrategy.DENSE
            override_reason = (
                "query has source_type/tags/owner/repo/source/state filters — "
                "only dense retrieval supports filters"
            )
        else:
            effective_strategy = decision.strategy
            override_reason = None

        self._last_observation = RoutingObservation(
            query=query,
            call_site=call_site,
            routing_enabled=self._config.routing_enabled,
            policy_strategy=decision.strategy,
            effective_strategy=effective_strategy,
            reason=decision.reason,
            override_reason=override_reason,
            signals=decision.signals,
        )
        self._log_observation(self._last_observation)

        if effective_strategy == RoutingStrategy.DENSE:
            return self._dense.retrieve(
                query,
                limit=limit,
                source_type=source_type,
                tags=tags,
                owner=owner,
                repo=repo,
                source=source,
                state=state,
                expand=expand,
                rerank=rerank,
                call_site=call_site,
            )

        if effective_strategy == RoutingStrategy.BM25:
            generate = self._bm25_generate
        else:
            generate = self._hybrid_generate

        return self._pipeline.run(
            query,
            generate=generate,
            limit=limit,
            expand=expand,
            rerank=rerank,
            call_site=call_site,
            strategy_label=effective_strategy.value,
        )

    @staticmethod
    def confidence_for(results: list[IncidentSearchResult]) -> tuple[float | None, str]:
        return IncidentSearchService.confidence_for(results)

    def _bm25_generate(self, phrase: str, limit: int) -> list[IncidentSearchResult]:
        if self._bm25 is None:
            raise ValueError("BM25 retriever not configured but routing selected BM25")
        results = self._bm25.retrieve(phrase, limit=limit)
        fetched = (
            _fetch_incident_result(self._dense.db, r.document_id, -r.score) for r in results
        )
        return [result for result in fetched if result is not None]

    def _hybrid_generate(self, phrase: str, limit: int) -> list[IncidentSearchResult]:
        if self._hybrid is None:
            raise ValueError("Hybrid retriever not configured but routing selected Hybrid")
        results = self._hybrid.retrieve(phrase, limit=limit)
        fetched = (_hybrid_to_incident_result(self._dense.db, r) for r in results)
        return [result for result in fetched if result is not None]

    def _log_observation(self, observation: RoutingObservation) -> None:
        logger.info(
            "retrieval.routed_search.routing_decision",
            extra={
                "call_site": observation.call_site or "unknown",
                "routing_enabled": observation.routing_enabled,
                "policy_strategy": observation.policy_strategy.value,
                "effective_strategy": observation.effective_strategy.value,
                "routing_rule": observation.reason,
                "override_reason": observation.override_reason,
                "routing_signals": {
                    "token_count": observation.signals.token_count,
                    "has_exact_error_signature": observation.signals.has_exact_error_signature,
                    "has_stack_trace": observation.signals.has_stack_trace,
                    "has_quoted_identifier": observation.signals.has_quoted_identifier,
                    "lexical_density": observation.signals.lexical_density,
                },
            },
        )
