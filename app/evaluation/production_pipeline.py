"""Hybrid Production Pipeline Adapter (Phase 17D).

Phase 17C's ``HybridRetrievalAdapter`` deliberately rejects ``expand=True``/
``rerank=True`` (Hybrid Retrieval, Phase 17B, implements neither). Phase 17D
needs to evaluate Hybrid *with* expansion and reranking layered on top, to
answer "does Hybrid still win once the full production pipeline runs."
Building that without modifying ``HybridRetriever`` or
``IncidentSearchService`` means re-composing the exact same
expand-then-merge-then-rerank algorithm ``IncidentSearchService.retrieve()``
already implements (Phases pre-16) — over ``HybridRetriever.retrieve()``
candidates instead of ``IncidentSearchService.search()`` candidates.

This module implements NONE of the following — it only recombines already-
built primitives:

- **No new retrieval algorithm.** Candidate generation is entirely
  ``HybridRetriever.retrieve()`` (Phase 17B, unmodified). Expansion is
  entirely ``LLMService.expand_search_query`` (pre-16, unmodified).
  Reranking is entirely ``LLMService.rerank_incident_search_results``
  (pre-16, unmodified).
- **No production wiring.** Nothing here is imported by API routes or the
  investigation agent. This adapter exists only so the Phase 16D harness
  can evaluate "Hybrid + Expansion + Rerank" as a configuration.

# Why this mirrors ``IncidentSearchService.retrieve()`` almost exactly

For the Phase 17D comparison to mean anything, "Hybrid + Expansion +
Rerank" must apply expansion and reranking the *same way* dense's existing
production pipeline does — same candidate-pool sizing
(``25`` when expanding/reranking, else ``limit``), same "keep the best
score across phrases" merge rule, same reranker-failure fallback (sorted
candidates, not an exception). Re-deriving a different expand/merge/rerank
policy for Hybrid would confound "does Hybrid benefit from these stages"
with "does this new, different policy behave differently than dense's" —
exactly the kind of confound Phase 17C's "isolate one variable at a time"
methodology was built to avoid.

# Why BM25-only candidates need a DB fetch for reranking payloads

``HybridSearchResult.dense_result`` is ``None`` for a candidate that only
came from BM25 (Phase 17B) — there is no ``Incident`` object attached for
the reranker's payload (title, symptoms, owner, ...) the way there is for
dense-sourced candidates. Rather than reranking BM25-only candidates with
a degraded, mostly-empty payload (which would unfairly bias the reranker
against exactly the candidates Phase 17C found BM25 contributes uniquely),
this adapter fetches the missing ``Incident`` row from ``db`` by id for any
candidate lacking ``dense_result`` — a one-off lookup per such candidate,
bounded by the (small, <=25) candidate pool, not a corpus-wide query.

# Pipeline lifecycle

```
HybridProductionAdapter(db, hybrid_retriever, llm_service)
  .retrieve(query, *, limit, expand, rerank, call_site)
    1. candidate_limit = 25 if (expand or rerank) else limit
    2. phrases = [query] + (llm.expand_search_query(query) if expand else [])
       (llm_service=None -> expand_search_query never called, phrases=[query];
        same graceful-degradation rule as IncidentSearchService._expand_query)
    3. for each phrase: hybrid_retriever.retrieve(phrase, limit=candidate_limit)
       merge into one candidate map, keyed by document_id, keeping the
       HIGHER rrf_score on a repeat (RRF: higher is better, unlike dense's
       distance where lower is better — the only sign flip versus
       IncidentSearchService._merge_candidates)
    4. candidates = sorted by rrf_score descending
    5. if rerank: LLM reranks the candidate pool (payloads built via
       Incident lookup, see above); falls back to score order on any
       exception or absent llm_service — identical fallback contract to
       IncidentSearchService._rerank
    6. -> top `limit`, converted to IncidentSearchResult-shaped objects
       (incident is a SimpleNamespace exposing only .id, matching Phase
       17C's BM25RetrievalAdapter/HybridRetrievalAdapter convention — the
       harness only ever reads result.incident.id)
```
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Incident
from app.services.hybrid_search import HybridRetriever, HybridSearchResult
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchResult

_EXPAND_CANDIDATE_LIMIT = 25


def _to_incident_search_result(document_id: str, score: float) -> IncidentSearchResult:
    return IncidentSearchResult(
        incident=SimpleNamespace(id=uuid.UUID(document_id)), distance=-score
    )


class HybridProductionAdapter:
    """Adapts ``HybridRetriever`` to the harness's
    ``.db`` + ``.retrieve(query, *, limit, expand, rerank, call_site)``
    contract, with expansion and reranking layered on top exactly as
    ``IncidentSearchService.retrieve()`` layers them on dense (see module
    docstring). Unlike Phase 17C's ``HybridRetrievalAdapter``, this adapter
    accepts ``expand=True``/``rerank=True``.
    """

    def __init__(
        self, db: Session, hybrid: HybridRetriever, llm_service: LLMService | None
    ) -> None:
        self.db = db
        self._hybrid = hybrid
        self._llm_service = llm_service

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        expand: bool = False,
        rerank: bool = False,
        call_site: str | None = None,
    ) -> list[IncidentSearchResult]:
        candidate_limit = _EXPAND_CANDIDATE_LIMIT if (expand or rerank) else limit

        phrases = [query]
        if expand:
            phrases += self._expand_query(query)

        candidate_map: dict[str, HybridSearchResult] = {}
        for phrase in phrases:
            results = self._hybrid.retrieve(phrase, limit=candidate_limit)
            self._merge_candidates(candidate_map, results)

        candidates = sorted(candidate_map.values(), key=lambda r: -r.rrf_score)

        if rerank:
            selected = self._rerank(query=query, candidates=candidates, limit=limit)
        else:
            selected = candidates[:limit]

        return [_to_incident_search_result(r.document_id, r.rrf_score) for r in selected]

    def _expand_query(self, query: str) -> list[str]:
        if self._llm_service is None:
            return []
        return self._llm_service.expand_search_query(query)

    def _merge_candidates(
        self, candidate_map: dict[str, HybridSearchResult], results: list[HybridSearchResult]
    ) -> None:
        for result in results:
            existing = candidate_map.get(result.document_id)
            if existing is None or result.rrf_score > existing.rrf_score:
                candidate_map[result.document_id] = result

    def _rerank(
        self, *, query: str, candidates: list[HybridSearchResult], limit: int
    ) -> list[HybridSearchResult]:
        candidates = sorted(candidates, key=lambda r: -r.rrf_score)
        if not candidates or self._llm_service is None:
            return candidates[:limit]

        try:
            payloads = [
                self._candidate_payload(index=index, result=result)
                for index, result in enumerate(candidates, start=1)
            ]
            selected_ids = self._llm_service.rerank_incident_search_results(
                query=query, candidates=payloads, limit=limit
            )
        except Exception:
            return candidates[:limit]

        result_by_id = {
            str(index): result for index, result in enumerate(candidates, start=1)
        }
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

    def _candidate_payload(self, *, index: int, result: HybridSearchResult) -> dict[str, Any]:
        incident = self._incident_for(result)
        symptoms = [symptom.text for symptom in incident.symptoms] if incident else []
        return {
            "candidate_id": str(index),
            "title": incident.title if incident else None,
            "owner": incident.owner if incident else None,
            "repo": incident.repo if incident else None,
            "source": incident.source if incident else None,
            "state": incident.state if incident else None,
            "symptoms": symptoms,
            "severity": incident.severity if incident else None,
            "status": incident.status if incident else None,
            "resolution_summary": incident.resolution_summary if incident else None,
            "hybrid_score": result.rrf_score,
        }

    def _incident_for(self, result: HybridSearchResult) -> Incident | None:
        if result.dense_result is not None:
            return result.dense_result.incident
        return self.db.get(Incident, uuid.UUID(result.document_id))
