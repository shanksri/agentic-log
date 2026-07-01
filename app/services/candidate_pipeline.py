"""Shared candidate-generation pipeline: expand -> generate -> merge -> rerank.

Extracted to eliminate a duplicate implementation of this exact algorithm that
previously existed independently in ``IncidentSearchService.retrieve()`` (dense,
pre-16) and ``RoutedSearchService``'s private ``_ProductionCandidatePipeline``
(BM25/Hybrid, Phase 18B). Both callers now delegate the phrase-expansion,
best-distance merge, and LLM-rerank-with-fallback logic to this module; each
keeps its own outer timing/logging (they emit different log event names and
fields), so this refactor is pure de-duplication with no change in observable
behavior for either caller.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.llm_service import LLMService
    from app.services.search import IncidentSearchResult

DEFAULT_CANDIDATE_LIMIT = 25


class CandidatePipeline:
    """Strategy-agnostic expand/merge/rerank algorithm, parameterized by a
    ``generate(phrase, limit) -> list[IncidentSearchResult]`` callable so it
    can sit in front of dense search, BM25, or Hybrid retrieval identically.
    """

    def __init__(self, *, llm_service: "LLMService | None") -> None:
        self.llm_service = llm_service

    def expand_query(self, query: str) -> list[str]:
        if self.llm_service is None:
            return []
        return self.llm_service.expand_search_query(query)

    def merge(
        self,
        candidate_map: dict[str, "IncidentSearchResult"],
        result: "IncidentSearchResult",
    ) -> None:
        incident_id = str(result.incident.id)
        existing = candidate_map.get(incident_id)
        if existing is None or result.distance < existing.distance:
            candidate_map[incident_id] = result

    def candidate_payload(self, *, index: int, result: "IncidentSearchResult") -> dict[str, Any]:
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

    def generate_candidates(
        self,
        query: str,
        *,
        generate: Callable[[str, int], list["IncidentSearchResult"]],
        limit: int,
        expand: bool,
        rerank: bool,
    ) -> tuple[list["IncidentSearchResult"], list[str]]:
        """Expand ``query`` into phrases (if ``expand``), call ``generate``
        once per phrase, merge into a best-distance-per-incident pool, and
        return the distance-sorted candidate list plus the phrases used (the
        latter so callers can log ``expansion_phrase_count`` themselves).
        """
        candidate_limit = DEFAULT_CANDIDATE_LIMIT if (expand or rerank) else limit
        phrases = [query]
        if expand:
            phrases += self.expand_query(query)

        candidate_map: dict[str, IncidentSearchResult] = {}
        for phrase in phrases:
            for result in generate(phrase, candidate_limit):
                self.merge(candidate_map, result)

        candidates = sorted(candidate_map.values(), key=lambda r: r.distance)
        return candidates, phrases

    def rerank(
        self,
        *,
        query: str,
        candidates: list["IncidentSearchResult"],
        limit: int,
    ) -> list["IncidentSearchResult"]:
        candidates = sorted(candidates, key=lambda r: r.distance)
        if not candidates or self.llm_service is None:
            return candidates[:limit]

        payloads = [
            self.candidate_payload(index=index, result=result)
            for index, result in enumerate(candidates, start=1)
        ]
        selected_ids = self.llm_service.rerank_incident_search_results(
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
