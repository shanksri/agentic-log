from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import Embedding, Incident
from app.services.candidate_pipeline import CandidatePipeline
from app.services.confidence import classify_confidence
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IncidentSearchResult:
    incident: Incident
    distance: float

    @property
    def similarity_score(self) -> float:
        return max(0.0, 1.0 - self.distance)


class IncidentSearchService:
    def __init__(
        self,
        db: Session,
        *,
        embedding_service: EmbeddingService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service
        self._pipeline = CandidatePipeline(llm_service=self.llm_service)

    def search(
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
        call_site: str | None = None,
    ) -> list[IncidentSearchResult]:
        started_at = time.monotonic()
        query_vector = self.embedding_service.embed_text(query)
        distance = Embedding.embedding.cosine_distance(query_vector).label("distance")

        statement: Select = (
            select(Incident, distance)
            .join(Embedding, Embedding.incident_id == Incident.id)
            .options(joinedload(Incident.symptoms))
            .where(Embedding.model_name == self.embedding_service.model_name)
            .order_by(distance)
            .limit(limit)
        )

        if source_type:
            statement = statement.where(Incident.source_type == source_type)
        if tags:
            statement = statement.where(Incident.tags.overlap(tags))
        if owner:
            statement = statement.where(Incident.owner == owner)
        if repo:
            statement = statement.where(Incident.repo == repo)
        if source:
            statement = statement.where(Incident.source == source)
        if state:
            statement = statement.where(Incident.state == state)

        rows = self.db.execute(statement).unique().all()
        results = [IncidentSearchResult(incident=row[0], distance=float(row[1])) for row in rows]

        duration_ms = (time.monotonic() - started_at) * 1000
        scores = [result.similarity_score for result in results]
        top1_score = scores[0] if scores else None
        logger.info(
            "retrieval.search",
            extra={
                "call_site": call_site or "unknown",
                "search_config": {"expansion": False, "reranking": False, "hybrid": False},
                "result_count": len(results),
                "top1_score": top1_score,
                "confidence_level": classify_confidence(top1_score),
                "top5_mean_score": sum(scores[:5]) / len(scores[:5]) if scores[:5] else None,
                "duration_ms": round(duration_ms, 2),
                "filters": {
                    "source_type": source_type,
                    "tags": tags,
                    "owner": owner,
                    "repo": repo,
                    "source": source,
                    "state": state,
                },
            },
        )
        return results

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
        """Canonical retrieval pipeline: dense search, with optional query
        expansion and LLM reranking layered on top.

        This is the single entry point all callers (search routes and
        investigation agents) should use going forward. `search()` remains
        available unchanged for backward compatibility and is used here as
        the base candidate-generation step.

        - expand=False, rerank=False: identical to plain `search()`.
        - expand=True: also searches LLM-generated related phrases and
          merges candidates, keeping the best distance per incident.
        - rerank=True: LLM reranks the candidate pool; falls back to
          distance order if no `llm_service` is configured or the LLM call
          fails.
        """
        started_at = time.monotonic()

        def generate(phrase: str, candidate_limit: int) -> list[IncidentSearchResult]:
            return self.search(
                phrase,
                limit=candidate_limit,
                source_type=source_type,
                tags=tags,
                owner=owner,
                repo=repo,
                source=source,
                state=state,
                call_site=call_site,
            )

        candidates, phrases = self._pipeline.generate_candidates(
            query, generate=generate, limit=limit, expand=expand, rerank=rerank
        )

        if rerank:
            try:
                results = self._pipeline.rerank(query=query, candidates=candidates, limit=limit)
            except Exception:
                logger.exception(
                    "retrieval.retrieve.rerank_failed",
                    extra={"call_site": call_site or "unknown"},
                )
                results = candidates[:limit]
        else:
            results = candidates[:limit]

        duration_ms = (time.monotonic() - started_at) * 1000
        scores = [result.similarity_score for result in results]
        top1_score = scores[0] if scores else None
        logger.info(
            "retrieval.retrieve",
            extra={
                "call_site": call_site or "unknown",
                "search_config": {
                    "expansion": expand,
                    "reranking": rerank,
                    "hybrid": False,
                },
                "expansion_phrase_count": len(phrases),
                "candidate_count": len(candidates),
                "result_count": len(results),
                "top1_score": top1_score,
                "confidence_level": classify_confidence(top1_score),
                "top5_mean_score": sum(scores[:5]) / len(scores[:5]) if scores[:5] else None,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return results

    def search_debug(
        self,
        query: str,
        *,
        owner: str | None = None,
        repo: str | None = None,
        source: str | None = None,
        state: str | None = None,
        call_site: str | None = None,
    ) -> list[IncidentSearchResult]:
        """Backward-compatible alias for the canonical retrieval pipeline
        with expansion and reranking enabled, limited to 5 results."""
        return self.retrieve(
            query,
            limit=5,
            owner=owner,
            repo=repo,
            source=source,
            state=state,
            expand=True,
            rerank=True,
            call_site=call_site,
        )

    @staticmethod
    def confidence_for(results: list[IncidentSearchResult]) -> tuple[float | None, str]:
        """Return (top1_score, confidence_level) for a result set.

        top1_score is None (confidence LOW) when no results were retrieved.
        """
        top1_score = results[0].similarity_score if results else None
        return top1_score, classify_confidence(top1_score)

