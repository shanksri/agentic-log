from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import DbSession
from app.api.schemas import (
    SearchDebugRequest,
    SearchDebugResponse,
    SearchDebugResult,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/incidents", response_model=SearchResponse)
def search_incidents(request: SearchRequest, db: DbSession) -> SearchResponse:
    results = IncidentSearchService(db).search(
        request.query,
        limit=request.limit,
        source_type=request.source_type,
        tags=request.tags,
        owner=request.owner,
        repo=request.repo,
        source=request.source,
        state=request.state,
    )
    top1_score, confidence_level = IncidentSearchService.confidence_for(results)
    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                incident=result.incident,
                similarity_score=result.similarity_score,
                distance=result.distance,
            )
            for result in results
        ],
        top1_score=top1_score,
        confidence_level=confidence_level,
    )


@router.post("/debug", response_model=SearchDebugResponse)
def search_debug(request: SearchDebugRequest, db: DbSession) -> SearchDebugResponse:
    results = IncidentSearchService(db, llm_service=LLMService()).search_debug(
        request.query,
        owner=request.owner,
        repo=request.repo,
        source=request.source,
        state=request.state,
    )
    top1_score, confidence_level = IncidentSearchService.confidence_for(results)
    return SearchDebugResponse(
        query=request.query,
        filters={
            "owner": request.owner,
            "repo": request.repo,
            "source": request.source,
            "state": request.state,
        },
        results=[
            SearchDebugResult(
                title=result.incident.title,
                repo=result.incident.repo,
                similarity_score=result.similarity_score,
            )
            for result in results
        ],
        top1_score=top1_score,
        confidence_level=confidence_level,
    )
