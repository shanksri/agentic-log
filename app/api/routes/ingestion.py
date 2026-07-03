from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_api_key
from app.api.dependencies import DbSession
from app.api.rate_limit import RATE_LIMIT_RESPONSES, ingestion_rate_limit
from app.api.schemas import (
    GitHubIngestRequest,
    GitHubIngestResponse,
    JiraIngestRequest,
    JiraIngestResponse,
)
from app.services.incident_ingestion import IncidentIngestionService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
    dependencies=[Depends(require_api_key), Depends(ingestion_rate_limit)],
    responses=RATE_LIMIT_RESPONSES,
)

# Both endpoints call out to a third-party API (GitHub/Jira) through the
# ingestion service's collectors. A connectivity failure there is an
# upstream-service problem, not a bug in this request — report it as 502
# rather than an unhandled 500, and never echo the raw exception (which may
# include request URLs/tokens) back to the client.


@router.post("/github", response_model=GitHubIngestResponse)
def ingest_github(request: GitHubIngestRequest, db: DbSession) -> GitHubIngestResponse:
    try:
        result = IncidentIngestionService(db).ingest_github_repo(
            request.owner,
            request.repo,
            state=request.state,
            limit=request.limit,
            include_comments=request.include_comments,
        )
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.exception("GitHub ingestion failed for %s/%s", request.owner, request.repo)
        raise HTTPException(status_code=502, detail="GitHub ingestion failed.") from exc
    return GitHubIngestResponse.model_validate(result)


@router.post("/jira", response_model=JiraIngestResponse)
def ingest_jira(request: JiraIngestRequest, db: DbSession) -> JiraIngestResponse:
    try:
        result = IncidentIngestionService(db).ingest_jira_project(
            base_url=request.base_url,
            project_key=request.project_key,
            limit=request.limit,
            force_backfill=request.force_backfill,
        )
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.exception("Jira ingestion failed for project %s", request.project_key)
        raise HTTPException(status_code=502, detail="Jira ingestion failed.") from exc
    return JiraIngestResponse.model_validate(result)
