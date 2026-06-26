from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import DbSession
from app.api.schemas import (
    GitHubIngestRequest,
    GitHubIngestResponse,
    JiraIngestRequest,
    JiraIngestResponse,
)
from app.services.incident_ingestion import IncidentIngestionService

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/github", response_model=GitHubIngestResponse)
def ingest_github(request: GitHubIngestRequest, db: DbSession) -> GitHubIngestResponse:
    result = IncidentIngestionService(db).ingest_github_repo(
        request.owner,
        request.repo,
        state=request.state,
        limit=request.limit,
        include_comments=request.include_comments,
    )
    return GitHubIngestResponse.model_validate(result)


@router.post("/jira", response_model=JiraIngestResponse)
def ingest_jira(request: JiraIngestRequest, db: DbSession) -> JiraIngestResponse:
    result = IncidentIngestionService(db).ingest_jira_project(
        base_url=request.base_url,
        project_key=request.project_key,
        limit=request.limit,
        force_backfill=request.force_backfill,
    )
    return JiraIngestResponse.model_validate(result)
