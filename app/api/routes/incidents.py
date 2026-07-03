from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from app.api.auth import require_api_key
from app.api.dependencies import DbSession
from app.api.rate_limit import RATE_LIMIT_RESPONSES, incidents_rate_limit
from app.api.schemas import IncidentResponse
from app.api.validation import validate_uuid
from app.db.models import Incident

router = APIRouter(
    prefix="/incidents",
    tags=["incidents"],
    dependencies=[Depends(require_api_key), Depends(incidents_rate_limit)],
    responses=RATE_LIMIT_RESPONSES,
)


@router.get("", response_model=list[IncidentResponse])
def list_incidents(db: DbSession, limit: int = Query(default=50, ge=1, le=200)) -> list[Incident]:
    return list(
        db.scalars(select(Incident).order_by(Incident.created_at.desc()).limit(limit))
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str, db: DbSession) -> Incident:
    parsed_id = validate_uuid(incident_id, field_name="incident_id")
    incident = db.get(Incident, parsed_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident
