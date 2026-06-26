from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.dependencies import DbSession
from app.api.schemas import IncidentResponse
from app.db.models import Incident

router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.get("", response_model=list[IncidentResponse])
def list_incidents(db: DbSession, limit: int = 50) -> list[Incident]:
    return list(
        db.scalars(select(Incident).order_by(Incident.created_at.desc()).limit(min(limit, 200)))
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str, db: DbSession) -> Incident:
    incident = db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident
