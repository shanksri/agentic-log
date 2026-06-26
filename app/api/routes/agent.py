from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import DbSession
from app.api.schemas import (
    AdvancedInvestigationRequest,
    AdvancedInvestigationResponse,
    InvestigationRequest,
    InvestigationResponse,
)
from app.services.advanced_investigation_agent import AdvancedInvestigationAgent
from app.services.investigation_agent import InvestigationAgent

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/investigate", response_model=InvestigationResponse)
def investigate(request: InvestigationRequest, db: DbSession) -> InvestigationResponse:
    analysis = InvestigationAgent(db).investigate(request.problem)
    return InvestigationResponse(analysis=analysis)


@router.post("/investigate-advanced", response_model=AdvancedInvestigationResponse)
def investigate_advanced(
    request: AdvancedInvestigationRequest,
    db: DbSession,
) -> AdvancedInvestigationResponse:
    result = AdvancedInvestigationAgent(db).investigate(request.problem)
    return AdvancedInvestigationResponse.model_validate(result)
