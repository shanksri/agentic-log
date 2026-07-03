from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.dependencies import DbSession
from app.api.schemas import (
    AdvancedInvestigationRequest,
    AdvancedInvestigationResponse,
    InvestigationRequest,
    InvestigationResponse,
    OrchestratedCritique,
    OrchestratedHypothesis,
    OrchestratedInvestigationRequest,
    OrchestratedInvestigationResponse,
)
from app.services.advanced_investigation_agent import AdvancedInvestigationAgent
from app.services.investigation_agent import InvestigationAgent
from app.services.investigation_orchestrator import MultiAgentInvestigationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

# Construction-time failures (missing OPENAI_API_KEY, etc.) are a
# service-unavailable condition — 503; anything raised once the agent is
# already running (mid-investigation LLM/embedding failure) is reported as a
# generic 500 rather than a raw traceback. Neither branch changes any agent's
# reasoning — Phase 23 does not touch agent/orchestrator control flow.


def _run_or_503(build_and_run, *, what: str):
    try:
        return build_and_run()
    except ValueError as exc:
        # Agent/LLMService constructors raise ValueError for missing
        # configuration (e.g. no OPENAI_API_KEY) — that is unavailability,
        # not a request error.
        logger.exception("%s unavailable", what)
        raise HTTPException(status_code=503, detail=f"{what} is temporarily unavailable.") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s failed", what)
        raise HTTPException(status_code=500, detail=f"{what} failed.") from exc


@router.post("/investigate", response_model=InvestigationResponse)
def investigate(request: InvestigationRequest, db: DbSession) -> InvestigationResponse:
    analysis = _run_or_503(
        lambda: InvestigationAgent(db).investigate(request.problem),
        what="Investigation",
    )
    return InvestigationResponse(analysis=analysis)


@router.post("/investigate-advanced", response_model=AdvancedInvestigationResponse)
def investigate_advanced(
    request: AdvancedInvestigationRequest,
    db: DbSession,
) -> AdvancedInvestigationResponse:
    result = _run_or_503(
        lambda: AdvancedInvestigationAgent(db).investigate(request.problem),
        what="Advanced investigation",
    )
    return AdvancedInvestigationResponse.model_validate(result)


@router.post("/investigate-orchestrated", response_model=OrchestratedInvestigationResponse)
def investigate_orchestrated(
    request: OrchestratedInvestigationRequest,
    db: DbSession,
) -> OrchestratedInvestigationResponse:
    """Canonical investigation endpoint: Phase 19A-19D's multi-agent
    orchestrator (planner, evidence-driven hypothesis generation, critic,
    iterative loop — see docs/architecture/19_multi_agent_investigation.md).

    Prefer this over ``/investigate`` and ``/investigate-advanced`` for new
    integrations; those remain available unmodified for existing callers.
    """
    session = _run_or_503(
        lambda: MultiAgentInvestigationOrchestrator(db).investigate(
            request.problem, n_hypotheses=request.n_hypotheses
        ),
        what="Orchestrated investigation",
    )
    investigation = session.final_report.investigation
    critique = session.final_report.critique

    return OrchestratedInvestigationResponse(
        problem=investigation.problem,
        selected_root_cause=(
            investigation.selected_hypothesis.root_cause
            if investigation.selected_hypothesis
            else None
        ),
        confidence=investigation.confidence,
        confidence_level=investigation.confidence_level,
        is_uncertain=investigation.is_uncertain,
        supporting_evidence=list(investigation.supporting_evidence),
        contradicting_evidence=list(investigation.contradicting_evidence),
        remaining_uncertainty=list(investigation.remaining_uncertainty),
        rejected_hypotheses=[
            OrchestratedHypothesis(
                id=hypothesis.id,
                root_cause=hypothesis.root_cause,
                rationale=hypothesis.rationale,
                validation_keywords=list(hypothesis.validation_keywords),
                raw_confidence=hypothesis.raw_confidence,
            )
            for hypothesis in investigation.rejected_hypotheses
        ],
        critique=OrchestratedCritique(
            verdict=critique.verdict.value,
            confidence=critique.confidence,
            explanation=critique.explanation,
            findings=list(critique.findings),
            unresolved_questions=list(critique.unresolved_questions),
            missing_evidence=list(critique.missing_evidence),
            recommended_actions=list(critique.recommended_actions),
        ),
        total_iterations=session.total_iterations,
        stopping_reason=session.stopping_reason.value,
        stop_explanation=session.stop_explanation,
    )
