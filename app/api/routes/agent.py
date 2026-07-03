from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_api_key
from app.api.dependencies import DbSession
from app.api.rate_limit import RATE_LIMIT_RESPONSES, agent_rate_limit
from app.api.schemas import (
    InvestigationRequest,
    InvestigationResponse,
    OrchestratedCritique,
    OrchestratedHypothesis,
)
from app.services.investigation_orchestrator import MultiAgentInvestigationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    dependencies=[Depends(require_api_key), Depends(agent_rate_limit)],
    responses=RATE_LIMIT_RESPONSES,
)

# Construction-time failures (missing OPENAI_API_KEY, etc.) are a
# service-unavailable condition — 503; anything raised once the agent is
# already running (mid-investigation LLM/embedding failure) is reported as a
# generic 500 rather than a raw traceback. Neither branch changes the
# orchestrator's reasoning — this is response-handling only.


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
    """The single canonical investigation endpoint (Phase 23A). Internally
    backed by Phase 19A-19D's ``MultiAgentInvestigationOrchestrator``
    (planner, evidence-driven hypothesis generation, critic, iterative
    loop) — see docs/architecture/19_multi_agent_investigation.md.

    Phase 23A history: this route previously coexisted with
    ``/investigate-advanced`` (a single-shot structured-report agent) and
    ``/investigate-orchestrated`` (this same orchestrator, under a
    different path) — three endpoints for one business capability
    ("investigate this problem and report a root cause") at three
    successive levels of sophistication. The orchestrator was already the
    documented "canonical" choice; the other two were earlier
    implementations, not distinct capabilities, so they were removed
    rather than kept as parallel routes. The underlying single-shot agent
    classes (``InvestigationAgent``, ``AdvancedInvestigationAgent`` in
    ``app/services/``) are unmodified and still directly unit-tested —
    only their public HTTP routes were retired.
    """
    session = _run_or_503(
        lambda: MultiAgentInvestigationOrchestrator(db).investigate(
            request.problem, n_hypotheses=request.n_hypotheses
        ),
        what="Investigation",
    )
    investigation = session.final_report.investigation
    critique = session.final_report.critique

    return InvestigationResponse(
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
