from __future__ import annotations

import json

import pytest

from app.evaluation.judge import STAGE_CRITIQUE, STAGE_DECISION, STAGE_HYPOTHESES, STAGE_PLAN
from app.evaluation.llm_judge import JudgeResponseError, LLMJudge
from app.services.critic_agent import CritiqueResult, CritiqueVerdict
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    HypothesisScore,
    InvestigationDecision,
    InvestigationHypothesis,
)
from app.services.investigation_orchestrator import InvestigationIteration, InvestigationSession
from app.services.investigation_orchestrator import StoppingReason
from app.services.planner_agent import InvestigationPlan, PlanningStrategy


def _good_response(score=8.0, explanation="solid reasoning"):
    return json.dumps({
        "score": score, "explanation": explanation,
        "strengths": [{"criterion": "diversity", "detail": "two causes"}],
        "weaknesses": [], "recommendations": [],
    })


class FakeLLMClient:
    """A deterministic test double satisfying the ``JudgeLLMClient``
    Protocol - no OpenAI/network call, exactly the contract this phase
    requires unit tests to never depend on a real LLM.
    """

    def __init__(self, response: str | None = None, *, responses: list[str] | None = None):
        self._response = response
        self._responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._responses is not None:
            return self._responses[len(self.prompts) - 1]
        return self._response


def _plan() -> InvestigationPlan:
    return InvestigationPlan(
        problem="p", strategy=PlanningStrategy.AUTHENTICATION, objective="o",
        priority_list=("a",), evidence_priorities=("e",), assumptions=("x",),
        expected_difficulty="medium", strategy_rationale="r",
    )


def _hypothesis(id_: str, root_cause: str) -> InvestigationHypothesis:
    return InvestigationHypothesis(
        id=id_, root_cause=root_cause, rationale="r", validation_keywords=("k",),
        raw_confidence=0.9,
    )


def _decision() -> InvestigationDecision:
    hypothesis = _hypothesis("h1", "cause a")
    score = HypothesisScore(
        hypothesis_id="h1", raw_confidence=0.9, retrieval_confidence_level="HIGH",
        evidence_confidence_level="HIGH", supporting_count=1, contradicting_count=0,
        missing_count=0, composite_score=0.8,
    )
    return InvestigationDecision(
        accepted=hypothesis, accepted_score=score, rejected=(), is_uncertain=False, rationale="r",
    )


def _critique() -> CritiqueResult:
    return CritiqueResult(
        verdict=CritiqueVerdict.APPROVED, confidence=0.8, findings=(), unresolved_questions=(),
        missing_evidence=(), recommended_actions=(), explanation="x",
    )


# ── Interface compliance / one call per evaluation ──────────────────────────────


def test_evaluate_plan_makes_exactly_one_completion_call() -> None:
    client = FakeLLMClient(_good_response())
    judge = LLMJudge(client)

    evaluation = judge.evaluate_plan("p", _plan())

    assert len(client.prompts) == 1
    assert evaluation.stage == STAGE_PLAN
    assert evaluation.score.value == 8.0
    assert evaluation.explanation == "solid reasoning"
    assert evaluation.strengths[0].criterion == "diversity"


def test_evaluate_hypotheses_includes_root_causes_in_prompt() -> None:
    client = FakeLLMClient(_good_response())
    judge = LLMJudge(client)
    hypotheses = (_hypothesis("h1", "expired token"), _hypothesis("h2", "revoked credential"))

    evaluation = judge.evaluate_hypotheses("p", _plan(), hypotheses)

    assert evaluation.stage == STAGE_HYPOTHESES
    assert "expired token" in client.prompts[0]
    assert "revoked credential" in client.prompts[0]


def test_evaluate_hypotheses_handles_empty_hypotheses() -> None:
    client = FakeLLMClient(_good_response())
    judge = LLMJudge(client)

    evaluation = judge.evaluate_hypotheses("p", _plan(), ())

    assert evaluation.stage == STAGE_HYPOTHESES
    assert "no hypotheses were generated" in client.prompts[0]


def test_evaluate_decision_includes_accepted_hypothesis_in_prompt() -> None:
    client = FakeLLMClient(_good_response())
    judge = LLMJudge(client)
    decision = _decision()

    evaluation = judge.evaluate_decision("p", (decision.accepted,), decision, {})

    assert evaluation.stage == STAGE_DECISION
    assert "cause a" in client.prompts[0]


def test_evaluate_decision_handles_uncertain_decision() -> None:
    client = FakeLLMClient(_good_response())
    judge = LLMJudge(client)
    decision = InvestigationDecision(
        accepted=None, accepted_score=None, rejected=(), is_uncertain=True, rationale="x",
    )

    judge.evaluate_decision("p", (), decision, {})

    assert "no hypothesis was accepted" in client.prompts[0]


def test_evaluate_critique_includes_verdict_in_prompt() -> None:
    client = FakeLLMClient(_good_response())
    judge = LLMJudge(client)

    evaluation = judge.evaluate_critique("p", _decision(), _critique())

    assert evaluation.stage == STAGE_CRITIQUE
    assert "approved" in client.prompts[0]


def test_evaluate_session_includes_timeline_in_prompt() -> None:
    client = FakeLLMClient(_good_response())
    judge = LLMJudge(client)
    iteration = InvestigationIteration(
        iteration_number=1, plan=_plan(), hypotheses=(_hypothesis("h1", "cause a"),),
        evaluations={}, decision=_decision(), critique=_critique(), progress_note="x",
        rationale="x",
    )
    session = InvestigationSession(
        final_report=None, iterations=(iteration,), stopping_reason=StoppingReason.CRITIC_APPROVED,
        total_iterations=1, stop_explanation="approved",
    )

    judge.evaluate_session("p", session)

    assert "cause a" in client.prompts[0]
    assert "critic_approved" in client.prompts[0]


# ── Response parsing ──────────────────────────────────────────────────────────────


def test_malformed_json_raises_judge_response_error() -> None:
    client = FakeLLMClient("not json at all")
    judge = LLMJudge(client)

    with pytest.raises(JudgeResponseError, match="not valid JSON"):
        judge.evaluate_plan("p", _plan())


def test_missing_score_field_raises_judge_response_error() -> None:
    client = FakeLLMClient(json.dumps({"explanation": "x"}))
    judge = LLMJudge(client)

    with pytest.raises(JudgeResponseError, match="missing"):
        judge.evaluate_plan("p", _plan())


def test_non_numeric_score_raises_judge_response_error() -> None:
    client = FakeLLMClient(json.dumps({"score": "high", "explanation": "x"}))
    judge = LLMJudge(client)

    with pytest.raises(JudgeResponseError, match="not numeric"):
        judge.evaluate_plan("p", _plan())


def test_malformed_findings_list_raises_judge_response_error() -> None:
    client = FakeLLMClient(json.dumps({"score": 5, "explanation": "x", "strengths": "oops"}))
    judge = LLMJudge(client)

    with pytest.raises(JudgeResponseError, match="must be a list"):
        judge.evaluate_plan("p", _plan())


def test_finding_missing_required_keys_raises_judge_response_error() -> None:
    client = FakeLLMClient(
        json.dumps({"score": 5, "explanation": "x", "strengths": [{"criterion": "diversity"}]})
    )
    judge = LLMJudge(client)

    with pytest.raises(JudgeResponseError, match="criterion.*detail"):
        judge.evaluate_plan("p", _plan())


def test_score_is_clamped_into_rubric_range() -> None:
    client = FakeLLMClient(_good_response(score=99.0))
    judge = LLMJudge(client)

    evaluation = judge.evaluate_plan("p", _plan())

    assert evaluation.score.value == 10.0
    assert evaluation.score.band == "Excellent"


def test_judge_response_error_is_a_value_error() -> None:
    assert issubclass(JudgeResponseError, ValueError)
