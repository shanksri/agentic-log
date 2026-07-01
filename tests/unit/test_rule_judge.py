from __future__ import annotations

from app.evaluation.judge import STAGE_CRITIQUE, STAGE_DECISION, STAGE_HYPOTHESES, STAGE_PLAN
from app.evaluation.rule_judge import RuleJudge
from app.services.critic_agent import CritiqueResult, CritiqueVerdict
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    HypothesisScore,
    InvestigationDecision,
    InvestigationHypothesis,
)
from app.services.investigation_orchestrator import InvestigationIteration, InvestigationSession
from app.services.planner_agent import InvestigationPlan, PlanningStrategy

judge = RuleJudge()


def _plan(strategy=PlanningStrategy.AUTHENTICATION, priorities=("a", "b")) -> InvestigationPlan:
    return InvestigationPlan(
        problem="p", strategy=strategy, objective="find the cause", priority_list=priorities,
        evidence_priorities=("e",), assumptions=("x",), expected_difficulty="medium",
        strategy_rationale="r",
    )


def _hypothesis(id_: str, root_cause: str, keywords=("k",)) -> InvestigationHypothesis:
    return InvestigationHypothesis(
        id=id_, root_cause=root_cause, rationale="r", validation_keywords=keywords,
        raw_confidence=0.9,
    )


def _score(id_: str, composite: float) -> HypothesisScore:
    return HypothesisScore(
        hypothesis_id=id_, raw_confidence=0.9, retrieval_confidence_level="HIGH",
        evidence_confidence_level="HIGH", supporting_count=1, contradicting_count=0,
        missing_count=0, composite_score=composite,
    )


def _evaluation(id_: str, supporting=1) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        hypothesis_id=id_, query="q",
        supporting_evidence=tuple(f"s{i}" for i in range(supporting)),
        contradicting_evidence=(), missing_evidence=(), evidence_confidence_level="HIGH",
        evidence_top1_score=0.9,
    )


def _critique(verdict: CritiqueVerdict, explanation="some explanation") -> CritiqueResult:
    return CritiqueResult(
        verdict=verdict, confidence=0.8, findings=(), unresolved_questions=(),
        missing_evidence=(), recommended_actions=(), explanation=explanation,
    )


# ── evaluate_plan ──────────────────────────────────────────────────────────────


def test_plan_with_known_strategy_and_priorities_scores_high() -> None:
    evaluation = judge.evaluate_plan("p", _plan())
    assert evaluation.stage == STAGE_PLAN
    assert evaluation.score.value == 10.0
    assert evaluation.score.band == "Excellent"
    assert not evaluation.weaknesses


def test_plan_with_unknown_strategy_scores_lower_and_recommends() -> None:
    evaluation = judge.evaluate_plan("p", _plan(strategy=PlanningStrategy.UNKNOWN))
    assert evaluation.score.value < 10.0
    assert any(w.criterion == "chosen_strategy" for w in evaluation.weaknesses)
    assert evaluation.recommendations


def test_plan_with_single_priority_loses_prioritization_points() -> None:
    high = judge.evaluate_plan("p", _plan(priorities=("a", "b")))
    low = judge.evaluate_plan("p", _plan(priorities=("a",)))
    assert low.score.value < high.score.value
    assert any(w.criterion == "prioritization" for w in low.weaknesses)


# ── evaluate_hypotheses ──────────────────────────────────────────────────────────


def test_empty_hypotheses_scores_minimum() -> None:
    evaluation = judge.evaluate_hypotheses("p", _plan(), ())
    assert evaluation.stage == STAGE_HYPOTHESES
    assert evaluation.score.value == 1.0
    assert evaluation.score.band == "Poor"


def test_diverse_well_keyed_hypotheses_score_higher() -> None:
    diverse = (_hypothesis("h1", "cause a"), _hypothesis("h2", "cause b"))
    single = (_hypothesis("h1", "cause a"), _hypothesis("h2", "cause a"))

    diverse_eval = judge.evaluate_hypotheses("p", _plan(), diverse)
    single_eval = judge.evaluate_hypotheses("p", _plan(), single)

    assert diverse_eval.score.value > single_eval.score.value
    assert any(s.criterion == "diversity" for s in diverse_eval.strengths)


def test_hypotheses_missing_keywords_lose_completeness_points() -> None:
    with_keywords = (_hypothesis("h1", "cause a", keywords=("k",)),)
    without_keywords = (_hypothesis("h1", "cause a", keywords=()),)

    with_eval = judge.evaluate_hypotheses("p", _plan(), with_keywords)
    without_eval = judge.evaluate_hypotheses("p", _plan(), without_keywords)

    assert with_eval.score.value > without_eval.score.value
    assert any(w.criterion == "completeness" for w in without_eval.weaknesses)


# ── evaluate_decision ─────────────────────────────────────────────────────────────


def test_uncertain_decision_scores_weak() -> None:
    decision = InvestigationDecision(
        accepted=None, accepted_score=None, rejected=(), is_uncertain=True, rationale="x",
    )
    evaluation = judge.evaluate_decision("p", (), decision, {})
    assert evaluation.stage == STAGE_DECISION
    assert evaluation.score.value == 4.0


def test_accepted_decision_rescales_composite_score() -> None:
    hypothesis = _hypothesis("h1", "cause a")
    decision = InvestigationDecision(
        accepted=hypothesis, accepted_score=_score("h1", 0.8), rejected=(), is_uncertain=False,
        rationale="x",
    )
    evaluation = judge.evaluate_decision("p", (hypothesis,), decision, {"h1": _evaluation("h1")})
    # SCORE_MIN + 0.8 * (SCORE_MAX - SCORE_MIN) = 1 + 0.8*9 = 8.2
    assert evaluation.score.value == 8.2
    assert any(s.criterion == "supporting_evidence" for s in evaluation.strengths)


def test_accepted_decision_with_no_supporting_evidence_is_flagged() -> None:
    hypothesis = _hypothesis("h1", "cause a")
    decision = InvestigationDecision(
        accepted=hypothesis, accepted_score=_score("h1", 0.8), rejected=(), is_uncertain=False,
        rationale="x",
    )
    evaluation = judge.evaluate_decision(
        "p", (hypothesis,), decision, {"h1": _evaluation("h1", supporting=0)}
    )
    assert any(w.criterion == "supporting_evidence" for w in evaluation.weaknesses)


# ── evaluate_critique ─────────────────────────────────────────────────────────────


def test_critique_score_follows_verdict_severity() -> None:
    approved = judge.evaluate_critique("p", _decision_stub(), _critique(CritiqueVerdict.APPROVED))
    inconclusive = judge.evaluate_critique(
        "p", _decision_stub(), _critique(CritiqueVerdict.INCONCLUSIVE)
    )
    assert approved.stage == STAGE_CRITIQUE
    assert approved.score.value > inconclusive.score.value


def test_critique_with_no_explanation_is_flagged() -> None:
    evaluation = judge.evaluate_critique(
        "p", _decision_stub(), _critique(CritiqueVerdict.APPROVED, explanation="")
    )
    assert any(w.criterion == "justification" for w in evaluation.weaknesses)


def _decision_stub() -> InvestigationDecision:
    return InvestigationDecision(
        accepted=None, accepted_score=None, rejected=(), is_uncertain=True, rationale="x",
    )


# ── evaluate_session ──────────────────────────────────────────────────────────────


def _iteration(n: int, *, accepted_id="h1") -> InvestigationIteration:
    hypothesis = _hypothesis("h1", "cause a")
    decision = InvestigationDecision(
        accepted=hypothesis if accepted_id else None,
        accepted_score=_score("h1", 0.9) if accepted_id else None,
        rejected=(), is_uncertain=accepted_id is None, rationale="x",
    )
    return InvestigationIteration(
        iteration_number=n, plan=_plan(), hypotheses=(hypothesis,),
        evaluations={"h1": _evaluation("h1")}, decision=decision,
        critique=_critique(CritiqueVerdict.APPROVED), progress_note="x", rationale="x",
    )


def test_session_with_one_iteration_has_no_efficiency_penalty() -> None:
    session = InvestigationSession(
        final_report=None, iterations=(_iteration(1),),
        stopping_reason=_stopping_reason(), total_iterations=1, stop_explanation="approved",
    )
    evaluation = judge.evaluate_session("p", session)
    assert not evaluation.weaknesses


def test_session_with_multiple_iterations_has_efficiency_penalty() -> None:
    session = InvestigationSession(
        final_report=None, iterations=(_iteration(1), _iteration(2), _iteration(3)),
        stopping_reason=_stopping_reason(), total_iterations=3, stop_explanation="approved",
    )
    one_iter = InvestigationSession(
        final_report=None, iterations=(_iteration(1),),
        stopping_reason=_stopping_reason(), total_iterations=1, stop_explanation="approved",
    )

    multi_eval = judge.evaluate_session("p", session)
    single_eval = judge.evaluate_session("p", one_iter)

    assert multi_eval.score.value < single_eval.score.value
    assert any(w.criterion == "efficiency" for w in multi_eval.weaknesses)


def _stopping_reason():
    from app.services.investigation_orchestrator import StoppingReason

    return StoppingReason.CRITIC_APPROVED


def test_judge_is_deterministic() -> None:
    session = InvestigationSession(
        final_report=None, iterations=(_iteration(1),),
        stopping_reason=_stopping_reason(), total_iterations=1, stop_explanation="approved",
    )
    first = judge.evaluate_session("p", session)
    second = judge.evaluate_session("p", session)
    assert first == second
