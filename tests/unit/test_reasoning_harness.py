from __future__ import annotations

import pytest

from app.evaluation.reasoning_dataset import InvestigationScenario, ReasoningGoldDataset
from app.evaluation.reasoning_harness import (
    evaluate_reasoning_dataset,
    evaluate_scenario,
)
from app.services.critic_agent import CritiqueResult, CritiqueVerdict, CritiquedInvestigationReport
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    HypothesisScore,
    InvestigationDecision,
    InvestigationHypothesis,
    InvestigationReport,
)
from app.services.investigation_orchestrator import (
    InvestigationIteration,
    InvestigationSession,
    StoppingReason,
)
from app.services.planner_agent import InvestigationPlan, PlanningStrategy

# ── Builders ──────────────────────────────────────────────────────────────────────


def _plan(strategy: PlanningStrategy, problem: str = "p") -> InvestigationPlan:
    return InvestigationPlan(
        problem=problem, strategy=strategy, objective="o", priority_list=("p",),
        evidence_priorities=("e",), assumptions=("a",), expected_difficulty="medium",
        strategy_rationale="r",
    )


def _hypothesis(id_: str, root_cause: str) -> InvestigationHypothesis:
    return InvestigationHypothesis(
        id=id_, root_cause=root_cause, rationale="r", validation_keywords=(), raw_confidence=0.9,
    )


def _critique(verdict: CritiqueVerdict) -> CritiqueResult:
    return CritiqueResult(
        verdict=verdict, confidence=0.8, findings=(), unresolved_questions=(),
        missing_evidence=(), recommended_actions=(), explanation="x",
    )


def _iteration(
    n: int, *, strategy: PlanningStrategy, hypotheses: tuple, accepted_id: str | None,
    verdict: CritiqueVerdict, problem: str = "p",
) -> InvestigationIteration:
    if accepted_id is None:
        decision = InvestigationDecision(
            accepted=None, accepted_score=None,
            rejected=tuple((h, _score(h.id, 0.3)) for h in hypotheses),
            is_uncertain=True, rationale="none accepted",
        )
    else:
        accepted = next(h for h in hypotheses if h.id == accepted_id)
        score = _score(accepted_id, 0.9)
        rejected = tuple((h, _score(h.id, 0.3)) for h in hypotheses if h.id != accepted_id)
        decision = InvestigationDecision(
            accepted=accepted, accepted_score=score, rejected=rejected, is_uncertain=False,
            rationale="accepted",
        )
    evaluations = {h.id: _evaluation(h.id) for h in hypotheses}
    return InvestigationIteration(
        iteration_number=n, plan=_plan(strategy, problem), hypotheses=hypotheses,
        evaluations=evaluations, decision=decision, critique=_critique(verdict),
        progress_note="x", rationale="x",
    )


def _score(id_: str, composite: float) -> HypothesisScore:
    return HypothesisScore(
        hypothesis_id=id_, raw_confidence=0.9, retrieval_confidence_level="HIGH",
        evidence_confidence_level="HIGH", supporting_count=1, contradicting_count=0,
        missing_count=0, composite_score=composite,
    )


def _evaluation(id_: str) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        hypothesis_id=id_, query="q", supporting_evidence=("s",), contradicting_evidence=(),
        missing_evidence=(), evidence_confidence_level="HIGH", evidence_top1_score=0.9,
    )


def _session(
    iterations: tuple[InvestigationIteration, ...], *, stopping_reason: StoppingReason,
    problem: str = "p",
) -> InvestigationSession:
    final = iterations[-1]
    selected = final.decision.accepted
    report = InvestigationReport(
        problem=problem, selected_hypothesis=selected,
        confidence=final.decision.accepted_score.composite_score if selected else 0.0,
        confidence_level="HIGH" if selected else "LOW",
        supporting_evidence=(), contradicting_evidence=(), remaining_uncertainty=(),
        is_uncertain=selected is None,
        rejected_hypotheses=tuple(h for h, _ in final.decision.rejected),
    )
    critiqued = CritiquedInvestigationReport(investigation=report, critique=final.critique)
    return InvestigationSession(
        final_report=critiqued, iterations=iterations, stopping_reason=stopping_reason,
        total_iterations=len(iterations), stop_explanation="x",
    )


class FakeOrchestrator:
    def __init__(self, session: InvestigationSession):
        self._session = session
        self.calls: list[dict] = []

    def investigate(self, problem, *, n_hypotheses=3, routing_observation=None):
        self.calls.append({"problem": problem, "n_hypotheses": n_hypotheses})
        return self._session


def _scenario(**overrides) -> InvestigationScenario:
    defaults = dict(
        id="s1", problem="login fails with expired token",
        expected_strategy="authentication", expected_root_causes=("expired token",),
        expected_verdict="approved", expected_stopping_reason="critic_approved",
    )
    defaults.update(overrides)
    return InvestigationScenario(**defaults)


# ── Perfect investigation ────────────────────────────────────────────────────────


def test_perfect_investigation_is_fully_correct() -> None:
    h1 = _hypothesis("h1", "expired token caused login failure")
    iteration = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id="h1",
        verdict=CritiqueVerdict.APPROVED,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.CRITIC_APPROVED)
    orchestrator = FakeOrchestrator(session)

    result = evaluate_scenario(_scenario(), orchestrator)

    assert result.planner_correct is True
    assert result.hypothesis_recall_hit is True
    assert result.hypothesis_precision == 1.0
    assert result.decision_correct is True
    assert result.critic_correct is True
    assert result.stopping_correct is True
    assert result.converged is True
    assert result.explanation == ()


# ── Planner mistakes ─────────────────────────────────────────────────────────────


def test_planner_mistake_is_detected_and_explained() -> None:
    h1 = _hypothesis("h1", "expired token caused login failure")
    iteration = _iteration(
        1, strategy=PlanningStrategy.NETWORK, hypotheses=(h1,), accepted_id="h1",
        verdict=CritiqueVerdict.APPROVED,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.CRITIC_APPROVED)
    orchestrator = FakeOrchestrator(session)

    result = evaluate_scenario(_scenario(), orchestrator)

    assert result.planner_correct is False
    assert any("planner mismatch" in line for line in result.explanation)


# ── Missing hypotheses ───────────────────────────────────────────────────────────


def test_missing_hypotheses_is_detected_and_explained() -> None:
    h1 = _hypothesis("h1", "completely unrelated guess")
    iteration = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id=None,
        verdict=CritiqueVerdict.INCONCLUSIVE,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.MAX_ITERATIONS)
    orchestrator = FakeOrchestrator(session)

    result = evaluate_scenario(_scenario(), orchestrator)

    assert result.hypothesis_recall_hit is False
    assert any("missing hypotheses" in line for line in result.explanation)


# ── Incorrect accepted hypothesis ───────────────────────────────────────────────


def test_incorrect_acceptance_is_detected_and_explained() -> None:
    h1 = _hypothesis("h1", "totally wrong root cause")
    iteration = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id="h1",
        verdict=CritiqueVerdict.APPROVED,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.CRITIC_APPROVED)
    orchestrator = FakeOrchestrator(session)

    result = evaluate_scenario(_scenario(), orchestrator)

    assert result.decision_correct is False
    assert any("incorrect acceptance" in line for line in result.explanation)


def test_incorrect_rejection_is_detected_and_explained() -> None:
    h1 = _hypothesis("h1", "expired token caused login failure")
    iteration = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id=None,
        verdict=CritiqueVerdict.INCONCLUSIVE,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.MAX_ITERATIONS)
    orchestrator = FakeOrchestrator(session)

    result = evaluate_scenario(_scenario(), orchestrator)

    assert result.decision_correct is False
    assert any("incorrect rejection" in line for line in result.explanation)


def test_negative_control_correctly_stays_uncertain() -> None:
    h1 = _hypothesis("h1", "some guess")
    iteration = _iteration(
        1, strategy=PlanningStrategy.UNKNOWN, hypotheses=(h1,), accepted_id=None,
        verdict=CritiqueVerdict.INCONCLUSIVE,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.MAX_ITERATIONS)
    orchestrator = FakeOrchestrator(session)
    scenario = _scenario(
        expected_strategy="unknown", expected_root_causes=(), expected_verdict="inconclusive",
        expected_stopping_reason="max_iterations",
    )

    result = evaluate_scenario(scenario, orchestrator)

    assert result.decision_correct is True


def test_incorrect_acceptance_when_nothing_was_expected() -> None:
    h1 = _hypothesis("h1", "some guess")
    iteration = _iteration(
        1, strategy=PlanningStrategy.UNKNOWN, hypotheses=(h1,), accepted_id="h1",
        verdict=CritiqueVerdict.APPROVED,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.CRITIC_APPROVED)
    orchestrator = FakeOrchestrator(session)
    scenario = _scenario(
        expected_strategy="unknown", expected_root_causes=(), expected_verdict="inconclusive",
        expected_stopping_reason="max_iterations",
    )

    result = evaluate_scenario(scenario, orchestrator)

    assert result.decision_correct is False
    assert any(
        "no hypothesis was expected to be accepted" in line for line in result.explanation
    )


# ── Critic failures ──────────────────────────────────────────────────────────────


def test_critic_verdict_mismatch_is_detected_and_explained() -> None:
    h1 = _hypothesis("h1", "expired token caused login failure")
    iteration = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id="h1",
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.MAX_ITERATIONS)
    orchestrator = FakeOrchestrator(session)

    result = evaluate_scenario(_scenario(), orchestrator)

    assert result.critic_correct is False
    assert any("critic verdict mismatch" in line for line in result.explanation)


# ── Orchestrator failures (stopping reason) ─────────────────────────────────────


def test_stopping_reason_mismatch_is_detected_and_explained() -> None:
    h1 = _hypothesis("h1", "expired token caused login failure")
    iteration = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id="h1",
        verdict=CritiqueVerdict.APPROVED,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.MAX_ITERATIONS)
    orchestrator = FakeOrchestrator(session)

    result = evaluate_scenario(_scenario(), orchestrator)

    assert result.stopping_correct is False
    assert any("incorrect stopping reason" in line for line in result.explanation)


def test_missing_convergence_note_when_max_iterations_was_unexpected() -> None:
    h1 = _hypothesis("h1", "totally unrelated guess")
    iteration = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id=None,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    session = _session((iteration,), stopping_reason=StoppingReason.MAX_ITERATIONS)
    orchestrator = FakeOrchestrator(session)
    scenario = _scenario(expected_stopping_reason="critic_approved")

    result = evaluate_scenario(scenario, orchestrator)

    assert any("missing convergence" in line for line in result.explanation)


# ── Multiple iterations ──────────────────────────────────────────────────────────


def test_uses_first_iteration_plan_for_planner_accuracy() -> None:
    h1 = _hypothesis("h1", "weak guess")
    h2 = _hypothesis("h2", "expired token caused login failure")
    iteration1 = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id=None,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    iteration2 = _iteration(
        2, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h2,), accepted_id="h2",
        verdict=CritiqueVerdict.APPROVED,
    )
    session = _session((iteration1, iteration2), stopping_reason=StoppingReason.CRITIC_APPROVED)
    orchestrator = FakeOrchestrator(session)

    result = evaluate_scenario(_scenario(), orchestrator)

    assert result.total_iterations == 2
    assert result.actual_root_causes == ("weak guess", "expired token caused login failure")
    assert result.planner_correct is True
    assert result.decision_correct is True
    assert result.hypothesis_precision == pytest.approx(0.5)


# ── Dataset-wide evaluation / benchmark comparison ──────────────────────────────


def test_evaluate_reasoning_dataset_aggregates_across_scenarios() -> None:
    h1 = _hypothesis("h1", "expired token caused login failure")
    good_iteration = _iteration(
        1, strategy=PlanningStrategy.AUTHENTICATION, hypotheses=(h1,), accepted_id="h1",
        verdict=CritiqueVerdict.APPROVED,
    )
    good_session = _session((good_iteration,), stopping_reason=StoppingReason.CRITIC_APPROVED)

    h2 = _hypothesis("h2", "totally wrong")
    bad_iteration = _iteration(
        1, strategy=PlanningStrategy.NETWORK, hypotheses=(h2,), accepted_id="h2",
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    bad_session = _session((bad_iteration,), stopping_reason=StoppingReason.MAX_ITERATIONS)

    class TwoScenarioOrchestrator:
        def investigate(self, problem, *, n_hypotheses=3, routing_observation=None):
            return good_session if "good" in problem else bad_session

    dataset = ReasoningGoldDataset(
        version="v1", description="d", created_at="2026-01-01",
        scenarios=(
            _scenario(id="s1", problem="good scenario login fails with expired token"),
            _scenario(id="s2", problem="bad scenario login fails with expired token"),
        ),
    )

    report = evaluate_reasoning_dataset(dataset, TwoScenarioOrchestrator())

    assert report.metrics.num_scenarios == 2
    assert report.metrics.planner_accuracy == pytest.approx(0.5)
    assert report.metrics.critic_accuracy == pytest.approx(0.5)
    assert report.metrics.convergence_rate == pytest.approx(0.5)
    assert len(report.results) == 2
