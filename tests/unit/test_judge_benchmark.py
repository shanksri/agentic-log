from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.judge import STAGE_DECISION, STAGE_PLAN, JudgeEvaluation, make_judge_score
from app.evaluation.judge_benchmark import (
    DeltaClassification,
    FileJudgedReasoningBenchmarkRepository,
    InMemoryJudgedReasoningBenchmarkRepository,
    JudgeAggregateMetrics,
    aggregate_judge_evaluations,
    compare_judge_aggregates,
    create_judged_benchmark_run,
)
from app.evaluation.reasoning_benchmark import create_reasoning_benchmark_run
from app.evaluation.reasoning_dataset import InvestigationScenario, ReasoningGoldDataset
from app.evaluation.reasoning_harness import (
    InvestigationEvaluationReport,
    ReasoningMetrics,
    evaluate_reasoning_dataset,
)
from app.services.critic_agent import CritiqueResult, CritiqueVerdict, CritiquedInvestigationReport
from app.services.hypothesis_investigation import (
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


def _evaluation(stage: str, score: float) -> JudgeEvaluation:
    return JudgeEvaluation(stage=stage, score=make_judge_score(score), explanation="x")


def _reasoning_report() -> InvestigationEvaluationReport:
    """A real (non-stub) InvestigationEvaluationReport, JSON-roundtrip-safe
    - the file-repository tests below serialize the whole tree, which a
    SimpleNamespace stand-in (used elsewhere for pure-aggregation tests)
    cannot survive.
    """
    hypothesis = InvestigationHypothesis(
        id="h1", root_cause="expired token", rationale="r", validation_keywords=(),
        raw_confidence=0.9,
    )
    decision = InvestigationDecision(
        accepted=hypothesis, accepted_score=None, rejected=(), is_uncertain=False, rationale="r",
    )
    plan = InvestigationPlan(
        problem="p", strategy=PlanningStrategy.AUTHENTICATION, objective="o",
        priority_list=("p",), evidence_priorities=("e",), assumptions=("a",),
        expected_difficulty="medium", strategy_rationale="r",
    )
    critique = CritiqueResult(
        verdict=CritiqueVerdict.APPROVED, confidence=0.9, findings=(), unresolved_questions=(),
        missing_evidence=(), recommended_actions=(), explanation="x",
    )
    iteration = InvestigationIteration(
        iteration_number=1, plan=plan, hypotheses=(hypothesis,), evaluations={},
        decision=decision, critique=critique, progress_note="x", rationale="x",
    )
    investigation_report = InvestigationReport(
        problem="p", selected_hypothesis=hypothesis, confidence=0.9, confidence_level="HIGH",
        supporting_evidence=(), contradicting_evidence=(), remaining_uncertainty=(),
        is_uncertain=False, rejected_hypotheses=(),
    )
    session = InvestigationSession(
        final_report=CritiquedInvestigationReport(
            investigation=investigation_report, critique=critique
        ),
        iterations=(iteration,), stopping_reason=StoppingReason.CRITIC_APPROVED,
        total_iterations=1, stop_explanation="x",
    )

    class _FakeOrchestrator:
        def investigate(self, problem, *, n_hypotheses=3, routing_observation=None):
            return session

    dataset = ReasoningGoldDataset(
        version="v1", description="d", created_at="2026-01-01",
        scenarios=(
            InvestigationScenario(
                id="s1", problem="p", expected_strategy="authentication",
                expected_root_causes=("expired token",), expected_verdict="approved",
                expected_stopping_reason="critic_approved",
            ),
        ),
    )
    return evaluate_reasoning_dataset(dataset, _FakeOrchestrator())


def _stub_reasoning_report() -> InvestigationEvaluationReport:
    """A SimpleNamespace-backed stand-in, fine for pure-aggregation/
    compatibility tests that never serialize the report.
    """
    from types import SimpleNamespace

    metrics = ReasoningMetrics(
        num_scenarios=1, planner_accuracy=1.0, hypothesis_recall=1.0, hypothesis_precision=1.0,
        decision_accuracy=1.0, critic_accuracy=1.0, stopping_accuracy=1.0, convergence_rate=1.0,
        mean_iteration_count=1.0,
    )
    return InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(SimpleNamespace(scenario_id="s1"),), metrics=metrics,
        started_at="t0", finished_at="t1", duration_seconds=0.1,
    )


def _reasoning_run(**kwargs):
    return create_reasoning_benchmark_run(
        experiment_name="exp", report=_stub_reasoning_report(), **kwargs
    )


def _real_reasoning_run(**kwargs):
    return create_reasoning_benchmark_run(
        experiment_name="exp", report=_reasoning_report(), **kwargs
    )


# ── Aggregation ────────────────────────────────────────────────────────────────


def test_aggregate_judge_evaluations_groups_by_stage() -> None:
    evaluations = (
        _evaluation(STAGE_PLAN, 8.0), _evaluation(STAGE_PLAN, 6.0),
        _evaluation(STAGE_DECISION, 9.0),
    )
    metrics = aggregate_judge_evaluations(evaluations)

    assert metrics.num_evaluations == 3
    assert metrics.mean_plan_score == pytest.approx(7.0)
    assert metrics.mean_decision_score == pytest.approx(9.0)
    assert metrics.mean_hypotheses_score is None
    assert metrics.mean_critique_score is None
    assert metrics.mean_session_score is None


def test_aggregate_judge_evaluations_handles_empty_input() -> None:
    metrics = aggregate_judge_evaluations(())
    assert metrics.num_evaluations == 0
    assert metrics.mean_plan_score is None


# ── Benchmark integration ────────────────────────────────────────────────────────


def test_create_judged_benchmark_run_computes_aggregate_automatically() -> None:
    run = create_judged_benchmark_run(
        experiment_name="exp", reasoning_run=_reasoning_run(),
        judge_evaluations=(_evaluation(STAGE_PLAN, 8.0),),
    )
    assert run.judge_aggregate is not None
    assert run.judge_aggregate.mean_plan_score == 8.0
    assert run.reasoning_run.report.dataset_version == "v1"


def test_create_judged_benchmark_run_without_judge_evaluations_has_no_aggregate() -> None:
    run = create_judged_benchmark_run(experiment_name="exp", reasoning_run=_reasoning_run())
    assert run.judge_evaluations == ()
    assert run.judge_aggregate is None


def test_judged_run_coexists_with_heuristic_metrics() -> None:
    """Both systems coexist: the heuristic ReasoningMetrics are untouched
    on the embedded reasoning_run, alongside the new judge fields.
    """
    run = create_judged_benchmark_run(
        experiment_name="exp", reasoning_run=_reasoning_run(),
        judge_evaluations=(_evaluation(STAGE_PLAN, 8.0),),
    )
    assert run.reasoning_run.report.metrics.planner_accuracy == 1.0
    assert run.judge_aggregate.mean_plan_score == 8.0


# ── Repository ────────────────────────────────────────────────────────────────────


def test_in_memory_repository_save_get_list_latest_delete() -> None:
    repo = InMemoryJudgedReasoningBenchmarkRepository()
    run1 = create_judged_benchmark_run(
        experiment_name="exp", reasoning_run=_reasoning_run(), timestamp="2026-01-01T00:00:00",
    )
    run2 = create_judged_benchmark_run(
        experiment_name="exp", reasoning_run=_reasoning_run(), timestamp="2026-01-02T00:00:00",
    )
    repo.save(run1)
    repo.save(run2)

    assert repo.get(run1.run_id) == run1
    assert repo.list_runs() == (run1, run2)
    assert repo.latest() == run2
    assert repo.delete(run1.run_id) is True
    assert repo.get(run1.run_id) is None


def test_in_memory_repository_rejects_duplicate_run_id() -> None:
    repo = InMemoryJudgedReasoningBenchmarkRepository()
    run = create_judged_benchmark_run(experiment_name="exp", reasoning_run=_reasoning_run())
    repo.save(run)
    with pytest.raises(ValueError):
        repo.save(run)


# ── Serialization ─────────────────────────────────────────────────────────────────


def test_file_repository_round_trips_a_judged_run(tmp_path: Path) -> None:
    repo = FileJudgedReasoningBenchmarkRepository(tmp_path)
    run = create_judged_benchmark_run(
        experiment_name="exp", reasoning_run=_real_reasoning_run(),
        judge_evaluations=(_evaluation(STAGE_PLAN, 8.0),),
    )
    repo.save(run)

    loaded = repo.get(run.run_id)

    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.judge_aggregate.mean_plan_score == 8.0
    assert loaded.judge_evaluations[0].stage == STAGE_PLAN
    assert loaded.reasoning_run.report.dataset_version == "v1"


def test_file_repository_round_trips_a_run_with_no_judge_evaluations(tmp_path: Path) -> None:
    repo = FileJudgedReasoningBenchmarkRepository(tmp_path)
    run = create_judged_benchmark_run(experiment_name="exp", reasoning_run=_real_reasoning_run())
    repo.save(run)

    loaded = repo.get(run.run_id)

    assert loaded is not None
    assert loaded.judge_aggregate is None
    assert loaded.judge_evaluations == ()


# ── Regression ─────────────────────────────────────────────────────────────────────


def test_compare_judge_aggregates_detects_improvement() -> None:
    baseline = JudgeAggregateMetrics(
        num_evaluations=1, mean_plan_score=5.0, mean_hypotheses_score=None,
        mean_decision_score=None, mean_critique_score=None, mean_session_score=None,
    )
    candidate = JudgeAggregateMetrics(
        num_evaluations=1, mean_plan_score=8.0, mean_hypotheses_score=None,
        mean_decision_score=None, mean_critique_score=None, mean_session_score=None,
    )

    deltas = compare_judge_aggregates(baseline, candidate)

    assert deltas["plan"].classification == DeltaClassification.IMPROVED
    assert deltas["plan"].delta == pytest.approx(3.0)
    assert deltas["hypotheses"].classification == DeltaClassification.UNCHANGED


def test_compare_judge_aggregates_detects_regression() -> None:
    baseline = JudgeAggregateMetrics(
        num_evaluations=1, mean_plan_score=8.0, mean_hypotheses_score=None,
        mean_decision_score=None, mean_critique_score=None, mean_session_score=None,
    )
    candidate = JudgeAggregateMetrics(
        num_evaluations=1, mean_plan_score=5.0, mean_hypotheses_score=None,
        mean_decision_score=None, mean_critique_score=None, mean_session_score=None,
    )

    deltas = compare_judge_aggregates(baseline, candidate)

    assert deltas["plan"].classification == DeltaClassification.REGRESSED


def test_compare_judge_aggregates_undefined_when_one_side_missing() -> None:
    baseline = JudgeAggregateMetrics(
        num_evaluations=0, mean_plan_score=None, mean_hypotheses_score=None,
        mean_decision_score=None, mean_critique_score=None, mean_session_score=None,
    )
    candidate = JudgeAggregateMetrics(
        num_evaluations=1, mean_plan_score=8.0, mean_hypotheses_score=None,
        mean_decision_score=None, mean_critique_score=None, mean_session_score=None,
    )

    deltas = compare_judge_aggregates(baseline, candidate)

    assert deltas["plan"].classification == DeltaClassification.UNDEFINED
