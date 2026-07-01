from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.reasoning_benchmark import (
    FileReasoningBenchmarkRepository,
    InMemoryReasoningBenchmarkRepository,
    compare_reasoning_runs,
    create_reasoning_benchmark_run,
    reasoning_regression_history,
)
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


def _real_report(planner_accuracy_one: bool = True) -> InvestigationEvaluationReport:
    """A real (non-stub) InvestigationEvaluationReport, JSON-roundtrip-safe,
    used only by the file-repository test below.
    """
    strategy = (
        PlanningStrategy.AUTHENTICATION if planner_accuracy_one else PlanningStrategy.NETWORK
    )
    hypothesis = InvestigationHypothesis(
        id="h1", root_cause="expired token", rationale="r", validation_keywords=(),
        raw_confidence=0.9,
    )
    decision = InvestigationDecision(
        accepted=hypothesis, accepted_score=None, rejected=(), is_uncertain=False,
        rationale="r",
    )
    plan = InvestigationPlan(
        problem="p", strategy=strategy, objective="o", priority_list=("p",),
        evidence_priorities=("e",), assumptions=("a",), expected_difficulty="medium",
        strategy_rationale="r",
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


def _report(**metric_overrides) -> InvestigationEvaluationReport:
    defaults = dict(
        num_scenarios=1, planner_accuracy=1.0, hypothesis_recall=1.0, hypothesis_precision=1.0,
        decision_accuracy=1.0, critic_accuracy=1.0, stopping_accuracy=1.0, convergence_rate=1.0,
        mean_iteration_count=1.0,
    )
    defaults.update(metric_overrides)
    from types import SimpleNamespace

    return InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(SimpleNamespace(scenario_id="s1"),), metrics=ReasoningMetrics(**defaults),
        started_at="t0", finished_at="t1", duration_seconds=0.1,
    )


def test_create_reasoning_benchmark_run_assigns_id_and_timestamp() -> None:
    run = create_reasoning_benchmark_run(experiment_name="exp", report=_report())
    assert run.run_id
    assert run.timestamp
    assert run.experiment_name == "exp"


def test_in_memory_repository_save_get_list_latest_delete() -> None:
    repo = InMemoryReasoningBenchmarkRepository()
    run1 = create_reasoning_benchmark_run(
        experiment_name="exp", report=_report(), timestamp="2026-01-01T00:00:00"
    )
    run2 = create_reasoning_benchmark_run(
        experiment_name="exp", report=_report(), timestamp="2026-01-02T00:00:00"
    )
    repo.save(run1)
    repo.save(run2)

    assert repo.get(run1.run_id) == run1
    assert repo.list_runs() == (run1, run2)
    assert repo.latest() == run2
    assert repo.delete(run1.run_id) is True
    assert repo.get(run1.run_id) is None
    assert repo.delete(run1.run_id) is False


def test_in_memory_repository_rejects_duplicate_run_id() -> None:
    repo = InMemoryReasoningBenchmarkRepository()
    run = create_reasoning_benchmark_run(experiment_name="exp", report=_report())
    repo.save(run)
    with pytest.raises(ValueError):
        repo.save(run)


def test_file_repository_round_trips_a_run(tmp_path: Path) -> None:
    repo = FileReasoningBenchmarkRepository(tmp_path)
    run = create_reasoning_benchmark_run(experiment_name="exp", report=_real_report())
    repo.save(run)

    loaded = repo.get(run.run_id)

    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.experiment_name == "exp"
    assert loaded.report.dataset_version == "v1"
    assert loaded.report.metrics.planner_accuracy == 1.0


def test_compare_reasoning_runs_delegates_to_compare_reasoning() -> None:
    baseline = create_reasoning_benchmark_run(
        experiment_name="exp", report=_report(planner_accuracy=0.5)
    )
    candidate = create_reasoning_benchmark_run(
        experiment_name="exp", report=_report(planner_accuracy=1.0)
    )

    result = compare_reasoning_runs(baseline, candidate)

    assert result.planner.verdict.value == "improved"


def test_reasoning_regression_history_compares_consecutive_runs() -> None:
    repo = InMemoryReasoningBenchmarkRepository()
    repo.save(create_reasoning_benchmark_run(
        experiment_name="exp", report=_report(planner_accuracy=0.5),
        timestamp="2026-01-01T00:00:00",
    ))
    repo.save(create_reasoning_benchmark_run(
        experiment_name="exp", report=_report(planner_accuracy=1.0),
        timestamp="2026-01-02T00:00:00",
    ))
    repo.save(create_reasoning_benchmark_run(
        experiment_name="exp", report=_report(planner_accuracy=1.0),
        timestamp="2026-01-03T00:00:00",
    ))

    history = reasoning_regression_history(repo)

    assert len(history) == 2
    assert history[0].planner.verdict.value == "improved"
    assert history[1].planner.verdict.value == "unchanged"


