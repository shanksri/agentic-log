from __future__ import annotations

from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics
from app.evaluation.reasoning_regression import Verdict, compare_reasoning


def _report(version: str = "v1", **metric_overrides) -> InvestigationEvaluationReport:
    defaults = dict(
        num_scenarios=2, planner_accuracy=1.0, hypothesis_recall=1.0, hypothesis_precision=1.0,
        decision_accuracy=1.0, critic_accuracy=1.0, stopping_accuracy=1.0, convergence_rate=1.0,
        mean_iteration_count=1.0,
    )
    defaults.update(metric_overrides)
    metrics = ReasoningMetrics(**defaults)
    return InvestigationEvaluationReport(
        dataset_version=version, dataset_description="d", n_hypotheses=3,
        results=(_fake_result("s1"), _fake_result("s2")), metrics=metrics,
        started_at="t0", finished_at="t1", duration_seconds=0.1,
    )


def _fake_result(scenario_id: str):
    # A minimal stand-in shaped enough for compatibility checks (only
    # .scenario_id is read by compare_reasoning's compatibility check).
    from types import SimpleNamespace

    return SimpleNamespace(scenario_id=scenario_id)


def test_identical_reports_are_unchanged() -> None:
    baseline = _report()
    candidate = _report()

    report = compare_reasoning(baseline, candidate)

    assert report.verdict == Verdict.UNCHANGED
    assert report.planner.verdict == Verdict.UNCHANGED
    assert report.decision.verdict == Verdict.UNCHANGED
    assert report.critic.verdict == Verdict.UNCHANGED


def test_planner_accuracy_improvement_is_detected() -> None:
    baseline = _report(planner_accuracy=0.5)
    candidate = _report(planner_accuracy=1.0)

    report = compare_reasoning(baseline, candidate)

    assert report.planner.verdict == Verdict.IMPROVED
    assert report.verdict == Verdict.IMPROVED


def test_decision_accuracy_regression_is_detected() -> None:
    baseline = _report(decision_accuracy=1.0)
    candidate = _report(decision_accuracy=0.5)

    report = compare_reasoning(baseline, candidate)

    assert report.decision.verdict == Verdict.REGRESSED
    assert report.verdict == Verdict.REGRESSED


def test_critic_accuracy_regression_is_detected() -> None:
    baseline = _report(critic_accuracy=1.0)
    candidate = _report(critic_accuracy=0.5)

    report = compare_reasoning(baseline, candidate)

    assert report.critic.verdict == Verdict.REGRESSED


def test_hypothesis_category_folds_recall_and_precision() -> None:
    baseline = _report(hypothesis_recall=0.5, hypothesis_precision=0.5)
    candidate = _report(hypothesis_recall=1.0, hypothesis_precision=0.5)

    report = compare_reasoning(baseline, candidate)

    assert report.hypothesis.verdict == Verdict.IMPROVED


def test_mixed_planner_and_decision_changes_yield_mixed_overall() -> None:
    baseline = _report(planner_accuracy=0.5, decision_accuracy=1.0)
    candidate = _report(planner_accuracy=1.0, decision_accuracy=0.5)

    report = compare_reasoning(baseline, candidate)

    assert report.verdict == Verdict.MIXED


def test_iteration_category_is_diagnostic_and_does_not_drive_overall() -> None:
    baseline = _report(mean_iteration_count=1.0, convergence_rate=1.0)
    candidate = _report(mean_iteration_count=3.0, convergence_rate=0.2)

    report = compare_reasoning(baseline, candidate)

    assert report.iteration.verdict == Verdict.REGRESSED
    assert report.verdict == Verdict.UNCHANGED


def test_iteration_count_fewer_iterations_is_improved() -> None:
    baseline = _report(mean_iteration_count=3.0)
    candidate = _report(mean_iteration_count=1.0)

    report = compare_reasoning(baseline, candidate)

    assert report.iteration.metrics["mean_iteration_count"].classification.value == "improved"


def test_incompatible_dataset_versions_are_rejected() -> None:
    baseline = _report(version="v1")
    candidate = _report(version="v2")

    report = compare_reasoning(baseline, candidate)

    assert report.verdict == Verdict.INCOMPATIBLE
    assert report.planner is None
    assert any("dataset version differs" in reason for reason in report.compatibility.reasons)


def test_incompatible_scenario_coverage_is_rejected() -> None:
    baseline = _report()
    candidate = InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(_fake_result("s1"), _fake_result("s3")), metrics=baseline.metrics,
        started_at="t0", finished_at="t1", duration_seconds=0.1,
    )

    report = compare_reasoning(baseline, candidate)

    assert report.verdict == Verdict.INCOMPATIBLE
    assert any("scenario coverage differs" in reason for reason in report.compatibility.reasons)


def test_undefined_metrics_do_not_count_as_regression() -> None:
    baseline = _report(hypothesis_recall=None, hypothesis_precision=None)
    candidate = _report(hypothesis_recall=None, hypothesis_precision=None)

    report = compare_reasoning(baseline, candidate)

    assert report.hypothesis.verdict == Verdict.UNCHANGED
