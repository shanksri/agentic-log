from __future__ import annotations

import pytest

from app.evaluation.judge_agreement import (
    AgreementPair,
    AgreementResult,
    BiasDirection,
    BiasFinding,
    ConsistencyResult,
)
from app.evaluation.judge_calibration import CalibrationResult
from app.evaluation.judge_validation_report import (
    Trustworthiness,
    assemble_validation_report,
    build_validation_report_from_benchmarks,
)
from app.evaluation.judge_benchmark import (
    InMemoryJudgedReasoningBenchmarkRepository,
    JudgeAggregateMetrics,
    create_judged_benchmark_run,
)
from app.evaluation.reasoning_benchmark import create_reasoning_benchmark_run
from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics
from app.evaluation.reasoning_regression import (
    ReasoningRegressionReport,
    CategoryDelta,
    CompatibilityCheck,
    MetricDelta,
    DeltaClassification,
)
from app.evaluation.reasoning_regression import Verdict as ReasoningVerdict


def _agreement(stage, *, within=1.0, n=5) -> AgreementResult:
    return AgreementResult(
        pair=AgreementPair.HUMAN_VS_LLM, stage=stage, n=n, differences=(0.5,) * n,
        mean_absolute_difference=0.5, agreement_within_tolerance=within, tolerance=1.0,
    )


def _consistency(stage, *, std_dev=0.1, n=5) -> ConsistencyResult:
    return ConsistencyResult(
        stage=stage, n=n, scores=(7.0,) * n, mean=7.0, variance=std_dev**2, std_dev=std_dev,
        minimum=7.0, maximum=7.0,
    )


def _bias(stage) -> BiasFinding:
    return BiasFinding(
        pair=AgreementPair.RULE_VS_LLM, stage=stage, mean_signed_difference=2.0,
        direction=BiasDirection.SECOND_HIGHER, n=5, description="x",
    )


def _calibration(direction, *, n=5) -> CalibrationResult:
    return CalibrationResult(
        metric_name="decision_accuracy", n=n, correlation=0.5, direction=direction, points=()
    )


# ── assemble_validation_report ──────────────────────────────────────────────────


def test_no_data_at_all_is_insufficient_data() -> None:
    report = assemble_validation_report()
    assert report.overall_trustworthiness == Trustworthiness.INSUFFICIENT_DATA
    assert report.confidence_level == 0.0
    assert "Insufficient" in report.recommended_production_usage


def test_clean_data_with_no_findings_is_high_trustworthiness() -> None:
    report = assemble_validation_report(
        agreement=(_agreement("plan", within=0.9),), consistency=(_consistency("plan"),),
    )
    assert report.overall_trustworthiness == Trustworthiness.HIGH
    assert "Safe for production" in report.recommended_production_usage


def test_bias_findings_reduce_trustworthiness() -> None:
    report = assemble_validation_report(
        agreement=(_agreement("plan", within=0.9),), bias=(_bias("plan"), _bias("decision")),
    )
    assert report.overall_trustworthiness in {Trustworthiness.MEDIUM, Trustworthiness.LOW}


def test_low_agreement_reduces_trustworthiness() -> None:
    clean = assemble_validation_report(agreement=(_agreement("plan", within=0.9),))
    degraded = assemble_validation_report(
        agreement=(_agreement("plan", within=0.3), _agreement("decision", within=0.2)),
    )
    assert clean.overall_trustworthiness == Trustworthiness.HIGH
    assert degraded.overall_trustworthiness == Trustworthiness.MEDIUM


def test_high_std_dev_reduces_trustworthiness() -> None:
    clean = assemble_validation_report(consistency=(_consistency("plan", std_dev=0.1),))
    degraded = assemble_validation_report(
        consistency=(_consistency("plan", std_dev=2.0), _consistency("decision", std_dev=2.0)),
    )
    assert clean.overall_trustworthiness == Trustworthiness.HIGH
    assert degraded.overall_trustworthiness == Trustworthiness.MEDIUM


def test_negative_calibration_reduces_trustworthiness() -> None:
    clean = assemble_validation_report(calibration=(_calibration("positive"),))
    degraded = assemble_validation_report(
        calibration=(_calibration("negative"), CalibrationResult(
            metric_name="recall_at_k", n=5, correlation=-0.5, direction="negative", points=(),
        )),
    )
    assert clean.overall_trustworthiness == Trustworthiness.HIGH
    assert degraded.overall_trustworthiness == Trustworthiness.MEDIUM


def test_positive_calibration_does_not_reduce_trustworthiness() -> None:
    clean = assemble_validation_report(agreement=(_agreement("plan", within=0.9),))
    with_positive_calibration = assemble_validation_report(
        agreement=(_agreement("plan", within=0.9),), calibration=(_calibration("positive"),),
    )
    assert clean.overall_trustworthiness == with_positive_calibration.overall_trustworthiness


def test_confidence_level_scales_with_total_n() -> None:
    small = assemble_validation_report(agreement=(_agreement("plan", n=2),))
    large = assemble_validation_report(agreement=(_agreement("plan", n=40),))
    assert small.confidence_level < large.confidence_level
    assert large.confidence_level == 1.0


def test_report_is_frozen() -> None:
    report = assemble_validation_report(agreement=(_agreement("plan"),))
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        report.confidence_level = 0.99  # type: ignore[misc]


def test_report_is_deterministic() -> None:
    first = assemble_validation_report(agreement=(_agreement("plan"),), bias=(_bias("plan"),))
    second = assemble_validation_report(agreement=(_agreement("plan"),), bias=(_bias("plan"),))
    assert first.overall_trustworthiness == second.overall_trustworthiness
    assert first.confidence_level == second.confidence_level


# ── Benchmark integration ────────────────────────────────────────────────────────


def _reasoning_report(decision_accuracy: float) -> InvestigationEvaluationReport:
    from types import SimpleNamespace

    metrics = ReasoningMetrics(
        num_scenarios=1, planner_accuracy=1.0, hypothesis_recall=1.0, hypothesis_precision=1.0,
        decision_accuracy=decision_accuracy, critic_accuracy=1.0, stopping_accuracy=1.0,
        convergence_rate=1.0, mean_iteration_count=1.0,
    )
    return InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(SimpleNamespace(scenario_id="s1"),), metrics=metrics,
        started_at="t0", finished_at="t1", duration_seconds=0.1,
    )


def _regression_report(verdict: ReasoningVerdict) -> ReasoningRegressionReport:
    baseline = _reasoning_report(0.5)
    candidate = _reasoning_report(0.9)
    delta = MetricDelta(
        baseline=0.5, candidate=0.9, delta=0.4, classification=DeltaClassification.IMPROVED
    )
    category = CategoryDelta(
        category="decision", metrics={"decision_accuracy": delta}, verdict=verdict
    )
    return ReasoningRegressionReport(
        baseline=baseline, candidate=candidate,
        compatibility=CompatibilityCheck(compatible=True, reasons=()), verdict=verdict,
        planner=category, hypothesis=category, decision=category, critic=category,
        iteration=category, summary="x",
    )


def test_build_validation_report_from_judged_benchmark_history() -> None:
    repo = InMemoryJudgedReasoningBenchmarkRepository()

    for i, (decision_accuracy, session_score) in enumerate([(0.3, 2.0), (0.6, 6.0), (0.9, 9.0)]):
        reasoning_run = create_reasoning_benchmark_run(
            experiment_name="exp", report=_reasoning_report(decision_accuracy),
            timestamp=f"2026-01-0{i+1}T00:00:00",
        )
        judge_aggregate = JudgeAggregateMetrics(
            num_evaluations=1, mean_plan_score=None, mean_hypotheses_score=None,
            mean_decision_score=None, mean_critique_score=None, mean_session_score=session_score,
        )
        run = create_judged_benchmark_run(
            experiment_name="exp", reasoning_run=reasoning_run,
            timestamp=f"2026-01-0{i+1}T00:00:00",
        )
        run = run.__class__(
            run_id=run.run_id, timestamp=run.timestamp, experiment_name=run.experiment_name,
            reasoning_run=reasoning_run, judge_evaluations=(), judge_aggregate=judge_aggregate,
        )
        repo.save(run)

    report = build_validation_report_from_benchmarks(judged_repo=repo)

    assert len(report.correlation) == 1
    assert report.correlation[0].direction == "positive"
    assert report.overall_trustworthiness != Trustworthiness.INSUFFICIENT_DATA


def test_build_validation_report_with_no_runs_is_insufficient_data() -> None:
    repo = InMemoryJudgedReasoningBenchmarkRepository()
    report = build_validation_report_from_benchmarks(judged_repo=repo)
    assert report.overall_trustworthiness == Trustworthiness.INSUFFICIENT_DATA


def test_build_validation_report_folds_in_existing_agreement_and_bias() -> None:
    repo = InMemoryJudgedReasoningBenchmarkRepository()
    report = build_validation_report_from_benchmarks(
        judged_repo=repo, existing_agreement=(_agreement("plan", within=0.9),),
        existing_bias=(_bias("plan"),),
    )
    assert len(report.agreement) == 1
    assert len(report.bias) == 1
