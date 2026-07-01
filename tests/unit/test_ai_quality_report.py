from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evaluation.ai_quality_report import (
    AIQualityReport,
    build_quality_report,
    build_quality_report_from_benchmarks,
)
from app.evaluation.benchmark import InMemoryBenchmarkRepository, create_benchmark_run
from app.evaluation.gold_dataset import CorpusFingerprintPlaceholder
from app.evaluation.gold_loader import GoldDatasetResolutionSummary
from app.evaluation.harness import (
    AggregateMetrics,
    CorpusStatistics,
    CoverageBreakdown,
    EvaluationConfig,
    EvaluationDatasetInfo,
    EvaluationReport,
    QueryEvaluationOutcome,
)
from app.evaluation.judge import STAGE_PLAN, JudgeEvaluation, make_judge_score
from app.evaluation.metrics import QueryMetricResult
from app.evaluation.reasoning_benchmark import (
    InMemoryReasoningBenchmarkRepository,
    create_reasoning_benchmark_run,
)
from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics


def _metric(query_id: str, recall: float | None) -> QueryMetricResult:
    return QueryMetricResult(
        query_id=query_id, k=10, num_relevant=1, num_unresolved_expected=0, num_retrieved=10,
        num_duplicate_retrieved=0, recall_at_k=recall, reciprocal_rank=recall, dcg_at_k=1.0,
        idcg_at_k=1.0, ndcg_at_k=recall,
    )


def _outcome(query_id: str, recall: float | None) -> QueryEvaluationOutcome:
    return QueryEvaluationOutcome(
        query_id=query_id, category="lexical-overlap", difficulty="easy", num_relevant=1,
        num_unresolved_expected=0, skipped=False, skip_reason=None,
        metric=_metric(query_id, recall),
    )


def _eval_report(outcomes: tuple[QueryEvaluationOutcome, ...]) -> EvaluationReport:
    return EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version="v1", description="d", created_at="t0", author=None,
            corpus_fingerprint=CorpusFingerprintPlaceholder(),
        ),
        config=EvaluationConfig(k=10, expand=False, rerank=False),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=CorpusFingerprintPlaceholder(), distinct_retrieved_incident_count=1,
        ),
        num_evaluated=len(outcomes), num_skipped=0,
        aggregate_metrics=AggregateMetrics(
            num_queries=len(outcomes), mean_recall_at_k=0.8, mean_reciprocal_rank=0.8,
            mean_ndcg_at_k=0.8, resolution_coverage=1.0, queries_with_unresolved_incidents=0,
        ),
        per_query=outcomes,
        coverage=CoverageBreakdown(
            total_queries=len(outcomes), no_match_expected_queries=0,
            fully_resolved_queries=len(outcomes), partially_resolved_queries=0,
            fully_unresolved_queries=0,
        ),
        resolution_summary=GoldDatasetResolutionSummary(
            total_expected_incidents=1, resolved_count=1, unresolved_identities=(),
        ),
        category_breakdown={}, difficulty_breakdown={}, started_at="t0", finished_at="t1",
        duration_seconds=0.1,
    )


def _reasoning_result(scenario_id="s1", *, decision_correct=True) -> object:
    return SimpleNamespace(
        scenario_id=scenario_id, expected_strategy="authentication",
        expected_root_causes=("x",), expected_verdict="approved",
        expected_stopping_reason="critic_approved", actual_strategy="authentication",
        actual_root_causes=("x",), actual_verdict="approved",
        actual_stopping_reason="critic_approved", total_iterations=1, planner_correct=True,
        hypothesis_recall_hit=True, hypothesis_precision=1.0, decision_correct=decision_correct,
        critic_correct=True, stopping_correct=True, converged=True, session=None,
        explanation=() if decision_correct else ("incorrect acceptance: x",),
    )


def _reasoning_report(results) -> InvestigationEvaluationReport:
    metrics = ReasoningMetrics(
        num_scenarios=len(results), planner_accuracy=1.0, hypothesis_recall=1.0,
        hypothesis_precision=1.0, decision_accuracy=1.0, critic_accuracy=1.0,
        stopping_accuracy=1.0, convergence_rate=1.0, mean_iteration_count=1.0,
    )
    return InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3, results=tuple(results),
        metrics=metrics, started_at="t0", finished_at="t1", duration_seconds=0.1,
    )


# ── build_quality_report (pure assembly) ────────────────────────────────────────


def test_no_failures_produces_empty_report() -> None:
    report = build_quality_report()
    assert report.failure_summary.total_failures == 0
    assert report.failure_clusters == ()
    assert report.recommendations == ()
    assert report.component_summaries == ()
    assert "No failures" in report.overall_summary


def test_retrieval_failures_feed_the_report() -> None:
    retrieval_report = _eval_report((_outcome("q1", 0.5),))
    report = build_quality_report(retrieval_reports=(retrieval_report,))
    assert report.failure_summary.total_failures == 1
    assert len(report.failure_clusters) == 1
    assert len(report.recommendations) == 1
    assert len(report.component_summaries) == 1


def test_reasoning_failures_feed_the_report() -> None:
    reasoning_report = _reasoning_report((_reasoning_result(decision_correct=False),))
    report = build_quality_report(reasoning_reports=(reasoning_report,))
    assert report.failure_summary.total_failures == 1
    assert report.component_summaries[0].component.value == "decision"


def test_judge_evaluations_feed_the_report() -> None:
    evaluation = JudgeEvaluation(stage=STAGE_PLAN, score=make_judge_score(2.0), explanation="x")
    report = build_quality_report(judge_evaluations=(evaluation,))
    assert report.failure_summary.total_failures == 1


def test_multiple_components_combine_into_one_report() -> None:
    retrieval_report = _eval_report((_outcome("q1", 0.5),))
    reasoning_report = _reasoning_report((_reasoning_result(decision_correct=False),))
    evaluation = JudgeEvaluation(stage=STAGE_PLAN, score=make_judge_score(2.0), explanation="x")

    report = build_quality_report(
        retrieval_reports=(retrieval_report,), reasoning_reports=(reasoning_report,),
        judge_evaluations=(evaluation,),
    )

    assert report.failure_summary.total_failures == 3
    assert len(report.component_summaries) == 3


def test_trend_summary_populated_with_two_or_more_historical_reports() -> None:
    report_a = _eval_report((_outcome("q1", 1.0),))
    report_b = _eval_report((_outcome("q1", 0.5),))

    report = build_quality_report(retrieval_reports=(report_a, report_b))

    assert report.trend_summary is not None
    assert report.trend_summary.failure_count_trend == (0, 1)


def test_trend_summary_absent_with_single_report_and_no_regression() -> None:
    report_a = _eval_report((_outcome("q1", 1.0),))
    report = build_quality_report(retrieval_reports=(report_a,))
    assert report.trend_summary is None


def test_regression_verdict_populates_trend_even_with_one_report() -> None:
    report_a = _eval_report((_outcome("q1", 1.0),))
    report = build_quality_report(retrieval_reports=(report_a,), regression_verdict="improved")
    assert report.trend_summary is not None
    assert report.trend_summary.regression_verdict == "improved"


def test_report_is_frozen() -> None:
    report = build_quality_report()
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        report.overall_summary = "changed"  # type: ignore[misc]


def test_report_is_deterministic_given_identical_inputs() -> None:
    retrieval_report = _eval_report((_outcome("q1", 0.5),))
    first = build_quality_report(retrieval_reports=(retrieval_report,))
    second = build_quality_report(retrieval_reports=(retrieval_report,))
    assert first.failure_summary == second.failure_summary
    assert first.failure_clusters == second.failure_clusters
    assert first.recommendations == second.recommendations


# ── Benchmark integration ────────────────────────────────────────────────────────


def test_single_benchmark_from_retrieval_repository() -> None:
    repo = InMemoryBenchmarkRepository()
    repo.save(create_benchmark_run(
        experiment_name="exp", report=_eval_report((_outcome("q1", 0.5),)),
    ))

    report = build_quality_report_from_benchmarks(retrieval_repo=repo)

    assert report.failure_summary.total_failures == 1


def test_multiple_benchmark_runs_via_include_history() -> None:
    repo = InMemoryBenchmarkRepository()
    repo.save(create_benchmark_run(
        experiment_name="exp", report=_eval_report((_outcome("q1", 1.0),)),
        timestamp="2026-01-01T00:00:00",
    ))
    repo.save(create_benchmark_run(
        experiment_name="exp", report=_eval_report((_outcome("q1", 0.5),)),
        timestamp="2026-01-02T00:00:00",
    ))

    report = build_quality_report_from_benchmarks(retrieval_repo=repo, include_history=True)

    assert report.trend_summary is not None
    assert report.trend_summary.failure_count_trend == (0, 1)


def test_reasoning_benchmark_repository_integration() -> None:
    repo = InMemoryReasoningBenchmarkRepository()
    repo.save(create_reasoning_benchmark_run(
        experiment_name="exp",
        report=_reasoning_report((_reasoning_result(decision_correct=False),)),
    ))

    report = build_quality_report_from_benchmarks(reasoning_repo=repo)

    assert report.failure_summary.total_failures == 1


def test_no_repositories_supplied_produces_empty_report() -> None:
    report = build_quality_report_from_benchmarks()
    assert report.failure_summary.total_failures == 0
    assert isinstance(report, AIQualityReport)


def test_empty_repository_produces_empty_report() -> None:
    repo = InMemoryBenchmarkRepository()
    report = build_quality_report_from_benchmarks(retrieval_repo=repo)
    assert report.failure_summary.total_failures == 0
