from __future__ import annotations

import uuid

import pytest

from app.evaluation.failure_analysis import (
    CauseLevel,
    Component,
    FailureCategory,
    Severity,
    analyze_judge_failures,
    analyze_reasoning_failures,
    analyze_retrieval_failures,
    classify_severity,
    cluster_failures,
    summarize_failures,
)
from app.evaluation.harness import (
    AggregateMetrics,
    CorpusStatistics,
    CoverageBreakdown,
    EvaluationConfig,
    EvaluationDatasetInfo,
    EvaluationReport,
    QueryEvaluationOutcome,
)
from app.evaluation.gold_dataset import CorpusFingerprintPlaceholder
from app.evaluation.gold_loader import GoldDatasetResolutionSummary
from app.evaluation.judge import STAGE_PLAN, JudgeEvaluation, JudgeFinding, make_judge_score
from app.evaluation.metrics import QueryMetricResult
from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics


# ── Severity ───────────────────────────────────────────────────────────────────


def test_classify_severity_bands() -> None:
    assert classify_severity(0.0) == Severity.LOW
    assert classify_severity(0.24) == Severity.LOW
    assert classify_severity(0.25) == Severity.MEDIUM
    assert classify_severity(0.49) == Severity.MEDIUM
    assert classify_severity(0.50) == Severity.HIGH
    assert classify_severity(0.74) == Severity.HIGH
    assert classify_severity(0.75) == Severity.CRITICAL
    assert classify_severity(1.0) == Severity.CRITICAL


def test_classify_severity_clamps_out_of_range() -> None:
    assert classify_severity(-1.0) == Severity.LOW
    assert classify_severity(5.0) == Severity.CRITICAL


# ── Retrieval failure detection ──────────────────────────────────────────────────


def _metric(query_id: str, recall: float | None) -> QueryMetricResult:
    return QueryMetricResult(
        query_id=query_id, k=10, num_relevant=1, num_unresolved_expected=0, num_retrieved=10,
        num_duplicate_retrieved=0, recall_at_k=recall, reciprocal_rank=recall,
        dcg_at_k=1.0, idcg_at_k=1.0, ndcg_at_k=recall,
    )


def _outcome(
    query_id: str, *, category="lexical-overlap", recall: float | None = 1.0,
    skipped=False, skip_reason=None, num_unresolved=0,
) -> QueryEvaluationOutcome:
    return QueryEvaluationOutcome(
        query_id=query_id, category=category, difficulty="easy", num_relevant=1,
        num_unresolved_expected=num_unresolved, skipped=skipped, skip_reason=skip_reason,
        metric=None if skipped else _metric(query_id, recall),
    )


def _eval_report(outcomes: tuple[QueryEvaluationOutcome, ...], category_breakdown=None):
    return EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version="v1", description="d", created_at="t0", author=None,
            corpus_fingerprint=CorpusFingerprintPlaceholder(),
        ),
        config=EvaluationConfig(k=10, expand=False, rerank=False),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=CorpusFingerprintPlaceholder(), distinct_retrieved_incident_count=1,
        ),
        num_evaluated=len(outcomes), num_skipped=sum(1 for o in outcomes if o.skipped),
        aggregate_metrics=AggregateMetrics(
            num_queries=len(outcomes), mean_recall_at_k=0.8, mean_reciprocal_rank=0.8,
            mean_ndcg_at_k=0.8, resolution_coverage=1.0, queries_with_unresolved_incidents=0,
        ),
        per_query=outcomes, coverage=CoverageBreakdown(
            total_queries=len(outcomes), no_match_expected_queries=0,
            fully_resolved_queries=len(outcomes), partially_resolved_queries=0,
            fully_unresolved_queries=0,
        ),
        resolution_summary=GoldDatasetResolutionSummary(
            total_expected_incidents=1, resolved_count=1, unresolved_identities=(),
        ),
        category_breakdown=category_breakdown or {},
        difficulty_breakdown={}, started_at="t0", finished_at="t1", duration_seconds=0.1,
    )


def test_perfect_query_produces_no_failure() -> None:
    report = _eval_report((_outcome("q1", recall=1.0),))
    assert analyze_retrieval_failures(report) == ()


def test_skipped_query_is_a_search_failure() -> None:
    report = _eval_report((_outcome("q1", skipped=True, skip_reason="boom", recall=None),))
    failures = analyze_retrieval_failures(report)
    assert len(failures) == 1
    assert failures[0].component == Component.RETRIEVAL
    assert failures[0].category == FailureCategory.SEARCH_FAILURE
    assert failures[0].severity == Severity.CRITICAL
    assert failures[0].cause_chain[0].level == CauseLevel.IMMEDIATE
    assert failures[0].cause_chain[-1].level == CauseLevel.SYSTEMIC


def test_incomplete_recall_is_a_failure_with_severity_from_deviation() -> None:
    report = _eval_report((_outcome("q1", recall=0.5),))
    failures = analyze_retrieval_failures(report)
    assert len(failures) == 1
    assert failures[0].category == FailureCategory.INCOMPLETE_RECALL
    assert failures[0].severity == Severity.HIGH  # deviation = 1.0 - 0.5 = 0.5 -> HIGH


def test_incomplete_recall_severity_matches_deviation_exactly() -> None:
    report = _eval_report((_outcome("q1", recall=0.1),))  # deviation 0.9 -> CRITICAL
    failures = analyze_retrieval_failures(report)
    assert failures[0].severity == Severity.CRITICAL


def test_unresolved_gold_entry_is_a_failure() -> None:
    report = _eval_report((_outcome("q1", recall=1.0, num_unresolved=1),))
    failures = analyze_retrieval_failures(report)
    assert any(f.category == FailureCategory.UNRESOLVED_GOLD_ENTRY for f in failures)


def test_systemic_cause_names_category_when_category_underperforms() -> None:
    from app.evaluation.harness import AggregateMetrics as Agg

    report = _eval_report(
        (_outcome("q1", category="paraphrase", recall=0.3),),
        category_breakdown={"paraphrase": Agg(
            num_queries=1, mean_recall_at_k=0.3, mean_reciprocal_rank=0.3, mean_ndcg_at_k=0.3,
            resolution_coverage=1.0, queries_with_unresolved_incidents=0,
        )},
    )
    failures = analyze_retrieval_failures(report)
    systemic = failures[0].cause_chain[-1].description
    assert "paraphrase" in systemic
    assert "underperforms" in systemic


# ── Reasoning failure detection ──────────────────────────────────────────────────


def _result(
    scenario_id="s1", *, planner_correct=True, hypothesis_recall_hit=True, decision_correct=True,
    critic_correct=True, stopping_correct=True, expected_root_causes=("x",),
    actual_root_causes=("x",), explanation=(),
) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        scenario_id=scenario_id, expected_strategy="authentication",
        expected_root_causes=expected_root_causes, expected_verdict="approved",
        expected_stopping_reason="critic_approved", actual_strategy=(
            "authentication" if planner_correct else "network"
        ),
        actual_root_causes=actual_root_causes,
        actual_verdict="approved" if critic_correct else "need_more_evidence",
        actual_stopping_reason="critic_approved" if stopping_correct else "max_iterations",
        total_iterations=1, planner_correct=planner_correct,
        hypothesis_recall_hit=hypothesis_recall_hit, hypothesis_precision=1.0,
        decision_correct=decision_correct, critic_correct=critic_correct,
        stopping_correct=stopping_correct, converged=True, session=None,
        explanation=explanation,
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


def test_perfect_result_produces_no_failures() -> None:
    report = _reasoning_report((_result(),))
    assert analyze_reasoning_failures(report) == ()


def test_planner_failure_is_detected() -> None:
    report = _reasoning_report((_result(planner_correct=False),))
    failures = analyze_reasoning_failures(report)
    assert len(failures) == 1
    assert failures[0].component == Component.PLANNER
    assert failures[0].category == FailureCategory.STRATEGY_MISMATCH
    assert failures[0].cause_chain[-1].description == "planner rule priority ordering"


def test_hypothesis_failure_is_detected() -> None:
    report = _reasoning_report((_result(hypothesis_recall_hit=False, actual_root_causes=("y",)),))
    failures = analyze_reasoning_failures(report)
    assert failures[0].component == Component.HYPOTHESIS_GENERATOR
    assert failures[0].category == FailureCategory.MISSING_HYPOTHESIS


def test_decision_failure_attributes_systemic_cause_to_planner_when_planner_also_failed() -> None:
    report = _reasoning_report((_result(planner_correct=False, decision_correct=False),))
    failures = analyze_reasoning_failures(report)
    decision_failure = next(f for f in failures if f.component == Component.DECISION)
    assert decision_failure.cause_chain[-1].description == "planner rule priority ordering"


def test_decision_failure_attributes_systemic_cause_to_evidence_when_planner_correct() -> None:
    report = _reasoning_report((_result(decision_correct=False),))
    failures = analyze_reasoning_failures(report)
    decision_failure = next(f for f in failures if f.component == Component.DECISION)
    assert "evidence" in decision_failure.cause_chain[-1].description


def test_critic_failure_is_detected() -> None:
    report = _reasoning_report((_result(critic_correct=False),))
    failures = analyze_reasoning_failures(report)
    assert any(f.component == Component.CRITIC for f in failures)


def test_orchestrator_failure_is_detected() -> None:
    report = _reasoning_report((_result(stopping_correct=False),))
    failures = analyze_reasoning_failures(report)
    assert any(f.component == Component.ORCHESTRATOR for f in failures)


def test_multiple_failing_checks_on_one_result_increase_severity() -> None:
    single = _reasoning_report((_result(decision_correct=False),))
    multi = _reasoning_report(
        (_result(planner_correct=False, decision_correct=False, critic_correct=False),)
    )
    single_failures = analyze_reasoning_failures(single)
    multi_failures = analyze_reasoning_failures(multi)
    severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    assert severity_rank[multi_failures[0].severity.value] >= severity_rank[
        single_failures[0].severity.value
    ]


def test_multiple_failing_checks_produce_multiple_records() -> None:
    report = _reasoning_report((_result(planner_correct=False, decision_correct=False),))
    failures = analyze_reasoning_failures(report)
    assert len(failures) == 2
    assert {f.component for f in failures} == {Component.PLANNER, Component.DECISION}


# ── Judge failure detection ──────────────────────────────────────────────────────


def _judge_eval(score: float, weaknesses=()) -> JudgeEvaluation:
    return JudgeEvaluation(
        stage=STAGE_PLAN, score=make_judge_score(score), explanation="x", weaknesses=weaknesses,
    )


def test_high_score_produces_no_failure() -> None:
    assert analyze_judge_failures((_judge_eval(9.0),)) == ()


def test_low_score_is_a_failure() -> None:
    failures = analyze_judge_failures((_judge_eval(2.0, weaknesses=(
        JudgeFinding("diversity", "only one cause"),
    )),))
    assert len(failures) == 1
    assert failures[0].component == Component.JUDGE
    assert failures[0].category == FailureCategory.LOW_CONFIDENCE
    assert failures[0].severity == Severity.CRITICAL


def test_judge_errors_produce_malformed_evaluation_failures() -> None:
    failures = analyze_judge_failures((), judge_errors=("bad json",))
    assert len(failures) == 1
    assert failures[0].category == FailureCategory.MALFORMED_EVALUATION
    assert failures[0].severity == Severity.CRITICAL


# ── Clustering ────────────────────────────────────────────────────────────────────


def test_cluster_failures_groups_by_component_and_category() -> None:
    report = _reasoning_report(
        (_result(planner_correct=False), _result(scenario_id="s2", planner_correct=False)),
    )
    failures = analyze_reasoning_failures(report)
    clusters = cluster_failures(failures)
    assert len(clusters) == 1
    assert clusters[0].component == Component.PLANNER
    assert len(clusters[0].failures) == 2


def test_cluster_severity_is_the_max_of_its_members() -> None:
    report = _reasoning_report(
        (
            _result(decision_correct=False),
            _result(scenario_id="s2", planner_correct=False, decision_correct=False),
        ),
    )
    failures = [
        f for f in analyze_reasoning_failures(report) if f.component == Component.DECISION
    ]
    clusters = cluster_failures(failures)
    assert len(clusters) == 1
    severities = {f.severity for f in failures}
    severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    assert severity_rank[clusters[0].severity.value] == max(
        severity_rank[s.value] for s in severities
    )


def test_cluster_failures_is_deterministic() -> None:
    report = _reasoning_report(
        (_result(planner_correct=False), _result(scenario_id="s2", critic_correct=False))
    )
    failures = analyze_reasoning_failures(report)
    first = cluster_failures(failures)
    second = cluster_failures(failures)
    assert first == second


def test_cluster_failures_handles_empty_input() -> None:
    assert cluster_failures(()) == ()


# ── Summary ───────────────────────────────────────────────────────────────────────


def test_summarize_failures_counts_by_dimension() -> None:
    report = _reasoning_report(
        (_result(planner_correct=False), _result(scenario_id="s2", critic_correct=False))
    )
    failures = analyze_reasoning_failures(report)
    summary = summarize_failures(failures)
    assert summary.total_failures == 2
    assert {c.component for c in summary.by_component} == {Component.PLANNER, Component.CRITIC}


def test_summarize_failures_handles_empty_input() -> None:
    summary = summarize_failures(())
    assert summary.total_failures == 0
    assert summary.by_component == ()
