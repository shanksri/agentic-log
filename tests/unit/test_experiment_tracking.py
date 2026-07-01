"""Tests for Phase 21F: Persistent Evaluation & Experiment Tracking."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.experiment_tracking import (
    ExperimentRepository,
    ExperimentRun,
    ExperimentStats,
    RunMetadata,
    _failed_queries,
    _failed_reasoning,
    _judge_disagreements,
    _to_jsonable,
    make_run_id,
)


# ── Helpers / Fakes ───────────────────────────────────────────────────────────


_FIXED_NOW = datetime(2026, 7, 1, 21, 30, 15, tzinfo=UTC)


def _make_execution_summary(duration: float = 1.23):
    """Build a minimal ExecutionSummary."""
    from app.evaluation.evaluation_pipeline import ExecutionSummary
    return ExecutionSummary(
        start_time="2026-07-01T21:30:14+00:00",
        end_time="2026-07-01T21:30:15+00:00",
        duration_seconds=duration,
        retrieval_queries=5,
        reasoning_scenarios=2,
        judge_evaluations=2,
        warnings=("w1",),
        errors=(),
    )


def _make_pipeline_result(*, retrieval=None, reasoning=None, judge=None,
                           quality=None, validation=None,
                           ret_regression=None, reas_regression=None,
                           ret_benchmark=None, reas_benchmark=None,
                           duration=1.23) -> Any:
    """Build a minimal EvaluationPipelineResult (all reports optional)."""
    from app.evaluation.evaluation_pipeline import EvaluationPipelineResult
    return EvaluationPipelineResult(
        retrieval_report=retrieval,
        retrieval_regression=ret_regression,
        retrieval_benchmark=ret_benchmark,
        reasoning_report=reasoning,
        reasoning_regression=reas_regression,
        reasoning_benchmark=reas_benchmark,
        judge_report=judge,
        judge_validation_report=validation,
        quality_report=quality,
        execution_summary=_make_execution_summary(duration),
    )


def _make_minimal_retrieval_report():
    """Return a real EvaluationReport with 0 queries."""
    from app.evaluation.gold_dataset import CorpusFingerprintPlaceholder
    from app.evaluation.gold_loader import GoldDatasetResolutionSummary
    from app.evaluation.harness import (
        AggregateMetrics, CoverageBreakdown, EvaluationConfig,
        EvaluationDatasetInfo, EvaluationReport, CorpusStatistics,
    )
    agg = AggregateMetrics(
        num_queries=2, mean_recall_at_k=0.8, mean_reciprocal_rank=0.7,
        mean_ndcg_at_k=0.75, resolution_coverage=1.0,
        queries_with_unresolved_incidents=0,
    )
    return EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version="v1", description="d", created_at="2026-01-01",
            author=None, corpus_fingerprint=CorpusFingerprintPlaceholder(),
        ),
        config=EvaluationConfig(k=10, expand=False, rerank=False),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=CorpusFingerprintPlaceholder(),
            distinct_retrieved_incident_count=3,
        ),
        num_evaluated=2, num_skipped=0, aggregate_metrics=agg,
        per_query=(), coverage=CoverageBreakdown(
            total_queries=2, no_match_expected_queries=0,
            fully_resolved_queries=2, partially_resolved_queries=0,
            fully_unresolved_queries=0,
        ),
        resolution_summary=GoldDatasetResolutionSummary(
            total_expected_incidents=0, resolved_count=0, unresolved_identities=(),
        ),
        category_breakdown={}, difficulty_breakdown={},
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )


def _make_minimal_reasoning_report():
    """Return a real InvestigationEvaluationReport with 0 results."""
    from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics
    metrics = ReasoningMetrics(
        num_scenarios=2, planner_accuracy=0.9, hypothesis_recall=0.8,
        hypothesis_precision=0.7, decision_accuracy=0.85, critic_accuracy=0.75,
        stopping_accuracy=0.9, convergence_rate=0.8, mean_iteration_count=1.5,
    )
    return InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(), metrics=metrics,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )


def _make_minimal_quality_report():
    from app.evaluation.ai_quality_report import AIQualityReport
    from app.evaluation.failure_analysis import FailureSummary, SeverityFailureCount, Severity
    return AIQualityReport(
        generated_at="2026-01-01T00:00:00+00:00",
        overall_summary="No failures detected across the analyzed evaluation artifacts.",
        failure_summary=FailureSummary(total_failures=0, by_component=(), by_category=(), by_severity=()),
        component_summaries=(),
        failure_clusters=(),
        recommendations=(),
        trend_summary=None,
    )


# ── make_run_id ───────────────────────────────────────────────────────────────


def test_make_run_id_format() -> None:
    rid = make_run_id("nightly", now=_FIXED_NOW)
    assert rid == "20260701_213015_nightly"


def test_make_run_id_sanitises_spaces() -> None:
    rid = make_run_id("my experiment", now=_FIXED_NOW)
    assert " " not in rid
    assert "my_experiment" in rid


def test_make_run_id_sanitises_slashes() -> None:
    rid = make_run_id("feat/branch", now=_FIXED_NOW)
    assert "/" not in rid


def test_make_run_id_default_uses_utc_now() -> None:
    rid = make_run_id("test")
    # Format: 8 digits _ 6 digits _ experiment_name
    parts = rid.split("_")
    assert len(parts) >= 3
    assert len(parts[0]) == 8   # YYYYMMDD
    assert len(parts[1]) == 6   # HHMMSS


# ── _to_jsonable ──────────────────────────────────────────────────────────────


def test_to_jsonable_primitives() -> None:
    assert _to_jsonable(42) == 42
    assert _to_jsonable("hello") == "hello"
    assert _to_jsonable(None) is None
    assert _to_jsonable(3.14) == 3.14


def test_to_jsonable_enum() -> None:
    from enum import Enum
    class Color(str, Enum):
        RED = "red"
    assert _to_jsonable(Color.RED) == "red"


def test_to_jsonable_dataclass() -> None:
    from dataclasses import dataclass
    @dataclass
    class Point:
        x: int
        y: int
    result = _to_jsonable(Point(x=1, y=2))
    assert result == {"x": 1, "y": 2}


def test_to_jsonable_nested() -> None:
    from dataclasses import dataclass
    @dataclass
    class Inner:
        v: int
    @dataclass
    class Outer:
        inner: Inner
        items: tuple[int, ...]
    result = _to_jsonable(Outer(inner=Inner(v=7), items=(1, 2, 3)))
    assert result == {"inner": {"v": 7}, "items": [1, 2, 3]}


def test_to_jsonable_mapping() -> None:
    assert _to_jsonable({"a": 1, "b": 2}) == {"a": 1, "b": 2}


# ── _failed_queries ───────────────────────────────────────────────────────────


def _q(query_id: str, recall: float | None = None, skipped: bool = False):
    metric = {"recall_at_k": recall} if recall is not None else {}
    return {"query_id": query_id, "metric": metric, "skipped": skipped}


def test_failed_queries_filters_low_recall() -> None:
    report = {"per_query": [_q("q1", 1.0), _q("q2", 0.5), _q("q3", 0.0)]}
    result = _failed_queries(report)
    ids = [r["query_id"] for r in result]
    assert "q2" in ids
    assert "q3" in ids
    assert "q1" not in ids


def test_failed_queries_includes_skipped() -> None:
    report = {"per_query": [_q("q1", skipped=True), _q("q2", 1.0)]}
    result = _failed_queries(report)
    assert any(r["query_id"] == "q1" for r in result)
    assert not any(r["query_id"] == "q2" for r in result)


def test_failed_queries_empty_per_query() -> None:
    assert _failed_queries({"per_query": []}) == []


def test_failed_queries_missing_per_query() -> None:
    assert _failed_queries({}) == []


# ── _failed_reasoning ─────────────────────────────────────────────────────────


def _rs(scenario_id: str, converged: bool | None = True,
        decision_correct: bool | None = True, planner_correct: bool | None = True):
    return {"scenario_id": scenario_id, "converged": converged,
            "decision_correct": decision_correct, "planner_correct": planner_correct}


def test_failed_reasoning_not_converged() -> None:
    report = {"results": [_rs("s1", converged=False), _rs("s2", converged=True)]}
    result = _failed_reasoning(report)
    assert any(r["scenario_id"] == "s1" for r in result)
    assert not any(r["scenario_id"] == "s2" for r in result)


def test_failed_reasoning_wrong_decision() -> None:
    report = {"results": [_rs("s1", decision_correct=False)]}
    result = _failed_reasoning(report)
    assert len(result) == 1


def test_failed_reasoning_wrong_planner() -> None:
    report = {"results": [_rs("s1", planner_correct=False)]}
    result = _failed_reasoning(report)
    assert len(result) == 1


def test_failed_reasoning_all_pass() -> None:
    report = {"results": [_rs("s1"), _rs("s2")]}
    assert _failed_reasoning(report) == []


def test_failed_reasoning_missing_results() -> None:
    assert _failed_reasoning({}) == []


# ── _judge_disagreements ──────────────────────────────────────────────────────


def _je(stage: str, score: float):
    return {"stage": stage, "score": {"value": score}, "explanation": "test"}


def test_judge_disagreements_below_threshold() -> None:
    report = {"judge_evaluations": [_je("session", 4.9), _je("session", 5.0)]}
    result = _judge_disagreements(report)
    assert len(result) == 1
    assert result[0]["score"]["value"] == 4.9


def test_judge_disagreements_none_below() -> None:
    report = {"judge_evaluations": [_je("session", 7.0), _je("session", 9.0)]}
    assert _judge_disagreements(report) == []


def test_judge_disagreements_missing_evaluations() -> None:
    assert _judge_disagreements({}) == []


# ── ExperimentRepository — directory creation ─────────────────────────────────


def test_save_creates_history_directory(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    result = _make_pipeline_result()
    repo.save(result, experiment_name="test", _now=_FIXED_NOW)
    assert (tmp_path / "runs" / "history").is_dir()


def test_save_creates_latest_directory(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    repo.save(_make_pipeline_result(), experiment_name="test", _now=_FIXED_NOW)
    assert (tmp_path / "runs" / "latest").is_dir()


def test_save_returns_run_id(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="nightly", _now=_FIXED_NOW)
    assert rid == "20260701_213015_nightly"


def test_save_creates_metadata_json(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="test", _now=_FIXED_NOW)
    meta_path = tmp_path / "runs" / "history" / rid / "metadata.json"
    assert meta_path.exists()
    data = json.loads(meta_path.read_text())
    assert data["run_id"] == rid
    assert data["experiment_name"] == "test"


def test_save_creates_summary_json(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(duration=2.5), experiment_name="t", _now=_FIXED_NOW)
    path = tmp_path / "runs" / "history" / rid / "summary.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["duration_seconds"] == pytest.approx(2.5)


def test_save_skips_absent_reports(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    run_dir = tmp_path / "runs" / "history" / rid
    # No retrieval report was supplied — file should not exist
    assert not (run_dir / "retrieval_report.json").exists()


def test_save_writes_retrieval_report(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    ret = _make_minimal_retrieval_report()
    rid = repo.save(
        _make_pipeline_result(retrieval=ret),
        experiment_name="t", _now=_FIXED_NOW,
    )
    path = tmp_path / "runs" / "history" / rid / "retrieval_report.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert "aggregate_metrics" in data


def test_save_writes_reasoning_report(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    reas = _make_minimal_reasoning_report()
    rid = repo.save(
        _make_pipeline_result(reasoning=reas),
        experiment_name="t", _now=_FIXED_NOW,
    )
    path = tmp_path / "runs" / "history" / rid / "reasoning_report.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert "metrics" in data


def test_save_writes_quality_report(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    q = _make_minimal_quality_report()
    rid = repo.save(
        _make_pipeline_result(quality=q),
        experiment_name="t", _now=_FIXED_NOW,
    )
    path = tmp_path / "runs" / "history" / rid / "quality_report.json"
    assert path.exists()


def test_save_writes_convenience_files(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    run_dir = tmp_path / "runs" / "history" / rid
    assert (run_dir / "failed_queries.json").exists()
    assert (run_dir / "failed_reasoning.json").exists()
    assert (run_dir / "judge_disagreements.json").exists()


def test_save_writes_failed_queries_content(tmp_path) -> None:
    """A retrieval report with one failing query should appear in failed_queries.json."""
    from app.evaluation.gold_dataset import CorpusFingerprintPlaceholder
    from app.evaluation.gold_loader import GoldDatasetResolutionSummary
    from app.evaluation.harness import (
        AggregateMetrics, CoverageBreakdown, EvaluationConfig,
        EvaluationDatasetInfo, EvaluationReport, CorpusStatistics,
        QueryEvaluationOutcome, QueryMetricResult,
    )

    # Build a per-query entry with recall=0.5 (failure)
    failing_metric = QueryMetricResult(
        query_id="q-bad", k=10, num_relevant=1, num_unresolved_expected=0,
        num_retrieved=10, num_duplicate_retrieved=0,
        recall_at_k=0.5, reciprocal_rank=0.5, dcg_at_k=0.5, idcg_at_k=1.0,
        ndcg_at_k=0.5,
    )
    failing_outcome = QueryEvaluationOutcome(
        query_id="q-bad", category="lexical-overlap", difficulty="easy",
        num_relevant=1, num_unresolved_expected=0, skipped=False,
        skip_reason=None, metric=failing_metric,
    )
    agg = AggregateMetrics(
        num_queries=1, mean_recall_at_k=0.5, mean_reciprocal_rank=0.5,
        mean_ndcg_at_k=0.5, resolution_coverage=1.0,
        queries_with_unresolved_incidents=0,
    )
    ret = EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version="v1", description="d", created_at="2026-01-01",
            author=None, corpus_fingerprint=CorpusFingerprintPlaceholder(),
        ),
        config=EvaluationConfig(k=10, expand=False, rerank=False),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=CorpusFingerprintPlaceholder(),
            distinct_retrieved_incident_count=1,
        ),
        num_evaluated=1, num_skipped=0, aggregate_metrics=agg,
        per_query=(failing_outcome,),
        coverage=CoverageBreakdown(
            total_queries=1, no_match_expected_queries=0,
            fully_resolved_queries=0, partially_resolved_queries=1,
            fully_unresolved_queries=0,
        ),
        resolution_summary=GoldDatasetResolutionSummary(
            total_expected_incidents=1, resolved_count=0, unresolved_identities=(),
        ),
        category_breakdown={}, difficulty_breakdown={},
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(
        _make_pipeline_result(retrieval=ret),
        experiment_name="t", _now=_FIXED_NOW,
    )
    run_dir = tmp_path / "runs" / "history" / rid
    failed = json.loads((run_dir / "failed_queries.json").read_text())
    assert len(failed) == 1
    assert failed[0]["query_id"] == "q-bad"


# ── latest overwrite ──────────────────────────────────────────────────────────


def test_latest_is_overwritten_on_second_save(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    repo.save(_make_pipeline_result(), experiment_name="run1", _now=now1)
    repo.save(_make_pipeline_result(), experiment_name="run2", _now=now2)
    latest_meta = json.loads((tmp_path / "runs" / "latest" / "metadata.json").read_text())
    assert "run2" in latest_meta["run_id"]


def test_latest_metadata_reflects_most_recent_run(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    repo.save(_make_pipeline_result(), experiment_name="alpha", _now=now1)
    rid2 = repo.save(_make_pipeline_result(), experiment_name="beta", _now=now2)
    run = repo.latest()
    assert run is not None
    assert run.metadata.run_id == rid2


# ── load ──────────────────────────────────────────────────────────────────────


def test_load_returns_none_for_unknown(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    assert repo.load("nonexistent") is None


def test_load_returns_experiment_run(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    run = repo.load(rid)
    assert isinstance(run, ExperimentRun)
    assert run.metadata.run_id == rid


def test_load_metadata_fields(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(
        _make_pipeline_result(), experiment_name="nightly",
        retrieval_dataset_version="v3", judge_name="rule",
        _now=_FIXED_NOW,
    )
    run = repo.load(rid)
    assert run.metadata.experiment_name == "nightly"
    assert run.metadata.retrieval_dataset_version == "v3"
    assert run.metadata.judge == "rule"
    assert run.metadata.duration == pytest.approx(1.23)


def test_load_reports_present_as_dicts(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(
        _make_pipeline_result(retrieval=_make_minimal_retrieval_report()),
        experiment_name="t", _now=_FIXED_NOW,
    )
    run = repo.load(rid)
    assert isinstance(run.retrieval_report, dict)


def test_load_absent_reports_are_none(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    run = repo.load(rid)
    assert run.retrieval_report is None
    assert run.reasoning_report is None
    assert run.judge_report is None


def test_load_failed_queries_tuple(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    run = repo.load(rid)
    assert isinstance(run.failed_queries, tuple)
    assert isinstance(run.failed_reasoning, tuple)
    assert isinstance(run.judge_disagreements, tuple)


# ── list_runs ─────────────────────────────────────────────────────────────────


def test_list_runs_empty_when_no_history(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    assert repo.list_runs() == ()


def test_list_runs_returns_all_ids(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    rid1 = repo.save(_make_pipeline_result(), experiment_name="a", _now=now1)
    rid2 = repo.save(_make_pipeline_result(), experiment_name="b", _now=now2)
    runs = repo.list_runs()
    assert rid1 in runs
    assert rid2 in runs


def test_list_runs_chronological_order(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    now3 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
    rid1 = repo.save(_make_pipeline_result(), experiment_name="a", _now=now1)
    rid2 = repo.save(_make_pipeline_result(), experiment_name="b", _now=now2)
    rid3 = repo.save(_make_pipeline_result(), experiment_name="c", _now=now3)
    runs = repo.list_runs()
    assert runs == (rid1, rid2, rid3)


# ── delete ────────────────────────────────────────────────────────────────────


def test_delete_removes_run(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    assert repo.delete(rid) is True
    assert repo.load(rid) is None


def test_delete_returns_false_for_unknown(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    assert repo.delete("does_not_exist") is False


def test_delete_removes_from_list(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    rid1 = repo.save(_make_pipeline_result(), experiment_name="a", _now=now1)
    rid2 = repo.save(_make_pipeline_result(), experiment_name="b", _now=now2)
    repo.delete(rid1)
    runs = repo.list_runs()
    assert rid1 not in runs
    assert rid2 in runs


def test_delete_latest_clears_latest_dir(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    repo.delete(rid)
    # latest/ should be gone or repopulated with previous; since no previous exists:
    latest_dir = tmp_path / "runs" / "latest"
    if latest_dir.exists():
        # If it still exists, its metadata must not point to the deleted run
        meta = json.loads((latest_dir / "metadata.json").read_text())
        assert meta["run_id"] != rid


def test_delete_latest_repopulates_from_remaining(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    rid1 = repo.save(_make_pipeline_result(), experiment_name="a", _now=now1)
    rid2 = repo.save(_make_pipeline_result(), experiment_name="b", _now=now2)
    repo.delete(rid2)  # delete latest
    run = repo.latest()
    assert run is not None
    assert run.metadata.run_id == rid1


# ── history persistence ───────────────────────────────────────────────────────


def test_history_persists_across_repo_instances(tmp_path) -> None:
    base = tmp_path / "runs"
    repo1 = ExperimentRepository(base_dir=base)
    rid = repo1.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)

    repo2 = ExperimentRepository(base_dir=base)
    run = repo2.load(rid)
    assert run is not None
    assert run.metadata.run_id == rid


def test_list_runs_persists_across_instances(tmp_path) -> None:
    base = tmp_path / "runs"
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    repo1 = ExperimentRepository(base_dir=base)
    rid1 = repo1.save(_make_pipeline_result(), experiment_name="a", _now=now1)
    rid2 = repo1.save(_make_pipeline_result(), experiment_name="b", _now=now2)

    repo2 = ExperimentRepository(base_dir=base)
    runs = repo2.list_runs()
    assert rid1 in runs
    assert rid2 in runs


# ── regression persistence ────────────────────────────────────────────────────


def test_regression_report_persisted_when_supplied(tmp_path) -> None:
    # Build a minimal RegressionReport
    ret_report = _make_minimal_retrieval_report()
    from app.evaluation.regression import compare
    regression = compare(ret_report, ret_report)  # same vs same = unchanged

    result = _make_pipeline_result(retrieval=ret_report, ret_regression=regression)
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(result, experiment_name="t", _now=_FIXED_NOW)
    path = tmp_path / "runs" / "history" / rid / "regression_report.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert "verdict" in data


def test_regression_report_absent_when_none(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    assert not (tmp_path / "runs" / "history" / rid / "regression_report.json").exists()


# ── statistics ────────────────────────────────────────────────────────────────


def test_stats_empty_repository(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    stats = repo.stats()
    assert stats.total_runs == 0
    assert stats.best_mrr is None
    assert stats.latest_run is None
    assert stats.trend == ()


def test_stats_counts_runs(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    repo.save(_make_pipeline_result(), experiment_name="a", _now=now1)
    repo.save(_make_pipeline_result(), experiment_name="b", _now=now2)
    stats = repo.stats()
    assert stats.total_runs == 2


def test_stats_best_mrr_from_retrieval_report(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    repo.save(
        _make_pipeline_result(retrieval=_make_minimal_retrieval_report()),
        experiment_name="t", _now=now1,
    )
    stats = repo.stats()
    assert stats.best_mrr == pytest.approx(0.7)
    assert stats.best_ndcg == pytest.approx(0.75)


def test_stats_latest_run_is_most_recent(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    now1 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)
    now2 = datetime(2026, 7, 1, 11, 0, 0, tzinfo=UTC)
    repo.save(_make_pipeline_result(), experiment_name="a", _now=now1)
    rid2 = repo.save(_make_pipeline_result(), experiment_name="b", _now=now2)
    stats = repo.stats()
    assert stats.latest_run == rid2


def test_stats_trend_is_chronological(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    nows = [datetime(2026, 7, 1, h, 0, 0, tzinfo=UTC) for h in (10, 11, 12)]
    rids = [repo.save(_make_pipeline_result(), experiment_name="t", _now=n) for n in nows]
    stats = repo.stats()
    assert stats.trend == tuple(rids)


def test_stats_returns_frozen_type(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    stats = repo.stats()
    assert isinstance(stats, ExperimentStats)


# ── metadata fields ───────────────────────────────────────────────────────────


def test_metadata_duration_from_summary(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(duration=4.56), experiment_name="t", _now=_FIXED_NOW)
    run = repo.load(rid)
    assert run.metadata.duration == pytest.approx(4.56)


def test_metadata_run_id_matches(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    run = repo.load(rid)
    assert run.metadata.run_id == rid


def test_metadata_reasoning_dataset_version(tmp_path) -> None:
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(
        _make_pipeline_result(), experiment_name="t",
        reasoning_dataset_version="r2", _now=_FIXED_NOW,
    )
    run = repo.load(rid)
    assert run.metadata.reasoning_dataset_version == "r2"


# ── CLI inspect ────────────────────────────────────────────────────────────────


def test_cli_latest_prints_metadata(tmp_path, capsys) -> None:
    from scripts.inspect_evaluation_run import main
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    repo.save(_make_pipeline_result(), experiment_name="cli-test", _now=_FIXED_NOW)
    code = main(["latest", "--dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert code == 0
    assert "cli-test" in out


def test_cli_run_id_loads_correct_run(tmp_path, capsys) -> None:
    from scripts.inspect_evaluation_run import main
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="abc", _now=_FIXED_NOW)
    code = main([rid, "--dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert code == 0
    assert rid in out


def test_cli_unknown_run_returns_error(tmp_path, capsys) -> None:
    from scripts.inspect_evaluation_run import main
    code = main(["no_such_run", "--dir", str(tmp_path / "runs")])
    assert code == 1


def test_cli_list_shows_all_runs(tmp_path, capsys) -> None:
    from scripts.inspect_evaluation_run import main
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    rid = repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    code = main(["--list", "--dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert code == 0
    assert rid in out


def test_cli_stats_flag(tmp_path, capsys) -> None:
    from scripts.inspect_evaluation_run import main
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    code = main(["--stats", "--dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert code == 0
    assert "Total runs" in out


def test_cli_no_args_returns_help(tmp_path, capsys) -> None:
    from scripts.inspect_evaluation_run import main
    code = main(["--dir", str(tmp_path / "runs")])
    assert code == 1


def test_cli_failed_queries_flag(tmp_path, capsys) -> None:
    from scripts.inspect_evaluation_run import main
    repo = ExperimentRepository(base_dir=tmp_path / "runs")
    repo.save(_make_pipeline_result(), experiment_name="t", _now=_FIXED_NOW)
    code = main(["latest", "--failed-queries", "--dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert code == 0
    assert "No failed queries" in out
