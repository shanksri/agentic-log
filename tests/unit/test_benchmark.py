from __future__ import annotations

import pytest

from app.evaluation.benchmark import (
    BenchmarkRepository,
    BenchmarkRun,
    FileBenchmarkRepository,
    InMemoryBenchmarkRepository,
    compare_latest_against_previous,
    compare_runs,
    create_benchmark_run,
    metric_history,
    regression_history,
)
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
from app.evaluation.metrics import QueryMetricResult
from app.evaluation.regression import RegressionReport, Verdict, compare


def _metric(*, recall: float = 1.0, rr: float = 1.0, ndcg: float = 1.0) -> QueryMetricResult:
    return QueryMetricResult(
        query_id="q-1", k=10, num_relevant=1, num_unresolved_expected=0, num_retrieved=1,
        num_duplicate_retrieved=0, recall_at_k=recall, reciprocal_rank=rr, dcg_at_k=1.0,
        idcg_at_k=1.0, ndcg_at_k=ndcg,
    )


def _outcome() -> QueryEvaluationOutcome:
    return QueryEvaluationOutcome(
        query_id="q-1", category="lexical-overlap", difficulty="easy", num_relevant=1,
        num_unresolved_expected=0, skipped=False, skip_reason=None, metric=_metric(),
    )


def make_report(*, recall: float = 1.0, rr: float = 1.0, ndcg: float = 1.0) -> EvaluationReport:
    return EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version="2.0.0", description="d", created_at="2026-01-01T00:00:00Z", author=None,
            corpus_fingerprint=CorpusFingerprintPlaceholder(),
        ),
        config=EvaluationConfig(k=10, expand=False, rerank=False),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=CorpusFingerprintPlaceholder(), distinct_retrieved_incident_count=1
        ),
        num_evaluated=1,
        num_skipped=0,
        aggregate_metrics=AggregateMetrics(
            num_queries=1, mean_recall_at_k=recall, mean_reciprocal_rank=rr,
            mean_ndcg_at_k=ndcg, resolution_coverage=1.0, queries_with_unresolved_incidents=0,
        ),
        per_query=(_outcome(),),
        coverage=CoverageBreakdown(
            total_queries=1, no_match_expected_queries=0, fully_resolved_queries=1,
            partially_resolved_queries=0, fully_unresolved_queries=0,
        ),
        resolution_summary=GoldDatasetResolutionSummary(
            total_expected_incidents=1, resolved_count=1, unresolved_identities=()
        ),
        category_breakdown={}, difficulty_breakdown={},
        started_at="2026-06-26T00:00:00+00:00", finished_at="2026-06-26T00:00:01+00:00",
        duration_seconds=1.0,
    )


# ── create_benchmark_run ───────────────────────────────────────────────────────


def test_create_benchmark_run_derives_config_from_report() -> None:
    report = make_report()
    run = create_benchmark_run(experiment_name="exp-1", report=report)

    assert isinstance(run, BenchmarkRun)
    assert run.config == report.config
    assert run.experiment_name == "exp-1"
    assert run.report is report
    assert run.git_commit_sha is None
    assert run.notes is None
    assert run.regression is None
    assert run.run_id  # auto-generated, non-empty
    assert run.timestamp  # auto-generated, non-empty


def test_create_benchmark_run_accepts_explicit_fields() -> None:
    report = make_report()
    run = create_benchmark_run(
        experiment_name="exp-1", report=report, run_id="run-123",
        timestamp="2026-01-01T00:00:00+00:00", git_commit_sha="abc123", notes="first run",
    )

    assert run.run_id == "run-123"
    assert run.timestamp == "2026-01-01T00:00:00+00:00"
    assert run.git_commit_sha == "abc123"
    assert run.notes == "first run"


def test_benchmark_run_is_immutable() -> None:
    run = create_benchmark_run(experiment_name="exp-1", report=make_report())
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        run.notes = "changed"  # type: ignore[misc]


# ── Repository: shared behavior, parametrized over both backends ────────────


@pytest.fixture(params=["memory", "file"])
def repository(request: pytest.FixtureRequest, tmp_path) -> BenchmarkRepository:
    if request.param == "memory":
        return InMemoryBenchmarkRepository()
    return FileBenchmarkRepository(tmp_path / "benchmarks")


def test_save_and_get_round_trips(repository: BenchmarkRepository) -> None:
    run = create_benchmark_run(experiment_name="exp-1", report=make_report(), run_id="run-1")
    repository.save(run)

    fetched = repository.get("run-1")

    assert fetched is not None
    assert fetched.run_id == "run-1"
    assert fetched.experiment_name == "exp-1"
    assert fetched.config == run.config
    assert fetched.report.aggregate_metrics.mean_recall_at_k == 1.0


def test_get_missing_run_returns_none(repository: BenchmarkRepository) -> None:
    assert repository.get("does-not-exist") is None


def test_save_duplicate_run_id_raises(repository: BenchmarkRepository) -> None:
    run = create_benchmark_run(experiment_name="exp-1", report=make_report(), run_id="run-1")
    repository.save(run)

    duplicate = create_benchmark_run(
        experiment_name="exp-1", report=make_report(), run_id="run-1"
    )
    with pytest.raises(ValueError, match="run-1"):
        repository.save(duplicate)


def test_list_runs_orders_by_timestamp_ascending(repository: BenchmarkRepository) -> None:
    run_a = create_benchmark_run(
        experiment_name="exp-1", report=make_report(), run_id="run-a",
        timestamp="2026-01-03T00:00:00+00:00",
    )
    run_b = create_benchmark_run(
        experiment_name="exp-1", report=make_report(), run_id="run-b",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    run_c = create_benchmark_run(
        experiment_name="exp-1", report=make_report(), run_id="run-c",
        timestamp="2026-01-02T00:00:00+00:00",
    )
    repository.save(run_a)
    repository.save(run_b)
    repository.save(run_c)

    ordered = repository.list_runs()

    assert [run.run_id for run in ordered] == ["run-b", "run-c", "run-a"]


def test_list_runs_filters_by_experiment_name(repository: BenchmarkRepository) -> None:
    repository.save(
        create_benchmark_run(experiment_name="exp-1", report=make_report(), run_id="run-1")
    )
    repository.save(
        create_benchmark_run(experiment_name="exp-2", report=make_report(), run_id="run-2")
    )

    exp1_runs = repository.list_runs(experiment_name="exp-1")
    assert [run.run_id for run in exp1_runs] == ["run-1"]


def test_latest_returns_most_recent_by_timestamp(repository: BenchmarkRepository) -> None:
    repository.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(), run_id="run-old",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    repository.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(), run_id="run-new",
            timestamp="2026-01-02T00:00:00+00:00",
        )
    )

    assert repository.latest().run_id == "run-new"


def test_latest_returns_none_when_empty(repository: BenchmarkRepository) -> None:
    assert repository.latest() is None


def test_delete_removes_run_and_returns_true(repository: BenchmarkRepository) -> None:
    repository.save(
        create_benchmark_run(experiment_name="exp-1", report=make_report(), run_id="run-1")
    )
    assert repository.delete("run-1") is True
    assert repository.get("run-1") is None


def test_delete_missing_run_returns_false(repository: BenchmarkRepository) -> None:
    assert repository.delete("does-not-exist") is False


# ── FileBenchmarkRepository: serialization/deserialization specifics ────────


def test_file_repository_round_trips_full_report_and_regression(tmp_path) -> None:
    baseline_report = make_report(recall=0.5, rr=0.5, ndcg=0.5)
    candidate_report = make_report(recall=0.8, rr=0.7, ndcg=0.75)
    regression = compare(baseline_report, candidate_report)

    run = create_benchmark_run(
        experiment_name="exp-1", report=candidate_report, regression=regression,
        git_commit_sha="deadbeef", notes="testing serialization", run_id="run-1",
    )
    repo = FileBenchmarkRepository(tmp_path / "benchmarks")
    repo.save(run)

    fetched = repo.get("run-1")

    assert fetched == run  # full structural equality after round-trip
    assert fetched.regression.verdict == Verdict.IMPROVED
    assert fetched.regression.overall.recall_at_k.delta == pytest.approx(0.3)
    assert fetched.report.aggregate_metrics.mean_recall_at_k == 0.8


def test_file_repository_writes_one_json_file_per_run(tmp_path) -> None:
    directory = tmp_path / "benchmarks"
    repo = FileBenchmarkRepository(directory)
    repo.save(create_benchmark_run(experiment_name="exp-1", report=make_report(), run_id="run-1"))

    assert (directory / "run-1.json").exists()


def test_file_repository_persists_across_instances(tmp_path) -> None:
    directory = tmp_path / "benchmarks"
    FileBenchmarkRepository(directory).save(
        create_benchmark_run(experiment_name="exp-1", report=make_report(), run_id="run-1")
    )

    reopened = FileBenchmarkRepository(directory)
    assert reopened.get("run-1") is not None
    assert reopened.list_runs()[0].run_id == "run-1"


# ── Comparison utilities: delegation, not duplication ─────────────────────────


def test_compare_runs_delegates_to_regression_compare() -> None:
    run_a = create_benchmark_run(
        experiment_name="exp-1", report=make_report(recall=0.5, rr=0.5, ndcg=0.5)
    )
    run_b = create_benchmark_run(
        experiment_name="exp-1", report=make_report(recall=0.9, rr=0.9, ndcg=0.9)
    )

    via_helper = compare_runs(run_a, run_b)
    via_direct_call = compare(run_a.report, run_b.report)

    assert isinstance(via_helper, RegressionReport)
    assert via_helper.verdict == via_direct_call.verdict == Verdict.IMPROVED
    assert via_helper.overall.recall_at_k.delta == via_direct_call.overall.recall_at_k.delta


def test_compare_latest_against_previous_uses_two_most_recent_runs() -> None:
    repo = InMemoryBenchmarkRepository()
    repo.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(recall=0.5, rr=0.5, ndcg=0.5),
            run_id="run-1", timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    repo.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(recall=0.9, rr=0.9, ndcg=0.9),
            run_id="run-2", timestamp="2026-01-02T00:00:00+00:00",
        )
    )

    result = compare_latest_against_previous(repo)

    assert result is not None
    assert result.verdict == Verdict.IMPROVED
    assert result.comparison.baseline_dataset_version == "2.0.0"


def test_compare_latest_against_previous_returns_none_with_fewer_than_two_runs() -> None:
    repo = InMemoryBenchmarkRepository()
    assert compare_latest_against_previous(repo) is None

    repo.save(create_benchmark_run(experiment_name="exp-1", report=make_report(), run_id="run-1"))
    assert compare_latest_against_previous(repo) is None


def test_metric_history_is_ordered_projection_not_recomputation() -> None:
    repo = InMemoryBenchmarkRepository()
    repo.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(recall=0.5, rr=0.6, ndcg=0.7),
            run_id="run-1", timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    repo.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(recall=0.8, rr=0.85, ndcg=0.9),
            run_id="run-2", timestamp="2026-01-02T00:00:00+00:00",
        )
    )

    history = metric_history(repo)

    assert [entry.run_id for entry in history] == ["run-1", "run-2"]
    assert history[0].mean_recall_at_k == 0.5
    assert history[1].mean_recall_at_k == 0.8


def test_regression_history_compares_every_consecutive_pair() -> None:
    repo = InMemoryBenchmarkRepository()
    repo.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(recall=0.3, rr=0.3, ndcg=0.3),
            run_id="run-1", timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    repo.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(recall=0.6, rr=0.6, ndcg=0.6),
            run_id="run-2", timestamp="2026-01-02T00:00:00+00:00",
        )
    )
    repo.save(
        create_benchmark_run(
            experiment_name="exp-1", report=make_report(recall=0.2, rr=0.2, ndcg=0.2),
            run_id="run-3", timestamp="2026-01-03T00:00:00+00:00",
        )
    )

    history = regression_history(repo)

    assert len(history) == 2
    assert history[0].verdict == Verdict.IMPROVED  # run-1 -> run-2
    assert history[1].verdict == Verdict.REGRESSED  # run-2 -> run-3


def test_regression_history_empty_with_fewer_than_two_runs() -> None:
    repo = InMemoryBenchmarkRepository()
    assert regression_history(repo) == ()

    repo.save(create_benchmark_run(experiment_name="exp-1", report=make_report(), run_id="run-1"))
    assert regression_history(repo) == ()
