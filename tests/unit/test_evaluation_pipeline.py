"""Tests for Phase 21E: End-to-End Evaluation Pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.evaluation.benchmark import InMemoryBenchmarkRepository
from app.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
    EvaluationPipelineConfig,
    EvaluationPipelineResult,
    ExecutionSummary,
    PipelineInputs,
    PipelineRepositories,
)
from app.evaluation.judge_benchmark import InMemoryJudgedReasoningBenchmarkRepository
from app.evaluation.reasoning_benchmark import InMemoryReasoningBenchmarkRepository


# ── Minimal fakes for all external dependencies ──────────────────────────────────


def _make_retrieval_report(version: str = "v1", n_queries: int = 2):
    """Build a minimal EvaluationReport that satisfies the pipeline's needs."""
    from app.evaluation.gold_dataset import CorpusFingerprintPlaceholder
    from app.evaluation.harness import (
        AggregateMetrics, CoverageBreakdown, EvaluationConfig,
        EvaluationDatasetInfo, EvaluationReport, CorpusStatistics,
    )
    from app.evaluation.gold_loader import GoldDatasetResolutionSummary

    agg = AggregateMetrics(
        num_queries=n_queries, mean_recall_at_k=0.8, mean_reciprocal_rank=0.7,
        mean_ndcg_at_k=0.75, resolution_coverage=1.0,
        queries_with_unresolved_incidents=0,
    )
    return EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version=version, description="d", created_at="2026-01-01",
            author=None, corpus_fingerprint=CorpusFingerprintPlaceholder(),
        ),
        config=EvaluationConfig(k=10, expand=False, rerank=False),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=CorpusFingerprintPlaceholder(),
            distinct_retrieved_incident_count=5,
        ),
        num_evaluated=n_queries, num_skipped=0, aggregate_metrics=agg,
        per_query=(), coverage=CoverageBreakdown(
            total_queries=n_queries, no_match_expected_queries=0,
            fully_resolved_queries=n_queries, partially_resolved_queries=0,
            fully_unresolved_queries=0,
        ),
        resolution_summary=GoldDatasetResolutionSummary(
            total_expected_incidents=0,
            resolved_count=0,
            unresolved_identities=(),
        ),
        category_breakdown={}, difficulty_breakdown={},
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )


def _make_reasoning_report(n_scenarios: int = 1):
    """Build a minimal InvestigationEvaluationReport."""
    from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics

    metrics = ReasoningMetrics(
        num_scenarios=n_scenarios, planner_accuracy=0.9,
        hypothesis_recall=0.8, hypothesis_precision=0.7,
        decision_accuracy=0.85, critic_accuracy=0.75,
        stopping_accuracy=0.9, convergence_rate=0.8,
        mean_iteration_count=1.5,
    )
    return InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(), metrics=metrics,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )


def _make_judge_evaluation(stage: str = "session"):
    """Build a minimal JudgeEvaluation."""
    from app.evaluation.judge import JudgeEvaluation, make_judge_score
    return JudgeEvaluation(
        stage=stage, score=make_judge_score(7.0),
        explanation="Good overall investigation.",
    )


class FakeGoldDataset:
    """Minimal duck-type for passing through the pipeline."""
    queries: tuple = ()
    version: str = "v1"


class FakeSearchService:
    """Returns a pre-built report; tracks calls."""
    def __init__(self, report) -> None:
        self._report = report
        self.calls: list = []

    def search(self, *args: Any, **kwargs: Any):
        self.calls.append(("search", args, kwargs))
        return []


class FakeEvaluate:
    """Replaces harness.evaluate with a callable returning a fixed report."""
    def __init__(self, report) -> None:
        self._report = report
        self.calls: list = []

    def __call__(self, dataset, search_service, **kwargs):
        self.calls.append((dataset, search_service, kwargs))
        return self._report


class FakeOrchestrator:
    """Duck-types _Orchestrator; not actually called in tests that mock
    evaluate_reasoning_dataset at the module level."""
    def investigate(self, problem: str, **kwargs) -> Any:
        raise NotImplementedError("should not be called directly in pipeline tests")


class FakeJudge:
    """Returns a fixed JudgeEvaluation from evaluate_session."""
    def __init__(self, evaluation) -> None:
        self._evaluation = evaluation
        self.calls: list = []

    def evaluate_session(self, problem: str, session: Any):
        self.calls.append((problem, session))
        return self._evaluation

    def evaluate_plan(self, *a, **k): ...
    def evaluate_hypotheses(self, *a, **k): ...
    def evaluate_decision(self, *a, **k): ...
    def evaluate_critique(self, *a, **k): ...


class FakeReasoningEvaluate:
    """Replaces evaluate_reasoning_dataset."""
    def __init__(self, report) -> None:
        self._report = report
        self.calls: list = []

    def __call__(self, dataset, orchestrator, **kwargs):
        self.calls.append((dataset, orchestrator, kwargs))
        return self._report


# ── Patching helpers ──────────────────────────────────────────────────────────────


def _patch_evaluate(monkeypatch, report):
    fake = FakeEvaluate(report)
    monkeypatch.setattr("app.evaluation.evaluation_pipeline.evaluate", fake)
    return fake


def _patch_reasoning_evaluate(monkeypatch, report):
    fake = FakeReasoningEvaluate(report)
    monkeypatch.setattr(
        "app.evaluation.evaluation_pipeline.evaluate_reasoning_dataset", fake
    )
    return fake


def _default_repos():
    return PipelineRepositories(
        retrieval_repo=InMemoryBenchmarkRepository(),
        reasoning_repo=InMemoryReasoningBenchmarkRepository(),
        judged_repo=InMemoryJudgedReasoningBenchmarkRepository(),
    )


def _pipeline(config=None, repos=None):
    return EvaluationPipeline(
        config=config or EvaluationPipelineConfig(),
        repositories=repos or _default_repos(),
    )


# ── Config ────────────────────────────────────────────────────────────────────────


def test_config_defaults() -> None:
    cfg = EvaluationPipelineConfig()
    assert cfg.run_retrieval is True
    assert cfg.run_reasoning is True
    assert cfg.run_judge is True
    assert cfg.run_failure_analysis is True
    assert cfg.run_validation is True
    assert cfg.persist_results is True
    assert cfg.retrieval_k == 10


def test_config_is_frozen() -> None:
    cfg = EvaluationPipelineConfig()
    with pytest.raises(Exception):
        cfg.run_retrieval = False  # type: ignore[misc]


# ── Retrieval stage ───────────────────────────────────────────────────────────────


def test_retrieval_stage_runs_and_persists(monkeypatch) -> None:
    report = _make_retrieval_report()
    fake = _patch_evaluate(monkeypatch, report)
    repos = _default_repos()
    p = _pipeline(EvaluationPipelineConfig(run_reasoning=False, run_judge=False), repos)

    inputs = PipelineInputs(
        gold_dataset=object(), search_service=object(),
    )
    result = p.run(inputs)

    assert fake.calls, "harness.evaluate was not called"
    assert result.retrieval_report is report
    assert result.retrieval_benchmark is not None
    assert repos.retrieval_repo.latest() is not None


def test_retrieval_skipped_when_flag_false(monkeypatch) -> None:
    fake = _patch_evaluate(monkeypatch, _make_retrieval_report())
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_reasoning=False,
                                            run_judge=False))
    result = p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert not fake.calls
    assert result.retrieval_report is None
    assert any("run_retrieval=False" in w for w in result.execution_summary.warnings)


def test_retrieval_skipped_when_no_dataset(monkeypatch) -> None:
    fake = _patch_evaluate(monkeypatch, _make_retrieval_report())
    p = _pipeline(EvaluationPipelineConfig(run_reasoning=False, run_judge=False))
    result = p.run(PipelineInputs(search_service=object()))
    assert not fake.calls
    assert result.retrieval_report is None
    assert any("gold_dataset" in w for w in result.execution_summary.warnings)


def test_retrieval_skipped_when_no_service(monkeypatch) -> None:
    fake = _patch_evaluate(monkeypatch, _make_retrieval_report())
    p = _pipeline(EvaluationPipelineConfig(run_reasoning=False, run_judge=False))
    result = p.run(PipelineInputs(gold_dataset=object()))
    assert not fake.calls
    assert result.retrieval_report is None
    assert any("search_service" in w for w in result.execution_summary.warnings)


def test_retrieval_regression_runs_when_previous_exists(monkeypatch) -> None:
    report1 = _make_retrieval_report()
    report2 = _make_retrieval_report()
    repos = _default_repos()

    _patch_evaluate(monkeypatch, report1)
    p1 = _pipeline(
        EvaluationPipelineConfig(run_reasoning=False, run_judge=False), repos
    )
    p1.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert repos.retrieval_repo.latest() is not None

    _patch_evaluate(monkeypatch, report2)
    p2 = _pipeline(
        EvaluationPipelineConfig(run_reasoning=False, run_judge=False), repos
    )
    result2 = p2.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert result2.retrieval_regression is not None


def test_retrieval_no_regression_on_first_run(monkeypatch) -> None:
    _patch_evaluate(monkeypatch, _make_retrieval_report())
    p = _pipeline(EvaluationPipelineConfig(run_reasoning=False, run_judge=False))
    result = p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert result.retrieval_regression is None


def test_retrieval_error_is_recorded_not_fatal(monkeypatch) -> None:
    def boom(dataset, svc, **kwargs):
        raise RuntimeError("DB exploded")
    monkeypatch.setattr("app.evaluation.evaluation_pipeline.evaluate", boom)
    p = _pipeline(EvaluationPipelineConfig(run_reasoning=False, run_judge=False))
    result = p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert result.retrieval_report is None
    assert any("Retrieval evaluation failed" in e for e in result.execution_summary.errors)


def test_persist_false_does_not_save_retrieval(monkeypatch) -> None:
    _patch_evaluate(monkeypatch, _make_retrieval_report())
    repos = _default_repos()
    p = _pipeline(
        EvaluationPipelineConfig(persist_results=False, run_reasoning=False, run_judge=False),
        repos,
    )
    result = p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert result.retrieval_benchmark is not None  # still built in memory
    assert repos.retrieval_repo.latest() is None    # not saved


# ── Reasoning stage ───────────────────────────────────────────────────────────────


def test_reasoning_stage_runs_and_persists(monkeypatch) -> None:
    report = _make_reasoning_report()
    fake = _patch_reasoning_evaluate(monkeypatch, report)
    repos = _default_repos()
    p = _pipeline(
        EvaluationPipelineConfig(run_retrieval=False, run_judge=False), repos
    )
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
    ))
    assert fake.calls
    assert result.reasoning_report is report
    assert result.reasoning_benchmark is not None
    assert repos.reasoning_repo.latest() is not None


def test_reasoning_skipped_when_flag_false(monkeypatch) -> None:
    fake = _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_reasoning=False,
                                            run_judge=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
    ))
    assert not fake.calls
    assert result.reasoning_report is None


def test_reasoning_skipped_when_no_dataset(monkeypatch) -> None:
    fake = _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_judge=False))
    result = p.run(PipelineInputs(orchestrator=FakeOrchestrator()))
    assert not fake.calls
    assert any("reasoning_dataset" in w for w in result.execution_summary.warnings)


def test_reasoning_skipped_when_no_orchestrator(monkeypatch) -> None:
    fake = _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_judge=False))
    result = p.run(PipelineInputs(reasoning_dataset=object()))
    assert not fake.calls
    assert any("orchestrator" in w for w in result.execution_summary.warnings)


def test_reasoning_regression_runs_when_previous_exists(monkeypatch) -> None:
    report1 = _make_reasoning_report()
    report2 = _make_reasoning_report()
    repos = _default_repos()

    _patch_reasoning_evaluate(monkeypatch, report1)
    p1 = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_judge=False), repos)
    p1.run(PipelineInputs(reasoning_dataset=object(), orchestrator=FakeOrchestrator()))

    _patch_reasoning_evaluate(monkeypatch, report2)
    p2 = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_judge=False), repos)
    result2 = p2.run(
        PipelineInputs(reasoning_dataset=object(), orchestrator=FakeOrchestrator())
    )
    assert result2.reasoning_regression is not None


def test_reasoning_error_is_recorded_not_fatal(monkeypatch) -> None:
    def boom(ds, orch, **kwargs):
        raise RuntimeError("OOM")
    monkeypatch.setattr(
        "app.evaluation.evaluation_pipeline.evaluate_reasoning_dataset", boom
    )
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_judge=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
    ))
    assert result.reasoning_report is None
    assert any("Reasoning evaluation failed" in e for e in result.execution_summary.errors)


# ── Judge stage ───────────────────────────────────────────────────────────────────


def test_judge_stage_runs_when_reasoning_available(monkeypatch) -> None:
    reasoning_report = _make_reasoning_report()
    _patch_reasoning_evaluate(monkeypatch, reasoning_report)
    eval_j = _make_judge_evaluation()
    judge = FakeJudge(eval_j)
    repos = _default_repos()
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False), repos)
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(), judge=judge,
    ))
    assert result.judge_report is not None
    assert repos.judged_repo.latest() is not None


def test_judge_skipped_when_flag_false(monkeypatch) -> None:
    _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    judge = FakeJudge(_make_judge_evaluation())
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_judge=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(), judge=judge,
    ))
    assert result.judge_report is None
    assert any("run_judge=False" in w for w in result.execution_summary.warnings)


def test_judge_skipped_when_no_judge_supplied(monkeypatch) -> None:
    _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
    ))
    assert result.judge_report is None
    assert any("no judge" in w for w in result.execution_summary.warnings)


def test_judge_skipped_when_reasoning_failed(monkeypatch) -> None:
    def boom(ds, orch, **kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr(
        "app.evaluation.evaluation_pipeline.evaluate_reasoning_dataset", boom
    )
    judge = FakeJudge(_make_judge_evaluation())
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(), judge=judge,
    ))
    assert result.judge_report is None
    assert any("reasoning benchmark not available" in w
               for w in result.execution_summary.warnings)


def test_judge_per_result_error_is_non_fatal(monkeypatch) -> None:
    from app.evaluation.reasoning_harness import InvestigationResult
    from types import SimpleNamespace

    result_obj = SimpleNamespace(
        scenario_id="s1", problem="p", session=object(),
    )
    # reasoning report with one result that will cause judge to fail
    from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics
    metrics = ReasoningMetrics(
        num_scenarios=1, planner_accuracy=None, hypothesis_recall=None,
        hypothesis_precision=None, decision_accuracy=None, critic_accuracy=None,
        stopping_accuracy=None, convergence_rate=None, mean_iteration_count=None,
    )
    reasoning_report = InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(result_obj,),  # type: ignore[arg-type]
        metrics=metrics,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=0.1,
    )
    _patch_reasoning_evaluate(monkeypatch, reasoning_report)

    class BoomJudge:
        def evaluate_session(self, p, s):
            raise RuntimeError("judge exploded")

    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(), judge=BoomJudge(),
    ))
    assert any("evaluate_session failed" in e for e in result.execution_summary.errors)
    # judge_report should still be created (with 0 evaluations)
    assert result.judge_report is not None
    assert result.judge_report.judge_evaluations == ()


def test_judge_persist_false_does_not_save(monkeypatch) -> None:
    _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    repos = _default_repos()
    p = _pipeline(
        EvaluationPipelineConfig(run_retrieval=False, persist_results=False), repos
    )
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
        judge=FakeJudge(_make_judge_evaluation()),
    ))
    assert result.judge_report is not None
    assert repos.judged_repo.latest() is None


# ── Failure analysis stage ────────────────────────────────────────────────────────


def test_quality_report_produced_when_retrieval_available(monkeypatch) -> None:
    _patch_evaluate(monkeypatch, _make_retrieval_report())
    p = _pipeline(EvaluationPipelineConfig(run_reasoning=False, run_judge=False))
    result = p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert result.quality_report is not None


def test_quality_report_skipped_when_flag_false(monkeypatch) -> None:
    _patch_evaluate(monkeypatch, _make_retrieval_report())
    p = _pipeline(EvaluationPipelineConfig(
        run_reasoning=False, run_judge=False, run_failure_analysis=False,
    ))
    result = p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert result.quality_report is None
    assert any("run_failure_analysis=False" in w
               for w in result.execution_summary.warnings)


def test_quality_report_built_even_with_no_inputs() -> None:
    p = _pipeline(EvaluationPipelineConfig(
        run_retrieval=False, run_reasoning=False, run_judge=False,
    ))
    result = p.run(PipelineInputs())
    assert result.quality_report is not None
    assert "No failures" in result.quality_report.overall_summary


# ── Validation stage ──────────────────────────────────────────────────────────────


def test_validation_report_produced_after_judge_runs(monkeypatch) -> None:
    _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    repos = _default_repos()
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False), repos)
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
        judge=FakeJudge(_make_judge_evaluation()),
    ))
    # With 1 run and INSUFFICIENT_DATA expected (single run, no correlation)
    assert result.judge_validation_report is not None


def test_validation_skipped_when_flag_false(monkeypatch) -> None:
    _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_validation=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
        judge=FakeJudge(_make_judge_evaluation()),
    ))
    assert result.judge_validation_report is None
    assert any("run_validation=False" in w for w in result.execution_summary.warnings)


def test_validation_skipped_when_no_judged_repo() -> None:
    repos = PipelineRepositories(
        retrieval_repo=None, reasoning_repo=None, judged_repo=None,
    )
    p = _pipeline(
        EvaluationPipelineConfig(run_retrieval=False, run_reasoning=False, run_judge=False),
        repos,
    )
    result = p.run(PipelineInputs())
    assert result.judge_validation_report is None
    assert any("no judged_repo" in w for w in result.execution_summary.warnings)


# ── Execution summary ─────────────────────────────────────────────────────────────


def test_execution_summary_always_present() -> None:
    p = _pipeline(EvaluationPipelineConfig(
        run_retrieval=False, run_reasoning=False, run_judge=False,
    ))
    result = p.run(PipelineInputs())
    assert isinstance(result.execution_summary, ExecutionSummary)
    assert result.execution_summary.start_time
    assert result.execution_summary.end_time
    assert result.execution_summary.duration_seconds >= 0


def test_execution_summary_counts_queries(monkeypatch) -> None:
    _patch_evaluate(monkeypatch, _make_retrieval_report(n_queries=7))
    p = _pipeline(EvaluationPipelineConfig(run_reasoning=False, run_judge=False))
    result = p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert result.execution_summary.retrieval_queries == 7


def test_execution_summary_counts_scenarios(monkeypatch) -> None:
    _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report(n_scenarios=4))
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False, run_judge=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
    ))
    assert result.execution_summary.reasoning_scenarios == 4


def test_execution_summary_counts_judge_evals(monkeypatch) -> None:
    from app.evaluation.reasoning_harness import (
        InvestigationEvaluationReport, ReasoningMetrics,
    )
    from types import SimpleNamespace

    metrics = ReasoningMetrics(
        num_scenarios=2, planner_accuracy=None, hypothesis_recall=None,
        hypothesis_precision=None, decision_accuracy=None, critic_accuracy=None,
        stopping_accuracy=None, convergence_rate=None, mean_iteration_count=None,
    )
    report = InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(
            SimpleNamespace(scenario_id="s1", problem="p1", session=object()),
            SimpleNamespace(scenario_id="s2", problem="p2", session=object()),
        ),  # type: ignore[arg-type]
        metrics=metrics,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=0.1,
    )
    _patch_reasoning_evaluate(monkeypatch, report)
    p = _pipeline(EvaluationPipelineConfig(run_retrieval=False))
    result = p.run(PipelineInputs(
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
        judge=FakeJudge(_make_judge_evaluation()),
    ))
    assert result.execution_summary.judge_evaluations == 2


# ── Result immutability ───────────────────────────────────────────────────────────


def test_pipeline_result_is_frozen() -> None:
    p = _pipeline(EvaluationPipelineConfig(
        run_retrieval=False, run_reasoning=False, run_judge=False,
    ))
    result = p.run(PipelineInputs())
    with pytest.raises(Exception):
        result.quality_report = None  # type: ignore[misc]


def test_execution_summary_is_frozen() -> None:
    p = _pipeline(EvaluationPipelineConfig(
        run_retrieval=False, run_reasoning=False, run_judge=False,
    ))
    result = p.run(PipelineInputs())
    with pytest.raises(Exception):
        result.execution_summary.retrieval_queries = 99  # type: ignore[misc]


# ── Repository interaction ────────────────────────────────────────────────────────


def test_pipeline_uses_injected_repositories(monkeypatch) -> None:
    _patch_evaluate(monkeypatch, _make_retrieval_report())
    repos = PipelineRepositories(
        retrieval_repo=InMemoryBenchmarkRepository(),
        reasoning_repo=None,
        judged_repo=None,
    )
    p = _pipeline(
        EvaluationPipelineConfig(run_reasoning=False, run_judge=False, run_validation=False),
        repos,
    )
    p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    assert repos.retrieval_repo.latest() is not None


def test_pipeline_handles_none_repos_gracefully(monkeypatch) -> None:
    _patch_evaluate(monkeypatch, _make_retrieval_report())
    repos = PipelineRepositories()  # all None
    p = _pipeline(
        EvaluationPipelineConfig(run_reasoning=False, run_judge=False, run_validation=False),
        repos,
    )
    result = p.run(PipelineInputs(gold_dataset=object(), search_service=object()))
    # Should succeed; retrieval benchmark built in memory, not persisted
    assert result.retrieval_benchmark is not None
    assert result.retrieval_regression is None


# ── End-to-end: full pipeline with all mocks ─────────────────────────────────────


def test_full_pipeline_with_all_stages_mocked(monkeypatch) -> None:
    retrieval_report = _make_retrieval_report()
    reasoning_report = _make_reasoning_report()
    _patch_evaluate(monkeypatch, retrieval_report)
    _patch_reasoning_evaluate(monkeypatch, reasoning_report)
    repos = _default_repos()
    p = _pipeline(EvaluationPipelineConfig(experiment_name="full-test"), repos)

    result = p.run(PipelineInputs(
        gold_dataset=object(),
        search_service=object(),
        reasoning_dataset=object(),
        orchestrator=FakeOrchestrator(),
        judge=FakeJudge(_make_judge_evaluation()),
    ))

    assert result.retrieval_report is retrieval_report
    assert result.reasoning_report is reasoning_report
    assert result.retrieval_benchmark is not None
    assert result.reasoning_benchmark is not None
    assert result.judge_report is not None
    assert result.quality_report is not None
    assert result.judge_validation_report is not None
    assert result.execution_summary.errors == ()


def test_pipeline_second_run_produces_regression(monkeypatch) -> None:
    """Two consecutive runs against the same repos produce regression reports."""
    repos = _default_repos()

    _patch_evaluate(monkeypatch, _make_retrieval_report())
    _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    cfg = EvaluationPipelineConfig(experiment_name="reg-test", run_judge=False)

    p1 = _pipeline(cfg, repos)
    r1 = p1.run(PipelineInputs(
        gold_dataset=object(), search_service=object(),
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
    ))
    assert r1.retrieval_regression is None
    assert r1.reasoning_regression is None

    _patch_evaluate(monkeypatch, _make_retrieval_report())
    _patch_reasoning_evaluate(monkeypatch, _make_reasoning_report())
    p2 = _pipeline(cfg, repos)
    r2 = p2.run(PipelineInputs(
        gold_dataset=object(), search_service=object(),
        reasoning_dataset=object(), orchestrator=FakeOrchestrator(),
    ))
    assert r2.retrieval_regression is not None
    assert r2.reasoning_regression is not None
