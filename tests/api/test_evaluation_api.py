"""API tests for Phase 21G: Evaluation REST API.

No database, no OpenAI, no retrieval.  All evaluation components are mocked;
only the Pydantic/FastAPI routing layer is exercised.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.auth import require_api_key
from app.db.session import get_db
from app.main import app
from app.api.routes.evaluation import _get_repo


# ── Shared test fixtures ──────────────────────────────────────────────────────


def _make_execution_summary(duration: float = 0.5):
    from app.evaluation.evaluation_pipeline import ExecutionSummary
    return ExecutionSummary(
        start_time="2026-07-01T00:00:00+00:00",
        end_time="2026-07-01T00:00:01+00:00",
        duration_seconds=duration,
        retrieval_queries=0,
        reasoning_scenarios=0,
        judge_evaluations=0,
        warnings=(),
        errors=(),
    )


def _make_pipeline_result(
    retrieval_report=None,
    reasoning_report=None,
    quality_report=None,
):
    from app.evaluation.evaluation_pipeline import EvaluationPipelineResult
    return EvaluationPipelineResult(
        retrieval_report=retrieval_report,
        retrieval_regression=None,
        retrieval_benchmark=None,
        reasoning_report=reasoning_report,
        reasoning_regression=None,
        reasoning_benchmark=None,
        judge_report=None,
        judge_validation_report=None,
        quality_report=quality_report,
        execution_summary=_make_execution_summary(),
    )


def _make_minimal_retrieval_report():
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
    corpus_fp = CorpusFingerprintPlaceholder()
    return EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version="v1", description="d", created_at="2026-01-01",
            author=None, corpus_fingerprint=corpus_fp,
        ),
        config=EvaluationConfig(k=10, expand=False, rerank=False),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=corpus_fp, distinct_retrieved_incident_count=3,
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
    from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics
    metrics = ReasoningMetrics(
        num_scenarios=1, planner_accuracy=1.0, hypothesis_recall=1.0,
        hypothesis_precision=1.0, decision_accuracy=1.0, critic_accuracy=1.0,
        stopping_accuracy=1.0, convergence_rate=1.0, mean_iteration_count=1.0,
    )
    return InvestigationEvaluationReport(
        dataset_version="v1", dataset_description="d", n_hypotheses=3,
        results=(), metrics=metrics,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )


# ── Client builders ───────────────────────────────────────────────────────────


def _fake_db():
    yield object()


class FakeExperimentRepo:
    """In-memory ExperimentRepository stub for API tests."""

    def __init__(self) -> None:
        from app.evaluation.experiment_tracking import (
            ExperimentRun, ExperimentStats, RunMetadata,
        )
        self._run_ids: list[str] = []
        self._runs: dict[str, Any] = {}

    def save(self, result, *, experiment_name="default", **kwargs) -> str:
        rid = f"20260701_000000_{experiment_name}"
        self._run_ids.append(rid)
        from app.evaluation.experiment_tracking import ExperimentRun, RunMetadata
        meta = RunMetadata(
            run_id=rid, timestamp="2026-07-01T00:00:00+00:00",
            git_commit=None, experiment_name=experiment_name,
            retrieval_dataset_version=None, reasoning_dataset_version=None,
            judge=None, duration=0.5, configuration={},
        )
        run = ExperimentRun(
            metadata=meta,
            summary={"duration_seconds": 0.5},
            retrieval_report=None, reasoning_report=None,
            judge_report=None, quality_report=None,
            validation_report=None, regression_report=None,
            failed_queries=(), failed_reasoning=(), judge_disagreements=(),
        )
        self._runs[rid] = run
        return rid

    def list_runs(self) -> tuple[str, ...]:
        return tuple(self._run_ids)

    def latest(self):
        if not self._run_ids:
            return None
        return self._runs[self._run_ids[-1]]

    def load(self, run_id: str):
        return self._runs.get(run_id)

    def delete(self, run_id: str) -> bool:
        if run_id in self._runs:
            del self._runs[run_id]
            self._run_ids.remove(run_id)
            return True
        return False

    def stats(self):
        from app.evaluation.experiment_tracking import ExperimentStats
        has_runs = bool(self._run_ids)
        return ExperimentStats(
            total_runs=len(self._run_ids),
            best_mrr=0.7 if has_runs else None,
            best_ndcg=0.75 if has_runs else None,
            best_reasoning_accuracy=0.9 if has_runs else None,
            latest_run=self._run_ids[-1] if has_runs else None,
            trend=tuple(self._run_ids),
        )


def _client(overrides: dict | None = None) -> tuple[TestClient, FakeExperimentRepo]:
    """Return (client, fake_repo) with DB and repo dependencies overridden.

    Phase 23B: also bypasses ``require_api_key`` (no-op override) — these
    tests exercise the evaluation routes, not authentication. See
    tests/api/test_authentication.py for the real auth behavior.
    """
    fake_repo = FakeExperimentRepo()
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[_get_repo] = lambda: fake_repo
    app.dependency_overrides[require_api_key] = lambda: None
    if overrides:
        app.dependency_overrides.update(overrides)
    return TestClient(app, raise_server_exceptions=False), fake_repo


# ── POST /evaluation/query ────────────────────────────────────────────────────


def test_query_eval_no_db_returns_503(monkeypatch) -> None:
    """If the search service fails to build, return 503."""
    client, _ = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation._build_search_service",
            lambda db: (_ for _ in ()).throw(
                __import__("fastapi").HTTPException(status_code=503, detail="no DB")
            ),
        )
        resp = client.post(
            "/evaluation/query",
            json={"query": "memory leak", "expected_incident_ids": [], "k": 5},
        )
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_query_eval_invalid_uuid_returns_422(monkeypatch) -> None:
    client, _ = _client()
    try:
        import uuid as _uuid
        fake_result = MagicMock()
        fake_result.incident.id = _uuid.uuid4()
        fake_result.incident.title = "Fake"
        fake_result.similarity_score = 0.9

        monkeypatch.setattr(
            "app.api.routes.evaluation._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [fake_result],
            ),
        )
        resp = client.post(
            "/evaluation/query",
            json={
                "query": "cpu spike",
                "expected_incident_ids": ["not-a-uuid"],
                "k": 10,
            },
        )
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_query_eval_success(monkeypatch) -> None:
    import uuid as _uuid

    client, _ = _client()
    try:
        iid = _uuid.uuid4()
        fake_result = MagicMock()
        fake_result.incident.id = iid
        fake_result.incident.title = "OOM crash"
        fake_result.similarity_score = 0.95

        monkeypatch.setattr(
            "app.api.routes.evaluation._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [fake_result],
            ),
        )
        resp = client.post(
            "/evaluation/query",
            json={
                "query": "out of memory",
                "expected_incident_ids": [str(iid)],
                "k": 10,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["query"] == "out of memory"
        assert data["recall_at_k"] == pytest.approx(1.0)
        assert data["rank_of_first_expected"] == 1
        assert len(data["retrieved"]) == 1
        assert data["retrieved"][0]["is_expected"] is True
    finally:
        app.dependency_overrides.clear()


def test_query_eval_empty_expected_no_crash(monkeypatch) -> None:
    import uuid as _uuid

    client, _ = _client()
    try:
        fake_result = MagicMock()
        fake_result.incident.id = _uuid.uuid4()
        fake_result.incident.title = "T"
        fake_result.similarity_score = 0.5

        monkeypatch.setattr(
            "app.api.routes.evaluation._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [fake_result],
            ),
        )
        resp = client.post(
            "/evaluation/query",
            json={"query": "crash", "expected_incident_ids": [], "k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rank_of_first_expected"] is None
    finally:
        app.dependency_overrides.clear()


# ── POST /evaluation/retrieval ────────────────────────────────────────────────


def test_retrieval_benchmark_dataset_not_found() -> None:
    client, _ = _client()
    try:
        resp = client.post(
            "/evaluation/retrieval",
            json={"dataset_path": "/no/such/file.json", "persist": False},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_retrieval_benchmark_success(monkeypatch, tmp_path) -> None:
    # Write a dummy dataset file so load_gold_dataset gets something
    ds_path = tmp_path / "test_dataset.json"
    ret_report = _make_minimal_retrieval_report()

    client, fake_repo = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation._load_gold_dataset",
            lambda path: object(),
        )
        monkeypatch.setattr(
            "app.api.routes.evaluation._build_search_service",
            lambda db: object(),
        )
        monkeypatch.setattr(
            "app.evaluation.harness.evaluate",
            lambda dataset, svc, **kw: ret_report,
        )
        monkeypatch.setattr(
            "app.api.routes.evaluation.evaluate",  # local import in route
            lambda dataset, svc, **kw: ret_report,
            raising=False,
        )

        # Patch the evaluate function that the route uses
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(
            ev_mod,
            "_build_search_service",
            lambda db: MagicMock(),
        )
        monkeypatch.setattr(
            ev_mod,
            "_load_gold_dataset",
            lambda path: MagicMock(),
        )

        # Patch the harness.evaluate call inside the route
        with patch("app.evaluation.harness.evaluate", return_value=ret_report):
            resp = client.post(
                "/evaluation/retrieval",
                json={
                    "dataset_path": str(tmp_path / "x.json"),
                    "persist": True,
                    "experiment_name": "test-ret",
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "evaluation_report" in data
        assert data["evaluation_report"]["aggregate_metrics"]["mean_reciprocal_rank"] == pytest.approx(0.7)
    finally:
        app.dependency_overrides.clear()


def test_retrieval_benchmark_persist_creates_run(monkeypatch) -> None:
    ret_report = _make_minimal_retrieval_report()
    client, fake_repo = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_load_gold_dataset", lambda p: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_search_service", lambda db: MagicMock())
        with patch("app.evaluation.harness.evaluate", return_value=ret_report):
            resp = client.post(
                "/evaluation/retrieval",
                json={"dataset_path": "any.json", "persist": True, "experiment_name": "test"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] is not None
    finally:
        app.dependency_overrides.clear()


def test_retrieval_benchmark_no_persist_no_run_id(monkeypatch) -> None:
    ret_report = _make_minimal_retrieval_report()
    client, fake_repo = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_load_gold_dataset", lambda p: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_search_service", lambda db: MagicMock())
        with patch("app.evaluation.harness.evaluate", return_value=ret_report):
            resp = client.post(
                "/evaluation/retrieval",
                json={"dataset_path": "any.json", "persist": False},
            )
        assert resp.status_code == 200
        assert resp.json()["run_id"] is None
    finally:
        app.dependency_overrides.clear()


# ── POST /evaluation/reasoning ────────────────────────────────────────────────


def test_reasoning_benchmark_dataset_not_found() -> None:
    client, _ = _client()
    try:
        resp = client.post(
            "/evaluation/reasoning",
            json={"dataset_path": "/no/such/file.json"},
        )
        assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_reasoning_benchmark_success(monkeypatch) -> None:
    reas_report = _make_minimal_reasoning_report()
    client, fake_repo = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_load_reasoning_dataset", lambda p: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_orchestrator", lambda db: MagicMock())
        with patch(
            "app.evaluation.reasoning_harness.evaluate_reasoning_dataset",
            return_value=reas_report,
        ):
            resp = client.post(
                "/evaluation/reasoning",
                json={
                    "dataset_path": "any.json",
                    "judge": "none",
                    "persist": True,
                    "experiment_name": "reas-test",
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "reasoning_report" in data
        assert data["reasoning_report"]["metrics"]["num_scenarios"] == 1
    finally:
        app.dependency_overrides.clear()


def test_reasoning_benchmark_orchestrator_unavailable(monkeypatch) -> None:
    from fastapi import HTTPException
    client, _ = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_load_reasoning_dataset", lambda p: MagicMock())
        monkeypatch.setattr(
            ev_mod, "_build_orchestrator",
            lambda db: (_ for _ in ()).throw(HTTPException(status_code=503, detail="no LLM")),
        )
        resp = client.post(
            "/evaluation/reasoning",
            json={"dataset_path": "any.json", "judge": "none", "persist": False},
        )
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


# ── POST /evaluation/full ─────────────────────────────────────────────────────


def test_full_pipeline_no_datasets_returns_empty_result(monkeypatch) -> None:
    """Full pipeline with no datasets should succeed but skip all stages."""
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        with patch(
            "app.evaluation.evaluation_pipeline.EvaluationPipeline.run",
            return_value=pipeline_result,
        ):
            resp = client.post(
                "/evaluation/full",
                json={"persist": False, "judge": "none"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_report"] is None
        assert data["reasoning_report"] is None
    finally:
        app.dependency_overrides.clear()


def test_full_pipeline_persist_saves_run(monkeypatch) -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        with patch(
            "app.evaluation.evaluation_pipeline.EvaluationPipeline.run",
            return_value=pipeline_result,
        ):
            resp = client.post(
                "/evaluation/full",
                json={"persist": True, "judge": "none", "experiment_name": "api-full"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] is not None
        assert "api-full" in data["run_id"]
    finally:
        app.dependency_overrides.clear()


def test_full_pipeline_with_retrieval_report(monkeypatch) -> None:
    ret = _make_minimal_retrieval_report()
    pipeline_result = _make_pipeline_result(retrieval_report=ret)
    client, fake_repo = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_load_gold_dataset", lambda p: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_search_service", lambda db: MagicMock())
        with patch(
            "app.evaluation.evaluation_pipeline.EvaluationPipeline.run",
            return_value=pipeline_result,
        ):
            resp = client.post(
                "/evaluation/full",
                json={
                    "retrieval_dataset": "some/dataset.json",
                    "persist": False,
                    "judge": "none",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_report"] is not None
        assert data["retrieval_report"]["aggregate_metrics"]["mean_ndcg_at_k"] == pytest.approx(0.75)
    finally:
        app.dependency_overrides.clear()


def test_full_pipeline_execution_summary_in_response(monkeypatch) -> None:
    pipeline_result = _make_pipeline_result()
    client, _ = _client()
    try:
        with patch(
            "app.evaluation.evaluation_pipeline.EvaluationPipeline.run",
            return_value=pipeline_result,
        ):
            resp = client.post("/evaluation/full", json={"persist": False, "judge": "none"})
        data = resp.json()
        assert "execution_summary" in data
        assert "duration_seconds" in data["execution_summary"]
    finally:
        app.dependency_overrides.clear()


def test_full_pipeline_invalid_judge_returns_400() -> None:
    client, _ = _client()
    try:
        resp = client.post(
            "/evaluation/full",
            json={"persist": False, "judge": "gpt4"},
        )
        # FastAPI Pydantic validation catches the pattern mismatch
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


# ── GET /evaluation/runs ──────────────────────────────────────────────────────


def test_list_runs_empty() -> None:
    client, _ = _client()
    try:
        resp = client.get("/evaluation/runs")
        assert resp.status_code == 200
        assert resp.json() == []
    finally:
        app.dependency_overrides.clear()


def test_list_runs_newest_first(monkeypatch) -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        # Pre-populate the fake repo with two runs
        fake_repo.save(pipeline_result, experiment_name="alpha")
        fake_repo.save(pipeline_result, experiment_name="beta")

        resp = client.get("/evaluation/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # Newest first — beta was saved last
        assert "beta" in data[0]["run_id"]
        assert "alpha" in data[1]["run_id"]
    finally:
        app.dependency_overrides.clear()


def test_list_runs_includes_required_fields(monkeypatch) -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        fake_repo.save(pipeline_result, experiment_name="check")
        resp = client.get("/evaluation/runs")
        data = resp.json()
        run = data[0]
        assert "run_id" in run
        assert "timestamp" in run
        assert "experiment_name" in run
        assert "duration" in run
    finally:
        app.dependency_overrides.clear()


# ── GET /evaluation/runs/latest ───────────────────────────────────────────────


def test_get_latest_no_runs_returns_404() -> None:
    client, _ = _client()
    try:
        resp = client.get("/evaluation/runs/latest")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_latest_returns_most_recent(monkeypatch) -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        fake_repo.save(pipeline_result, experiment_name="first")
        rid2 = fake_repo.save(pipeline_result, experiment_name="second")
        resp = client.get("/evaluation/runs/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["run_id"] == rid2
    finally:
        app.dependency_overrides.clear()


# ── GET /evaluation/runs/{run_id} ─────────────────────────────────────────────


def test_get_run_not_found() -> None:
    client, _ = _client()
    try:
        resp = client.get("/evaluation/runs/no_such_run")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_run_returns_metadata() -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        rid = fake_repo.save(pipeline_result, experiment_name="detail-test")
        resp = client.get(f"/evaluation/runs/{rid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["run_id"] == rid
        assert data["metadata"]["experiment_name"] == "detail-test"
    finally:
        app.dependency_overrides.clear()


def test_get_run_includes_summary() -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        rid = fake_repo.save(pipeline_result, experiment_name="t")
        resp = client.get(f"/evaluation/runs/{rid}")
        data = resp.json()
        assert "summary" in data
    finally:
        app.dependency_overrides.clear()


# ── GET /evaluation/runs/{run_id}: failed-queries/failed-reasoning/ ──────────
# judge-disagreements views (Phase 23A: folded into RunDetailResponse —
# previously three separate GET endpoints, now always-present fields on
# the same run-detail response tested above).


def test_get_run_includes_empty_failure_views_for_new_run() -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        rid = fake_repo.save(pipeline_result, experiment_name="t")
        resp = client.get(f"/evaluation/runs/{rid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["failed_queries"] == []
        assert data["failed_reasoning"] == []
        assert data["judge_disagreements"] == []
    finally:
        app.dependency_overrides.clear()


def test_get_run_diagnostics_omitted_by_default() -> None:
    """The diagnostics dashboard is real computation (Phase 22C's
    build_health_report), so it must NOT be computed on a plain
    GET /runs/{run_id} call — only when explicitly requested.
    """
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        rid = fake_repo.save(pipeline_result, experiment_name="t")
        resp = client.get(f"/evaluation/runs/{rid}")
        assert resp.status_code == 200
        assert resp.json()["diagnostics"] is None
    finally:
        app.dependency_overrides.clear()


def test_get_run_diagnostics_included_when_requested() -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        rid = fake_repo.save(pipeline_result, experiment_name="t")
        resp = client.get(f"/evaluation/runs/{rid}", params={"include_diagnostics": "true"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["diagnostics"] is not None
        assert data["diagnostics"]["run_id"] == rid
        assert "overall_health" in data["diagnostics"]
    finally:
        app.dependency_overrides.clear()


def test_get_latest_run_diagnostics_included_when_requested() -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        fake_repo.save(pipeline_result, experiment_name="t")
        resp = client.get("/evaluation/runs/latest", params={"include_diagnostics": "true"})
        assert resp.status_code == 200
        assert resp.json()["diagnostics"] is not None
    finally:
        app.dependency_overrides.clear()


# ── GET /evaluation/stats ─────────────────────────────────────────────────────


def test_stats_empty_repo() -> None:
    client, _ = _client()
    try:
        resp = client.get("/evaluation/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 0
        assert data["best_mrr"] is None
        assert data["latest_run"] is None
        assert data["trend"] == []
    finally:
        app.dependency_overrides.clear()


def test_stats_after_saves() -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        fake_repo.save(pipeline_result, experiment_name="a")
        fake_repo.save(pipeline_result, experiment_name="b")
        resp = client.get("/evaluation/stats")
        data = resp.json()
        assert data["total_runs"] == 2
        assert data["best_mrr"] == pytest.approx(0.7)
        assert data["best_ndcg"] == pytest.approx(0.75)
        assert len(data["trend"]) == 2
    finally:
        app.dependency_overrides.clear()


def test_stats_latest_run_present() -> None:
    pipeline_result = _make_pipeline_result()
    client, fake_repo = _client()
    try:
        rid = fake_repo.save(pipeline_result, experiment_name="z")
        resp = client.get("/evaluation/stats")
        data = resp.json()
        assert data["latest_run"] == rid
    finally:
        app.dependency_overrides.clear()


# ── Route registration / Swagger ──────────────────────────────────────────────


def test_evaluation_tag_appears_in_openapi() -> None:
    client, _ = _client()
    try:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        evaluation_paths = [p for p in paths if p.startswith("/evaluation")]
        assert len(evaluation_paths) > 0
    finally:
        app.dependency_overrides.clear()


def test_all_expected_routes_registered() -> None:
    client, _ = _client()
    try:
        resp = client.get("/openapi.json")
        paths = set(resp.json()["paths"].keys())
        expected = {
            "/evaluation/query",
            "/evaluation/retrieval",
            "/evaluation/reasoning",
            "/evaluation/full",
            "/evaluation/runs",
            "/evaluation/runs/latest",
            "/evaluation/runs/{run_id}",
            "/evaluation/stats",
        }
        for path in expected:
            assert path in paths, f"Missing route: {path}"
    finally:
        app.dependency_overrides.clear()


def test_removed_filtered_view_routes_no_longer_registered() -> None:
    """Phase 23A: these four routes were consolidated into
    GET /evaluation/runs/{run_id} (see RunDetailResponse) and must not
    reappear as separate paths.
    """
    client, _ = _client()
    try:
        resp = client.get("/openapi.json")
        paths = set(resp.json()["paths"].keys())
        removed = {
            "/evaluation/runs/{run_id}/failed-queries",
            "/evaluation/runs/{run_id}/failed-reasoning",
            "/evaluation/runs/{run_id}/judge-disagreements",
            "/evaluation/runs/{run_id}/diagnostics",
        }
        for path in removed:
            assert path not in paths, f"Route should have been removed: {path}"
    finally:
        app.dependency_overrides.clear()


def test_build_orchestrator_constructs_without_typeerror() -> None:
    """Regression test: every other test exercising ``/evaluation/reasoning``
    and ``/evaluation/full`` monkeypatches ``_build_orchestrator`` entirely,
    which previously hid a constructor-arity bug (``MultiAgentInvestigationOrchestrator``
    takes ``db`` positionally and ``search_service``/``llm_service`` as
    keyword-only args; this helper was calling it with two bare positional
    args and always raising ``TypeError``, silently converted into a 503 by
    the surrounding ``except Exception``). This test calls the real,
    unpatched helper to make sure construction actually succeeds.
    """
    from app.api.routes.evaluation import _build_orchestrator
    from app.services.investigation_orchestrator import MultiAgentInvestigationOrchestrator

    orchestrator = _build_orchestrator(None)
    assert isinstance(orchestrator, MultiAgentInvestigationOrchestrator)


# ── POST /evaluation/full — generation (Phase 22A) ────────────────────────────


def _make_generation_report():
    from app.evaluation.generation_harness import (
        GenerationEvaluationReport,
        GenerationQueryResult,
    )
    from app.evaluation.generation_metrics import (
        GenerationMetrics,
        aggregate_generation_metrics,
    )
    from app.evaluation.grounding_metrics import (
        GroundingMetrics,
        aggregate_grounding_metrics,
    )
    generation = GenerationMetrics(
        bert_score_precision=0.9, bert_score_recall=0.7, bert_score_f1=0.7875
    )
    grounding = GroundingMetrics(
        faithfulness=1.0, answer_relevancy=0.85, context_precision=0.5,
        context_recall=0.75, context_entity_recall=None,
    )
    return GenerationEvaluationReport(
        dataset_version="2.1.0", dataset_description="d", k=5,
        num_answered=1, num_generation_scored=1, num_grounding_scored=1,
        num_skipped=0, num_failed=0,
        results=(
            GenerationQueryResult(
                query_id="q-1", query="broker down",
                generated_answer="restart it",
                reference_answer="restart the broker",
                num_contexts=2,
                generation=generation, grounding=grounding,
                skipped=False, skip_reason=None, notes=(),
            ),
        ),
        generation_aggregate=aggregate_generation_metrics([generation]),
        grounding_aggregate=aggregate_grounding_metrics([grounding]),
        started_at="2026-07-02T00:00:00+00:00",
        finished_at="2026-07-02T00:00:01+00:00",
        duration_seconds=1.0,
    )


def test_full_pipeline_generation_report_in_response(monkeypatch) -> None:
    from app.evaluation.evaluation_pipeline import EvaluationPipelineResult
    gen_report = _make_generation_report()
    pipeline_result = EvaluationPipelineResult(
        retrieval_report=None, retrieval_regression=None, retrieval_benchmark=None,
        reasoning_report=None, reasoning_regression=None, reasoning_benchmark=None,
        judge_report=None, judge_validation_report=None, quality_report=None,
        execution_summary=_make_execution_summary(),
        generation_report=gen_report,
    )
    client, _ = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_load_gold_dataset", lambda p: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_search_service", lambda db: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_answer_generator", lambda: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_token_embedder", lambda: None)
        with patch(
            "app.evaluation.evaluation_pipeline.EvaluationPipeline.run",
            return_value=pipeline_result,
        ):
            resp = client.post(
                "/evaluation/full",
                json={
                    "retrieval_dataset": "some/dataset.json",
                    "generation": True,
                    "persist": False,
                    "judge": "none",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        report = data["generation_report"]
        assert report is not None
        assert report["num_answered"] == 1
        # Generation half (BERTScore) and grounding half (RAGAS) both present.
        assert report["generation_aggregate"]["bert_score_f1"]["mean"] == pytest.approx(0.7875)
        assert report["grounding_aggregate"]["faithfulness"]["mean"] == pytest.approx(1.0)
        assert report["grounding_aggregate"]["context_precision"]["mean"] == pytest.approx(0.5)
        assert report["grounding_aggregate"]["context_entity_recall"] is None
    finally:
        app.dependency_overrides.clear()


def test_full_pipeline_generation_flag_passed_to_config(monkeypatch) -> None:
    captured: dict = {}
    client, _ = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_build_search_service", lambda db: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_answer_generator", lambda: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_token_embedder", lambda: None)

        from app.evaluation.evaluation_pipeline import EvaluationPipeline
        original_run = EvaluationPipeline.run

        def spy_run(self, inputs):
            captured["run_generation"] = self.config.run_generation
            captured["answer_generator"] = inputs.answer_generator
            return _make_pipeline_result()

        with patch.object(EvaluationPipeline, "run", spy_run):
            resp = client.post(
                "/evaluation/full",
                json={"generation": True, "persist": False, "judge": "none"},
            )
        assert resp.status_code == 200
        assert captured["run_generation"] is True
        assert captured["answer_generator"] is not None
    finally:
        app.dependency_overrides.clear()


def test_full_pipeline_default_request_has_generation_off_backward_compatible(
    monkeypatch,
) -> None:
    # A pre-22A request body (no "generation" key) must behave exactly as
    # before: run_generation False, no generation_report in the result, and
    # the response's new field is null.
    captured: dict = {}
    client, _ = _client()
    try:
        from app.evaluation.evaluation_pipeline import EvaluationPipeline

        def spy_run(self, inputs):
            captured["run_generation"] = self.config.run_generation
            return _make_pipeline_result()

        with patch.object(EvaluationPipeline, "run", spy_run):
            resp = client.post(
                "/evaluation/full", json={"persist": False, "judge": "none"}
            )
        assert resp.status_code == 200
        assert captured["run_generation"] is False
        assert resp.json()["generation_report"] is None
    finally:
        app.dependency_overrides.clear()


def test_run_detail_tolerates_runs_without_generation_report() -> None:
    # The FakeExperimentRepo's ExperimentRun predates 22A field population —
    # RunDetailResponse must surface generation_report as null, not error.
    client, fake_repo = _client()
    try:
        fake_repo.save(_make_pipeline_result(), experiment_name="old")
        resp = client.get("/evaluation/runs/latest")
        assert resp.status_code == 200
        assert resp.json()["generation_report"] is None
    finally:
        app.dependency_overrides.clear()


# ── Phase 22B: generation_mode / generation_repetitions ───────────────────────


def test_full_pipeline_mode_and_repetitions_passed_to_config(monkeypatch) -> None:
    captured: dict = {}
    client, _ = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_build_search_service", lambda db: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_answer_generator", lambda: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_token_embedder", lambda: None)
        monkeypatch.setattr(ev_mod, "_build_grounding_llm", lambda: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_sentence_embedder", lambda: MagicMock())

        from app.evaluation.evaluation_pipeline import EvaluationPipeline

        def spy_run(self, inputs):
            captured["generation_mode"] = self.config.generation_mode
            captured["generation_repetitions"] = self.config.generation_repetitions
            return _make_pipeline_result()

        with patch.object(EvaluationPipeline, "run", spy_run):
            resp = client.post(
                "/evaluation/full",
                json={
                    "generation": True,
                    "generation_mode": "standard",
                    "generation_repetitions": 3,
                    "persist": False,
                    "judge": "none",
                },
            )
        assert resp.status_code == 200
        assert captured["generation_mode"] == "standard"
        assert captured["generation_repetitions"] == 3
    finally:
        app.dependency_overrides.clear()


def test_full_pipeline_defaults_fast_mode_single_repetition(monkeypatch) -> None:
    # Backward compatibility: a request body without the Phase 22B fields
    # gets fast mode and 1 repetition — conservative production defaults.
    captured: dict = {}
    client, _ = _client()
    try:
        import app.api.routes.evaluation as ev_mod
        monkeypatch.setattr(ev_mod, "_build_search_service", lambda db: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_answer_generator", lambda: MagicMock())
        monkeypatch.setattr(ev_mod, "_build_token_embedder", lambda: None)
        monkeypatch.setattr(ev_mod, "_build_grounding_llm", lambda: MagicMock())

        embedder_built: list[bool] = []
        monkeypatch.setattr(
            ev_mod, "_build_sentence_embedder",
            lambda: embedder_built.append(True) or MagicMock(),
        )

        from app.evaluation.evaluation_pipeline import EvaluationPipeline

        def spy_run(self, inputs):
            captured["generation_mode"] = self.config.generation_mode
            captured["generation_repetitions"] = self.config.generation_repetitions
            return _make_pipeline_result()

        with patch.object(EvaluationPipeline, "run", spy_run):
            resp = client.post(
                "/evaluation/full",
                json={"generation": True, "persist": False, "judge": "none"},
            )
        assert resp.status_code == 200
        assert captured["generation_mode"] == "fast"
        assert captured["generation_repetitions"] == 1
        # fast mode never uses Answer Relevancy -> the sentence-embedding
        # model must not be loaded at all (Phase 22B cost discipline).
        assert embedder_built == []
    finally:
        app.dependency_overrides.clear()


def test_full_pipeline_rejects_invalid_generation_mode() -> None:
    client, _ = _client()
    try:
        resp = client.post(
            "/evaluation/full",
            json={"generation": True, "generation_mode": "turbo", "judge": "none"},
        )
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()
