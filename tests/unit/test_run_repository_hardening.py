"""Phase 23: resilience/security hardening tests for the shared
``FileRunRepositoryMixin`` (used by Benchmark/Reasoning/Judged repositories)
and ``ExperimentRepository`` (Phase 21F) — corrupted files on disk, and
defense-in-depth against a ``run_id`` that resolves outside the storage
directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.evaluation.run_repository import FileRunRepositoryMixin


@dataclass(frozen=True)
class _FakeRun:
    run_id: str
    timestamp: str
    payload: str
    experiment_name: str = "default"


class _FakeFileRepo(FileRunRepositoryMixin):
    _run_type = _FakeRun


def _repo(tmp_path: Path) -> _FakeFileRepo:
    return _FakeFileRepo(tmp_path / "runs")


# ── Happy path (unchanged behavior) ─────────────────────────────────────────


def test_save_get_round_trip(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run = _FakeRun(run_id="20260701_000000_test", timestamp="t", payload="hello")
    repo.save(run)
    loaded = repo.get("20260701_000000_test")
    assert loaded == run


def test_save_duplicate_raises_value_error(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run = _FakeRun(run_id="dup", timestamp="t", payload="a")
    repo.save(run)
    with pytest.raises(ValueError, match="already exists"):
        repo.save(run)


def test_get_missing_run_returns_none(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert repo.get("no_such_run") is None


def test_delete_missing_run_returns_false(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert repo.delete("no_such_run") is False


# ── Corrupted evaluation data ────────────────────────────────────────────────


def test_get_corrupted_json_returns_none_not_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / "broken.json").write_text("{not valid json", encoding="utf-8")
    assert repo.get("broken") is None


def test_get_schema_mismatch_returns_none_not_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    # Valid JSON, but missing required fields for _FakeRun.
    (tmp_path / "runs" / "wrongshape.json").write_text('{"unexpected": true}', encoding="utf-8")
    assert repo.get("wrongshape") is None


def test_list_runs_skips_corrupted_files_alongside_valid_ones(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    good = _FakeRun(run_id="good", timestamp="2026-01-01T00:00:00", payload="ok")
    repo.save(good)
    (tmp_path / "runs" / "broken.json").write_text("not json at all", encoding="utf-8")

    runs = repo.list_runs()

    assert runs == (good,)


# ── Path traversal defense-in-depth ─────────────────────────────────────────


def test_get_with_traversal_run_id_does_not_escape_directory(tmp_path: Path) -> None:
    """A run_id that resolves outside the repository directory must be
    treated as not-found, never read (even if a same-named .json file
    happens to exist one level up).
    """
    repo = _repo(tmp_path)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    # A file that WOULD be reachable via naive "../" traversal from runs/.
    (tmp_path / "secret.json").write_text('{"run_id": "leak"}', encoding="utf-8")

    assert repo.get("../secret") is None


def test_save_with_traversal_run_id_raises_instead_of_escaping(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run = _FakeRun(run_id="../../escaped", timestamp="t", payload="x")
    with pytest.raises(ValueError, match="not a valid identifier"):
        repo.save(run)
    # Nothing was written outside the repository directory.
    assert not (tmp_path / "escaped.json").exists()


def test_delete_with_traversal_run_id_returns_false(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    assert repo.delete("../secret") is False
    assert (tmp_path / "secret.json").exists()  # untouched


# ── ExperimentRepository (Phase 21F) — same hardening, directory-per-run ───


def _minimal_pipeline_result():
    from datetime import UTC, datetime

    from app.evaluation.evaluation_pipeline import EvaluationPipelineResult, ExecutionSummary

    now = datetime.now(UTC).isoformat()
    return EvaluationPipelineResult(
        retrieval_report=None,
        retrieval_regression=None,
        retrieval_benchmark=None,
        reasoning_report=None,
        reasoning_regression=None,
        reasoning_benchmark=None,
        judge_report=None,
        judge_validation_report=None,
        quality_report=None,
        execution_summary=ExecutionSummary(
            start_time=now, end_time=now, duration_seconds=0.0,
            retrieval_queries=0, reasoning_scenarios=0, judge_evaluations=0,
            warnings=(), errors=(),
        ),
    )


def test_experiment_repository_load_with_traversal_run_id_returns_none(tmp_path: Path) -> None:
    from app.evaluation.experiment_tracking import ExperimentRepository

    repo = ExperimentRepository(base_dir=tmp_path / "evaluation_runs")
    # A directory outside history/ that happens to look like a run.
    outside = tmp_path / "outside_run"
    outside.mkdir()
    (outside / "metadata.json").write_text(
        '{"run_id": "outside_run", "timestamp": "t", "git_commit": null, '
        '"experiment_name": "x", "retrieval_dataset_version": null, '
        '"reasoning_dataset_version": null, "judge": null, "duration": 0.0, '
        '"configuration": {}}',
        encoding="utf-8",
    )

    assert repo.load("../outside_run") is None


def test_experiment_repository_save_load_round_trip(tmp_path: Path) -> None:
    from app.evaluation.experiment_tracking import ExperimentRepository

    repo = ExperimentRepository(base_dir=tmp_path / "evaluation_runs")
    run_id = repo.save(_minimal_pipeline_result(), experiment_name="test", git_commit="")
    run = repo.load(run_id)
    assert run is not None
    assert run.metadata.run_id == run_id


def test_experiment_repository_delete_with_traversal_run_id_returns_false(tmp_path: Path) -> None:
    from app.evaluation.experiment_tracking import ExperimentRepository

    repo = ExperimentRepository(base_dir=tmp_path / "evaluation_runs")
    outside = tmp_path / "outside_run"
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")

    assert repo.delete("../outside_run") is False
    assert marker.exists()
