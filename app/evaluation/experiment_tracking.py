"""Persistent Evaluation & Experiment Tracking (Phase 21F).

Provides a production-grade persistence layer around
``EvaluationPipelineResult`` (Phase 21E).  Every report already computed by
the pipeline is written to disk in full; nothing is recomputed here.

# Directory layout

```
.evaluation_runs/
    latest/                        ← always the most recent run (symlink-like copy)
        metadata.json
        summary.json
        retrieval_report.json      ← absent when retrieval stage was skipped
        reasoning_report.json
        judge_report.json
        quality_report.json
        validation_report.json
        regression_report.json
        failed_queries.json        ← convenience: per_query entries with recall < 1
        failed_reasoning.json      ← convenience: InvestigationResults that failed
        judge_disagreements.json   ← convenience: JudgeEvaluations with score < 5
    history/
        20260701_213015_nightly/   ← one sub-directory per run
            (same files as latest/)
```

# Design constraints

- MUST NOT import ``IncidentSearchService``, ``LLMService``, or any agent class.
- MUST NOT re-run or re-derive any metric, failure, or recommendation.
- Everything written is a JSON-serialised view of what the pipeline already
  produced; nothing is added, averaged, or reinterpreted.
- The three "convenience" JSON files are pure filters of existing data — the
  filter predicates are documented on the helper functions below so a reader
  knows exactly what "failed" and "disagreement" mean in this context.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.evaluation.evaluation_pipeline import EvaluationPipelineResult

# ── Run ID ────────────────────────────────────────────────────────────────────


def make_run_id(experiment_name: str, *, now: datetime | None = None) -> str:
    """Return ``YYYYMMDD_HHMMSS_<experiment_name>`` — e.g.
    ``20260701_213015_nightly``.  ``now`` defaults to the current UTC time
    and is exposed for testing.
    """
    ts = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    # Sanitise experiment name: replace whitespace/slashes with underscores.
    safe = experiment_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{ts}_{safe}"


def _git_commit() -> str | None:
    """Best-effort: return the current HEAD commit sha, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except Exception:  # noqa: BLE001
        return None


# ── Serialisation (mirrors benchmark.py / reasoning_benchmark.py pattern) ────


def _to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into a JSON-safe form.

    Handles: dataclasses → dicts, Enums → their ``.value``, tuples/lists →
    lists, Mappings → dicts, everything else passed through unchanged.
    """
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Failure / disagreement filters ───────────────────────────────────────────


def _failed_queries(retrieval_report_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-query entries where recall_at_k < 1.0 or the query was
    skipped (i.e. it did not achieve full recall, which we treat as a
    retrieval failure for inspection purposes).
    """
    per_query = retrieval_report_dict.get("per_query") or []
    result = []
    for q in per_query:
        if q.get("skipped"):
            result.append(q)
            continue
        metric = q.get("metric") or {}
        recall = metric.get("recall_at_k")
        if recall is not None and recall < 1.0:
            result.append(q)
    return result


def _failed_reasoning(reasoning_report_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Return InvestigationResult entries where the investigation did not
    fully succeed: ``converged`` is False, or ``decision_correct`` is False,
    or ``planner_correct`` is False.  A result with any ``None`` field
    (metric not measured) is excluded unless another failure criterion fires.
    """
    results = reasoning_report_dict.get("results") or []
    out = []
    for r in results:
        if (
            r.get("converged") is False
            or r.get("decision_correct") is False
            or r.get("planner_correct") is False
        ):
            out.append(r)
    return out


def _judge_disagreements(judge_report_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Return JudgeEvaluation entries where the numeric score is below 5.0
    (the midpoint of the 1–10 scale), which we treat as the judge finding
    the investigation insufficient.
    """
    evaluations = judge_report_dict.get("judge_evaluations") or []
    out = []
    for e in evaluations:
        score_block = e.get("score") or {}
        score_val = score_block.get("value")
        if score_val is not None and score_val < 5.0:
            out.append(e)
    return out


# ── Metadata ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunMetadata:
    """Structured header for one experiment run."""

    run_id: str
    timestamp: str
    git_commit: str | None
    experiment_name: str
    retrieval_dataset_version: str | None
    reasoning_dataset_version: str | None
    judge: str | None
    duration: float
    configuration: dict[str, Any]


def _build_metadata(
    *,
    run_id: str,
    timestamp: str,
    result: EvaluationPipelineResult,
    experiment_name: str,
    git_commit: str | None,
    retrieval_dataset_version: str | None,
    reasoning_dataset_version: str | None,
    judge_name: str | None,
) -> RunMetadata:
    summary = result.execution_summary
    config: dict[str, Any] = {}
    # Pull in the pipeline config if we can find it on a benchmark run.
    if result.retrieval_benchmark is not None:
        cfg = result.retrieval_benchmark.config
        config["retrieval_k"] = cfg.k
        config["retrieval_expand"] = cfg.expand
        config["retrieval_rerank"] = cfg.rerank
    return RunMetadata(
        run_id=run_id,
        timestamp=timestamp,
        git_commit=git_commit,
        experiment_name=experiment_name,
        retrieval_dataset_version=retrieval_dataset_version,
        reasoning_dataset_version=reasoning_dataset_version,
        judge=judge_name,
        duration=summary.duration_seconds,
        configuration=config,
    )


# ── ExperimentRun (loaded view) ───────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentRun:
    """A fully-loaded experiment run.  Report fields are plain dicts
    (JSON-parsed) rather than typed dataclasses, so this module does not
    need to import every evaluation module's types for round-trip
    deserialisation.
    """

    metadata: RunMetadata
    summary: dict[str, Any]
    retrieval_report: dict[str, Any] | None
    reasoning_report: dict[str, Any] | None
    judge_report: dict[str, Any] | None
    quality_report: dict[str, Any] | None
    validation_report: dict[str, Any] | None
    regression_report: dict[str, Any] | None
    failed_queries: tuple[dict[str, Any], ...]
    failed_reasoning: tuple[dict[str, Any], ...]
    judge_disagreements: tuple[dict[str, Any], ...]
    # Phase 22A — defaulted so runs persisted before generation evaluation
    # existed (no generation_report.json on disk) load unchanged as None.
    generation_report: dict[str, Any] | None = None


# ── Statistics ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentStats:
    """Headline statistics across the full run history."""

    total_runs: int
    best_mrr: float | None
    best_ndcg: float | None
    best_reasoning_accuracy: float | None
    latest_run: str | None          # run_id of the most recent run
    trend: tuple[str, ...]          # run_ids oldest → newest


# ── ExperimentRepository ──────────────────────────────────────────────────────


_REPORT_FILES = {
    "retrieval_report": "retrieval_report.json",
    "reasoning_report": "reasoning_report.json",
    "judge_report": "judge_report.json",
    "quality_report": "quality_report.json",
    "validation_report": "validation_report.json",
    "regression_report": "regression_report.json",
    "generation_report": "generation_report.json",  # Phase 22A
}


class ExperimentRepository:
    """Persists ``EvaluationPipelineResult``s under ``base_dir``.

    The repository never re-runs or re-derives anything.  It converts the
    already-computed dataclass tree to JSON (via ``_to_jsonable``), writes it
    to disk, then reads it back on demand.

    Usage::

        repo = ExperimentRepository(Path(".evaluation_runs"))
        run_id = repo.save(
            result,
            experiment_name="nightly",
            retrieval_dataset_version="v3",
        )
        run = repo.load(run_id)
        print(run.metadata.duration)
    """

    def __init__(self, base_dir: Path = Path(".evaluation_runs")) -> None:
        self._base = Path(base_dir)
        self._history = self._base / "history"
        self._latest = self._base / "latest"

    # ── save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        result: EvaluationPipelineResult,
        *,
        experiment_name: str = "default",
        git_commit: str | None = None,
        retrieval_dataset_version: str | None = None,
        reasoning_dataset_version: str | None = None,
        judge_name: str | None = None,
        run_id: str | None = None,
        _now: datetime | None = None,
    ) -> str:
        """Persist ``result`` and return the assigned ``run_id``.

        If ``git_commit`` is ``None`` the repository will attempt
        ``git rev-parse --short HEAD`` automatically; pass ``""`` to
        suppress the lookup.
        """
        now = _now or datetime.now(UTC)
        rid = run_id or make_run_id(experiment_name, now=now)
        timestamp = now.isoformat()
        resolved_git = git_commit if git_commit is not None else _git_commit()

        meta = _build_metadata(
            run_id=rid,
            timestamp=timestamp,
            result=result,
            experiment_name=experiment_name,
            git_commit=resolved_git,
            retrieval_dataset_version=retrieval_dataset_version,
            reasoning_dataset_version=reasoning_dataset_version,
            judge_name=judge_name,
        )

        run_dir = self._history / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # metadata
        _write_json(run_dir / "metadata.json", _to_jsonable(meta))

        # execution summary
        _write_json(run_dir / "summary.json", _to_jsonable(result.execution_summary))

        # main reports (skip None)
        report_map: dict[str, Any] = {
            "retrieval_report": result.retrieval_report,
            "reasoning_report": result.reasoning_report,
            "judge_report": result.judge_report,
            "quality_report": result.quality_report,
            "validation_report": result.judge_validation_report,
            "regression_report": (
                result.retrieval_regression or result.reasoning_regression
            ),
            "generation_report": result.generation_report,  # Phase 22A
        }
        serialised: dict[str, dict[str, Any] | None] = {}
        for key, obj in report_map.items():
            if obj is not None:
                data = _to_jsonable(obj)
                _write_json(run_dir / _REPORT_FILES[key], data)
                serialised[key] = data
            else:
                serialised[key] = None

        # convenience failure files (filter only; never recompute)
        ret_data = serialised.get("retrieval_report")
        reas_data = serialised.get("reasoning_report")
        judge_data = serialised.get("judge_report")

        failed_q = _failed_queries(ret_data) if ret_data else []
        failed_r = _failed_reasoning(reas_data) if reas_data else []
        disagree = _judge_disagreements(judge_data) if judge_data else []

        _write_json(run_dir / "failed_queries.json", failed_q)
        _write_json(run_dir / "failed_reasoning.json", failed_r)
        _write_json(run_dir / "judge_disagreements.json", disagree)

        # overwrite latest/ atomically (copy the whole directory)
        self._overwrite_latest(run_dir)

        return rid

    def _overwrite_latest(self, run_dir: Path) -> None:
        if self._latest.exists():
            shutil.rmtree(self._latest)
        shutil.copytree(run_dir, self._latest)

    # ── list / latest / load / delete ─────────────────────────────────────────

    def list_runs(self) -> tuple[str, ...]:
        """Return all run IDs in chronological order (oldest first), derived
        from their ``metadata.json`` timestamp rather than directory mtime.
        """
        if not self._history.exists():
            return ()
        entries: list[tuple[str, str]] = []  # (timestamp, run_id)
        for d in self._history.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                ts = json.loads(meta_path.read_text(encoding="utf-8"))["timestamp"]
            except (KeyError, json.JSONDecodeError):
                ts = ""
            entries.append((ts, d.name))
        entries.sort(key=lambda x: x[0])
        return tuple(rid for _, rid in entries)

    def latest(self) -> ExperimentRun | None:
        """Return the most recently saved run, or ``None``."""
        if not self._latest.exists():
            return None
        return self._load_from_dir(self._latest)

    def load(self, run_id: str) -> ExperimentRun | None:
        """Load the run with this ``run_id``.  Returns ``None`` if it does
        not exist.
        """
        run_dir = self._history / run_id
        if not run_dir.exists():
            return None
        return self._load_from_dir(run_dir)

    def delete(self, run_id: str) -> bool:
        """Remove the run directory.  Also clears ``latest/`` if it points
        to the deleted run.  Returns ``True`` if the run existed.
        """
        run_dir = self._history / run_id
        if not run_dir.exists():
            return False
        shutil.rmtree(run_dir)
        # Clear latest if it matches
        if self._latest.exists():
            latest_meta = self._latest / "metadata.json"
            if latest_meta.exists():
                try:
                    data = json.loads(latest_meta.read_text(encoding="utf-8"))
                    if data.get("run_id") == run_id:
                        shutil.rmtree(self._latest)
                        # Repopulate latest with the new most-recent run
                        remaining = self.list_runs()
                        if remaining:
                            newest = self._history / remaining[-1]
                            shutil.copytree(newest, self._latest)
                except Exception:  # noqa: BLE001
                    pass
        return True

    def stats(self) -> ExperimentStats:
        """Compute headline statistics across the full run history."""
        run_ids = self.list_runs()
        if not run_ids:
            return ExperimentStats(
                total_runs=0, best_mrr=None, best_ndcg=None,
                best_reasoning_accuracy=None, latest_run=None, trend=(),
            )
        best_mrr: float | None = None
        best_ndcg: float | None = None
        best_ra: float | None = None

        for rid in run_ids:
            run_dir = self._history / rid
            ret_path = run_dir / "retrieval_report.json"
            if ret_path.exists():
                try:
                    d = _read_json(ret_path)
                    agg = d.get("aggregate_metrics") or {}
                    mrr = agg.get("mean_reciprocal_rank")
                    ndcg = agg.get("mean_ndcg_at_k")
                    if mrr is not None:
                        best_mrr = mrr if best_mrr is None else max(best_mrr, mrr)
                    if ndcg is not None:
                        best_ndcg = ndcg if best_ndcg is None else max(best_ndcg, ndcg)
                except Exception:  # noqa: BLE001
                    pass
            reas_path = run_dir / "reasoning_report.json"
            if reas_path.exists():
                try:
                    d = _read_json(reas_path)
                    m = d.get("metrics") or {}
                    da = m.get("decision_accuracy")
                    if da is not None:
                        best_ra = da if best_ra is None else max(best_ra, da)
                except Exception:  # noqa: BLE001
                    pass

        return ExperimentStats(
            total_runs=len(run_ids),
            best_mrr=best_mrr,
            best_ndcg=best_ndcg,
            best_reasoning_accuracy=best_ra,
            latest_run=run_ids[-1],
            trend=run_ids,
        )

    # ── internal loader ───────────────────────────────────────────────────────

    def _load_from_dir(self, run_dir: Path) -> ExperimentRun:
        meta_raw = _read_json(run_dir / "metadata.json")
        meta = RunMetadata(
            run_id=meta_raw["run_id"],
            timestamp=meta_raw["timestamp"],
            git_commit=meta_raw.get("git_commit"),
            experiment_name=meta_raw["experiment_name"],
            retrieval_dataset_version=meta_raw.get("retrieval_dataset_version"),
            reasoning_dataset_version=meta_raw.get("reasoning_dataset_version"),
            judge=meta_raw.get("judge"),
            duration=meta_raw.get("duration", 0.0),
            configuration=meta_raw.get("configuration") or {},
        )
        summary = _read_json(run_dir / "summary.json")

        def _opt(filename: str) -> dict[str, Any] | None:
            p = run_dir / filename
            return _read_json(p) if p.exists() else None

        def _list(filename: str) -> tuple[dict[str, Any], ...]:
            p = run_dir / filename
            return tuple(_read_json(p)) if p.exists() else ()

        return ExperimentRun(
            metadata=meta,
            summary=summary,
            retrieval_report=_opt("retrieval_report.json"),
            reasoning_report=_opt("reasoning_report.json"),
            judge_report=_opt("judge_report.json"),
            quality_report=_opt("quality_report.json"),
            validation_report=_opt("validation_report.json"),
            regression_report=_opt("regression_report.json"),
            failed_queries=_list("failed_queries.json"),
            failed_reasoning=_list("failed_reasoning.json"),
            judge_disagreements=_list("judge_disagreements.json"),
            generation_report=_opt("generation_report.json"),
        )
