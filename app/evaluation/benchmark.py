"""Benchmark History & Experiment Tracking (Phase 16F).

Organizes and persists evaluation runs over time. This module implements
NONE of the following — it only stores and looks up already-completed
results, and delegates comparison to Phase 16E:

- **Retrieval** — never imports or calls ``IncidentSearchService``.
- **Evaluation** — never calls ``app.evaluation.harness.evaluate``; it only
  accepts an already-built ``EvaluationReport`` as input to
  ``create_benchmark_run``.
- **Metrics** — never computes Recall/MRR/NDCG; it only reads already-
  computed values off the ``EvaluationReport``s it stores.
- **Regression computation** — never reimplements comparison logic. Every
  comparison utility in this module (``compare_runs``,
  ``compare_latest_against_previous``, ``regression_history``) delegates to
  ``app.evaluation.regression.compare`` (Phase 16E) and nothing else.

# Benchmark lifecycle

```
EvaluationReport (Phase 16D)  ──┐
RegressionReport (Phase 16E,    │
  optional — vs. some baseline) ┤
                                 ▼
                  create_benchmark_run(experiment_name=, report=,
                                        regression=, git_commit_sha=, notes=)
                                 │
                                 ▼
                          BenchmarkRun (frozen)
                                 │
                                 ▼
                   repository.save(run)   [InMemory or File backend]
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     repository.get(id)   repository.list_runs()   repository.latest()
                                 │
                                 ▼
        compare_runs / compare_latest_against_previous / regression_history
                  (all delegate to app.evaluation.regression.compare)
```

# Why ``BenchmarkRun.config`` duplicates ``BenchmarkRun.report.config``

``BenchmarkRun`` exposes ``config: EvaluationConfig`` as its own top-level
field even though ``report.config`` already carries the same information —
the same "denormalized for convenience" choice made for
``ResolvedGoldQuery.query`` (Phase 16B) and the embedded
``baseline``/``candidate`` reports on ``RegressionReport`` (Phase 16E): a
reader scanning run history (e.g. ``metric_history``) shouldn't need to
drill into a nested report just to see what ``k``/``expand``/``rerank`` a
run used. To prevent this duplication from drifting, ``BenchmarkRun`` is
never constructed with an independently-supplied ``config`` — the only
supported constructor, ``create_benchmark_run``, always derives it from
``report.config`` directly, so the two values can never disagree.
"""

from __future__ import annotations

import json
import types
import typing
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.evaluation.harness import EvaluationConfig, EvaluationReport
from app.evaluation.regression import RegressionReport, compare


# ── BenchmarkRun ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkRun:
    """A single, immutable evaluation experiment.

    Construct via ``create_benchmark_run`` rather than directly, so
    ``config`` is always derived from ``report.config`` (see module
    docstring) and ``run_id``/``timestamp`` get sensible defaults.
    """

    run_id: str
    timestamp: str
    experiment_name: str
    config: EvaluationConfig
    report: EvaluationReport
    git_commit_sha: str | None
    notes: str | None
    regression: RegressionReport | None


def create_benchmark_run(
    *,
    experiment_name: str,
    report: EvaluationReport,
    regression: RegressionReport | None = None,
    git_commit_sha: str | None = None,
    notes: str | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
) -> BenchmarkRun:
    """Build a ``BenchmarkRun`` from an already-completed ``EvaluationReport``
    (and, optionally, an already-computed ``RegressionReport`` comparing it
    against some baseline). ``run_id`` defaults to a fresh UUID4 string;
    ``timestamp`` defaults to the current UTC time in ISO-8601 form.
    """
    return BenchmarkRun(
        run_id=run_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        experiment_name=experiment_name,
        config=report.config,
        report=report,
        git_commit_sha=git_commit_sha,
        notes=notes,
        regression=regression,
    )


# ── Repository interface ────────────────────────────────────────────────────────


class BenchmarkRepository(ABC):
    """Storage interface for ``BenchmarkRun``s.

    Defined as an ``ABC`` (not a structural ``Protocol``) so that a future
    SQLite/Postgres/S3-backed implementation is required to implement every
    method explicitly — an incomplete subclass fails at instantiation, not
    silently at first use. Callers should depend on this interface, not on
    ``InMemoryBenchmarkRepository``/``FileBenchmarkRepository`` directly.
    """

    @abstractmethod
    def save(self, run: BenchmarkRun) -> None:
        """Persist ``run``. Raises ``ValueError`` if a run with the same
        ``run_id`` already exists — runs are never silently overwritten.
        """

    @abstractmethod
    def get(self, run_id: str) -> BenchmarkRun | None:
        """Return the run with this id, or ``None`` if it does not exist."""

    @abstractmethod
    def list_runs(self, *, experiment_name: str | None = None) -> tuple[BenchmarkRun, ...]:
        """Return all stored runs (optionally filtered to one
        ``experiment_name``), ordered by ``timestamp`` ascending (oldest
        first). This ordering is the contract every comparison utility below
        relies on.
        """

    @abstractmethod
    def latest(self, *, experiment_name: str | None = None) -> BenchmarkRun | None:
        """Return the most recent run (by ``timestamp``), or ``None`` if
        there are no runs (optionally scoped to one ``experiment_name``).
        """

    @abstractmethod
    def delete(self, run_id: str) -> bool:
        """Remove the run with this id. Returns ``True`` if a run was
        removed, ``False`` if no run with that id existed.
        """


class InMemoryBenchmarkRepository(BenchmarkRepository):
    """Process-local, non-persistent ``BenchmarkRepository``. Useful for
    tests and for short-lived scripts that don't need runs to survive past
    the process.
    """

    def __init__(self) -> None:
        self._runs: dict[str, BenchmarkRun] = {}

    def save(self, run: BenchmarkRun) -> None:
        if run.run_id in self._runs:
            raise ValueError(f"a run with id {run.run_id!r} already exists")
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> BenchmarkRun | None:
        return self._runs.get(run_id)

    def list_runs(self, *, experiment_name: str | None = None) -> tuple[BenchmarkRun, ...]:
        runs = self._runs.values()
        if experiment_name is not None:
            runs = (run for run in runs if run.experiment_name == experiment_name)
        return tuple(sorted(runs, key=lambda run: run.timestamp))

    def latest(self, *, experiment_name: str | None = None) -> BenchmarkRun | None:
        runs = self.list_runs(experiment_name=experiment_name)
        return runs[-1] if runs else None

    def delete(self, run_id: str) -> bool:
        return self._runs.pop(run_id, None) is not None


class FileBenchmarkRepository(BenchmarkRepository):
    """JSON-file-backed ``BenchmarkRepository``: one ``{run_id}.json`` file
    per run in ``directory``. Serialization is generic (see
    ``_to_jsonable``/``_from_jsonable`` below) — every nested dataclass and
    enum across Phases 16A-16E's report types round-trips without any
    per-type serialization code in this module.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, run_id: str) -> Path:
        return self._directory / f"{run_id}.json"

    def save(self, run: BenchmarkRun) -> None:
        path = self._path_for(run.run_id)
        if path.exists():
            raise ValueError(f"a run with id {run.run_id!r} already exists")
        path.write_text(json.dumps(_to_jsonable(run), indent=2), encoding="utf-8")

    def get(self, run_id: str) -> BenchmarkRun | None:
        path = self._path_for(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return _from_jsonable(data, BenchmarkRun)

    def list_runs(self, *, experiment_name: str | None = None) -> tuple[BenchmarkRun, ...]:
        runs = [self.get(path.stem) for path in self._directory.glob("*.json")]
        present = [run for run in runs if run is not None]
        if experiment_name is not None:
            present = [run for run in present if run.experiment_name == experiment_name]
        return tuple(sorted(present, key=lambda run: run.timestamp))

    def latest(self, *, experiment_name: str | None = None) -> BenchmarkRun | None:
        runs = self.list_runs(experiment_name=experiment_name)
        return runs[-1] if runs else None

    def delete(self, run_id: str) -> bool:
        path = self._path_for(run_id)
        if path.exists():
            path.unlink()
            return True
        return False


# ── Generic dataclass <-> JSON-safe conversion ────────────────────────────────


def _to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into something ``json.dumps`` accepts:
    dataclasses -> dicts of their fields, enums -> their ``.value``,
    tuples -> lists, dicts -> dicts (values converted), everything else
    passed through unchanged (str/int/float/bool/None).
    """
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _from_jsonable(data: Any, target_type: Any) -> Any:
    """The inverse of ``_to_jsonable``, guided by ``target_type``'s resolved
    type hints. Generic across every dataclass in Phases 16A-16E's report
    trees because it walks ``dataclasses.fields``/``typing.get_type_hints``
    rather than hand-coding each type — adding a field to any report
    dataclass needs no change here.
    """
    if data is None:
        return None

    origin = typing.get_origin(target_type)

    if origin in (typing.Union, types.UnionType):
        non_none_args = [arg for arg in typing.get_args(target_type) if arg is not type(None)]
        return _from_jsonable(data, non_none_args[0])

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(data)

    if is_dataclass(target_type):
        hints = typing.get_type_hints(target_type)
        kwargs = {
            f.name: _from_jsonable(data[f.name], hints[f.name]) for f in fields(target_type)
        }
        return target_type(**kwargs)

    if origin is tuple:
        (item_type, *_rest) = typing.get_args(target_type)
        return tuple(_from_jsonable(item, item_type) for item in data)

    if origin is dict:
        _key_type, value_type = typing.get_args(target_type)
        return {key: _from_jsonable(item, value_type) for key, item in data.items()}

    return data


# ── Comparison utilities (delegate entirely to Phase 16E) ────────────────────


@dataclass(frozen=True)
class MetricHistoryEntry:
    """One run's headline metrics, projected out for trend-watching without
    needing to traverse a full ``EvaluationReport``. Purely a read-through
    projection — every value here is copied from
    ``run.report.aggregate_metrics``, never recomputed.
    """

    run_id: str
    timestamp: str
    experiment_name: str
    mean_recall_at_k: float | None
    mean_reciprocal_rank: float | None
    mean_ndcg_at_k: float | None


def compare_runs(baseline: BenchmarkRun, candidate: BenchmarkRun) -> RegressionReport:
    """Compare two arbitrary runs. Delegates entirely to
    ``app.evaluation.regression.compare`` — this function exists only to
    save a caller the ``.report`` indirection.
    """
    return compare(baseline.report, candidate.report)


def compare_latest_against_previous(
    repository: BenchmarkRepository, *, experiment_name: str | None = None
) -> RegressionReport | None:
    """Compare the two most recent runs (by ``timestamp``). Returns ``None``
    if fewer than two runs exist — there is nothing to compare, and
    fabricating a "no comparison possible" ``RegressionReport`` would
    require inventing a baseline that doesn't exist.
    """
    runs = repository.list_runs(experiment_name=experiment_name)
    if len(runs) < 2:
        return None
    return compare_runs(runs[-2], runs[-1])


def metric_history(
    repository: BenchmarkRepository, *, experiment_name: str | None = None
) -> tuple[MetricHistoryEntry, ...]:
    """Headline-metric trend across all stored runs, oldest first."""
    runs = repository.list_runs(experiment_name=experiment_name)
    return tuple(
        MetricHistoryEntry(
            run_id=run.run_id,
            timestamp=run.timestamp,
            experiment_name=run.experiment_name,
            mean_recall_at_k=run.report.aggregate_metrics.mean_recall_at_k,
            mean_reciprocal_rank=run.report.aggregate_metrics.mean_reciprocal_rank,
            mean_ndcg_at_k=run.report.aggregate_metrics.mean_ndcg_at_k,
        )
        for run in runs
    )


def regression_history(
    repository: BenchmarkRepository, *, experiment_name: str | None = None
) -> tuple[RegressionReport, ...]:
    """Every consecutive-pair comparison across the run history, oldest
    pair first: ``compare(run[0], run[1]), compare(run[1], run[2]), ...``.
    Each comparison delegates to ``compare_runs`` (and therefore to Phase
    16E) individually — no aggregation or reinterpretation happens here.
    """
    runs = repository.list_runs(experiment_name=experiment_name)
    return tuple(compare_runs(runs[i], runs[i + 1]) for i in range(len(runs) - 1))
