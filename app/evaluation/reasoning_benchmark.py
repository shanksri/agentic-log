"""Reasoning Benchmark Integration (Phase 20A).

Extends Phase 16F's benchmark history pattern to reasoning evaluation,
WITHOUT modifying ``app.evaluation.benchmark`` (Phase 16F). ``BenchmarkRun``
and ``BenchmarkRepository`` there are hard-typed to retrieval's
``EvaluationReport``/``RegressionReport`` — adding reasoning fields to them
would be a modification, not an extension, so this module instead:

1. Mirrors the same run/repository pattern with a parallel
   ``ReasoningBenchmarkRun``/``ReasoningBenchmarkRepository`` for THIS
   phase's ``InvestigationEvaluationReport``/``ReasoningRegressionReport`` —
   duplicated, not derived, because Phase 16F's ``ABC`` is hard-typed to
   the retrieval report types (see module docstring's "Why duplicate, not
   subclass").
2. Introduces ``CombinedBenchmarkRun``, a small composing dataclass that
   carries an OPTIONAL retrieval ``BenchmarkRun`` (16F, unmodified) AND/OR
   an OPTIONAL ``ReasoningBenchmarkRun`` (this phase) side by side — "a
   benchmark may now optionally contain retrieval metrics and reasoning
   metrics," satisfied by composition over two independently-already-
   built run objects, never by adding a field to ``BenchmarkRun`` itself.

# Why duplicate, not subclass

``BenchmarkRepository`` (16F) declares ``save(self, run: BenchmarkRun)``,
etc., with ``BenchmarkRun`` baked into every method signature. A
``ReasoningBenchmarkRepository`` cannot satisfy that same ABC for a
different run type without either (a) modifying the ABC to be generic
(a modification to Phase 16F, explicitly disallowed) or (b) violating the
Liskov substitution the ABC's type hints promise. Duplicating the small,
already-simple interface is the only option that satisfies "do not modify
any previous phase" while still genuinely integrating with — i.e. mirroring
the same lifecycle and API shape as — the existing benchmark framework.

# Benchmark integration workflow

```
InvestigationEvaluationReport (this phase)  ──┐
ReasoningRegressionReport (this phase,         │
  optional - vs. some baseline)                ┤
                                                 ▼
                       create_reasoning_benchmark_run(experiment_name=,
                                                       report=, regression=)
                                                 │
                                                 ▼
                                  ReasoningBenchmarkRun (frozen)
                                                 │
                                                 ▼
                     reasoning_repository.save(run)
                                                 │
                                                 ▼
       compose with an OPTIONAL retrieval BenchmarkRun (16F, unmodified):
                                                 │
                                                 ▼
              combine_benchmark_runs(retrieval=, reasoning=)
                                                 │
                                                 ▼
                              CombinedBenchmarkRun
                    (retrieval and/or reasoning - at least one required)
```
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

from app.evaluation.benchmark import BenchmarkRun
from app.evaluation.reasoning_harness import InvestigationEvaluationReport
from app.evaluation.reasoning_regression import ReasoningRegressionReport, compare_reasoning


# ── ReasoningBenchmarkRun ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReasoningBenchmarkRun:
    """A single, immutable reasoning-evaluation experiment — mirrors
    ``BenchmarkRun`` (16F) field-for-field, scoped to
    ``InvestigationEvaluationReport``/``ReasoningRegressionReport`` instead
    of their retrieval counterparts.
    """

    run_id: str
    timestamp: str
    experiment_name: str
    report: InvestigationEvaluationReport
    regression: ReasoningRegressionReport | None
    git_commit_sha: str | None
    notes: str | None


def create_reasoning_benchmark_run(
    *,
    experiment_name: str,
    report: InvestigationEvaluationReport,
    regression: ReasoningRegressionReport | None = None,
    git_commit_sha: str | None = None,
    notes: str | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
) -> ReasoningBenchmarkRun:
    return ReasoningBenchmarkRun(
        run_id=run_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        experiment_name=experiment_name,
        report=report,
        regression=regression,
        git_commit_sha=git_commit_sha,
        notes=notes,
    )


# ── Repository interface ────────────────────────────────────────────────────────


class ReasoningBenchmarkRepository(ABC):
    """Storage interface for ``ReasoningBenchmarkRun``s — mirrors
    ``BenchmarkRepository`` (16F); see module docstring's "Why duplicate,
    not subclass".
    """

    @abstractmethod
    def save(self, run: ReasoningBenchmarkRun) -> None: ...

    @abstractmethod
    def get(self, run_id: str) -> ReasoningBenchmarkRun | None: ...

    @abstractmethod
    def list_runs(
        self, *, experiment_name: str | None = None
    ) -> tuple[ReasoningBenchmarkRun, ...]: ...

    @abstractmethod
    def latest(self, *, experiment_name: str | None = None) -> ReasoningBenchmarkRun | None: ...

    @abstractmethod
    def delete(self, run_id: str) -> bool: ...


class InMemoryReasoningBenchmarkRepository(ReasoningBenchmarkRepository):
    def __init__(self) -> None:
        self._runs: dict[str, ReasoningBenchmarkRun] = {}

    def save(self, run: ReasoningBenchmarkRun) -> None:
        if run.run_id in self._runs:
            raise ValueError(f"a run with id {run.run_id!r} already exists")
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> ReasoningBenchmarkRun | None:
        return self._runs.get(run_id)

    def list_runs(
        self, *, experiment_name: str | None = None
    ) -> tuple[ReasoningBenchmarkRun, ...]:
        runs = self._runs.values()
        if experiment_name is not None:
            runs = (run for run in runs if run.experiment_name == experiment_name)
        return tuple(sorted(runs, key=lambda run: run.timestamp))

    def latest(self, *, experiment_name: str | None = None) -> ReasoningBenchmarkRun | None:
        runs = self.list_runs(experiment_name=experiment_name)
        return runs[-1] if runs else None

    def delete(self, run_id: str) -> bool:
        return self._runs.pop(run_id, None) is not None


class FileReasoningBenchmarkRepository(ReasoningBenchmarkRepository):
    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, run_id: str) -> Path:
        return self._directory / f"{run_id}.json"

    def save(self, run: ReasoningBenchmarkRun) -> None:
        path = self._path_for(run.run_id)
        if path.exists():
            raise ValueError(f"a run with id {run.run_id!r} already exists")
        path.write_text(json.dumps(_to_jsonable(run), indent=2), encoding="utf-8")

    def get(self, run_id: str) -> ReasoningBenchmarkRun | None:
        path = self._path_for(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return _from_jsonable(data, ReasoningBenchmarkRun)

    def list_runs(
        self, *, experiment_name: str | None = None
    ) -> tuple[ReasoningBenchmarkRun, ...]:
        runs = [self.get(path.stem) for path in self._directory.glob("*.json")]
        present = [run for run in runs if run is not None]
        if experiment_name is not None:
            present = [run for run in present if run.experiment_name == experiment_name]
        return tuple(sorted(present, key=lambda run: run.timestamp))

    def latest(self, *, experiment_name: str | None = None) -> ReasoningBenchmarkRun | None:
        runs = self.list_runs(experiment_name=experiment_name)
        return runs[-1] if runs else None

    def delete(self, run_id: str) -> bool:
        path = self._path_for(run_id)
        if path.exists():
            path.unlink()
            return True
        return False


# ── Generic dataclass <-> JSON-safe conversion (mirrors benchmark.py) ─────────


def _to_jsonable(value: Any) -> Any:
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


# ── Combined retrieval + reasoning runs ─────────────────────────────────────────


@dataclass(frozen=True)
class CombinedBenchmarkRun:
    """A benchmark entry that optionally carries a retrieval
    ``BenchmarkRun`` (16F, unmodified) and/or a ``ReasoningBenchmarkRun``
    (this phase) - composition, never a new field bolted onto either run
    type. At least one of ``retrieval``/``reasoning`` must be present (see
    ``combine_benchmark_runs``).
    """

    run_id: str
    timestamp: str
    experiment_name: str
    retrieval: BenchmarkRun | None
    reasoning: ReasoningBenchmarkRun | None


def combine_benchmark_runs(
    *,
    experiment_name: str,
    retrieval: BenchmarkRun | None = None,
    reasoning: ReasoningBenchmarkRun | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
) -> CombinedBenchmarkRun:
    """Compose an already-built retrieval ``BenchmarkRun`` and/or
    ``ReasoningBenchmarkRun`` into one ``CombinedBenchmarkRun``. Raises
    ``ValueError`` if neither is supplied - a combined run with no metrics
    at all is not a valid benchmark entry.
    """
    if retrieval is None and reasoning is None:
        raise ValueError("combine_benchmark_runs requires at least one of retrieval/reasoning")
    return CombinedBenchmarkRun(
        run_id=run_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        experiment_name=experiment_name,
        retrieval=retrieval,
        reasoning=reasoning,
    )


# ── Comparison utilities (delegate entirely to this phase's own compare) ──────


def compare_reasoning_runs(
    baseline: ReasoningBenchmarkRun, candidate: ReasoningBenchmarkRun
) -> ReasoningRegressionReport:
    """Compare two reasoning runs. Delegates entirely to
    ``app.evaluation.reasoning_regression.compare_reasoning``.
    """
    return compare_reasoning(baseline.report, candidate.report)


def reasoning_regression_history(
    repository: ReasoningBenchmarkRepository, *, experiment_name: str | None = None
) -> tuple[ReasoningRegressionReport, ...]:
    runs = repository.list_runs(experiment_name=experiment_name)
    return tuple(compare_reasoning_runs(runs[i], runs[i + 1]) for i in range(len(runs) - 1))
