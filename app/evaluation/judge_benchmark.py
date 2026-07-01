"""Judge Benchmark Integration (Phase 20B).

Extends Phase 20A's reasoning benchmark pattern to judge evaluations,
WITHOUT modifying ``app.evaluation.reasoning_benchmark`` (Phase 20A) or
``app.evaluation.benchmark`` (Phase 16F). ``ReasoningBenchmarkRun`` (20A)
is hard-typed to ``InvestigationEvaluationReport``/``ReasoningRegressionReport``
— exactly the same reason Phase 20A itself gave for not modifying Phase
16F's ``BenchmarkRun``. This module instead introduces
``JudgedReasoningBenchmarkRun``, which composes an already-built
``ReasoningBenchmarkRun`` (20A, by reference, unmodified) with an OPTIONAL
tuple of ``JudgeEvaluation``s and an OPTIONAL ``JudgeAggregateMetrics`` —
"the two systems should coexist," satisfied by composition: every
heuristic field Phase 20A already computed is still there, untouched, and
the judge fields sit alongside it, never replacing it.

# Benchmark integration workflow

```
ReasoningBenchmarkRun (20A, unmodified)  ──┐
tuple[JudgeEvaluation, ...] (this phase,    │
  zero or more, e.g. one evaluate_session   │
  call per scenario)                        ┤
                                              ▼
                  create_judged_benchmark_run(experiment_name=,
                                               reasoning_run=,
                                               judge_evaluations=)
                                              │
                                              ▼
                       JudgedReasoningBenchmarkRun (frozen)
                  (judge_aggregate computed automatically from
                   judge_evaluations via aggregate_judge_evaluations)
                                              │
                                              ▼
                  judged_repository.save(run)
                                              │
                                              ▼
            compare_judge_aggregates(baseline, candidate)
                       -> per-stage JudgeMetricDelta
```

# Aggregation

``aggregate_judge_evaluations(evaluations)`` groups by ``.stage`` and
computes the mean ``score.value`` per stage (``JudgeAggregateMetrics``) —
the same "mean over defined values, report counts" convention Phase 16D's
``AggregateMetrics``/Phase 20A's ``ReasoningMetrics`` already use. A stage
with zero evaluations is reported as ``None`` (undefined), never
fabricated as 0.0.

# Regression

``compare_judge_aggregates(baseline, candidate)`` reuses the exact same
improved/regressed/unchanged/undefined classification rule Phase 16E/20A
already established (reimplemented locally here, for the same "do not
import another module's private underscore-prefixed helpers" reason Phase
20A's own regression module documents), applied to each stage's mean
score independently. There is no single "overall judge verdict" — per-
stage deltas are reported as-is; folding five independent semantic
judgments into one verdict would discard exactly the stage-level
granularity this phase's interface design (five separate ``evaluate_*``
methods) was built to preserve.

# Serialization

``FileJudgedReasoningBenchmarkRepository`` reuses the same generic
dataclass<->JSON conversion Phase 16F/20A already established
(``_to_jsonable``/``_from_jsonable``, duplicated here for the same reason
Phase 20A duplicates Phase 16F's: no shared base class exists to extend
without modifying an earlier phase).
"""

from __future__ import annotations

import json
import types
import typing
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.evaluation.judge import JudgeEvaluation
from app.evaluation.reasoning_benchmark import ReasoningBenchmarkRun

EPSILON = 1e-9


class DeltaClassification(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    UNDEFINED = "undefined"


# ── Aggregation ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeAggregateMetrics:
    """Mean judge score per stage, across a set of ``JudgeEvaluation``s -
    see module docstring's "Aggregation".
    """

    num_evaluations: int
    mean_plan_score: float | None
    mean_hypotheses_score: float | None
    mean_decision_score: float | None
    mean_critique_score: float | None
    mean_session_score: float | None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_judge_evaluations(
    evaluations: Sequence[JudgeEvaluation],
) -> JudgeAggregateMetrics:
    by_stage: dict[str, list[float]] = {}
    for evaluation in evaluations:
        by_stage.setdefault(evaluation.stage, []).append(evaluation.score.value)
    return JudgeAggregateMetrics(
        num_evaluations=len(evaluations),
        mean_plan_score=_mean(by_stage.get("plan", [])),
        mean_hypotheses_score=_mean(by_stage.get("hypotheses", [])),
        mean_decision_score=_mean(by_stage.get("decision", [])),
        mean_critique_score=_mean(by_stage.get("critique", [])),
        mean_session_score=_mean(by_stage.get("session", [])),
    )


# ── JudgedReasoningBenchmarkRun ─────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgedReasoningBenchmarkRun:
    """Composes a Phase 20A ``ReasoningBenchmarkRun`` (unmodified, by
    reference) with this phase's judge output. Both ``judge_evaluations``
    and ``judge_aggregate`` are optional - "the two systems should
    coexist," and a caller may run the heuristic harness without ever
    invoking a Judge.
    """

    run_id: str
    timestamp: str
    experiment_name: str
    reasoning_run: ReasoningBenchmarkRun
    judge_evaluations: tuple[JudgeEvaluation, ...]
    judge_aggregate: JudgeAggregateMetrics | None


def create_judged_benchmark_run(
    *,
    experiment_name: str,
    reasoning_run: ReasoningBenchmarkRun,
    judge_evaluations: Sequence[JudgeEvaluation] = (),
    run_id: str | None = None,
    timestamp: str | None = None,
) -> JudgedReasoningBenchmarkRun:
    evaluations = tuple(judge_evaluations)
    return JudgedReasoningBenchmarkRun(
        run_id=run_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        experiment_name=experiment_name,
        reasoning_run=reasoning_run,
        judge_evaluations=evaluations,
        judge_aggregate=aggregate_judge_evaluations(evaluations) if evaluations else None,
    )


# ── Repository ─────────────────────────────────────────────────────────────────


class JudgedReasoningBenchmarkRepository(ABC):
    @abstractmethod
    def save(self, run: JudgedReasoningBenchmarkRun) -> None: ...

    @abstractmethod
    def get(self, run_id: str) -> JudgedReasoningBenchmarkRun | None: ...

    @abstractmethod
    def list_runs(
        self, *, experiment_name: str | None = None
    ) -> tuple[JudgedReasoningBenchmarkRun, ...]: ...

    @abstractmethod
    def latest(
        self, *, experiment_name: str | None = None
    ) -> JudgedReasoningBenchmarkRun | None: ...

    @abstractmethod
    def delete(self, run_id: str) -> bool: ...


class InMemoryJudgedReasoningBenchmarkRepository(JudgedReasoningBenchmarkRepository):
    def __init__(self) -> None:
        self._runs: dict[str, JudgedReasoningBenchmarkRun] = {}

    def save(self, run: JudgedReasoningBenchmarkRun) -> None:
        if run.run_id in self._runs:
            raise ValueError(f"a run with id {run.run_id!r} already exists")
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> JudgedReasoningBenchmarkRun | None:
        return self._runs.get(run_id)

    def list_runs(
        self, *, experiment_name: str | None = None
    ) -> tuple[JudgedReasoningBenchmarkRun, ...]:
        runs = self._runs.values()
        if experiment_name is not None:
            runs = (run for run in runs if run.experiment_name == experiment_name)
        return tuple(sorted(runs, key=lambda run: run.timestamp))

    def latest(
        self, *, experiment_name: str | None = None
    ) -> JudgedReasoningBenchmarkRun | None:
        runs = self.list_runs(experiment_name=experiment_name)
        return runs[-1] if runs else None

    def delete(self, run_id: str) -> bool:
        return self._runs.pop(run_id, None) is not None


class FileJudgedReasoningBenchmarkRepository(JudgedReasoningBenchmarkRepository):
    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, run_id: str) -> Path:
        return self._directory / f"{run_id}.json"

    def save(self, run: JudgedReasoningBenchmarkRun) -> None:
        path = self._path_for(run.run_id)
        if path.exists():
            raise ValueError(f"a run with id {run.run_id!r} already exists")
        path.write_text(json.dumps(_to_jsonable(run), indent=2), encoding="utf-8")

    def get(self, run_id: str) -> JudgedReasoningBenchmarkRun | None:
        path = self._path_for(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return _from_jsonable(data, JudgedReasoningBenchmarkRun)

    def list_runs(
        self, *, experiment_name: str | None = None
    ) -> tuple[JudgedReasoningBenchmarkRun, ...]:
        runs = [self.get(path.stem) for path in self._directory.glob("*.json")]
        present = [run for run in runs if run is not None]
        if experiment_name is not None:
            present = [run for run in present if run.experiment_name == experiment_name]
        return tuple(sorted(present, key=lambda run: run.timestamp))

    def latest(
        self, *, experiment_name: str | None = None
    ) -> JudgedReasoningBenchmarkRun | None:
        runs = self.list_runs(experiment_name=experiment_name)
        return runs[-1] if runs else None

    def delete(self, run_id: str) -> bool:
        path = self._path_for(run_id)
        if path.exists():
            path.unlink()
            return True
        return False


# ── Generic dataclass <-> JSON-safe conversion (mirrors 16F/20A) ──────────────


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


# ── Regression ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeMetricDelta:
    baseline: float | None
    candidate: float | None
    delta: float | None
    classification: DeltaClassification


def _classify(baseline: float | None, candidate: float | None) -> DeltaClassification:
    if baseline is None and candidate is None:
        return DeltaClassification.UNCHANGED
    if baseline is None or candidate is None:
        return DeltaClassification.UNDEFINED
    delta = candidate - baseline
    if abs(delta) <= EPSILON:
        return DeltaClassification.UNCHANGED
    return DeltaClassification.IMPROVED if delta > 0 else DeltaClassification.REGRESSED


def _delta(baseline: float | None, candidate: float | None) -> JudgeMetricDelta:
    classification = _classify(baseline, candidate)
    delta = candidate - baseline if (baseline is not None and candidate is not None) else None
    return JudgeMetricDelta(
        baseline=baseline, candidate=candidate, delta=delta, classification=classification
    )


def compare_judge_aggregates(
    baseline: JudgeAggregateMetrics, candidate: JudgeAggregateMetrics
) -> dict[str, JudgeMetricDelta]:
    """Per-stage mean-score deltas - see module docstring's "Regression".
    Higher is always better for every stage's mean score (1-10 scale).
    """
    return {
        "plan": _delta(baseline.mean_plan_score, candidate.mean_plan_score),
        "hypotheses": _delta(baseline.mean_hypotheses_score, candidate.mean_hypotheses_score),
        "decision": _delta(baseline.mean_decision_score, candidate.mean_decision_score),
        "critique": _delta(baseline.mean_critique_score, candidate.mean_critique_score),
        "session": _delta(baseline.mean_session_score, candidate.mean_session_score),
    }
