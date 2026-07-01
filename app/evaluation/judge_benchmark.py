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
dataclass<->JSON conversion Phase 16F/20A already established, via the
shared ``app.evaluation.serialization``/``app.evaluation.run_repository``
modules (moved there to remove what were three duplicated copies of this
logic; see those modules' docstrings).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from app.evaluation.judge import JudgeEvaluation
from app.evaluation.reasoning_benchmark import ReasoningBenchmarkRun
from app.evaluation.run_repository import FileRunRepositoryMixin, InMemoryRunRepositoryMixin

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


class InMemoryJudgedReasoningBenchmarkRepository(
    InMemoryRunRepositoryMixin, JudgedReasoningBenchmarkRepository
):
    """Process-local, non-persistent ``JudgedReasoningBenchmarkRepository``.
    Method bodies live on ``InMemoryRunRepositoryMixin`` (see
    ``app.evaluation.run_repository``), shared with Phase 16F/20A's
    equivalent repositories.
    """


class FileJudgedReasoningBenchmarkRepository(
    FileRunRepositoryMixin, JudgedReasoningBenchmarkRepository
):
    """JSON-file-backed ``JudgedReasoningBenchmarkRepository``. Method
    bodies live on ``FileRunRepositoryMixin``, shared with Phase 16F/20A's
    equivalent repositories.
    """

    _run_type = JudgedReasoningBenchmarkRun


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
