"""Reasoning Benchmark Integration (Phase 20A).

Extends Phase 16F's benchmark history pattern to reasoning evaluation,
WITHOUT modifying ``app.evaluation.benchmark`` (Phase 16F). ``BenchmarkRun``
and ``BenchmarkRepository`` there are hard-typed to retrieval's
``EvaluationReport``/``RegressionReport`` — adding reasoning fields to them
would be a modification, not an extension, so this module instead mirrors
the same run/repository pattern with a parallel
``ReasoningBenchmarkRun``/``ReasoningBenchmarkRepository`` for THIS phase's
``InvestigationEvaluationReport``/``ReasoningRegressionReport`` — duplicated,
not derived, because Phase 16F's ``ABC`` is hard-typed to the retrieval
report types (see module docstring's "Why duplicate, not subclass"). The
``InMemory``/``File`` repository *implementations* delegate their method
bodies to ``app.evaluation.run_repository``, shared with Phase 16F/20B's
equivalent repositories — only the ``ABC``s themselves are duplicated, for
the Liskov-substitution reason below.

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
```
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

from app.evaluation.reasoning_harness import InvestigationEvaluationReport
from app.evaluation.reasoning_regression import ReasoningRegressionReport, compare_reasoning
from app.evaluation.run_repository import FileRunRepositoryMixin, InMemoryRunRepositoryMixin


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


class InMemoryReasoningBenchmarkRepository(InMemoryRunRepositoryMixin, ReasoningBenchmarkRepository):
    """Process-local, non-persistent ``ReasoningBenchmarkRepository``.
    Method bodies live on ``InMemoryRunRepositoryMixin`` (see
    ``app.evaluation.run_repository``), shared with Phase 16F/20B's
    equivalent repositories.
    """


class FileReasoningBenchmarkRepository(FileRunRepositoryMixin, ReasoningBenchmarkRepository):
    """JSON-file-backed ``ReasoningBenchmarkRepository``. Method bodies live
    on ``FileRunRepositoryMixin``, shared with Phase 16F/20B's equivalent
    repositories.
    """

    _run_type = ReasoningBenchmarkRun


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
