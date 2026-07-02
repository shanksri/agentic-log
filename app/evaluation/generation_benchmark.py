"""Generation Benchmark Integration (Phase 22A).

Extends the benchmark-history pattern (Phase 16F retrieval → 20A reasoning →
20B judged) to generation evaluation, WITHOUT modifying any of those three.
Same reasoning as each predecessor gave: their ``*Run``/``*Repository`` types
are hard-typed to their own report types, so this module mirrors the shape
with a parallel ``GenerationBenchmarkRun``/``GenerationBenchmarkRepository``
for Phase 22A's ``GenerationEvaluationReport``.

Unlike those predecessors, the repository *implementations* are NOT
duplicated — the ``InMemory``/``File`` method bodies come from the shared
``app.evaluation.run_repository`` mixins (extracted from the 16F/20A/20B
triplication precisely so a fourth benchmark family would not need a fourth
copy). Only the small, family-specific ABC is defined here, for the same
Liskov-substitution reason Phase 20A's "Why duplicate, not subclass"
documents.

# Workflow

```
GenerationEvaluationReport (Phase 22A harness)
    │
    ▼
create_generation_benchmark_run(experiment_name=, report=, ...)
    │
    ▼
GenerationBenchmarkRun (frozen)
    │
    ▼
generation_repository.save(run)   [InMemory or File backend]
```

No generation regression runner exists yet (see the harness module's
"deliberately does NOT do" list); persisted runs give a future comparison
phase its history.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

from app.evaluation.generation_harness import GenerationEvaluationReport
from app.evaluation.run_repository import FileRunRepositoryMixin, InMemoryRunRepositoryMixin


# ── GenerationBenchmarkRun ────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationBenchmarkRun:
    """A single, immutable generation-evaluation experiment — mirrors
    ``BenchmarkRun`` (16F) / ``ReasoningBenchmarkRun`` (20A) field-for-field,
    scoped to ``GenerationEvaluationReport``.
    """

    run_id: str
    timestamp: str
    experiment_name: str
    report: GenerationEvaluationReport
    git_commit_sha: str | None
    notes: str | None


def create_generation_benchmark_run(
    *,
    experiment_name: str,
    report: GenerationEvaluationReport,
    git_commit_sha: str | None = None,
    notes: str | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
) -> GenerationBenchmarkRun:
    return GenerationBenchmarkRun(
        run_id=run_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        experiment_name=experiment_name,
        report=report,
        git_commit_sha=git_commit_sha,
        notes=notes,
    )


# ── Repository interface ──────────────────────────────────────────────────────


class GenerationBenchmarkRepository(ABC):
    """Storage interface for ``GenerationBenchmarkRun``s — mirrors
    ``BenchmarkRepository`` (16F); see Phase 20A's "Why duplicate, not
    subclass" for why each benchmark family keeps its own typed ABC.
    """

    @abstractmethod
    def save(self, run: GenerationBenchmarkRun) -> None: ...

    @abstractmethod
    def get(self, run_id: str) -> GenerationBenchmarkRun | None: ...

    @abstractmethod
    def list_runs(
        self, *, experiment_name: str | None = None
    ) -> tuple[GenerationBenchmarkRun, ...]: ...

    @abstractmethod
    def latest(
        self, *, experiment_name: str | None = None
    ) -> GenerationBenchmarkRun | None: ...

    @abstractmethod
    def delete(self, run_id: str) -> bool: ...


class InMemoryGenerationBenchmarkRepository(
    InMemoryRunRepositoryMixin, GenerationBenchmarkRepository
):
    """Process-local, non-persistent ``GenerationBenchmarkRepository``.
    Method bodies live on ``InMemoryRunRepositoryMixin`` (see
    ``app.evaluation.run_repository``), shared with the Phase 16F/20A/20B
    repositories.
    """


class FileGenerationBenchmarkRepository(
    FileRunRepositoryMixin, GenerationBenchmarkRepository
):
    """JSON-file-backed ``GenerationBenchmarkRepository``. Method bodies
    live on ``FileRunRepositoryMixin``, shared with the Phase 16F/20A/20B
    repositories.
    """

    _run_type = GenerationBenchmarkRun
