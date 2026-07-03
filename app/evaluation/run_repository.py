"""Shared InMemory/File repository method bodies for the "run + repository"
persistence pattern duplicated (by necessity — see each phase's own module
docstring) across ``BenchmarkRepository`` (Phase 16F), ``ReasoningBenchmarkRepository``
(Phase 20A), and ``JudgedReasoningBenchmarkRepository`` (Phase 20B).

Each phase keeps its own named ``<X>Repository`` ABC (so ``isinstance``
checks, type hints, and existing imports are unaffected) and its own named
``InMemory<X>Repository``/``File<X>Repository`` classes — only the method
*bodies* are shared, via these two mixins. A concrete class like
``InMemoryBenchmarkRepository`` becomes::

    class InMemoryBenchmarkRepository(InMemoryRunRepositoryMixin, BenchmarkRepository):
        \"\"\"...\"\"\"

with no method bodies of its own; Python's MRO resolves every abstract
method on ``BenchmarkRepository`` to the mixin's concrete implementation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.evaluation.serialization import from_jsonable, to_jsonable

logger = logging.getLogger(__name__)


class InMemoryRunRepositoryMixin:
    """Process-local, non-persistent storage: ``save``/``get``/``list_runs``/
    ``latest``/``delete`` over an in-memory ``dict`` keyed by ``run_id``.
    """

    def __init__(self) -> None:
        self._runs: dict[str, Any] = {}

    def save(self, run: Any) -> None:
        if run.run_id in self._runs:
            raise ValueError(f"a run with id {run.run_id!r} already exists")
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> Any | None:
        return self._runs.get(run_id)

    def list_runs(self, *, experiment_name: str | None = None) -> tuple[Any, ...]:
        runs = self._runs.values()
        if experiment_name is not None:
            runs = (run for run in runs if run.experiment_name == experiment_name)
        return tuple(sorted(runs, key=lambda run: run.timestamp))

    def latest(self, *, experiment_name: str | None = None) -> Any | None:
        runs = self.list_runs(experiment_name=experiment_name)
        return runs[-1] if runs else None

    def delete(self, run_id: str) -> bool:
        return self._runs.pop(run_id, None) is not None


class FileRunRepositoryMixin:
    """JSON-file-backed storage: one ``{run_id}.json`` file per run in
    ``directory``. A concrete subclass must set the class attribute
    ``_run_type`` to the dataclass ``from_jsonable`` should reconstruct on
    ``get``/``list_runs``.
    """

    _run_type: type

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, run_id: str) -> Path | None:
        """Return the on-disk path for ``run_id``, or ``None`` if the
        resulting path would resolve outside ``self._directory`` (a
        defense-in-depth check — Phase 23 hardening — against a ``run_id``
        containing traversal segments, e.g. ``..`` combined with a
        platform path separator).
        """
        directory = self._directory.resolve()
        candidate = (self._directory / f"{run_id}.json").resolve()
        if candidate != directory and directory not in candidate.parents:
            return None
        return candidate

    def save(self, run: Any) -> None:
        path = self._path_for(run.run_id)
        if path is None:
            raise ValueError(f"run id {run.run_id!r} is not a valid identifier")
        if path.exists():
            raise ValueError(f"a run with id {run.run_id!r} already exists")
        path.write_text(json.dumps(to_jsonable(run), indent=2), encoding="utf-8")

    def get(self, run_id: str) -> Any | None:
        path = self._path_for(run_id)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping unreadable/corrupted run file: %s", path.name)
            return None
        try:
            return from_jsonable(data, self._run_type)
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping run file with unexpected schema: %s", path.name)
            return None

    def list_runs(self, *, experiment_name: str | None = None) -> tuple[Any, ...]:
        runs = [self.get(path.stem) for path in self._directory.glob("*.json")]
        present = [run for run in runs if run is not None]
        if experiment_name is not None:
            present = [run for run in present if run.experiment_name == experiment_name]
        return tuple(sorted(present, key=lambda run: run.timestamp))

    def latest(self, *, experiment_name: str | None = None) -> Any | None:
        runs = self.list_runs(experiment_name=experiment_name)
        return runs[-1] if runs else None

    def delete(self, run_id: str) -> bool:
        path = self._path_for(run_id)
        if path is not None and path.exists():
            path.unlink()
            return True
        return False
