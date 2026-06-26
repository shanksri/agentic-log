"""SourceRegistry — maps source_type strings to SourceAdapter instances."""

from __future__ import annotations

from app.ingestion.adapters.base import SourceAdapter


class _SourceRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, SourceAdapter] = {}

    def register(self, adapter: SourceAdapter) -> None:
        self._adapters[adapter.source_type] = adapter

    def get(self, source_type: str) -> SourceAdapter:
        try:
            return self._adapters[source_type]
        except KeyError:
            registered = ", ".join(sorted(self._adapters)) or "(none)"
            raise KeyError(
                f"No adapter registered for source_type={source_type!r}. "
                f"Registered: {registered}"
            ) from None

    def registered_types(self) -> list[str]:
        return sorted(self._adapters)


SourceRegistry = _SourceRegistry()
