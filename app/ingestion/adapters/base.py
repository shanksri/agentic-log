"""Generic ingestion abstractions.

NormalizedIncident is the canonical, source-agnostic representation that every
adapter must produce.  SourceAdapter is the ABC all adapters must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator


@dataclass(frozen=True)
class NormalizedIncident:
    source_type: str
    source_external_id: str
    source_url: str | None
    title: str
    description: str
    severity: str
    status: str
    incident_type: str
    environment: dict[str, Any]
    affected_components: list[str]
    tags: list[str]
    symptoms: list[str]
    root_cause_summary: str | None
    resolution_summary: str | None
    canonical_text: str
    confidence_score: float
    is_gold_labeled: bool
    created_at_source: datetime | None
    updated_at_source: datetime | None
    # Source-specific fields that don't fit the generic schema
    source_metadata: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(ABC):
    """Abstract base for all source adapters.

    Each concrete adapter owns:
    - ``source_type`` — the string stored in ``incidents.source_type``
    - ``collect()`` — yields raw payloads from the external system
    - ``normalize()`` — converts one raw payload to a ``NormalizedIncident``
    """

    source_type: str

    @abstractmethod
    def collect(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw payload dicts from the external source."""

    @abstractmethod
    def normalize(self, raw: dict[str, Any]) -> NormalizedIncident:
        """Convert one raw payload to a NormalizedIncident."""
