from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from app.db.models import IncidentSource
from app.ingestion.adapters.base import NormalizedIncident, SourceAdapter
from app.services.incident_ingestion import IncidentIngestionService
from app.services.watermark import MODE_BACKFILL, MODE_INCREMENTAL, WatermarkService

FIXED_NOW = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
PREVIOUS = datetime(2026, 6, 19, 8, 30, 0, tzinfo=timezone.utc)


# ── WatermarkService (pure logic) ────────────────────────────────────────────

def test_resolve_null_watermark_is_backfill() -> None:
    source = IncidentSource(source_type="github", name="acme/api")
    mode, since = WatermarkService().resolve(source)
    assert mode == MODE_BACKFILL
    assert since is None


def test_resolve_existing_watermark_is_incremental() -> None:
    source = IncidentSource(source_type="github", name="acme/api")
    source.last_ingested_at = PREVIOUS
    mode, since = WatermarkService().resolve(source)
    assert mode == MODE_INCREMENTAL
    assert since == PREVIOUS


def test_resolve_force_backfill_ignores_existing_watermark() -> None:
    source = IncidentSource(source_type="github", name="acme/api")
    source.last_ingested_at = PREVIOUS
    mode, since = WatermarkService().resolve(source, force_backfill=True)
    assert mode == MODE_BACKFILL
    assert since is None


def test_advance_sets_last_ingested_at() -> None:
    source = IncidentSource(source_type="github", name="acme/api")
    WatermarkService().advance(source, FIXED_NOW)
    assert source.last_ingested_at == FIXED_NOW


# ── Orchestration (watermark advancement rules) ──────────────────────────────

def _normalized(external_id: str) -> NormalizedIncident:
    return NormalizedIncident(
        source_type="fake",
        source_external_id=external_id,
        source_url=None,
        title="t",
        description="d",
        severity="unknown",
        status="open",
        incident_type="bug",
        environment={},
        affected_components=[],
        tags=[],
        symptoms=[],
        root_cause_summary=None,
        resolution_summary=None,
        canonical_text="canonical",
        confidence_score=0.5,
        is_gold_labeled=False,
        created_at_source=None,
        updated_at_source=None,
        source_metadata={},
    )


class FakeAdapter(SourceAdapter):
    source_type = "fake"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self.collected_since: datetime | None = None
        self.collect_called = False

    def collect(self, config: dict[str, Any], **kwargs: Any):
        self.collect_called = True
        self.collected_since = kwargs.get("since")
        yield from self._payloads

    def normalize(self, raw: dict[str, Any]) -> NormalizedIncident:
        return _normalized(raw["id"])


class FakeSession:
    """Minimal SQLAlchemy-Session stand-in: every lookup misses (insert path)."""

    def __init__(self) -> None:
        self.commits = 0

    def scalar(self, _stmt: Any) -> None:
        return None

    def add(self, _obj: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.commits += 1


class FakeEmbedding:
    model_name = "fake-model"

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    def embed_text(self, _text: str) -> list[float]:
        if self._fail:
            raise RuntimeError("embedding backend down")
        return [0.0] * 384


def _service(*, fail_embed: bool = False) -> IncidentIngestionService:
    return IncidentIngestionService(
        db=FakeSession(),  # type: ignore[arg-type]
        embedding_service=FakeEmbedding(fail=fail_embed),  # type: ignore[arg-type]
        now=lambda: FIXED_NOW,
    )


def test_first_ingestion_runs_backfill_and_sets_watermark() -> None:
    source = IncidentSource(source_type="fake", name="acme/api")
    adapter = FakeAdapter([{"id": "x#1"}, {"id": "x#2"}])
    service = _service()

    result = service.ingest_with_adapter(source, adapter, config={})

    assert result["mode"] == MODE_BACKFILL
    assert adapter.collected_since is None  # no since filter on backfill
    assert result["previous_watermark"] is None
    assert result["new_watermark"] == FIXED_NOW.isoformat()
    assert source.last_ingested_at == FIXED_NOW
    assert result["fetched"] == 2
    assert result["inserted"] == 2


def test_incremental_ingestion_passes_since_and_advances_watermark() -> None:
    source = IncidentSource(source_type="fake", name="acme/api")
    source.last_ingested_at = PREVIOUS
    adapter = FakeAdapter([{"id": "x#3"}])
    service = _service()

    result = service.ingest_with_adapter(source, adapter, config={})

    assert result["mode"] == MODE_INCREMENTAL
    assert adapter.collected_since == PREVIOUS  # incremental passes the watermark
    assert result["previous_watermark"] == PREVIOUS.isoformat()
    assert source.last_ingested_at == FIXED_NOW  # advanced to run_start_time


def test_failed_ingestion_leaves_watermark_unchanged() -> None:
    source = IncidentSource(source_type="fake", name="acme/api")
    source.last_ingested_at = PREVIOUS
    adapter = FakeAdapter([{"id": "x#4"}])
    service = _service(fail_embed=True)  # ingest() raises during embedding

    with pytest.raises(RuntimeError):
        service.ingest_with_adapter(source, adapter, config={})

    assert source.last_ingested_at == PREVIOUS  # NOT advanced


def test_empty_incremental_run_still_advances_watermark() -> None:
    source = IncidentSource(source_type="fake", name="acme/api")
    source.last_ingested_at = PREVIOUS
    adapter = FakeAdapter([])  # nothing changed since last run
    service = _service()

    result = service.ingest_with_adapter(source, adapter, config={})

    assert adapter.collect_called is True
    assert result["fetched"] == 0
    assert result["inserted"] == 0
    assert source.last_ingested_at == FIXED_NOW  # watermark still advances
