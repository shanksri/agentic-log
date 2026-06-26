"""WatermarkService — incremental-ingestion bookkeeping.

Responsibilities:
- read ``incident_sources.last_ingested_at``
- decide whether a run is a full backfill or an incremental run
- advance the watermark after a successful ingestion commit

Watermark strategy is ``run_start_time`` (see Phase 11A failure-mode analysis):
the new watermark is the wall-clock time captured at the start of the run, NOT
the maximum ``updated_at`` observed. The caller is responsible for capturing
``run_start_time`` before the first external API call and for only invoking
``advance()`` after the incident commit has succeeded.
"""

from __future__ import annotations

from datetime import datetime

from app.db.models import IncidentSource

MODE_BACKFILL = "backfill"
MODE_INCREMENTAL = "incremental"


class WatermarkService:
    def resolve(
        self,
        source: IncidentSource,
        *,
        force_backfill: bool = False,
    ) -> tuple[str, datetime | None]:
        """Return ``(mode, since)`` for the upcoming run.

        - NULL watermark or ``force_backfill`` → ``("backfill", None)``
        - otherwise → ``("incremental", last_ingested_at)``
        """
        if force_backfill or source.last_ingested_at is None:
            return MODE_BACKFILL, None
        return MODE_INCREMENTAL, source.last_ingested_at

    def advance(self, source: IncidentSource, new_watermark: datetime) -> None:
        """Set the watermark on the source row.

        Does NOT commit — the caller commits, so that a failure between the
        incident commit and this call leaves the watermark un-advanced and the
        next run safely re-processes the overlap window.
        """
        source.last_ingested_at = new_watermark
