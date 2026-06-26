from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Embedding, Incident, IncidentSource, RawDocument, Symptom
from app.ingestion.adapters import SourceRegistry
from app.ingestion.adapters.base import NormalizedIncident, SourceAdapter
from app.ingestion.normalizers.github_normalizer import GitHubNormalizer
from app.services.deduplication import DeduplicationService
from app.services.embedding_service import EmbeddingService
from app.services.watermark import WatermarkService
from app.utils.json_sanitizer import sanitize_json_with_stats

logger = logging.getLogger(__name__)


class IncidentIngestionService:
    def __init__(
        self,
        db: Session,
        *,
        normalizer: GitHubNormalizer | None = None,
        deduplication: DeduplicationService | None = None,
        embedding_service: EmbeddingService | None = None,
        watermark: WatermarkService | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.db = db
        self.normalizer = normalizer or GitHubNormalizer()
        self.deduplication = deduplication or DeduplicationService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.watermark = watermark or WatermarkService()
        self._now = now or (lambda: datetime.now(timezone.utc))

    # ── Generic entry point (Phase 11A-1) ────────────────────────────────────

    def ingest(
        self,
        source: IncidentSource,
        payloads: list[dict[str, Any]],
        normalized_items: list[NormalizedIncident],
    ) -> dict[str, int | str]:
        """Ingest pre-collected, pre-normalized incidents.

        The caller (adapter dispatcher) is responsible for collection and
        normalization.  This method handles deduplication, upsert, and
        embedding — all source-agnostic operations.
        """
        inserted = 0
        updated = 0
        skipped = 0

        for payload, normalized in zip(payloads, normalized_items):
            raw_document = self._upsert_raw_document(source, payload, normalized)
            outcome = self._upsert_incident(raw_document, normalized)
            if outcome == "inserted":
                inserted += 1
            elif outcome == "updated":
                updated += 1
            else:
                skipped += 1

        self.db.commit()
        return {
            "source": f"{source.source_type}:{source.name}",
            "fetched": len(payloads),
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
        }

    # ── Watermark-aware adapter dispatch (Phase 11A-2) ───────────────────────

    def ingest_with_adapter(
        self,
        source: IncidentSource,
        adapter: SourceAdapter,
        config: dict[str, Any],
        *,
        force_backfill: bool = False,
    ) -> dict[str, int | str | None]:
        """Run one watermark-aware ingestion through the given adapter.

        Watermark advancement rules:
          - run_start_time is captured before any external API call
          - the watermark is advanced only AFTER the incident commit succeeds
          - a failure anywhere before advance() leaves the watermark unchanged
        """
        mode, since = self.watermark.resolve(source, force_backfill=force_backfill)
        previous_watermark = source.last_ingested_at
        run_start_time = self._now()

        payloads = list(adapter.collect(config, since=since))
        normalized_items = [adapter.normalize(payload) for payload in payloads]

        # ingest() commits the incidents. If it raises, advance() is never
        # reached and the watermark stays put.
        result: dict[str, int | str | None] = dict(
            self.ingest(source, payloads, normalized_items)
        )

        self.watermark.advance(source, run_start_time)
        self.db.commit()

        result["mode"] = mode
        result["previous_watermark"] = (
            previous_watermark.isoformat() if previous_watermark else None
        )
        result["new_watermark"] = run_start_time.isoformat()

        # Surface collector diagnostics when the adapter exposes them (14C).
        # Adapters without diagnostics (e.g. Jira) leave the payload unchanged,
        # preserving backward compatibility.
        diagnostics = getattr(adapter, "last_diagnostics", None)
        if diagnostics is not None:
            result["exit_reason"] = diagnostics.exit_reason
            result["pages_traversed"] = diagnostics.pages_traversed
            result["raw_items_scanned"] = diagnostics.raw_items_scanned
            result["effective_yield"] = diagnostics.effective_yield

        logger.info(
            "ingestion_complete",
            extra={
                "source": f"{source.source_type}:{source.name}",
                "mode": mode,
                "previous_watermark": result["previous_watermark"],
                "new_watermark": result["new_watermark"],
                "incidents_collected": result["fetched"],
                "inserted": result["inserted"],
                "updated": result["updated"],
                "skipped": result["skipped"],
            },
        )
        return result

    # ── Core dispatch: the single execution path for BOTH modes (13B) ────────

    def _dispatch(
        self,
        source: IncidentSource,
        config: dict[str, Any],
        *,
        force_backfill: bool = False,
    ) -> dict[str, int | str | None]:
        """Resolve the adapter via the registry and run the shared pipeline.

        Used by every ingestion entry point — ad-hoc (Mode A) and stored-source
        (Mode B) alike. There is no source-type branching and no hardcoded
        adapter instantiation: the adapter is always resolved from
        ``SourceRegistry`` by ``source.source_type``.
        """
        adapter = SourceRegistry.get(source.source_type)  # raises if unregistered
        return self.ingest_with_adapter(source, adapter, config, force_backfill=force_backfill)

    # ── Mode B: stored-source ingestion (Phase 13A) ──────────────────────────

    def ingest_source(
        self,
        source_id: Any,
        *,
        force_backfill: bool = False,
    ) -> dict[str, int | str | None]:
        """Ingest a source identified only by its incident_sources row id.

        Config is read from ``source.config``. Primary path for scheduled jobs,
        automation, and incremental syncs.
        """
        source = self.db.get(IncidentSource, source_id)
        if source is None:
            raise ValueError(f"No incident_source with id={source_id!r}")

        config: dict[str, Any] = dict(source.config or {})

        # Convenience: GitHub sources may omit the token from the stored config
        # and inherit it from the environment, so a row only needs owner/repo.
        if source.source_type == "github" and not config.get("token"):
            config["token"] = settings.github_token

        result = self._dispatch(source, config, force_backfill=force_backfill)
        result["source"] = f"{source.source_type}:{source.name}"
        return result

    # ── Mode A: ad-hoc payload ingestion (no pre-existing row required) ───────

    def ingest_github_repo(
        self,
        owner: str,
        repo: str,
        *,
        state: str,
        limit: int,
        include_comments: bool,
        force_backfill: bool = False,
    ) -> dict[str, int | str | None]:
        """Ad-hoc GitHub ingestion from a payload (Mode A).

        The caller supplies owner/repo/limit directly; the incident_sources
        row is auto-managed (get-or-create) and no config is persisted to it.
        """
        source = self._get_or_create_source(owner, repo)
        config: dict[str, Any] = {
            "owner": owner,
            "repo": repo,
            "state": state,
            "limit": limit,
            "include_comments": include_comments,
            "token": settings.github_token,
        }
        result = self._dispatch(source, config, force_backfill=force_backfill)
        # Preserve the historic "github:owner/repo" source label.
        result["source"] = f"github:{owner}/{repo}"
        return result

    def ingest_jira_project(
        self,
        base_url: str,
        project_key: str,
        *,
        limit: int = 50,
        status_filter: list[str] | None = None,
        token: str | None = None,
        force_backfill: bool = False,
    ) -> dict[str, int | str | None]:
        """Ad-hoc Jira ingestion from a payload (Mode A)."""
        source = self._get_or_create_source_generic(
            source_type="jira",
            name=f"Jira {project_key}",
            base_url=base_url,
        )
        config: dict[str, Any] = {
            "base_url": base_url,
            "project_key": project_key,
            "limit": limit,
            "status_filter": status_filter,
            "token": token,
        }
        result = self._dispatch(source, config, force_backfill=force_backfill)
        result["source"] = f"jira:{project_key}"
        return result

    # ── Source row helpers ───────────────────────────────────────────────────

    def _get_or_create_source(self, owner: str, repo: str) -> IncidentSource:
        return self._get_or_create_source_generic(
            source_type="github",
            name=f"GitHub {owner}/{repo}",
            base_url=f"https://github.com/{owner}/{repo}",
        )

    def _get_or_create_source_generic(
        self,
        *,
        source_type: str,
        name: str,
        base_url: str | None,
    ) -> IncidentSource:
        source = self.db.scalar(
            select(IncidentSource).where(
                IncidentSource.source_type == source_type,
                IncidentSource.name == name,
            )
        )
        if source:
            return source
        source = IncidentSource(source_type=source_type, name=name, base_url=base_url)
        self.db.add(source)
        self.db.flush()
        return source

    # ── Upsert helpers (source-agnostic) ────────────────────────────────────

    def _upsert_raw_document(
        self,
        source: IncidentSource,
        payload: dict[str, Any],
        normalized: NormalizedIncident,
    ) -> RawDocument:
        # Persistence boundary owns sanitization. Both the stored payload and
        # the dedup hash MUST derive from the same sanitized object, or updates
        # would fire forever on payloads that contained control characters.
        sanitized_payload, removed = sanitize_json_with_stats(payload)
        if removed:
            logger.debug(
                "payload_sanitized external_id=%s removed_control_chars=%d",
                normalized.source_external_id,
                removed,
            )
        payload_hash = self.deduplication.payload_hash(sanitized_payload)
        raw_document = self.db.scalar(
            select(RawDocument).where(
                RawDocument.source_id == source.id,
                RawDocument.external_id == normalized.source_external_id,
            )
        )
        if raw_document:
            raw_document.payload = sanitized_payload
            raw_document.payload_hash = payload_hash
            raw_document.source_url = normalized.source_url
            self.db.flush()
            return raw_document

        raw_document = RawDocument(
            source_id=source.id,
            external_id=normalized.source_external_id,
            source_url=normalized.source_url,
            payload=sanitized_payload,
            payload_hash=payload_hash,
        )
        self.db.add(raw_document)
        self.db.flush()
        return raw_document

    def _upsert_incident(
        self,
        raw_document: RawDocument,
        normalized: NormalizedIncident,
    ) -> str:
        deduplication_key = self.deduplication.incident_key(normalized)
        existing = self.db.scalar(
            select(Incident).where(Incident.deduplication_key == deduplication_key)
        )
        text_hash = self.deduplication.text_hash(normalized.canonical_text)

        if existing and self._is_embedding_fresh(existing, text_hash):
            return "skipped"

        if existing:
            incident = existing
            outcome = "updated"
        else:
            incident = Incident(deduplication_key=deduplication_key)
            self.db.add(incident)
            outcome = "inserted"

        incident.raw_document_id = raw_document.id
        incident.source_type = normalized.source_type
        incident.source_external_id = normalized.source_external_id
        incident.source_url = normalized.source_url
        incident.title = normalized.title
        incident.description = normalized.description
        incident.severity = normalized.severity
        incident.status = normalized.status
        incident.incident_type = normalized.incident_type
        incident.environment = normalized.environment
        incident.affected_components = normalized.affected_components
        incident.tags = normalized.tags
        incident.root_cause_summary = normalized.root_cause_summary
        incident.resolution_summary = normalized.resolution_summary
        incident.canonical_text = normalized.canonical_text
        incident.confidence_score = normalized.confidence_score
        incident.is_gold_labeled = normalized.is_gold_labeled
        incident.created_at_source = normalized.created_at_source
        incident.updated_at_source = normalized.updated_at_source
        incident.source_metadata = normalized.source_metadata

        # `source` is a generic source identifier (mirrors source_type) and must
        # be populated for EVERY source so the search `source=` filter works
        # across source types — not only GitHub.
        incident.source = normalized.source_type

        # GitHub-specific legacy columns remain GitHub-only.
        meta = normalized.source_metadata
        if normalized.source_type == "github":
            incident.owner = meta.get("owner")
            incident.repo = meta.get("repo")
            incident.state = meta.get("state")

        incident.symptoms.clear()
        for symptom_text in normalized.symptoms:
            incident.symptoms.append(Symptom(text=symptom_text))

        self.db.flush()
        self._upsert_embedding(incident, text_hash)
        return outcome

    def _is_embedding_fresh(self, incident: Incident, text_hash: str) -> bool:
        return any(
            embedding.model_name == self.embedding_service.model_name
            and embedding.text_hash == text_hash
            for embedding in incident.embeddings
        )

    def _upsert_embedding(self, incident: Incident, text_hash: str) -> None:
        vector = self.embedding_service.embed_text(incident.canonical_text)
        embedding = self.db.scalar(
            select(Embedding).where(
                Embedding.incident_id == incident.id,
                Embedding.model_name == self.embedding_service.model_name,
            )
        )
        if embedding:
            embedding.embedding = vector
            embedding.text_hash = text_hash
            return
        self.db.add(
            Embedding(
                incident_id=incident.id,
                model_name=self.embedding_service.model_name,
                embedding=vector,
                text_hash=text_hash,
            )
        )
