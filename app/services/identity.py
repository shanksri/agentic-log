"""IdentityResolver — Phase 16A stable-identity layer.

Responsibilities:
- resolve a stable identity ``(source_type, source_external_id)`` to the
  current runtime identity (``ResolvedIdentity``)
- derive the stable identity from an existing ``Incident`` row

This is read-only, in-memory-against-existing-tables infrastructure. It
introduces no schema change, no migration, and no new persisted artifact. It
does not change ingestion, retrieval, investigation, or evaluation behavior —
nothing in the runtime calls it yet. Its purpose is to give the future
evaluation platform (gold v2 / harness, Phase 16B+) a way to anchor gold
queries to incidents by stable identity instead of by UUID, since UUIDs
regenerate on re-ingestion while ``(source_type, source_external_id)`` does
not (see docs/architecture/05_deduplication.md, 15_evaluation_framework.md).

The runtime identifier for an incident remains its ``id`` (UUID). Stable
identity is purely an evaluation-facing concept.

Resolution results are returned as ``ResolvedIdentity``, a plain dataclass,
never as the ``Incident`` ORM entity. This keeps every consumer of this
module (the future gold harness, regression runner, etc.) free of SQLAlchemy
session/lazy-load concerns and free of any temptation to read incident
content through the identity layer — see ``ResolvedIdentity``'s docstring.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Incident


class _HasSourceIdentity(Protocol):
    source_type: str
    source_external_id: str


@dataclass(frozen=True)
class StableIdentity:
    """The identity-anchored key ``(source_type, source_external_id)``.

    Equivalent to the pair hashed into ``Incident.deduplication_key`` by
    ``DeduplicationService.incident_key`` (app/services/deduplication.py), but
    expressed as a comparable/hashable value object rather than a hash, so it
    can be used directly as a gold-set key.
    """

    source_type: str
    source_external_id: str

    def __str__(self) -> str:
        return f"{self.source_type}:{self.source_external_id}"


@dataclass(frozen=True)
class ResolvedIdentity:
    """The outcome of resolving a ``StableIdentity`` to a current incident.

    Carries only the identity-resolution result — the stable identity that
    was resolved (as found on the matched row, not merely echoed from the
    query input) plus the runtime UUID it currently maps to.

    This intentionally contains NO incident metadata: no title, severity,
    status, canonical_text, symptoms, or any other content field. Consumers
    that need incident content must fetch it separately via ``incident_id``
    through the normal data-access path (e.g. a direct lookup or
    ``IncidentSearchService``). Keeping this DTO narrow is what lets the
    identity layer — and anything built on it (gold v2, the harness, the
    regression runner) — stay persistence-agnostic and free of SQLAlchemy
    coupling. Do not widen this dataclass with incident content fields.
    """

    source_type: str
    source_external_id: str
    incident_id: uuid.UUID


class IdentityResolver:
    """Resolves between stable identity and the current runtime identity.

    Strictly read-only: every method issues a ``SELECT`` against the existing
    ``incidents`` table (via the caller-supplied session) and never writes,
    flushes, or commits. Queries project only ``id``, ``source_type``, and
    ``source_external_id`` — the full ``Incident`` entity is never loaded.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    def resolve(self, identity: StableIdentity) -> ResolvedIdentity | None:
        """Return the ``ResolvedIdentity`` for a stable identity, or ``None``
        if no incident with that identity currently exists (e.g. it was never
        ingested, or was removed).
        """
        stmt = select(
            Incident.id, Incident.source_type, Incident.source_external_id
        ).where(
            Incident.source_type == identity.source_type,
            Incident.source_external_id == identity.source_external_id,
        )
        row = self._db.execute(stmt).first()
        if row is None:
            return None
        return ResolvedIdentity(
            source_type=row.source_type,
            source_external_id=row.source_external_id,
            incident_id=row.id,
        )

    def resolve_many(
        self, identities: list[StableIdentity]
    ) -> dict[StableIdentity, ResolvedIdentity | None]:
        """Batch form of ``resolve``. One query for all identities.

        Returns a dict covering every requested identity, with ``None`` for
        any identity that did not resolve to a current incident.
        """
        if not identities:
            return {}

        source_types = {identity.source_type for identity in identities}
        external_ids = {identity.source_external_id for identity in identities}
        stmt = select(
            Incident.id, Incident.source_type, Incident.source_external_id
        ).where(
            Incident.source_type.in_(source_types),
            Incident.source_external_id.in_(external_ids),
        )
        rows = self._db.execute(stmt).all()
        by_identity = {
            StableIdentity(row.source_type, row.source_external_id): ResolvedIdentity(
                source_type=row.source_type,
                source_external_id=row.source_external_id,
                incident_id=row.id,
            )
            for row in rows
        }
        return {identity: by_identity.get(identity) for identity in identities}

    @staticmethod
    def identity_for(incident: _HasSourceIdentity) -> StableIdentity:
        """Derive the stable identity of an already-loaded incident."""
        return StableIdentity(incident.source_type, incident.source_external_id)
