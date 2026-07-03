"""Human-Friendly Interactive Evaluation API (Phase 21H).

Adds a three-step annotation workflow on top of the Phase 21G evaluation API
WITHOUT modifying any existing endpoint.  A developer can now evaluate a
retrieval query without ever touching a UUID:

  Step 1 — POST /evaluation/query/preview
             Run retrieval once.  Get back human-readable incident titles,
             scores, and a session_id.

  Step 2 — (Human) inspect results, select correct incident(s).

  Step 3 — POST /evaluation/query/{session_id}/evaluate
             Reuse already-retrieved results.  Supply only the IDs of the
             correct incidents.  Get back Recall / MRR / NDCG / failures.

Convenience endpoints:

  POST /evaluation/query/by-title
       Resolve incident titles to UUIDs then score — no UUID hunting at all.

  GET  /evaluation/query/{session_id}
       Inspect an open session (query, retrieved results, status).

# Session storage

Sessions are stored in a module-level dict (process-local, no persistence
required per the brief).  Each session expires after ``SESSION_TTL_SECONDS``
(default 1800 = 30 min).  Expired sessions are lazily pruned on access.  The
store is exposed through a FastAPI dependency so tests can substitute a fresh
dict per test.

# Design constraints

- MUST NOT modify or duplicate any endpoint from Phase 21G (evaluation.py).
- MUST NOT rerun retrieval inside the /evaluate step — stored results are used.
- MUST NOT compute metrics, re-derive failures, or duplicate serialisation
  beyond what already exists in Phase 21G's helpers.
- Title resolution uses only ``db.query(Incident).filter(...)`` — no new
  retrieval, no LLM call, no extra service.
- This router shares the ``/evaluation`` prefix and ``evaluation`` tag with
  Phase 21G so Swagger groups them together.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import DbSession
from app.api.routes.evaluation import (
    QueryEvalResponse,
    _build_search_service,
    _score_query_against_expected,
)
from app.api.validation import validate_uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# ── Session store ─────────────────────────────────────────────────────────────

SESSION_TTL_SECONDS: int = 1800  # 30 minutes


@dataclass
class _SearchHit:
    """Plain-data snapshot of one retrieved incident — no SQLAlchemy types."""

    incident_id: str
    title: str
    similarity_score: float
    rank: int
    repo: str | None
    source: str | None
    source_type: str


@dataclass
class PreviewSession:
    """In-memory state for one interactive evaluation session."""

    session_id: str
    query: str
    k: int
    hits: list[_SearchHit]
    created_at: str
    expires_at: str
    status: str = "pending"  # "pending" | "evaluated"


# Module-level store — replaced per-test via the dependency.
_DEFAULT_STORE: dict[str, PreviewSession] = {}


def _get_session_store() -> dict[str, PreviewSession]:
    return _DEFAULT_STORE


SessionStore = Depends(_get_session_store)


def _prune_expired(store: dict[str, PreviewSession]) -> None:
    now = datetime.now(UTC).isoformat()
    expired = [sid for sid, s in store.items() if s.expires_at < now]
    for sid in expired:
        del store[sid]


_MAX_SESSION_ID_LENGTH = 200


def _require_session(
    session_id: str,
    store: dict[str, PreviewSession],
) -> PreviewSession:
    if not session_id or len(session_id) > _MAX_SESSION_ID_LENGTH:
        raise HTTPException(status_code=422, detail="session_id is malformed.")
    _prune_expired(store)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found or has expired.",
        )
    return session


# ── Request / response models ─────────────────────────────────────────────────


class PreviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=100)


class PreviewHit(BaseModel):
    incident_id: str
    title: str
    similarity_score: float
    rank: int
    repo: str | None = None
    source: str | None = None
    source_type: str


class PreviewResponse(BaseModel):
    session_id: str
    query: str
    k: int
    expires_at: str
    retrieved: list[PreviewHit]


class EvaluateSessionRequest(BaseModel):
    selected_incident_ids: list[str] = Field(
        max_length=1000,
        description=(
            "UUIDs of the incidents you consider correct for this query.  "
            "Pass an empty list to signal that none of the retrieved results "
            "are relevant (no-match-expected)."
        ),
    )


class SessionStatusResponse(BaseModel):
    session_id: str
    query: str
    k: int
    status: str
    created_at: str
    expires_at: str
    retrieved: list[PreviewHit]


class ByTitleRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    expected_titles: list[str] = Field(
        min_length=1,
        max_length=1000,
        description="Exact incident titles to resolve to UUIDs.",
    )
    k: int = Field(default=10, ge=1, le=100)


# ── Shared scoring helper ─────────────────────────────────────────────────────


def _score_hits_against(
    hits: list[_SearchHit],
    expected_uuids: list[uuid.UUID],
    k: int,
    query: str,
) -> QueryEvalResponse:
    """Compute metrics for ``hits`` vs ``expected_uuids`` without re-running
    retrieval. Delegates to Phase 21G's shared
    ``_score_query_against_expected`` (identical synthetic-``GoldQuery``
    scoring glue built once, not duplicated per phase).
    """
    retrieved = [(uuid.UUID(h.incident_id), h.title, h.similarity_score) for h in hits]
    return _score_query_against_expected(
        query_id="interactive-session",
        query=query,
        k=k,
        retrieved=retrieved,
        expected_uuids=expected_uuids,
    )


# ── POST /evaluation/query/preview ───────────────────────────────────────────
# Must be registered BEFORE /evaluation/query/{session_id}/evaluate and
# GET /evaluation/query/{session_id} so FastAPI does not mistake "preview"
# for a session_id.


@router.post("/query/preview", response_model=PreviewResponse)
def preview_query(
    request: PreviewRequest,
    db: DbSession,
    store: dict[str, PreviewSession] = SessionStore,
) -> PreviewResponse:
    """Step 1: run retrieval and create an interactive evaluation session.

    Returns a ``session_id`` plus ranked, human-readable incident results.
    No metrics are computed here — that happens in the ``/evaluate`` step
    once the reviewer has selected the correct incident(s).
    """
    search_service = _build_search_service(db)
    try:
        raw_results = search_service.search(
            request.query, limit=request.k, call_site="evaluation_interactive"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed for query %r", request.query)
        raise HTTPException(status_code=500, detail="Retrieval failed.") from exc

    hits = [
        _SearchHit(
            incident_id=str(r.incident.id),
            title=r.incident.title,
            similarity_score=r.similarity_score,
            rank=i + 1,
            repo=getattr(r.incident, "repo", None),
            source=getattr(r.incident, "source", None),
            source_type=getattr(r.incident, "source_type", ""),
        )
        for i, r in enumerate(raw_results)
    ]

    now = datetime.now(UTC)
    session_id = str(uuid.uuid4())
    session = PreviewSession(
        session_id=session_id,
        query=request.query,
        k=request.k,
        hits=hits,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=SESSION_TTL_SECONDS)).isoformat(),
    )
    _prune_expired(store)
    store[session_id] = session

    return PreviewResponse(
        session_id=session_id,
        query=request.query,
        k=request.k,
        expires_at=session.expires_at,
        retrieved=[
            PreviewHit(
                incident_id=h.incident_id,
                title=h.title,
                similarity_score=h.similarity_score,
                rank=h.rank,
                repo=h.repo,
                source=h.source,
                source_type=h.source_type,
            )
            for h in hits
        ],
    )


# ── POST /evaluation/query/by-title ──────────────────────────────────────────


@router.post("/query/by-title", response_model=QueryEvalResponse)
def evaluate_query_by_title(
    request: ByTitleRequest,
    db: DbSession,
) -> QueryEvalResponse:
    """Convenience: resolve expected incident titles to UUIDs, run retrieval,
    and score in one call — no manual UUID copying needed.

    Title matching is exact and case-insensitive.  If a title matches multiple
    incidents the most recently created one is used.  Titles that match no
    incident in the database are silently skipped (they may be typos or from
    a different corpus).
    """
    from sqlalchemy import func
    from app.db.models import Incident

    # Resolve titles → UUIDs via the database.
    # Lower-case both sides for case-insensitive exact matching without relying
    # on dialect-specific operators (func.lower works across SQLite and Postgres).
    title_lower = [t.lower() for t in request.expected_titles]
    try:
        rows = (
            db.query(Incident.id, Incident.title)
            .filter(func.lower(Incident.title).in_(title_lower))
            .all()
        )
        # Preserve order of expected_titles; deduplicate per title.
        by_lower: dict[str, uuid.UUID] = {}
        for row in rows:
            key = row.title.lower() if row.title else ""
            if key not in by_lower:
                by_lower[key] = row.id
        resolved_uuids = [by_lower[t] for t in title_lower if t in by_lower]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Title resolution failed")
        raise HTTPException(
            status_code=500, detail="Title resolution failed."
        ) from exc

    # Run retrieval
    search_service = _build_search_service(db)
    try:
        raw_results = search_service.search(
            request.query, limit=request.k, call_site="evaluation_by_title"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed for query %r", request.query)
        raise HTTPException(status_code=500, detail="Retrieval failed.") from exc

    hits = [
        _SearchHit(
            incident_id=str(r.incident.id),
            title=r.incident.title,
            similarity_score=r.similarity_score,
            rank=i + 1,
            repo=getattr(r.incident, "repo", None),
            source=getattr(r.incident, "source", None),
            source_type=getattr(r.incident, "source_type", ""),
        )
        for i, r in enumerate(raw_results)
    ]

    return _score_hits_against(hits, resolved_uuids, request.k, request.query)


# ── POST /evaluation/query/{session_id}/evaluate ──────────────────────────────


@router.post(
    "/query/{session_id}/evaluate",
    response_model=QueryEvalResponse,
)
def evaluate_session(
    session_id: str,
    request: EvaluateSessionRequest,
    store: dict[str, PreviewSession] = SessionStore,
) -> QueryEvalResponse:
    """Step 3: score a preview session against the reviewer's selection.

    Uses the retrieval results already cached in ``session_id`` — retrieval
    is NOT repeated.  Marks the session as ``evaluated``.
    """
    session = _require_session(session_id, store)

    expected_uuids: list[uuid.UUID] = [
        validate_uuid(eid, field_name="selected_incident_ids")
        for eid in request.selected_incident_ids
    ]

    result = _score_hits_against(
        session.hits, expected_uuids, session.k, session.query
    )

    # Update status in-place (dataclass is mutable)
    session.status = "evaluated"

    return result


# ── GET /evaluation/query/{session_id} ───────────────────────────────────────


@router.get("/query/{session_id}", response_model=SessionStatusResponse)
def get_session(
    session_id: str,
    store: dict[str, PreviewSession] = SessionStore,
) -> SessionStatusResponse:
    """Inspect an open evaluation session — query, retrieved results, status."""
    session = _require_session(session_id, store)
    return SessionStatusResponse(
        session_id=session.session_id,
        query=session.query,
        k=session.k,
        status=session.status,
        created_at=session.created_at,
        expires_at=session.expires_at,
        retrieved=[
            PreviewHit(
                incident_id=h.incident_id,
                title=h.title,
                similarity_score=h.similarity_score,
                rank=h.rank,
                repo=h.repo,
                source=h.source,
                source_type=h.source_type,
            )
            for h in session.hits
        ],
    )
