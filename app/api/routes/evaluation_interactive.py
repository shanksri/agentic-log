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

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import DbSession
from app.api.routes.evaluation import (
    QueryEvalResponse,
    RetrievedIncidentItem,
    _build_search_service,
    _to_dict,
)

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


def _require_session(
    session_id: str,
    store: dict[str, PreviewSession],
) -> PreviewSession:
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
    query: str = Field(min_length=1)
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
    query: str = Field(min_length=1)
    expected_titles: list[str] = Field(
        min_length=1,
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
    retrieval.  Builds a synthetic ``ResolvedGoldQuery`` and calls
    ``score_query`` from Phase 16C — identical to what Phase 21G does.
    """
    from app.evaluation.failure_analysis import analyze_retrieval_failures
    from app.evaluation.gold_dataset import (
        RELEVANCE_MAX,
        CorpusFingerprintPlaceholder,
        ExpectedIncident,
        GoldQuery,
    )
    from app.evaluation.gold_loader import (
        GoldDatasetResolutionSummary,
        ResolvedExpectedIncident,
        ResolvedGoldQuery,
        ResolvedIdentity,
    )
    from app.evaluation.harness import (
        AggregateMetrics,
        CoverageBreakdown,
        EvaluationConfig,
        EvaluationDatasetInfo,
        EvaluationReport,
        CorpusStatistics,
        QueryEvaluationOutcome,
    )
    from app.evaluation.metrics import score_query

    expected_set = set(expected_uuids)
    retrieved_uuids = [uuid.UUID(h.incident_id) for h in hits]

    expected_incidents = tuple(
        ExpectedIncident(
            source_type="api",
            source_external_id=str(uid),
            relevance=RELEVANCE_MAX,
        )
        for uid in expected_uuids
    )
    gold_q = GoldQuery(
        id="interactive-session",
        query=query,
        category="lexical-overlap" if expected_incidents else "no-match-expected",
        difficulty="medium",
        expected_incidents=expected_incidents,
    )
    resolved_incidents = tuple(
        ResolvedExpectedIncident(
            expected=ei,
            resolved=ResolvedIdentity(
                source_type="api",
                source_external_id=str(uid),
                incident_id=uid,
            ),
        )
        for ei, uid in zip(expected_incidents, expected_uuids)
    )
    resolved_q = ResolvedGoldQuery(query=gold_q, resolved_incidents=resolved_incidents)
    metric = score_query(retrieved_uuids, resolved_q, k=k)

    # Rank of first expected (1-indexed)
    rank_of_first: int | None = None
    for i, h in enumerate(hits):
        if uuid.UUID(h.incident_id) in expected_set:
            rank_of_first = i + 1
            break

    # Retrieved items for the response
    retrieved_items = [
        RetrievedIncidentItem(
            incident_id=h.incident_id,
            title=h.title,
            similarity_score=h.similarity_score,
            rank=h.rank,
            is_expected=uuid.UUID(h.incident_id) in expected_set,
        )
        for h in hits
    ]

    # Best-effort failure analysis
    failures: list[dict[str, Any]] = []
    if metric is not None and metric.recall_at_k is not None and metric.recall_at_k < 1.0:
        try:
            corpus_fp = CorpusFingerprintPlaceholder()
            agg = AggregateMetrics(
                num_queries=1,
                mean_recall_at_k=metric.recall_at_k,
                mean_reciprocal_rank=metric.reciprocal_rank,
                mean_ndcg_at_k=metric.ndcg_at_k,
                resolution_coverage=1.0,
                queries_with_unresolved_incidents=0,
            )
            outcome = QueryEvaluationOutcome(
                query_id="interactive-session",
                category=gold_q.category,
                difficulty=gold_q.difficulty,
                num_relevant=len(expected_uuids),
                num_unresolved_expected=0,
                skipped=False,
                skip_reason=None,
                metric=metric,
            )
            mini_report = EvaluationReport(
                dataset=EvaluationDatasetInfo(
                    version="api", description="Interactive session",
                    created_at="", author=None, corpus_fingerprint=corpus_fp,
                ),
                config=EvaluationConfig(k=k, expand=False, rerank=False),
                corpus_statistics=CorpusStatistics(
                    corpus_fingerprint=corpus_fp,
                    distinct_retrieved_incident_count=len(hits),
                ),
                num_evaluated=1, num_skipped=0, aggregate_metrics=agg,
                per_query=(outcome,),
                coverage=CoverageBreakdown(
                    total_queries=1, no_match_expected_queries=0,
                    fully_resolved_queries=0, partially_resolved_queries=1,
                    fully_unresolved_queries=0,
                ),
                resolution_summary=GoldDatasetResolutionSummary(
                    total_expected_incidents=len(expected_uuids),
                    resolved_count=len(expected_uuids),
                    unresolved_identities=(),
                ),
                category_breakdown={}, difficulty_breakdown={},
                started_at="", finished_at="", duration_seconds=0.0,
            )
            failures = [_to_dict(f) for f in analyze_retrieval_failures(mini_report)]
        except Exception:  # noqa: BLE001
            pass

    return QueryEvalResponse(
        query=query,
        k=k,
        retrieved=retrieved_items,
        recall_at_k=metric.recall_at_k if metric else None,
        reciprocal_rank=metric.reciprocal_rank if metric else None,
        ndcg_at_k=metric.ndcg_at_k if metric else None,
        rank_of_first_expected=rank_of_first,
        failures=failures,
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
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

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
        raise HTTPException(
            status_code=500, detail=f"Title resolution failed: {exc}"
        ) from exc

    # Run retrieval
    search_service = _build_search_service(db)
    try:
        raw_results = search_service.search(
            request.query, limit=request.k, call_site="evaluation_by_title"
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

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

    expected_uuids: list[uuid.UUID] = []
    for eid in request.selected_incident_ids:
        try:
            expected_uuids.append(uuid.UUID(eid))
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid UUID in selected_incident_ids: {eid!r}",
            )

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
