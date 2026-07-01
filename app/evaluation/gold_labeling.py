"""Assisted Gold Labeling Framework (Phase 21D).

Assists a human reviewer in populating the ``expected_incidents`` field of
``GoldQuery`` objects produced by Phase 21C's ``export_gold_dataset()``.

Without this framework, a reviewer must manually run a search, scan results,
and copy UUIDs — slow and error-prone.  With it, the system runs retrieval
automatically for each ``CandidateQuery``, presents the top-K results in rank
order, and the reviewer simply *selects* the correct incident ID(s).  The
framework then assembles a ``GoldQuery`` with a fully-populated
``expected_incidents`` tuple.

# Updated architecture

```
CandidateQuery  (from Phase 21C)
        │
        ▼
GoldLabelRetriever.retrieve_candidates(query, limit)
        │    ┌──────────────────────────┐
        │    │  DenseGoldLabelRetriever │  wraps IncidentSearchService.search()
        │    │  HybridGoldLabelRetriever│  wraps HybridRetriever.retrieve()
        │    └──────────────────────────┘
        ▼
tuple[CandidateIncident, ...]  (ranked 1-based)
        │
        ▼
GoldLabelSession  (status=PENDING)
        │
        ▼
Human reviewer calls .label(session_id, selected_ids)
  or                 .skip(session_id)
        │
        ▼
GoldLabelSession  (status=LABELED or SKIPPED)
        │
        ▼
GoldLabelingWorkflow.export_labeled_queries()
        │
        ▼
tuple[GoldQuery, ...]  (expected_incidents fully populated — no placeholder ())
```

No automatic acceptance.  The framework never selects incident IDs.  Human
reviewer always makes the final decision.

# Retriever abstraction

``GoldLabelRetriever`` is an ABC with one method:

    retrieve_candidates(query, limit) -> tuple[CandidateIncident, ...]

``DenseGoldLabelRetriever`` wraps ``IncidentSearchService.search()``.
``HybridGoldLabelRetriever`` wraps ``HybridRetriever.retrieve()``.

Neither duplicates retrieval logic — they only translate the existing return
types into ``CandidateIncident`` (a plain-data, schema-only type this module
introduces so the labeling workflow has no SQLAlchemy dependency in its
public API).

# Review flow

1. For each ``CandidateQuery``, call ``GoldLabelingWorkflow.add_query(q)``
   — this fires retrieval and creates a ``PENDING`` session.
2. Reviewer calls ``workflow.sessions()`` to list the queue.
3. For each session: ``workflow.label(session_id, ["INC-001", "INC-007"])``
   or ``workflow.skip(session_id)``.
4. Multi-selection is fully supported: 0, 1, or many incident IDs.
5. ``workflow.export_labeled_queries()`` returns only ``LABELED`` sessions
   converted to ``GoldQuery``; ``SKIPPED`` sessions are retained in the
   queue so they remain visible for future passes.

# Traceability

Every exported ``GoldQuery`` carries a ``LabelingProvenance`` object stored
alongside it in ``LabeledGoldQuery`` — preserving the original AI-generated
query text, the retrieval strategy used, and the IDs selected.  This allows
future auditors to re-run the same retrieval and compare candidate sets
without accessing the original session state.

# Statistics

``LabelingStats`` is computed on-demand:
- total sessions, labeled, skipped, pending
- average candidates presented per session
- average incidents selected per labeled session
- single-label %, multi-label % (of labeled sessions)

# Forbidden imports

This module must never import: evaluation harness, regression, Judge, Planner,
Critic, Benchmark.  It depends only on:
    - IncidentSearchService (app.services.search)
    - HybridRetriever (app.services.hybrid_search)
    - Gold Dataset models (app.evaluation.gold_dataset)
    - CandidateQuery (app.evaluation.dataset_authoring)
    - Standard library

# Risks discovered

- ``DenseGoldLabelRetriever`` requires a live database and embedding service;
  tests must inject a fake to avoid network/db dependencies.
- ``CandidateIncident.incident_id`` is a string extracted from the SQLAlchemy
  model's UUID; UUID formatting is caller-controlled, which could cause
  mismatch if callers mix ``str(uuid)`` with ``uuid.hex``.
- ``GoldLabelingWorkflow.add_query()`` fires retrieval eagerly and
  synchronously — a slow embedding service will block the caller.  A future
  phase could introduce async or batched retrieval without modifying this
  module's public API.
- The labeling workflow is in-process only: a fresh ``GoldLabelingWorkflow``
  instance has no memory of previous sessions.
- Default relevance grade is 3 (exact match) for all selected incidents; a
  future phase could allow per-selection relevance grading without breaking
  existing callers (add an optional ``relevance`` parameter to ``.label()``).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from app.evaluation.dataset_authoring import CandidateQuery
from app.evaluation.gold_dataset import (
    RELEVANCE_MAX,
    ExpectedIncident,
    GoldQuery,
)

# ── Public constants ────────────────────────────────────────────────────────────

DEFAULT_LIMIT = 10

#: Relevance grade assigned to every reviewer-selected incident.
#: 3 = exact/primary match on Phase 16B's 1-3 scale.  A future phase may
#: allow per-selection grading; this default is always the safest choice.
DEFAULT_RELEVANCE: int = RELEVANCE_MAX

RETRIEVAL_STRATEGY_DENSE = "dense"
RETRIEVAL_STRATEGY_HYBRID = "hybrid"


# ── CandidateIncident ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateIncident:
    """A single retrieval result presented to the human reviewer.

    ``incident_id`` is the string form of the incident's UUID; ``source``
    carries ``source_type:source_external_id`` so the reviewer can identify
    the upstream record without a DB lookup.  ``score`` and ``rank``
    (1-based) are provided for transparency — the reviewer can see how
    confident the retrieval system was without the system acting on that
    confidence.
    """

    incident_id: str
    title: str
    source: str
    score: float
    rank: int
    explanation: str = ""

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.incident_id:
            problems.append("incident_id must be non-empty")
        if not self.title:
            problems.append(f"candidate {self.incident_id!r}: title must be non-empty")
        if self.rank < 1:
            problems.append(
                f"candidate {self.incident_id!r}: rank must be >= 1, got {self.rank}"
            )
        return problems

    def is_valid(self) -> bool:
        return not self.issues()


# ── LabelDecision ────────────────────────────────────────────────────────────────


class LabelDecision(str, Enum):
    PENDING = "pending"
    LABELED = "labeled"
    SKIPPED = "skipped"


# ── GoldLabelSession ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GoldLabelSession:
    """One labeling task: a single query with its retrieved candidates and the
    reviewer's selection.

    ``selected_incident_ids`` is empty until a reviewer calls
    ``GoldLabelingWorkflow.label()``.  Supports 0, 1, or many IDs —
    zero is valid when the reviewer determines no retrieved candidate is
    correct (which will produce a ``no-match-expected`` GoldQuery on export).
    ``status`` transitions: PENDING → LABELED or SKIPPED.
    Original state is never mutated; reviews create replacement objects.
    """

    session_id: str
    query_id: str
    query: str
    retrieval_strategy: str
    candidates: tuple[CandidateIncident, ...]
    selected_incident_ids: tuple[str, ...] = field(default_factory=tuple)
    status: LabelDecision = LabelDecision.PENDING

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.session_id:
            problems.append("session_id must be non-empty")
        if not self.query_id:
            problems.append("query_id must be non-empty")
        if not self.query:
            problems.append(f"session {self.session_id!r}: query must be non-empty")
        if self.status == LabelDecision.LABELED:
            for inc_id in self.selected_incident_ids:
                if not inc_id:
                    problems.append(
                        f"session {self.session_id!r}: selected_incident_ids "
                        "contains an empty string"
                    )
        return problems

    def is_valid(self) -> bool:
        return not self.issues()


# ── LabelingProvenance ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelingProvenance:
    """Audit trail attached to every exported ``LabeledGoldQuery``.

    Preserves the original AI-generated query text (before any human edit),
    the retrieval strategy used during labeling, and the IDs selected, so
    future auditors can reproduce the same candidate pool without access to
    the live session state.
    """

    query_id: str
    original_query: str
    retrieval_strategy: str
    selected_incident_ids: tuple[str, ...]
    labeled_at: str


# ── LabeledGoldQuery ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabeledGoldQuery:
    """A ``GoldQuery`` with full ``expected_incidents`` plus its audit trail.

    This is the primary export type; callers that only need the
    ``GoldQuery`` portion (e.g. to pass to ``GoldDataset``) can use
    ``.gold_query`` directly.
    """

    gold_query: GoldQuery
    provenance: LabelingProvenance


# ── LabelingStats ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelingStats:
    """On-demand statistics snapshot for the current labeling workflow.

    Computed from live session state — never accumulated separately.
    """

    total_sessions: int
    labeled: int
    skipped: int
    pending: int
    avg_candidates_presented: float
    avg_selected_per_labeled: float
    single_label_pct: float
    multi_label_pct: float


# ── GoldLabelRetriever ABC ───────────────────────────────────────────────────────


class GoldLabelRetriever(ABC):
    """Abstract retriever for the gold labeling workflow.

    Concrete implementations wrap existing retrieval services without
    duplicating any retrieval logic.  Each implementation translates the
    service's own result type into ``CandidateIncident`` objects.
    """

    @abstractmethod
    def retrieve_candidates(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[CandidateIncident, ...]:
        """Return the top ``limit`` candidates for ``query`` in rank order.

        Rank 1 is the highest-scoring result.  Each candidate carries a
        ``score`` (float) and ``rank`` (1-based int) so the reviewer sees
        how confident retrieval was, without the system acting on that
        confidence.
        """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Short identifier recorded in ``GoldLabelSession.retrieval_strategy``."""


# ── Concrete retrievers ──────────────────────────────────────────────────────────


class DenseGoldLabelRetriever(GoldLabelRetriever):
    """Wraps ``IncidentSearchService.search()`` (dense vector retrieval).

    Translates ``IncidentSearchResult`` objects into ``CandidateIncident``
    objects.  No retrieval logic is duplicated — this is a pure adapter.
    """

    def __init__(self, search_service: object) -> None:
        # Accepts the service as `object` so this module can be imported
        # without SQLAlchemy being available (e.g. in tests that inject a
        # fake).  The actual type is IncidentSearchService from
        # app.services.search — enforced at runtime by duck typing.
        self._service = search_service

    @property
    def strategy_name(self) -> str:
        return RETRIEVAL_STRATEGY_DENSE

    def retrieve_candidates(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[CandidateIncident, ...]:
        results = self._service.search(query, limit=limit, call_site="gold_labeling")
        candidates: list[CandidateIncident] = []
        for rank, result in enumerate(results, start=1):
            incident = result.incident
            source = f"{incident.source_type}:{incident.source_external_id}"
            candidates.append(CandidateIncident(
                incident_id=str(incident.id),
                title=incident.title or "",
                source=source,
                score=result.similarity_score,
                rank=rank,
                explanation=f"dense similarity {result.similarity_score:.4f}",
            ))
        return tuple(candidates)


class HybridGoldLabelRetriever(GoldLabelRetriever):
    """Wraps ``HybridRetriever.retrieve()`` (dense + BM25 via RRF).

    Translates ``HybridSearchResult`` objects into ``CandidateIncident``
    objects.  No retrieval logic is duplicated — this is a pure adapter.
    ``dense_result`` is the preferred source of incident metadata; if absent
    (BM25-only result), the incident_id comes from ``document_id`` and title
    is left empty (reviewers can resolve via the corpus).
    """

    def __init__(self, hybrid_retriever: object) -> None:
        self._retriever = hybrid_retriever

    @property
    def strategy_name(self) -> str:
        return RETRIEVAL_STRATEGY_HYBRID

    def retrieve_candidates(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[CandidateIncident, ...]:
        results = self._retriever.retrieve(query, limit=limit)
        candidates: list[CandidateIncident] = []
        for rank, result in enumerate(results, start=1):
            if result.dense_result is not None:
                incident = result.dense_result.incident
                title = incident.title or ""
                source = f"{incident.source_type}:{incident.source_external_id}"
            else:
                title = ""
                source = ""
            explanation = (
                f"rrf_score={result.rrf_score:.4f} "
                f"dense_rank={result.dense_rank} "
                f"bm25_rank={result.bm25_rank}"
            )
            candidates.append(CandidateIncident(
                incident_id=result.document_id,
                title=title,
                source=source,
                score=result.rrf_score,
                rank=rank,
                explanation=explanation,
            ))
        return tuple(candidates)


# ── GoldLabelingWorkflow ─────────────────────────────────────────────────────────


class GoldLabelingWorkflow:
    """Stateful labeling queue that drives the assisted gold-labeling loop.

    State model:
    - ``_sessions``: dict[session_id → GoldLabelSession]  (insertion order)
    Individual ``GoldLabelSession`` objects are frozen dataclasses; reviews
    create replacement objects and replace the dict entry — no mutation.

    Usage::

        workflow = GoldLabelingWorkflow(DenseGoldLabelRetriever(search_service))

        for candidate_query in my_candidates:
            workflow.add_query(candidate_query)

        for session in workflow.pending_sessions():
            # present session.candidates to the reviewer ...
            workflow.label(session.session_id, ["uuid-of-incident"])
            # or workflow.skip(session.session_id)

        labeled = workflow.export_labeled_queries()
        # labeled is a tuple[LabeledGoldQuery, ...]
    """

    def __init__(
        self,
        retriever: GoldLabelRetriever,
        *,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        self._retriever = retriever
        self._limit = limit
        self._sessions: dict[str, GoldLabelSession] = {}

    # ── Queue population ────────────────────────────────────────────────────────

    def add_query(self, candidate: CandidateQuery) -> GoldLabelSession:
        """Retrieve candidates for ``candidate.query`` and create a PENDING session.

        Returns the new session.  Fires retrieval eagerly and synchronously.
        """
        retrieved = self._retriever.retrieve_candidates(candidate.query, self._limit)
        session = GoldLabelSession(
            session_id=str(uuid.uuid4()),
            query_id=candidate.id,
            query=candidate.effective_query,
            retrieval_strategy=self._retriever.strategy_name,
            candidates=retrieved,
            status=LabelDecision.PENDING,
        )
        self._sessions[session.session_id] = session
        return session

    # ── Review ──────────────────────────────────────────────────────────────────

    def label(
        self,
        session_id: str,
        selected_incident_ids: list[str] | tuple[str, ...],
    ) -> GoldLabelSession:
        """Mark a session LABELED with the reviewer's chosen incident IDs.

        ``selected_incident_ids`` may be empty (reviewer confirms no match
        exists).  Returns the updated session.
        Raises ``KeyError`` for unknown session IDs.
        """
        session = self._get(session_id)
        updated = GoldLabelSession(
            session_id=session.session_id,
            query_id=session.query_id,
            query=session.query,
            retrieval_strategy=session.retrieval_strategy,
            candidates=session.candidates,
            selected_incident_ids=tuple(selected_incident_ids),
            status=LabelDecision.LABELED,
        )
        self._sessions[session_id] = updated
        return updated

    def skip(self, session_id: str) -> GoldLabelSession:
        """Mark a session SKIPPED; it remains visible for future review passes.

        Returns the updated session.  Raises ``KeyError`` for unknown IDs.
        """
        session = self._get(session_id)
        updated = GoldLabelSession(
            session_id=session.session_id,
            query_id=session.query_id,
            query=session.query,
            retrieval_strategy=session.retrieval_strategy,
            candidates=session.candidates,
            selected_incident_ids=session.selected_incident_ids,
            status=LabelDecision.SKIPPED,
        )
        self._sessions[session_id] = updated
        return updated

    def _get(self, session_id: str) -> GoldLabelSession:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"No labeling session found with id {session_id!r}")

    # ── Queue inspection ────────────────────────────────────────────────────────

    def sessions(self) -> tuple[GoldLabelSession, ...]:
        return tuple(self._sessions.values())

    def pending_sessions(self) -> tuple[GoldLabelSession, ...]:
        return tuple(s for s in self._sessions.values() if s.status == LabelDecision.PENDING)

    def labeled_sessions(self) -> tuple[GoldLabelSession, ...]:
        return tuple(s for s in self._sessions.values() if s.status == LabelDecision.LABELED)

    def skipped_sessions(self) -> tuple[GoldLabelSession, ...]:
        return tuple(s for s in self._sessions.values() if s.status == LabelDecision.SKIPPED)

    # ── Export ──────────────────────────────────────────────────────────────────

    def export_labeled_queries(self) -> tuple[LabeledGoldQuery, ...]:
        """Export all LABELED sessions as ``LabeledGoldQuery`` objects.

        Only LABELED sessions are exported; SKIPPED sessions remain in the
        queue for future review passes.  Raises ``ValueError`` if there are no
        labeled sessions.

        When a reviewer selected zero incident IDs, the exported ``GoldQuery``
        uses category ``"no-match-expected"`` and an empty ``expected_incidents``
        tuple — consistent with Phase 16B's negative-control convention.
        """
        labeled = self.labeled_sessions()
        if not labeled:
            raise ValueError(
                "No labeled sessions to export. "
                "Label at least one session before exporting."
            )
        results: list[LabeledGoldQuery] = []
        for session in labeled:
            results.append(self._session_to_labeled(session))
        return tuple(results)

    def _session_to_labeled(self, session: GoldLabelSession) -> LabeledGoldQuery:
        selected_ids = session.selected_incident_ids
        if selected_ids:
            expected = tuple(
                ExpectedIncident(
                    source_type="",
                    source_external_id=inc_id,
                    relevance=DEFAULT_RELEVANCE,
                )
                for inc_id in selected_ids
            )
            # Determine category from number of selections.
            category = "lexical-overlap" if len(selected_ids) == 1 else "multi-concept"
        else:
            expected = ()
            category = "no-match-expected"

        # Carry the category and difficulty from the original session's
        # retrieved candidates; fall back to defaults.
        gold_query = GoldQuery(
            id=session.query_id,
            query=session.query,
            category=category,
            difficulty="medium",
            expected_incidents=expected,
        )
        provenance = LabelingProvenance(
            query_id=session.query_id,
            original_query=session.query,
            retrieval_strategy=session.retrieval_strategy,
            selected_incident_ids=selected_ids,
            labeled_at=datetime.now(UTC).isoformat(),
        )
        return LabeledGoldQuery(gold_query=gold_query, provenance=provenance)

    # ── Statistics ──────────────────────────────────────────────────────────────

    def stats(self) -> LabelingStats:
        all_sessions = list(self._sessions.values())
        total = len(all_sessions)
        labeled_list = [s for s in all_sessions if s.status == LabelDecision.LABELED]
        skipped = sum(1 for s in all_sessions if s.status == LabelDecision.SKIPPED)
        pending = sum(1 for s in all_sessions if s.status == LabelDecision.PENDING)

        avg_candidates = (
            sum(len(s.candidates) for s in all_sessions) / total if total else 0.0
        )
        labeled_count = len(labeled_list)
        if labeled_count:
            avg_selected = (
                sum(len(s.selected_incident_ids) for s in labeled_list) / labeled_count
            )
            single = sum(1 for s in labeled_list if len(s.selected_incident_ids) == 1)
            multi = sum(1 for s in labeled_list if len(s.selected_incident_ids) > 1)
            single_pct = single / labeled_count
            multi_pct = multi / labeled_count
        else:
            avg_selected = 0.0
            single_pct = 0.0
            multi_pct = 0.0

        return LabelingStats(
            total_sessions=total,
            labeled=labeled_count,
            skipped=skipped,
            pending=pending,
            avg_candidates_presented=avg_candidates,
            avg_selected_per_labeled=avg_selected,
            single_label_pct=single_pct,
            multi_label_pct=multi_pct,
        )
