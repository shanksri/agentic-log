"""Tests for Phase 21D: Assisted Gold Labeling Framework."""
from __future__ import annotations

import pytest

from app.evaluation.dataset_authoring import CandidateQuery, ReviewDecision
from app.evaluation.gold_labeling import (
    CandidateIncident,
    DenseGoldLabelRetriever,
    GoldLabelRetriever,
    GoldLabelSession,
    GoldLabelingWorkflow,
    HybridGoldLabelRetriever,
    LabelDecision,
    LabeledGoldQuery,
    LabelingProvenance,
    LabelingStats,
    DEFAULT_LIMIT,
    RETRIEVAL_STRATEGY_DENSE,
    RETRIEVAL_STRATEGY_HYBRID,
)
from app.evaluation.gold_dataset import ExpectedIncident, GoldQuery


# ── Test fakes ───────────────────────────────────────────────────────────────────


class FakeIncident:
    """Minimal stub matching the attributes DenseGoldLabelRetriever reads."""
    def __init__(self, uid: str, title: str = "Incident Title") -> None:
        self.id = uid
        self.title = title
        self.source_type = "pagerduty"
        self.source_external_id = f"PD-{uid[:4]}"


class FakeDenseResult:
    def __init__(self, uid: str, score: float = 0.9, title: str = "Incident Title") -> None:
        self.incident = FakeIncident(uid, title)
        self.similarity_score = score


class FakeSearchService:
    """Stub for IncidentSearchService — returns pre-built results."""
    def __init__(self, results: list[FakeDenseResult] | None = None) -> None:
        self._results = results or []
        self.calls: list[dict] = []

    def search(self, query: str, *, limit: int = 10, call_site: str = "") -> list:
        self.calls.append({"query": query, "limit": limit, "call_site": call_site})
        return self._results[:limit]


class FakeHybridResult:
    def __init__(
        self,
        doc_id: str,
        rrf_score: float = 0.5,
        dense_rank: int | None = 1,
        bm25_rank: int | None = 2,
        title: str = "Hybrid Incident",
    ) -> None:
        self.document_id = doc_id
        self.rrf_score = rrf_score
        self.dense_rank = dense_rank
        self.bm25_rank = bm25_rank
        self.dense_result = FakeDenseResult(doc_id, 0.8, title) if dense_rank else None
        self.bm25_result = None


class FakeHybridRetriever:
    """Stub for HybridRetriever."""
    def __init__(self, results: list[FakeHybridResult] | None = None) -> None:
        self._results = results or []
        self.calls: list[dict] = []

    def retrieve(self, query: str, *, limit: int = 10) -> list:
        self.calls.append({"query": query, "limit": limit})
        return self._results[:limit]


def _dense_retriever(results: list | None = None) -> DenseGoldLabelRetriever:
    return DenseGoldLabelRetriever(FakeSearchService(results or _default_dense()))


def _hybrid_retriever(results: list | None = None) -> HybridGoldLabelRetriever:
    return HybridGoldLabelRetriever(FakeHybridRetriever(results or _default_hybrid()))


def _default_dense() -> list[FakeDenseResult]:
    return [
        FakeDenseResult("uuid-001", 0.95, "DB connection pool exhausted"),
        FakeDenseResult("uuid-002", 0.80, "Slow query causing timeouts"),
        FakeDenseResult("uuid-003", 0.65, "Disk IO saturation on primary"),
    ]


def _default_hybrid() -> list[FakeHybridResult]:
    return [
        FakeHybridResult("uuid-001", 0.90, dense_rank=1, bm25_rank=1),
        FakeHybridResult("uuid-002", 0.70, dense_rank=2, bm25_rank=3),
    ]


def _candidate_query(
    query: str = "database connection failure",
    incident_id: str = "INC-001",
) -> CandidateQuery:
    return CandidateQuery(
        id="cq-" + incident_id,
        incident_id=incident_id,
        query=query,
        category="lexical-overlap",
        difficulty="easy",
        rationale="test",
        generation_method="exact_keyword",
        status=ReviewDecision.ACCEPTED,
    )


def _workflow(results: list | None = None) -> GoldLabelingWorkflow:
    return GoldLabelingWorkflow(_dense_retriever(results))


# ── CandidateIncident ────────────────────────────────────────────────────────────


def test_candidate_incident_is_frozen() -> None:
    c = CandidateIncident("id1", "Title", "pd:PD-1", 0.9, 1)
    with pytest.raises(Exception):
        c.rank = 2  # type: ignore[misc]


def test_candidate_incident_issues_empty_id() -> None:
    c = CandidateIncident("", "title", "src", 0.9, 1)
    assert any("incident_id" in i for i in c.issues())


def test_candidate_incident_issues_rank_zero() -> None:
    c = CandidateIncident("id", "title", "src", 0.9, 0)
    assert any("rank" in i for i in c.issues())


def test_candidate_incident_valid() -> None:
    c = CandidateIncident("id", "title", "pd:X", 0.9, 1)
    assert c.is_valid()


# ── DenseGoldLabelRetriever ──────────────────────────────────────────────────────


def test_dense_retriever_strategy_name() -> None:
    assert _dense_retriever().strategy_name == RETRIEVAL_STRATEGY_DENSE


def test_dense_retriever_returns_candidates_in_rank_order() -> None:
    retriever = _dense_retriever()
    candidates = retriever.retrieve_candidates("db failure", limit=3)
    assert len(candidates) == 3
    assert all(isinstance(c, CandidateIncident) for c in candidates)
    assert [c.rank for c in candidates] == [1, 2, 3]


def test_dense_retriever_scores_descend() -> None:
    retriever = _dense_retriever()
    candidates = retriever.retrieve_candidates("db failure", limit=3)
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_dense_retriever_respects_limit() -> None:
    retriever = _dense_retriever()
    candidates = retriever.retrieve_candidates("q", limit=2)
    assert len(candidates) == 2


def test_dense_retriever_populates_title_and_source() -> None:
    retriever = _dense_retriever()
    (top,) = retriever.retrieve_candidates("q", limit=1)
    assert top.title == "DB connection pool exhausted"
    assert top.source.startswith("pagerduty:")


def test_dense_retriever_passes_call_site() -> None:
    svc = FakeSearchService(_default_dense())
    retriever = DenseGoldLabelRetriever(svc)
    retriever.retrieve_candidates("q", limit=1)
    assert svc.calls[0]["call_site"] == "gold_labeling"


def test_dense_retriever_empty_results() -> None:
    retriever = DenseGoldLabelRetriever(FakeSearchService([]))
    assert retriever.retrieve_candidates("q") == ()


# ── HybridGoldLabelRetriever ─────────────────────────────────────────────────────


def test_hybrid_retriever_strategy_name() -> None:
    assert _hybrid_retriever().strategy_name == RETRIEVAL_STRATEGY_HYBRID


def test_hybrid_retriever_returns_candidates_in_rank_order() -> None:
    retriever = _hybrid_retriever()
    candidates = retriever.retrieve_candidates("db failure", limit=2)
    assert len(candidates) == 2
    assert [c.rank for c in candidates] == [1, 2]


def test_hybrid_retriever_populates_rrf_score() -> None:
    retriever = _hybrid_retriever()
    (top,) = retriever.retrieve_candidates("q", limit=1)
    assert top.score == 0.90


def test_hybrid_retriever_populates_title_from_dense_result() -> None:
    results = [FakeHybridResult("id-1", 0.9, dense_rank=1, bm25_rank=1, title="Real Title")]
    retriever = _hybrid_retriever(results)
    (top,) = retriever.retrieve_candidates("q", limit=1)
    assert top.title == "Real Title"


def test_hybrid_retriever_handles_bm25_only_result() -> None:
    bm25_only = FakeHybridResult("bm25-only", 0.4, dense_rank=None, bm25_rank=1)
    retriever = _hybrid_retriever([bm25_only])
    (top,) = retriever.retrieve_candidates("q", limit=1)
    assert top.incident_id == "bm25-only"
    assert top.title == ""


def test_hybrid_retriever_explanation_contains_rrf_info() -> None:
    retriever = _hybrid_retriever()
    (top,) = retriever.retrieve_candidates("q", limit=1)
    assert "rrf_score" in top.explanation
    assert "dense_rank" in top.explanation


# ── GoldLabelRetriever is abstract ───────────────────────────────────────────────


def test_gold_label_retriever_is_abstract() -> None:
    with pytest.raises(TypeError):
        GoldLabelRetriever()  # type: ignore[abstract]


# ── GoldLabelingWorkflow.add_query ───────────────────────────────────────────────


def test_add_query_creates_pending_session() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    assert isinstance(session, GoldLabelSession)
    assert session.status == LabelDecision.PENDING


def test_add_query_stores_query_text() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query(query="api gateway 502"))
    assert session.query == "api gateway 502"


def test_add_query_uses_effective_query_for_edited_candidate() -> None:
    candidate = CandidateQuery(
        id="cq-1", incident_id="INC-1", query="original text",
        category="paraphrase", difficulty="medium", rationale="r",
        generation_method="paraphrase", status=ReviewDecision.EDITED,
        edited_query="refined query text",
    )
    wf = _workflow()
    session = wf.add_query(candidate)
    assert session.query == "refined query text"


def test_add_query_fires_retrieval_with_query_text() -> None:
    svc = FakeSearchService(_default_dense())
    wf = GoldLabelingWorkflow(DenseGoldLabelRetriever(svc))
    wf.add_query(_candidate_query(query="my specific query"))
    assert svc.calls[0]["query"] == "my specific query"


def test_add_query_populates_candidates() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    assert len(session.candidates) == 3
    assert all(isinstance(c, CandidateIncident) for c in session.candidates)


def test_add_query_accumulates_sessions() -> None:
    wf = _workflow()
    wf.add_query(_candidate_query(incident_id="A"))
    wf.add_query(_candidate_query(incident_id="B"))
    assert len(wf.sessions()) == 2


# ── Label ────────────────────────────────────────────────────────────────────────


def test_label_session_marks_labeled() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    updated = wf.label(session.session_id, ["uuid-001"])
    assert updated.status == LabelDecision.LABELED
    assert updated.selected_incident_ids == ("uuid-001",)


def test_label_session_multi_selection() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    updated = wf.label(session.session_id, ["uuid-001", "uuid-002"])
    assert updated.selected_incident_ids == ("uuid-001", "uuid-002")


def test_label_session_zero_selection_is_valid() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    updated = wf.label(session.session_id, [])
    assert updated.status == LabelDecision.LABELED
    assert updated.selected_incident_ids == ()


def test_label_session_does_not_mutate_original() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    original_id = session.session_id
    wf.label(session.session_id, ["uuid-001"])
    # The session object returned by add_query remains frozen
    assert session.status == LabelDecision.PENDING
    # But the workflow's stored session is updated
    assert wf._sessions[original_id].status == LabelDecision.LABELED


def test_label_unknown_session_raises() -> None:
    wf = _workflow()
    with pytest.raises(KeyError, match="nonexistent"):
        wf.label("nonexistent", ["uuid-001"])


# ── Skip ─────────────────────────────────────────────────────────────────────────


def test_skip_session_marks_skipped() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    updated = wf.skip(session.session_id)
    assert updated.status == LabelDecision.SKIPPED


def test_skip_session_remains_visible_in_sessions() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    wf.skip(session.session_id)
    assert len(wf.sessions()) == 1
    assert wf.sessions()[0].status == LabelDecision.SKIPPED


def test_skip_unknown_session_raises() -> None:
    wf = _workflow()
    with pytest.raises(KeyError):
        wf.skip("no-such-id")


# ── Queue filtering ───────────────────────────────────────────────────────────────


def test_pending_sessions_returns_only_pending() -> None:
    wf = _workflow()
    s1 = wf.add_query(_candidate_query(incident_id="A"))
    s2 = wf.add_query(_candidate_query(incident_id="B"))
    wf.label(s1.session_id, ["uuid-001"])
    pending = wf.pending_sessions()
    assert len(pending) == 1
    assert pending[0].session_id == s2.session_id


def test_labeled_sessions_returns_only_labeled() -> None:
    wf = _workflow()
    s1 = wf.add_query(_candidate_query(incident_id="A"))
    s2 = wf.add_query(_candidate_query(incident_id="B"))
    wf.label(s1.session_id, ["uuid-001"])
    wf.skip(s2.session_id)
    assert len(wf.labeled_sessions()) == 1
    assert len(wf.skipped_sessions()) == 1


# ── Export ────────────────────────────────────────────────────────────────────────


def test_export_returns_only_labeled_sessions() -> None:
    wf = _workflow()
    s1 = wf.add_query(_candidate_query(incident_id="A"))
    s2 = wf.add_query(_candidate_query(incident_id="B"))
    s3 = wf.add_query(_candidate_query(incident_id="C"))
    wf.label(s1.session_id, ["uuid-001"])
    wf.skip(s2.session_id)
    # s3 remains PENDING
    exported = wf.export_labeled_queries()
    assert len(exported) == 1
    assert isinstance(exported[0], LabeledGoldQuery)


def test_export_populates_expected_incidents() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    wf.label(session.session_id, ["uuid-001", "uuid-002"])
    (labeled,) = wf.export_labeled_queries()
    gq = labeled.gold_query
    assert isinstance(gq, GoldQuery)
    assert len(gq.expected_incidents) == 2
    incident_ids = {e.source_external_id for e in gq.expected_incidents}
    assert "uuid-001" in incident_ids
    assert "uuid-002" in incident_ids


def test_export_zero_selection_produces_no_match_expected() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    wf.label(session.session_id, [])
    (labeled,) = wf.export_labeled_queries()
    gq = labeled.gold_query
    assert gq.category == "no-match-expected"
    assert gq.expected_incidents == ()


def test_export_single_selection_category() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    wf.label(session.session_id, ["uuid-001"])
    (labeled,) = wf.export_labeled_queries()
    assert labeled.gold_query.category == "lexical-overlap"


def test_export_multi_selection_category() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    wf.label(session.session_id, ["uuid-001", "uuid-002"])
    (labeled,) = wf.export_labeled_queries()
    assert labeled.gold_query.category == "multi-concept"


def test_export_expected_incidents_use_max_relevance() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    wf.label(session.session_id, ["uuid-001"])
    (labeled,) = wf.export_labeled_queries()
    assert labeled.gold_query.expected_incidents[0].relevance == 3


def test_export_provenance_is_populated() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query(query="my query"))
    wf.label(session.session_id, ["uuid-001"])
    (labeled,) = wf.export_labeled_queries()
    prov = labeled.provenance
    assert isinstance(prov, LabelingProvenance)
    assert prov.original_query == "my query"
    assert prov.retrieval_strategy == RETRIEVAL_STRATEGY_DENSE
    assert "uuid-001" in prov.selected_incident_ids


def test_export_no_labeled_sessions_raises() -> None:
    wf = _workflow()
    wf.add_query(_candidate_query())
    with pytest.raises(ValueError, match="No labeled sessions"):
        wf.export_labeled_queries()


def test_export_skipped_sessions_excluded() -> None:
    wf = _workflow()
    s1 = wf.add_query(_candidate_query(incident_id="A"))
    s2 = wf.add_query(_candidate_query(incident_id="B"))
    wf.label(s1.session_id, ["uuid-001"])
    wf.skip(s2.session_id)
    exported = wf.export_labeled_queries()
    assert len(exported) == 1


# ── Statistics ────────────────────────────────────────────────────────────────────


def test_stats_empty_workflow() -> None:
    wf = GoldLabelingWorkflow(_dense_retriever([]))
    s = wf.stats()
    assert s.total_sessions == 0
    assert s.labeled == 0
    assert s.avg_candidates_presented == 0.0
    assert s.avg_selected_per_labeled == 0.0


def test_stats_counts_all_statuses() -> None:
    wf = _workflow()
    s1 = wf.add_query(_candidate_query(incident_id="A"))
    s2 = wf.add_query(_candidate_query(incident_id="B"))
    s3 = wf.add_query(_candidate_query(incident_id="C"))
    wf.label(s1.session_id, ["uuid-001"])
    wf.skip(s2.session_id)
    stats = wf.stats()
    assert stats.total_sessions == 3
    assert stats.labeled == 1
    assert stats.skipped == 1
    assert stats.pending == 1


def test_stats_avg_candidates_presented() -> None:
    wf = GoldLabelingWorkflow(_dense_retriever(_default_dense()))
    wf.add_query(_candidate_query(incident_id="A"))  # 3 candidates
    wf.add_query(_candidate_query(incident_id="B"))  # 3 candidates
    stats = wf.stats()
    assert stats.avg_candidates_presented == pytest.approx(3.0)


def test_stats_avg_selected_per_labeled() -> None:
    wf = _workflow()
    s1 = wf.add_query(_candidate_query(incident_id="A"))
    s2 = wf.add_query(_candidate_query(incident_id="B"))
    wf.label(s1.session_id, ["uuid-001", "uuid-002"])
    wf.label(s2.session_id, ["uuid-003"])
    stats = wf.stats()
    assert stats.avg_selected_per_labeled == pytest.approx(1.5)


def test_stats_single_and_multi_label_pct() -> None:
    wf = _workflow()
    s1 = wf.add_query(_candidate_query(incident_id="A"))
    s2 = wf.add_query(_candidate_query(incident_id="B"))
    s3 = wf.add_query(_candidate_query(incident_id="C"))
    wf.label(s1.session_id, ["uuid-001"])
    wf.label(s2.session_id, ["uuid-001", "uuid-002"])
    wf.label(s3.session_id, [])
    stats = wf.stats()
    assert stats.single_label_pct == pytest.approx(1 / 3)
    assert stats.multi_label_pct == pytest.approx(1 / 3)


# ── Immutability ──────────────────────────────────────────────────────────────────


def test_gold_label_session_is_frozen() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    with pytest.raises(Exception):
        session.status = LabelDecision.LABELED  # type: ignore[misc]


def test_labeled_gold_query_is_frozen() -> None:
    wf = _workflow()
    session = wf.add_query(_candidate_query())
    wf.label(session.session_id, ["uuid-001"])
    (labeled,) = wf.export_labeled_queries()
    with pytest.raises(Exception):
        labeled.gold_query = None  # type: ignore[misc]


def test_labeling_stats_is_frozen() -> None:
    wf = _workflow()
    s = wf.stats()
    with pytest.raises(Exception):
        s.total_sessions = 99  # type: ignore[misc]


# ── Traceability ──────────────────────────────────────────────────────────────────


def test_provenance_preserves_query_id() -> None:
    wf = _workflow()
    cq = _candidate_query()
    session = wf.add_query(cq)
    wf.label(session.session_id, ["uuid-001"])
    (labeled,) = wf.export_labeled_queries()
    assert labeled.provenance.query_id == cq.id


def test_provenance_is_frozen() -> None:
    prov = LabelingProvenance(
        query_id="q1", original_query="q", retrieval_strategy="dense",
        selected_incident_ids=("id1",), labeled_at="2026-01-01",
    )
    with pytest.raises(Exception):
        prov.query_id = "other"  # type: ignore[misc]


# ── Hybrid retriever integration ─────────────────────────────────────────────────


def test_workflow_with_hybrid_retriever() -> None:
    wf = GoldLabelingWorkflow(_hybrid_retriever(), limit=5)
    session = wf.add_query(_candidate_query())
    assert session.retrieval_strategy == RETRIEVAL_STRATEGY_HYBRID
    assert len(session.candidates) == 2  # two results from _default_hybrid()


def test_workflow_records_retrieval_strategy_in_provenance() -> None:
    wf = GoldLabelingWorkflow(_hybrid_retriever())
    session = wf.add_query(_candidate_query())
    wf.label(session.session_id, ["uuid-001"])
    (labeled,) = wf.export_labeled_queries()
    assert labeled.provenance.retrieval_strategy == RETRIEVAL_STRATEGY_HYBRID
