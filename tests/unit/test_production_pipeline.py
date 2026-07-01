from __future__ import annotations

import uuid

import pytest

from app.evaluation.production_pipeline import HybridProductionAdapter
from app.services.hybrid_search import HybridSearchResult
from app.services.search import IncidentSearchResult


def _result(document_id: str, score: float, *, dense_result=None) -> HybridSearchResult:
    return HybridSearchResult(
        document_id=document_id,
        rrf_score=score,
        dense_rank=None,
        bm25_rank=None,
        dense_result=dense_result,
        bm25_result=None,
    )


class FakeHybridRetriever:
    def __init__(self, responses: dict[str, list[HybridSearchResult]]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def retrieve(self, query: str, *, limit: int = 10) -> list[HybridSearchResult]:
        self.calls.append({"query": query, "limit": limit})
        return self._responses.get(query, [])


class FakeLLMService:
    def __init__(
        self,
        *,
        expansions: dict[str, list[str]] | None = None,
        rerank_selected_ids: list[str] | None = None,
        rerank_raises: Exception | None = None,
    ) -> None:
        self._expansions = expansions or {}
        self._rerank_selected_ids = rerank_selected_ids
        self._rerank_raises = rerank_raises
        self.rerank_calls: list[dict] = []

    def expand_search_query(self, query: str) -> list[str]:
        return self._expansions.get(query, [])

    def rerank_incident_search_results(self, *, query, candidates, limit):
        self.rerank_calls.append({"query": query, "candidates": candidates, "limit": limit})
        if self._rerank_raises is not None:
            raise self._rerank_raises
        return self._rerank_selected_ids or []


class FakeDb:
    def __init__(self, incidents: dict[uuid.UUID, object] | None = None) -> None:
        self._incidents = incidents or {}
        self.get_calls: list[uuid.UUID] = []

    def get(self, model, incident_id):
        self.get_calls.append(incident_id)
        return self._incidents.get(incident_id)


def _incident(*, title="t", owner="o", repo="r", source="s", state="open", symptoms=()):
    from types import SimpleNamespace

    return SimpleNamespace(
        title=title, owner=owner, repo=repo, source=source, state=state,
        severity="high", status="open", resolution_summary="fixed",
        symptoms=[SimpleNamespace(text=s) for s in symptoms],
    )


# ── No expansion, no rerank: pass-through ────────────────────────────────────


def test_no_expand_no_rerank_calls_hybrid_once_with_limit() -> None:
    id_a = str(uuid.uuid4())
    hybrid = FakeHybridRetriever({"q": [_result(id_a, 0.5)]})
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=None)

    results = adapter.retrieve("q", limit=3)

    assert hybrid.calls == [{"query": "q", "limit": 3}]
    [result] = results
    assert isinstance(result, IncidentSearchResult)
    assert result.incident.id == uuid.UUID(id_a)


def test_results_sorted_by_rrf_score_descending() -> None:
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    hybrid = FakeHybridRetriever({"q": [_result(id_a, 0.1), _result(id_b, 0.9)]})
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=None)

    results = adapter.retrieve("q", limit=10)

    assert [r.incident.id for r in results] == [uuid.UUID(id_b), uuid.UUID(id_a)]


# ── Expansion ──────────────────────────────────────────────────────────────────


def test_expand_uses_candidate_limit_25_and_queries_every_phrase() -> None:
    hybrid = FakeHybridRetriever({"q": [], "related": []})
    llm = FakeLLMService(expansions={"q": ["related"]})
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=llm)

    adapter.retrieve("q", limit=5, expand=True)

    assert hybrid.calls == [{"query": "q", "limit": 25}, {"query": "related", "limit": 25}]


def test_expand_without_llm_service_degrades_to_original_query_only() -> None:
    hybrid = FakeHybridRetriever({"q": []})
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=None)

    adapter.retrieve("q", limit=5, expand=True)

    assert hybrid.calls == [{"query": "q", "limit": 25}]


def test_expand_merges_candidates_keeping_higher_rrf_score() -> None:
    shared_id = str(uuid.uuid4())
    hybrid = FakeHybridRetriever(
        {"q": [_result(shared_id, 0.2)], "related": [_result(shared_id, 0.8)]}
    )
    llm = FakeLLMService(expansions={"q": ["related"]})
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=llm)

    [result] = adapter.retrieve("q", limit=5, expand=True)

    assert result.distance == pytest.approx(-0.8)  # kept the higher score


def test_expand_deduplicates_candidate_appearing_in_both_phrases() -> None:
    shared_id = str(uuid.uuid4())
    hybrid = FakeHybridRetriever(
        {"q": [_result(shared_id, 0.2)], "related": [_result(shared_id, 0.8)]}
    )
    llm = FakeLLMService(expansions={"q": ["related"]})
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=llm)

    results = adapter.retrieve("q", limit=5, expand=True)

    assert len(results) == 1


# ── Reranking ──────────────────────────────────────────────────────────────────


def test_rerank_reorders_per_llm_selected_ids() -> None:
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    hybrid = FakeHybridRetriever({"q": [_result(id_a, 0.9), _result(id_b, 0.1)]})
    # Score order would be [a, b]; LLM picks b first.
    llm = FakeLLMService(rerank_selected_ids=["2", "1"])
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=llm)

    results = adapter.retrieve("q", limit=5, rerank=True)

    assert [r.incident.id for r in results] == [uuid.UUID(id_b), uuid.UUID(id_a)]


def test_rerank_without_llm_service_falls_back_to_score_order() -> None:
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    hybrid = FakeHybridRetriever({"q": [_result(id_a, 0.1), _result(id_b, 0.9)]})
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=None)

    results = adapter.retrieve("q", limit=5, rerank=True)

    assert [r.incident.id for r in results] == [uuid.UUID(id_b), uuid.UUID(id_a)]


def test_rerank_llm_failure_falls_back_to_score_order() -> None:
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    hybrid = FakeHybridRetriever({"q": [_result(id_a, 0.1), _result(id_b, 0.9)]})
    llm = FakeLLMService(rerank_raises=RuntimeError("LLM down"))
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=llm)

    results = adapter.retrieve("q", limit=5, rerank=True)

    assert [r.incident.id for r in results] == [uuid.UUID(id_b), uuid.UUID(id_a)]


def test_rerank_uses_dense_result_incident_for_payload() -> None:
    incident = _incident(title="dense title", symptoms=["s1"])
    from types import SimpleNamespace

    dense_result = SimpleNamespace(incident=incident, distance=0.1)
    id_a = str(uuid.uuid4())
    hybrid = FakeHybridRetriever({"q": [_result(id_a, 0.5, dense_result=dense_result)]})
    llm = FakeLLMService(rerank_selected_ids=["1"])
    adapter = HybridProductionAdapter(FakeDb(), hybrid, llm_service=llm)

    adapter.retrieve("q", limit=5, rerank=True)

    [payload] = llm.rerank_calls[0]["candidates"]
    assert payload["title"] == "dense title"
    assert payload["symptoms"] == ["s1"]


def test_rerank_fetches_incident_from_db_for_bm25_only_candidate() -> None:
    id_a = uuid.uuid4()
    incident = _incident(title="bm25-only title")
    db = FakeDb({id_a: incident})
    hybrid = FakeHybridRetriever({"q": [_result(str(id_a), 0.5)]})  # dense_result=None
    llm = FakeLLMService(rerank_selected_ids=["1"])
    adapter = HybridProductionAdapter(db, hybrid, llm_service=llm)

    adapter.retrieve("q", limit=5, rerank=True)

    [payload] = llm.rerank_calls[0]["candidates"]
    assert payload["title"] == "bm25-only title"
    assert db.get_calls == [id_a]


# ── db passthrough ─────────────────────────────────────────────────────────────


def test_adapter_exposes_db() -> None:
    db = FakeDb()
    adapter = HybridProductionAdapter(db, FakeHybridRetriever({}), llm_service=None)
    assert adapter.db is db
