from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evaluation.gold_dataset import GoldDataset, GoldQuery
from app.evaluation.overlap_analysis import compute_overlap
from app.services.bm25_search import BM25Document, BM25Retriever


def _incident_result(document_id: str) -> SimpleNamespace:
    return SimpleNamespace(incident=SimpleNamespace(id=document_id), distance=0.1)


class FakeDenseService:
    def __init__(self, responses: dict[str, list]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def search(self, query: str, *, limit: int = 10, call_site: str | None = None, **kwargs):
        self.calls.append({"query": query, "limit": limit, "call_site": call_site})
        return self._responses.get(query, [])


def _dataset(*queries: GoldQuery) -> GoldDataset:
    return GoldDataset(
        version="2.1.0", description="d", created_at="2026-06-27T00:00:00Z", queries=queries
    )


def _query(query_id: str, text: str) -> GoldQuery:
    return GoldQuery(id=query_id, query=text, category="no-match-expected", difficulty="hard")


def _bm25(*docs: tuple[str, str]) -> BM25Retriever:
    return BM25Retriever.from_documents([BM25Document(document_id=d, text=t) for d, t in docs])


def test_full_overlap_jaccard_is_one() -> None:
    dense = FakeDenseService({"term": [_incident_result("a")]})
    bm25 = _bm25(("a", "term"))
    dataset = _dataset(_query("q1", "term"))

    report = compute_overlap(dataset, dense, bm25, limit=10)

    [result] = report.per_query
    assert result.jaccard == pytest.approx(1.0)
    assert result.overlap_count == 1
    assert result.dense_only_count == 0
    assert result.bm25_only_count == 0


def test_disjoint_results_jaccard_is_zero() -> None:
    dense = FakeDenseService({"term": [_incident_result("a")]})
    bm25 = _bm25(("b", "term"))
    dataset = _dataset(_query("q1", "term"))

    report = compute_overlap(dataset, dense, bm25, limit=10)

    [result] = report.per_query
    assert result.jaccard == pytest.approx(0.0)
    assert result.overlap_count == 0
    assert result.dense_only_count == 1
    assert result.bm25_only_count == 1


def test_partial_overlap_hand_computed_jaccard() -> None:
    # dense: {a, b}   bm25: {b, c}   overlap={b} union={a,b,c} -> jaccard = 1/3
    dense = FakeDenseService({"term": [_incident_result("a"), _incident_result("b")]})
    bm25 = _bm25(("b", "term term"), ("c", "term"))
    dataset = _dataset(_query("q1", "term"))

    report = compute_overlap(dataset, dense, bm25, limit=10)

    [result] = report.per_query
    assert result.jaccard == pytest.approx(1.0 / 3.0)
    assert result.overlap_count == 1
    assert result.dense_only_count == 1
    assert result.bm25_only_count == 1


def test_both_empty_jaccard_is_zero_not_division_error() -> None:
    dense = FakeDenseService({"q": []})
    bm25 = _bm25(("z", "unrelated"))
    dataset = _dataset(_query("q1", "q"))

    report = compute_overlap(dataset, dense, bm25, limit=10)

    [result] = report.per_query
    assert result.jaccard == 0.0


def test_aggregate_means_computed_across_queries() -> None:
    # Neither "q1" nor "q2" (the query text) tokenizes to anything present in
    # the BM25 corpus ("alpha"/"beta"), so bm25_ids is {} for both queries —
    # every dense candidate is dense-only, hand-computable below.
    dense = FakeDenseService(
        {"q1": [_incident_result("a")], "q2": [_incident_result("x"), _incident_result("y")]}
    )
    bm25 = _bm25(("a", "alpha"), ("x", "beta"))
    dataset = _dataset(_query("q1", "q1"), _query("q2", "q2"))

    report = compute_overlap(dataset, dense, bm25, limit=10)

    assert report.num_queries == 2
    assert report.mean_dense_only_count == pytest.approx((1 + 2) / 2)
    assert report.mean_bm25_only_count == pytest.approx(0.0)
    assert report.mean_overlap_count == pytest.approx(0.0)
    assert report.mean_jaccard == pytest.approx(0.0)


def test_dense_search_called_with_limit_and_not_retrieve() -> None:
    dense = FakeDenseService({"q": []})
    bm25 = _bm25(("z", "unrelated"))
    dataset = _dataset(_query("q1", "q"))

    compute_overlap(dataset, dense, bm25, limit=5)

    assert dense.calls == [{"query": "q", "limit": 5, "call_site": "phase17c_overlap_analysis"}]


def test_empty_dataset_returns_empty_report() -> None:
    dense = FakeDenseService({})
    bm25 = _bm25(("z", "unrelated"))
    dataset = _dataset()

    report = compute_overlap(dataset, dense, bm25, limit=10)

    assert report.num_queries == 0
    assert report.mean_jaccard is None
    assert report.per_query == ()
