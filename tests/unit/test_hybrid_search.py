from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.services.bm25_search import BM25Document, BM25Retriever
from app.services.hybrid_search import HybridConfig, HybridRetriever, HybridSearchResult


def _incident_result(document_id: str, distance: float) -> SimpleNamespace:
    incident = SimpleNamespace(id=document_id)
    return SimpleNamespace(incident=incident, distance=distance)


@dataclass
class _Call:
    query: str
    limit: int
    call_site: str | None


class FakeDenseService:
    """Duck-typed stand-in for ``IncidentSearchService``. Only ``.search()``
    is exercised by ``HybridRetriever`` — never ``.retrieve()`` — so this
    fake intentionally implements only ``.search()``.
    """

    def __init__(self, responses: dict[str, list | Exception]) -> None:
        self._responses = responses
        self.calls: list[_Call] = []

    def search(self, query: str, *, limit: int = 10, call_site: str | None = None, **kwargs):
        self.calls.append(_Call(query=query, limit=limit, call_site=call_site))
        outcome = self._responses.get(query, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _bm25(*docs: tuple[str, str]) -> BM25Retriever:
    return BM25Retriever.from_documents([BM25Document(document_id=d, text=t) for d, t in docs])


# ── Pure fusion behavior ─────────────────────────────────────────────────────


def test_retrieve_fuses_disjoint_results_from_both_retrievers() -> None:
    dense = FakeDenseService({"shared term": [_incident_result("a", 0.1)]})
    bm25 = _bm25(("b", "shared term"))
    retriever = HybridRetriever(dense, bm25)

    results = retriever.retrieve("shared term")

    assert {r.document_id for r in results} == {"a", "b"}


def test_document_in_both_retrievers_appears_exactly_once() -> None:
    dense = FakeDenseService({"memory leak": [_incident_result("a", 0.1)]})
    bm25 = _bm25(("a", "memory leak"))
    retriever = HybridRetriever(dense, bm25)

    results = retriever.retrieve("memory leak")

    matching = [r for r in results if r.document_id == "a"]
    assert len(matching) == 1
    assert matching[0].dense_rank == 1
    assert matching[0].bm25_rank == 1


def test_rrf_score_overlap_beats_single_source_hand_verified() -> None:
    # dense: A(rank1), B(rank2)   bm25: B(rank1), C(rank2)   k=60
    dense = FakeDenseService(
        {"alpha": [_incident_result("A", 0.1), _incident_result("B", 0.2)]}
    )
    bm25 = _bm25(("B", "alpha alpha alpha"), ("C", "alpha"))
    retriever = HybridRetriever(dense, bm25, config=HybridConfig(rrf_k=60.0, final_limit=10))

    results = retriever.retrieve("alpha")
    by_id = {r.document_id: r.rrf_score for r in results}

    # Independently hand-computed expected RRF scores (k=60):
    expected_a = 1.0 / (60 + 1)
    expected_b = 1.0 / (60 + 2) + 1.0 / (60 + 1)
    expected_c = 1.0 / (60 + 2)

    assert by_id["A"] == pytest.approx(expected_a)
    assert by_id["B"] == pytest.approx(expected_b)
    assert by_id["C"] == pytest.approx(expected_c)
    assert [r.document_id for r in results] == ["B", "A", "C"]


def test_document_only_in_dense_has_none_bm25_rank_and_result() -> None:
    dense = FakeDenseService({"memory leak": [_incident_result("a", 0.1)]})
    bm25 = _bm25(("z", "unrelated"))
    retriever = HybridRetriever(dense, bm25)

    [result] = [
        r for r in retriever.retrieve("memory leak") if r.document_id == "a"
    ]

    assert result.bm25_rank is None
    assert result.bm25_result is None
    assert result.dense_rank == 1
    assert result.dense_result is not None


def test_document_only_in_bm25_has_none_dense_rank_and_result() -> None:
    dense = FakeDenseService({"memory leak": []})
    bm25 = _bm25(("a", "memory leak"))
    retriever = HybridRetriever(dense, bm25)

    [result] = retriever.retrieve("memory leak")

    assert result.dense_rank is None
    assert result.dense_result is None
    assert result.bm25_rank == 1
    assert result.bm25_result is not None


# ── Deduplication and ordering ───────────────────────────────────────────────


def test_results_sorted_descending_by_rrf_score() -> None:
    dense = FakeDenseService(
        {"q": [_incident_result("a", 0.1), _incident_result("b", 0.2), _incident_result("c", 0.3)]}
    )
    bm25 = _bm25(("z", "unrelated"))
    retriever = HybridRetriever(dense, bm25)

    results = retriever.retrieve("q")

    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ties_broken_deterministically_by_document_id() -> None:
    # "z-doc" reaches rank 1 via dense only; "a-doc" reaches rank 1 via BM25
    # only (its only competitor, "unrelated", does not match the query). Both
    # get an identical single RRF contribution of 1/(k+1) -> a genuine tie.
    dense = FakeDenseService({"shared": [_incident_result("z-doc", 0.1)]})
    bm25 = _bm25(("a-doc", "shared term"), ("unrelated", "nothing"))
    retriever = HybridRetriever(dense, bm25)

    results = retriever.retrieve("shared")

    assert len(results) == 2
    assert results[0].rrf_score == pytest.approx(results[1].rrf_score)
    assert [r.document_id for r in results] == ["a-doc", "z-doc"]


# ── Limit handling ───────────────────────────────────────────────────────────


def test_retrieve_respects_final_limit_from_config() -> None:
    dense = FakeDenseService(
        {"q": [_incident_result(str(i), 0.1) for i in range(5)]}
    )
    bm25 = _bm25(("z", "unrelated"))
    retriever = HybridRetriever(dense, bm25, config=HybridConfig(final_limit=2))

    results = retriever.retrieve("q")

    assert len(results) == 2


def test_retrieve_limit_override_takes_precedence_over_config() -> None:
    dense = FakeDenseService(
        {"q": [_incident_result(str(i), 0.1) for i in range(5)]}
    )
    bm25 = _bm25(("z", "unrelated"))
    retriever = HybridRetriever(dense, bm25, config=HybridConfig(final_limit=2))

    results = retriever.retrieve("q", limit=4)

    assert len(results) == 4


def test_retrieve_rejects_non_positive_limit_override() -> None:
    dense = FakeDenseService({"q": []})
    bm25 = _bm25(("z", "unrelated"))
    retriever = HybridRetriever(dense, bm25)

    with pytest.raises(ValueError):
        retriever.retrieve("q", limit=0)


def test_dense_search_called_with_configured_dense_limit_and_call_site() -> None:
    dense = FakeDenseService({"q": []})
    bm25 = _bm25(("z", "unrelated"))
    retriever = HybridRetriever(dense, bm25, config=HybridConfig(dense_limit=7))

    retriever.retrieve("q")

    assert dense.calls[0].limit == 7
    assert dense.calls[0].call_site == "hybrid_retriever"


# ── Graceful degradation on retriever failure ────────────────────────────────


def test_dense_failure_degrades_to_bm25_only_results() -> None:
    dense = FakeDenseService({"q": RuntimeError("dense backend down")})
    bm25 = _bm25(("a", "memory leak"))
    retriever = HybridRetriever(dense, bm25)

    results = retriever.retrieve("memory leak")

    assert [r.document_id for r in results] == ["a"]
    assert results[0].dense_rank is None


def test_bm25_failure_degrades_to_dense_only_results() -> None:
    dense = FakeDenseService({"q": [_incident_result("a", 0.1)]})

    class FailingBM25Retriever:
        def retrieve(self, query: str, *, limit: int = 10):
            raise RuntimeError("bm25 index not built")

    retriever = HybridRetriever(dense, FailingBM25Retriever())

    results = retriever.retrieve("q")

    assert [r.document_id for r in results] == ["a"]
    assert results[0].bm25_rank is None


def test_both_retrievers_empty_returns_empty_list() -> None:
    dense = FakeDenseService({"q": []})
    bm25 = _bm25(("z", "unrelated"))
    retriever = HybridRetriever(dense, bm25)

    assert retriever.retrieve("nonexistent vocabulary") == []


# ── HybridConfig validation ──────────────────────────────────────────────────


def test_config_rejects_non_positive_dense_limit() -> None:
    with pytest.raises(ValueError):
        HybridConfig(dense_limit=0)


def test_config_rejects_non_positive_bm25_limit() -> None:
    with pytest.raises(ValueError):
        HybridConfig(bm25_limit=0)


def test_config_rejects_non_positive_rrf_k() -> None:
    with pytest.raises(ValueError):
        HybridConfig(rrf_k=0)


def test_config_rejects_non_positive_final_limit() -> None:
    with pytest.raises(ValueError):
        HybridConfig(final_limit=0)


def test_config_defaults_are_sensible() -> None:
    config = HybridConfig()
    assert config.dense_limit == 25
    assert config.bm25_limit == 25
    assert config.rrf_k == 60.0
    assert config.final_limit == 10


# ── Result type ───────────────────────────────────────────────────────────────


def test_hybrid_search_result_is_frozen() -> None:
    result = HybridSearchResult(
        document_id="a", rrf_score=1.0, dense_rank=1, bm25_rank=None,
        dense_result=None, bm25_result=None,
    )
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        result.rrf_score = 2.0  # type: ignore[misc]
