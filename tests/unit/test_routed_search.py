from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.services.hybrid_search import HybridSearchResult
from app.services.routed_search import RoutedSearchConfig, RoutedSearchService, RoutingObservation
from app.services.routing import (
    DefaultRuleBasedRoutingPolicy,
    RoutingDecision,
    RoutingEngine,
    RoutingPolicy,
    RoutingSignals,
    RoutingStrategy,
    extract_routing_signals,
)
from app.services.search import IncidentSearchResult


def _incident(*, title="t", owner=None, repo=None, source=None, state=None, symptoms=()):
    return SimpleNamespace(
        id=uuid.uuid4(), title=title, owner=owner, repo=repo, source=source, state=state,
        severity="high", status="open", resolution_summary="fixed",
        symptoms=[SimpleNamespace(text=s) for s in symptoms],
    )


def _result(incident, distance: float) -> IncidentSearchResult:
    return IncidentSearchResult(incident=incident, distance=distance)


class FakeDb:
    def __init__(self, incidents: dict[uuid.UUID, object] | None = None) -> None:
        self._incidents = incidents or {}
        self.get_calls: list[uuid.UUID] = []

    def get(self, model, incident_id):
        self.get_calls.append(incident_id)
        return self._incidents.get(incident_id)


class FakeDenseService:
    def __init__(self, db, *, responses=None, llm_service=None):
        self.db = db
        self.llm_service = llm_service
        self._responses = responses or {}
        self.retrieve_calls: list[dict] = []

    def retrieve(self, query, *, limit=10, source_type=None, tags=None, owner=None, repo=None,
                 source=None, state=None, expand=False, rerank=False, call_site=None):
        self.retrieve_calls.append({
            "query": query, "limit": limit, "source_type": source_type, "tags": tags,
            "owner": owner, "repo": repo, "source": source, "state": state,
            "expand": expand, "rerank": rerank, "call_site": call_site,
        })
        return self._responses.get(query, [])


class FakeBM25Result:
    def __init__(self, document_id: str, score: float) -> None:
        self.document_id = document_id
        self.score = score


class FakeBM25Retriever:
    def __init__(self, responses: dict[str, list[FakeBM25Result]]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def retrieve(self, query, *, limit=10):
        self.calls.append({"query": query, "limit": limit})
        return self._responses.get(query, [])


def _hybrid_result(document_id: str, score: float, *, dense_result=None) -> HybridSearchResult:
    return HybridSearchResult(
        document_id=document_id, rrf_score=score, dense_rank=None, bm25_rank=None,
        dense_result=dense_result, bm25_result=None,
    )


class FakeHybridRetriever:
    def __init__(self, responses: dict[str, list[HybridSearchResult]]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def retrieve(self, query, *, limit=10):
        self.calls.append({"query": query, "limit": limit})
        return self._responses.get(query, [])


class FakeLLMService:
    def __init__(self, *, expansions=None, rerank_selected_ids=None, rerank_raises=None):
        self._expansions = expansions or {}
        self._rerank_selected_ids = rerank_selected_ids
        self._rerank_raises = rerank_raises
        self.rerank_calls: list[dict] = []

    def expand_search_query(self, query):
        return self._expansions.get(query, [])

    def rerank_incident_search_results(self, *, query, candidates, limit):
        self.rerank_calls.append({"query": query, "candidates": candidates, "limit": limit})
        if self._rerank_raises is not None:
            raise self._rerank_raises
        return self._rerank_selected_ids or []


class _FixedPolicy(RoutingPolicy):
    def __init__(self, strategy: RoutingStrategy, reason: str = "fixed") -> None:
        self._strategy = strategy
        self._reason = reason

    def decide(self, query: str, signals: RoutingSignals) -> RoutingDecision:
        return RoutingDecision(strategy=self._strategy, reason=self._reason, signals=signals)


def _engine(strategy: RoutingStrategy) -> RoutingEngine:
    return RoutingEngine(_FixedPolicy(strategy))


# ── Routing disabled: identical to calling dense directly ───────────────────────


def test_routing_disabled_delegates_to_dense_unchanged() -> None:
    incident = _incident()
    db = FakeDb()
    dense = FakeDenseService(db, responses={"q": [_result(incident, 0.1)]})
    bm25 = FakeBM25Retriever({})
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=False),
    )

    results = service.retrieve("q", limit=5, expand=True, rerank=True, call_site="api")

    assert results == [_result(incident, 0.1)]
    assert dense.retrieve_calls == [{
        "query": "q", "limit": 5, "source_type": None, "tags": None, "owner": None,
        "repo": None, "source": None, "state": None, "expand": True, "rerank": True,
        "call_site": "api",
    }]
    assert bm25.calls == []  # never touched — proves the policy's BM25 decision was overridden


def test_routing_disabled_still_records_observation_for_shadow_monitoring() -> None:
    db = FakeDb()
    dense = FakeDenseService(db, responses={"q": []})
    service = RoutedSearchService(
        dense, routing_engine=_engine(RoutingStrategy.HYBRID),
        config=RoutedSearchConfig(routing_enabled=False),
    )

    service.retrieve("q")

    observation = service.last_observation
    assert observation.routing_enabled is False
    assert observation.policy_strategy == RoutingStrategy.HYBRID  # what the policy WOULD pick
    assert observation.effective_strategy == RoutingStrategy.DENSE  # what was actually used
    assert observation.override_reason is not None
    assert observation.signals is not None


def test_regression_routing_disabled_matches_calling_dense_directly() -> None:
    incident = _incident()
    db = FakeDb()
    dense = FakeDenseService(db, responses={"memory leak": [_result(incident, 0.2)]})
    service = RoutedSearchService(
        dense, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=False),
    )

    direct = dense.retrieve("memory leak", limit=10, call_site="x")
    routed = service.retrieve("memory leak", limit=10, call_site="x")

    assert routed == direct


# ── Routing enabled: Dense ───────────────────────────────────────────────────────


def test_routing_enabled_dense_decision_delegates_to_dense() -> None:
    incident = _incident()
    db = FakeDb()
    dense = FakeDenseService(db, responses={"q": [_result(incident, 0.1)]})
    service = RoutedSearchService(
        dense, routing_engine=_engine(RoutingStrategy.DENSE),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    results = service.retrieve("q")

    assert results == [_result(incident, 0.1)]
    assert service.last_observation.effective_strategy == RoutingStrategy.DENSE
    assert service.last_observation.override_reason is None


# ── Routing enabled: BM25 ─────────────────────────────────────────────────────────


def test_routing_enabled_bm25_fetches_real_incidents() -> None:
    incident_a = _incident(title="bm25 match")
    db = FakeDb({incident_a.id: incident_a})
    dense = FakeDenseService(db, responses={})
    bm25 = FakeBM25Retriever({"q": [FakeBM25Result(str(incident_a.id), 3.5)]})
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    [result] = service.retrieve("q", limit=5)

    assert isinstance(result, IncidentSearchResult)
    assert result.incident is incident_a
    assert result.distance == pytest.approx(-3.5)
    assert dense.retrieve_calls == []  # dense never invoked
    assert service.last_observation.effective_strategy == RoutingStrategy.BM25


def test_bm25_routing_without_configured_retriever_raises() -> None:
    dense = FakeDenseService(FakeDb(), responses={})
    service = RoutedSearchService(
        dense, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )
    with pytest.raises(ValueError, match="BM25"):
        service.retrieve("q")


def test_bm25_missing_incident_in_db_is_skipped_not_crashed() -> None:
    db = FakeDb({})  # incident not found
    dense = FakeDenseService(db, responses={})
    bm25 = FakeBM25Retriever({"q": [FakeBM25Result(str(uuid.uuid4()), 1.0)]})
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    results = service.retrieve("q")

    assert results == []


# ── Routing enabled: Hybrid ────────────────────────────────────────────────────────


def test_routing_enabled_hybrid_uses_dense_result_incident_when_present() -> None:
    incident = _incident(title="hybrid dense-sourced")
    dense_result = _result(incident, 0.05)
    db = FakeDb()
    dense = FakeDenseService(db, responses={})
    hybrid_results = [_hybrid_result(str(incident.id), 0.5, dense_result=dense_result)]
    hybrid = FakeHybridRetriever({"q": hybrid_results})
    service = RoutedSearchService(
        dense, hybrid=hybrid, routing_engine=_engine(RoutingStrategy.HYBRID),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    [result] = service.retrieve("q")

    assert result.incident is incident
    assert result.distance == pytest.approx(-0.5)


def test_routing_enabled_hybrid_fetches_incident_for_bm25_only_candidate() -> None:
    incident = _incident(title="hybrid bm25-sourced")
    db = FakeDb({incident.id: incident})
    dense = FakeDenseService(db, responses={})
    hybrid = FakeHybridRetriever({"q": [_hybrid_result(str(incident.id), 0.3)]})  # no dense_result
    service = RoutedSearchService(
        dense, hybrid=hybrid, routing_engine=_engine(RoutingStrategy.HYBRID),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    [result] = service.retrieve("q")

    assert result.incident is incident
    assert db.get_calls == [incident.id]


def test_hybrid_routing_without_configured_retriever_raises() -> None:
    dense = FakeDenseService(FakeDb(), responses={})
    service = RoutedSearchService(
        dense, routing_engine=_engine(RoutingStrategy.HYBRID),
        config=RoutedSearchConfig(routing_enabled=True),
    )
    with pytest.raises(ValueError, match="Hybrid"):
        service.retrieve("q")


# ── Filters force dense even when routing is enabled ────────────────────────────


@pytest.mark.parametrize(
    "kwargs", [{"owner": "k8s"}, {"repo": "kubernetes"}, {"source_type": "github"},
               {"tags": ["bug"]}, {"source": "github"}, {"state": "open"}]
)
def test_any_filter_forces_dense_despite_bm25_decision(kwargs) -> None:
    incident = _incident()
    db = FakeDb()
    dense = FakeDenseService(db, responses={"q": [_result(incident, 0.1)]})
    bm25 = FakeBM25Retriever({})
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    results = service.retrieve("q", **kwargs)

    assert results == [_result(incident, 0.1)]
    assert bm25.calls == []
    assert service.last_observation.effective_strategy == RoutingStrategy.DENSE
    assert "filters" in service.last_observation.override_reason


# ── Expansion compatibility (BM25/Hybrid) ───────────────────────────────────────


def test_bm25_expansion_merges_candidates_across_phrases_keeping_lowest_distance() -> None:
    shared = _incident(title="shared")
    db = FakeDb({shared.id: shared})
    llm = FakeLLMService(expansions={"q": ["related"]})
    dense = FakeDenseService(db, responses={}, llm_service=llm)
    bm25 = FakeBM25Retriever({
        "q": [FakeBM25Result(str(shared.id), 1.0)],
        "related": [FakeBM25Result(str(shared.id), 9.0)],
    })
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    [result] = service.retrieve("q", limit=5, expand=True)

    assert result.distance == pytest.approx(-9.0)  # higher score -> lower (better) distance kept
    assert bm25.calls == [{"query": "q", "limit": 25}, {"query": "related", "limit": 25}]


def test_hybrid_expansion_without_llm_service_uses_only_original_query() -> None:
    incident = _incident()
    db = FakeDb({incident.id: incident})
    dense = FakeDenseService(db, responses={}, llm_service=None)
    hybrid = FakeHybridRetriever({"q": [_hybrid_result(str(incident.id), 0.4)]})
    service = RoutedSearchService(
        dense, hybrid=hybrid, routing_engine=_engine(RoutingStrategy.HYBRID),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    service.retrieve("q", expand=True)

    assert hybrid.calls == [{"query": "q", "limit": 25}]


# ── Reranking compatibility (BM25/Hybrid) ───────────────────────────────────────


def test_bm25_rerank_reorders_per_llm_and_uses_identical_payload_shape() -> None:
    inc_a = _incident(title="a", symptoms=["s1"])
    inc_b = _incident(title="b")
    db = FakeDb({inc_a.id: inc_a, inc_b.id: inc_b})
    llm = FakeLLMService(rerank_selected_ids=["2", "1"])
    dense = FakeDenseService(db, responses={}, llm_service=llm)
    bm25_results = [FakeBM25Result(str(inc_a.id), 5.0), FakeBM25Result(str(inc_b.id), 1.0)]
    bm25 = FakeBM25Retriever({"q": bm25_results})
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    results = service.retrieve("q", limit=5, rerank=True)

    assert [r.incident.id for r in results] == [inc_b.id, inc_a.id]
    [payload_a, payload_b] = llm.rerank_calls[0]["candidates"]
    assert set(payload_a.keys()) == {
        "candidate_id", "title", "owner", "repo", "source", "state", "symptoms",
        "severity", "status", "resolution_summary", "similarity_score",
    }
    assert payload_a["symptoms"] == ["s1"]


def test_rerank_llm_failure_falls_back_to_distance_order() -> None:
    inc_a = _incident(title="a")
    inc_b = _incident(title="b")
    db = FakeDb({inc_a.id: inc_a, inc_b.id: inc_b})
    llm = FakeLLMService(rerank_raises=RuntimeError("LLM down"))
    dense = FakeDenseService(db, responses={}, llm_service=llm)
    bm25_results = [FakeBM25Result(str(inc_a.id), 1.0), FakeBM25Result(str(inc_b.id), 9.0)]
    bm25 = FakeBM25Retriever({"q": bm25_results})
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    results = service.retrieve("q", limit=5, rerank=True)

    assert [r.incident.id for r in results] == [inc_b.id, inc_a.id]  # distance order preserved


# ── Confidence compatibility ──────────────────────────────────────────────────────


def test_confidence_for_works_identically_regardless_of_strategy() -> None:
    incident = _incident()
    db = FakeDb({incident.id: incident})
    dense = FakeDenseService(db, responses={})
    bm25 = FakeBM25Retriever({"q": [FakeBM25Result(str(incident.id), 1.0)]})
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    results = service.retrieve("q")
    top1_score, confidence_level = RoutedSearchService.confidence_for(results)

    assert isinstance(top1_score, float)
    assert confidence_level in {"LOW", "MEDIUM", "HIGH"}


def test_confidence_for_no_results_is_low() -> None:
    top1_score, confidence_level = RoutedSearchService.confidence_for([])
    assert top1_score is None
    assert confidence_level == "LOW"


# ── Identical downstream interface across strategies ────────────────────────────


def test_all_three_strategies_return_same_result_shape() -> None:
    incident = _incident()
    db = FakeDb({incident.id: incident})
    llm = FakeLLMService()
    dense = FakeDenseService(db, responses={"q": [_result(incident, 0.1)]}, llm_service=llm)
    bm25 = FakeBM25Retriever({"q": [FakeBM25Result(str(incident.id), 1.0)]})
    hybrid = FakeHybridRetriever({"q": [_hybrid_result(str(incident.id), 0.5)]})

    for strategy in (RoutingStrategy.DENSE, RoutingStrategy.BM25, RoutingStrategy.HYBRID):
        service = RoutedSearchService(
            dense, bm25=bm25, hybrid=hybrid, routing_engine=_engine(strategy),
            config=RoutedSearchConfig(routing_enabled=True),
        )
        [result] = service.retrieve("q")
        assert isinstance(result, IncidentSearchResult)
        assert result.incident is incident
        assert isinstance(result.distance, float)
        assert isinstance(result.similarity_score, float)


def test_no_downstream_leak_of_which_policy_was_used() -> None:
    incident = _incident()
    db = FakeDb({incident.id: incident})
    dense = FakeDenseService(db, responses={})
    bm25 = FakeBM25Retriever({"q": [FakeBM25Result(str(incident.id), 1.0)]})
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=_engine(RoutingStrategy.BM25),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    [result] = service.retrieve("q")

    assert not hasattr(result, "strategy")
    assert not hasattr(result, "routing_decision")


# ── Real DefaultRuleBasedRoutingPolicy end-to-end (no stubbed engine) ───────────


def test_end_to_end_short_query_routes_to_bm25_via_real_policy() -> None:
    incident = _incident()
    db = FakeDb({incident.id: incident})
    dense = FakeDenseService(db, responses={})
    bm25 = FakeBM25Retriever({"memory leak": [FakeBM25Result(str(incident.id), 1.0)]})
    real_engine = RoutingEngine(DefaultRuleBasedRoutingPolicy())
    service = RoutedSearchService(
        dense, bm25=bm25, routing_engine=real_engine,
        config=RoutedSearchConfig(routing_enabled=True),
    )

    [result] = service.retrieve("memory leak")

    assert result.incident is incident
    assert service.last_observation.effective_strategy == RoutingStrategy.BM25


def test_end_to_end_medium_query_with_no_signal_routes_to_dense_via_real_policy() -> None:
    incident = _incident()
    db = FakeDb()
    query = "background scheduler refuses to launch after upgrade"
    dense = FakeDenseService(db, responses={query: [_result(incident, 0.1)]})
    real_engine = RoutingEngine(DefaultRuleBasedRoutingPolicy())
    service = RoutedSearchService(
        dense, routing_engine=real_engine, config=RoutedSearchConfig(routing_enabled=True),
    )

    results = service.retrieve(query)

    assert results == [_result(incident, 0.1)]
    assert service.last_observation.effective_strategy == RoutingStrategy.DENSE


def test_signals_match_extract_routing_signals_directly() -> None:
    db = FakeDb()
    dense = FakeDenseService(db, responses={"memory leak": []})
    real_engine = RoutingEngine(DefaultRuleBasedRoutingPolicy())
    service = RoutedSearchService(
        dense, bm25=FakeBM25Retriever({}), routing_engine=real_engine,
        config=RoutedSearchConfig(routing_enabled=True),
    )

    service.retrieve("memory leak")

    assert service.last_observation.signals == extract_routing_signals("memory leak")
