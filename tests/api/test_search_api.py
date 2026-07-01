"""Integration tests proving the adaptive-routing production adoption:
``/search/incidents`` and ``/search/debug`` now execute through
``RoutedSearchService`` (Dense/BM25/Hybrid), not a plain
``IncidentSearchService``.

- routing disabled (default) -> byte-for-byte identical to the pre-adoption
  dense-only response; BM25/Hybrid are never touched.
- routing enabled -> the routing engine's actual decision is executed (the
  selected backend is called, the others are not), and the resulting
  ``RoutingObservation`` reflects it.

Uses the REAL ``RoutedSearchService``/``RoutingEngine``/
``DefaultRuleBasedRoutingPolicy`` classes (not stubs of the routing
decision itself) with fake Dense/BM25/Hybrid backends, exercised through
the actual FastAPI route via ``TestClient`` — no database, no OpenAI.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.services.hybrid_search import HybridSearchResult
from app.services.routed_search import RoutedSearchConfig, RoutedSearchService
from app.services.routing import DefaultRuleBasedRoutingPolicy, RoutingEngine
from app.services.search import IncidentSearchResult


def _incident(*, title: str, incident_id: uuid.UUID | None = None) -> SimpleNamespace:
    """A fake Incident with every field IncidentResponse (from_attributes)
    needs to serialize successfully through the real API response layer.
    """
    return SimpleNamespace(
        id=incident_id or uuid.uuid4(),
        source_type="github",
        source_external_id="123",
        source_url=None,
        owner="apache",
        repo="airflow",
        source="github",
        state="closed",
        title=title,
        description="desc",
        severity="high",
        status="resolved",
        incident_type="bug",
        environment={},
        affected_components=[],
        tags=[],
        canonical_text=title,
        created_at_source=datetime.now(UTC),
        updated_at_source=datetime.now(UTC),
        symptoms=[],
        resolution_summary="fixed",
    )


def _result(title: str, distance: float) -> IncidentSearchResult:
    return IncidentSearchResult(incident=_incident(title=title), distance=distance)


class FakeDense:
    """Stand-in for IncidentSearchService: records every call so tests can
    prove exactly which backend actually executed."""

    def __init__(self) -> None:
        self.db = "fake-db"
        self.llm_service = None
        self.retrieve_calls: list[dict] = []
        self._responses: dict[str, list[IncidentSearchResult]] = {}

    def set_response(self, query: str, results: list[IncidentSearchResult]) -> None:
        self._responses[query] = results

    def retrieve(self, query, *, limit=10, source_type=None, tags=None, owner=None,
                 repo=None, source=None, state=None, expand=False, rerank=False,
                 call_site=None):
        self.retrieve_calls.append({
            "query": query, "limit": limit, "expand": expand, "rerank": rerank,
            "call_site": call_site,
        })
        return self._responses.get(query, [])[:limit]


class FakeBM25:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._responses: dict[str, list] = {}

    def set_response(self, query: str, results: list) -> None:
        self._responses[query] = results

    def retrieve(self, query, *, limit=10):
        self.calls.append({"query": query, "limit": limit})
        return self._responses.get(query, [])


class FakeHybrid:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, query, *, limit=10):
        self.calls.append({"query": query, "limit": limit})
        return []


def _client() -> TestClient:
    app.dependency_overrides[get_db] = lambda: MagicMock()
    return TestClient(app)


def _install_routed_service(monkeypatch, service: RoutedSearchService) -> None:
    import app.api.routes.search as search_mod

    monkeypatch.setattr(search_mod, "build_routed_search_service", lambda db, **_: service)


# ── Routing disabled: identical to pre-adoption dense-only behavior ──────────


def test_search_incidents_routing_disabled_matches_dense_only(monkeypatch) -> None:
    dense = FakeDense()
    incident_id = uuid.uuid4()
    dense.set_response("scheduler timeout", [_result("Scheduler heartbeat missed", 0.1)])
    bm25 = FakeBM25()
    hybrid = FakeHybrid()
    service = RoutedSearchService(
        dense, bm25=bm25, hybrid=hybrid,
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=False),
    )
    _install_routed_service(monkeypatch, service)

    client = _client()
    try:
        resp = client.post("/search/incidents", json={"query": "scheduler timeout"})
        assert resp.status_code == 200
        body = resp.json()

        # Same result the dense backend alone would have produced.
        assert len(body["results"]) == 1
        assert body["results"][0]["incident"]["title"] == "Scheduler heartbeat missed"
        assert body["results"][0]["distance"] == 0.1

        # Dense was called with expand=False, rerank=False -- documented as
        # identical to plain search() -- and BM25/Hybrid were never touched.
        assert len(dense.retrieve_calls) == 1
        assert dense.retrieve_calls[0]["expand"] is False
        assert dense.retrieve_calls[0]["rerank"] is False
        assert bm25.calls == []
        assert hybrid.calls == []

        # Observation still recorded (shadow observability) but confirms the
        # override reason is exactly "routing disabled".
        assert service.last_observation.routing_enabled is False
        assert service.last_observation.effective_strategy.value == "dense"
        assert "routing disabled" in service.last_observation.override_reason
    finally:
        app.dependency_overrides.clear()


def test_search_debug_routing_disabled_matches_dense_only(monkeypatch) -> None:
    dense = FakeDense()
    dense.set_response("scheduler timeout", [_result("Scheduler heartbeat missed", 0.1)])
    bm25 = FakeBM25()
    hybrid = FakeHybrid()
    service = RoutedSearchService(
        dense, bm25=bm25, hybrid=hybrid,
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=False),
    )
    _install_routed_service(monkeypatch, service)

    client = _client()
    try:
        resp = client.post("/search/debug", json={"query": "scheduler timeout"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["title"] == "Scheduler heartbeat missed"

        # search_debug is the expand=True, rerank=True, limit=5 alias.
        assert len(dense.retrieve_calls) == 1
        assert dense.retrieve_calls[0]["expand"] is True
        assert dense.retrieve_calls[0]["rerank"] is True
        assert dense.retrieve_calls[0]["limit"] == 5
        assert bm25.calls == []
        assert hybrid.calls == []
    finally:
        app.dependency_overrides.clear()


def test_search_incidents_filters_force_dense_even_when_routing_enabled(monkeypatch) -> None:
    """A filtered query must never route to BM25/Hybrid (neither supports
    filters) regardless of routing_enabled -- preserves today's filter
    behavior exactly.
    """
    dense = FakeDense()
    dense.set_response("dbx", [_result("Postgres pool exhausted", 0.2)])
    bm25 = FakeBM25()
    hybrid = FakeHybrid()
    service = RoutedSearchService(
        dense, bm25=bm25, hybrid=hybrid,
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=True),
    )
    _install_routed_service(monkeypatch, service)

    client = _client()
    try:
        resp = client.post(
            "/search/incidents", json={"query": "dbx", "owner": "apache"}
        )
        assert resp.status_code == 200
        assert bm25.calls == []
        assert len(dense.retrieve_calls) == 1
        assert service.last_observation.effective_strategy.value == "dense"
        assert "filters" in service.last_observation.override_reason
    finally:
        app.dependency_overrides.clear()


# ── Routing enabled: the selected strategy actually executes ────────────────


def test_search_incidents_routing_enabled_executes_bm25_and_populates_observation(
    monkeypatch,
) -> None:
    dense = FakeDense()
    bm25 = FakeBM25()
    hybrid = FakeHybrid()
    bm25_incident_id = uuid.uuid4()
    bm25.set_response(
        "db timeout",
        [SimpleNamespace(document_id=str(bm25_incident_id), score=4.2)],
    )
    service = RoutedSearchService(
        dense, bm25=bm25, hybrid=hybrid,
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=True),
    )
    # RoutedSearchService fetches the real Incident row for a BM25 hit via
    # its own dense.db.get(...) -- fake it the same way the module does.
    fetched_incident = _incident(title="DB connection pool exhausted", incident_id=bm25_incident_id)
    dense.db = SimpleNamespace(get=lambda model, iid: fetched_incident if iid == bm25_incident_id else None)
    _install_routed_service(monkeypatch, service)

    client = _client()
    try:
        # "db timeout" -> 2 tokens -> DefaultRuleBasedRoutingPolicy's
        # short-query rule (<=3 tokens) selects BM25.
        resp = client.post("/search/incidents", json={"query": "db timeout"})
        assert resp.status_code == 200
        body = resp.json()

        # BM25 was actually invoked; dense's retrieve() (the candidate-
        # generation primitive) was not -- proves the selected strategy, not
        # dense, produced these results.
        assert len(bm25.calls) == 1
        assert bm25.calls[0]["query"] == "db timeout"
        assert dense.retrieve_calls == []
        assert hybrid.calls == []

        assert len(body["results"]) == 1
        assert body["results"][0]["incident"]["title"] == "DB connection pool exhausted"

        # RoutingObservation is populated and reflects the real decision.
        obs = service.last_observation
        assert obs is not None
        assert obs.routing_enabled is True
        assert obs.policy_strategy.value == "bm25"
        assert obs.effective_strategy.value == "bm25"
        assert obs.override_reason is None
        assert obs.signals.token_count == 2
    finally:
        app.dependency_overrides.clear()


def test_search_incidents_routing_enabled_executes_hybrid(monkeypatch) -> None:
    dense = FakeDense()
    bm25 = FakeBM25()
    hybrid = FakeHybrid()
    dense_incident_id = uuid.uuid4()
    dense_result = _result("Long multi concept incident", 0.3)
    hybrid.retrieve = lambda query, *, limit=10: (
        hybrid.calls.append({"query": query, "limit": limit}) or [
            HybridSearchResult(
                document_id=str(dense_result.incident.id), rrf_score=0.02,
                dense_rank=1, bm25_rank=None,
                dense_result=dense_result, bm25_result=None,
            )
        ]
    )
    service = RoutedSearchService(
        dense, bm25=bm25, hybrid=hybrid,
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=True),
    )
    _install_routed_service(monkeypatch, service)

    client = _client()
    try:
        # 12+ tokens -> DefaultRuleBasedRoutingPolicy's long-query rule
        # selects HYBRID.
        long_query = "why does the airflow scheduler heartbeat keep missing during high load database migrations"
        resp = client.post("/search/incidents", json={"query": long_query})
        assert resp.status_code == 200
        body = resp.json()

        assert len(hybrid.calls) == 1
        assert bm25.calls == []
        assert dense.retrieve_calls == []
        assert len(body["results"]) == 1
        assert body["results"][0]["incident"]["title"] == "Long multi concept incident"

        obs = service.last_observation
        assert obs.effective_strategy.value == "hybrid"
        assert obs.policy_strategy.value == "hybrid"
    finally:
        app.dependency_overrides.clear()
