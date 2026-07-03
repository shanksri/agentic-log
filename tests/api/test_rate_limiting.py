"""Phase 23C: tests for centralized, endpoint-aware rate limiting
(``app/api/rate_limit.py``).

Auth is bypassed here (``require_api_key`` overridden to a no-op, like
every other non-auth-focused test file) so these tests isolate rate-limit
behavior specifically. Real limits (100/min, 20/min, ...) are far too slow
to exercise directly in a test, so every test overrides the relevant
``Settings.rate_limit_*_per_minute`` value to a small number (2-3) via
``monkeypatch`` — the mechanism under test is identical regardless of the
configured number.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import app.api.rate_limit as rate_limit_module
from fastapi.testclient import TestClient

from app.api.auth import require_api_key
from app.core.config import settings
from app.db.session import get_db
from app.main import app


def _client(db: MagicMock | None = None) -> TestClient:
    app.dependency_overrides[get_db] = lambda: (db if db is not None else MagicMock())
    app.dependency_overrides[require_api_key] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


def _clear() -> None:
    app.dependency_overrides.clear()
    rate_limit_module.reset_rate_limits()


def _incidents_db() -> MagicMock:
    db = MagicMock()
    db.scalars.return_value = []
    return db


# ── Below / at / above limit ─────────────────────────────────────────────────


def test_requests_below_limit_all_succeed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 3)
    client = _client(_incidents_db())
    try:
        for _ in range(2):
            resp = client.get("/incidents")
            assert resp.status_code == 200
    finally:
        _clear()


def test_requests_exactly_at_limit_all_succeed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 3)
    client = _client(_incidents_db())
    try:
        for _ in range(3):
            resp = client.get("/incidents")
            assert resp.status_code == 200
    finally:
        _clear()


def test_request_above_limit_returns_429_with_headers_and_message(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 3)
    client = _client(_incidents_db())
    try:
        for _ in range(3):
            assert client.get("/incidents").status_code == 200
        resp = client.get("/incidents")
        assert resp.status_code == 429
        assert "incidents" in resp.json()["detail"]
        assert "3 requests" in resp.json()["detail"]
        assert resp.headers.get("Retry-After") is not None
        assert resp.headers.get("X-RateLimit-Limit") == "3"
        assert resp.headers.get("X-RateLimit-Remaining") == "0"
        assert resp.headers.get("X-RateLimit-Reset") is not None
    finally:
        _clear()


def test_successful_response_carries_rate_limit_headers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 5)
    client = _client(_incidents_db())
    try:
        resp = client.get("/incidents")
        assert resp.status_code == 200
        assert resp.headers.get("X-RateLimit-Limit") == "5"
        assert resp.headers.get("X-RateLimit-Remaining") == "4"
    finally:
        _clear()


# ── Independent limits per endpoint group ────────────────────────────────────


def test_different_endpoint_groups_have_independent_quotas(monkeypatch) -> None:
    """Exhausting /incidents must not affect /search/incidents — different
    groups, different buckets.
    """
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 1)
    monkeypatch.setattr(settings, "rate_limit_search_per_minute", 5)
    fake_search_service = MagicMock()
    fake_search_service.search.return_value = []
    monkeypatch.setattr(
        "app.api.routes.search.build_routed_search_service", lambda db, **kw: fake_search_service
    )
    client = _client(_incidents_db())
    try:
        assert client.get("/incidents").status_code == 200
        assert client.get("/incidents").status_code == 429  # incidents exhausted

        # search is a completely different group — still fresh
        resp = client.post("/search/incidents", json={"query": "database timeout"})
        assert resp.status_code == 200
    finally:
        _clear()


def test_evaluation_query_and_evaluation_full_have_independent_quotas(monkeypatch) -> None:
    """Two different rate-limited routes on the SAME router (evaluation.py)
    must still have independent per-route limits.
    """
    from tests.api.test_evaluation_api import FakeExperimentRepo, _get_repo

    monkeypatch.setattr(settings, "rate_limit_evaluation_query_per_minute", 1)
    monkeypatch.setattr(settings, "rate_limit_evaluation_full_per_minute", 1)
    monkeypatch.setattr(
        "app.api.routes.evaluation._build_search_service", lambda db: MagicMock()
    )
    app.dependency_overrides[_get_repo] = lambda: FakeExperimentRepo()
    client = _client()
    try:
        assert client.post("/evaluation/query", json={"query": "x"}).status_code == 200
        assert client.post("/evaluation/query", json={"query": "x"}).status_code == 429

        # /full is a different group on the same router — still fresh
        resp = client.post("/evaluation/full", json={"persist": False, "judge": "none"})
        assert resp.status_code == 200
    finally:
        _clear()


# ── Independent quotas per API key ───────────────────────────────────────────


def test_different_api_keys_receive_independent_quotas(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 1)
    client = _client(_incidents_db())
    try:
        resp1 = client.get("/incidents", headers={"Authorization": "Bearer key-one"})
        assert resp1.status_code == 200
        resp2 = client.get("/incidents", headers={"Authorization": "Bearer key-one"})
        assert resp2.status_code == 429  # key-one's quota is exhausted

        resp3 = client.get("/incidents", headers={"Authorization": "Bearer key-two"})
        assert resp3.status_code == 200  # key-two has its own, untouched quota
    finally:
        _clear()


def test_requests_without_a_key_fall_back_to_ip_identity(monkeypatch) -> None:
    """Unit-level proof of the IP-fallback branch — see rate_limit.py's
    module docstring for why this isn't reachable over HTTP on the
    currently-wired (always-authenticated) routers.
    """
    from fastapi import Request

    scope = {
        "type": "http", "method": "GET", "path": "/incidents", "headers": [],
        "client": ("203.0.113.5", 12345),
    }
    request = Request(scope)
    identity = rate_limit_module._resolve_identity(request)
    assert identity == "ip:203.0.113.5"


# ── Health remains unlimited ─────────────────────────────────────────────────


def test_health_endpoint_remains_unlimited(monkeypatch) -> None:
    # Configure every group to an absurdly low limit — health must ignore all of it.
    for field in (
        "rate_limit_incidents_per_minute", "rate_limit_search_per_minute",
        "rate_limit_agent_per_minute",
    ):
        monkeypatch.setattr(settings, field, 1)
    client = _client()
    try:
        for _ in range(25):
            resp = client.get("/health")
            assert resp.status_code == 200
            assert "X-RateLimit-Limit" not in resp.headers
    finally:
        _clear()


# ── Reset after time window ───────────────────────────────────────────────────


def test_limit_resets_after_window_expires(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 1)
    fake_time = {"now": 0.0}  # start of a fixed window — see the sibling test's note
    monkeypatch.setattr(rate_limit_module._backend, "_clock", lambda: fake_time["now"])
    client = _client(_incidents_db())
    try:
        assert client.get("/incidents").status_code == 200
        assert client.get("/incidents").status_code == 429

        fake_time["now"] += 61  # advance past the 60s fixed window
        resp = client.get("/incidents")
        assert resp.status_code == 200
    finally:
        _clear()


def test_limit_does_not_reset_before_window_expires(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 1)
    # 0.0 is the start of a fixed window ([0, 60)) — chosen deliberately so
    # "+30" stays inside the same window regardless of wall-clock alignment
    # (an arbitrary epoch-based start could land close to a window boundary
    # and flip this test's outcome depending on real time, as happened
    # during development with a wall-clock-derived starting value).
    fake_time = {"now": 0.0}
    monkeypatch.setattr(rate_limit_module._backend, "_clock", lambda: fake_time["now"])
    client = _client(_incidents_db())
    try:
        assert client.get("/incidents").status_code == 200
        fake_time["now"] += 30  # still inside the same 60s window
        resp = client.get("/incidents")
        assert resp.status_code == 429
    finally:
        _clear()


# ── Global kill switch (configurable through Settings) ───────────────────────


def test_rate_limit_enabled_false_disables_enforcement(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_incidents_per_minute", 1)
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    client = _client(_incidents_db())
    try:
        for _ in range(5):
            resp = client.get("/incidents")
            assert resp.status_code == 200
            assert "X-RateLimit-Limit" not in resp.headers
    finally:
        _clear()


# ── In-memory backend unit tests (the swappable abstraction itself) ─────────


def test_backend_check_reports_correct_remaining_and_reset() -> None:
    backend = rate_limit_module.InMemoryRateLimitBackend(clock=lambda: 120.0)
    d1 = backend.check("group:id", limit=2, window_seconds=60)
    assert d1.allowed is True
    assert d1.remaining == 1
    assert d1.reset_at == 180.0  # window covering [120, 180)

    d2 = backend.check("group:id", limit=2, window_seconds=60)
    assert d2.allowed is True
    assert d2.remaining == 0

    d3 = backend.check("group:id", limit=2, window_seconds=60)
    assert d3.allowed is False
    assert d3.remaining == 0


def test_backend_reset_clears_all_counters() -> None:
    backend = rate_limit_module.InMemoryRateLimitBackend(clock=lambda: 0.0)
    backend.check("group:id", limit=1, window_seconds=60)
    assert backend.check("group:id", limit=1, window_seconds=60).allowed is False
    backend.reset()
    assert backend.check("group:id", limit=1, window_seconds=60).allowed is True
