"""Phase 23B: tests for the centralized Bearer API-key authentication
dependency (``app/api/auth.py``).

Unlike every other API test file, this one does NOT override
``require_api_key`` to a no-op — it exercises the real dependency against
real (or deliberately malformed/missing) ``Authorization`` headers. Every
other test file bypasses auth by design (see each file's ``_client()``
docstring) so this is the single place the auth behavior itself is proven.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import get_db
from app.main import app

TEST_KEY = "test-suite-api-key-do-not-use-in-prod"

# One representative, correctly-shaped request per protected router group —
# used to prove the dependency is wired (and executes) on every one of them,
# not just the router exercised by other test files' happy paths.
PROTECTED_ENDPOINTS: list[tuple[str, str, dict | None]] = [
    ("GET", "/incidents", None),
    ("GET", f"/incidents/{uuid.uuid4()}", None),
    ("POST", "/ingestion/github", {"owner": "apache", "repo": "airflow"}),
    ("POST", "/search/incidents", {"query": "database timeout"}),
    ("POST", "/agent/investigate", {"problem": "users cannot log in"}),
    ("POST", "/evaluation/query", {"query": "x"}),
    ("POST", "/evaluation/query/preview", {"query": "x"}),
]


def _client() -> TestClient:
    """Real client — auth is deliberately NOT bypassed here. ``get_db`` is
    still overridden so a request that *passes* auth doesn't need a real
    database to avoid an unrelated 500.
    """
    app.dependency_overrides[get_db] = lambda: MagicMock()
    return TestClient(app, raise_server_exceptions=False)


def _clear() -> None:
    app.dependency_overrides.clear()


# ── Missing Authorization header ─────────────────────────────────────────────


def test_missing_header_returns_401_on_every_protected_router(monkeypatch) -> None:
    """Proves the dependency actually executes for every protected router
    group, not just the one other tests happen to exercise.
    """
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    client = _client()
    try:
        for method, path, body in PROTECTED_ENDPOINTS:
            resp = client.request(method, path, json=body)
            assert resp.status_code == 401, f"{method} {path} did not require auth"
            assert resp.json()["detail"] == "Not authenticated."
    finally:
        _clear()


def test_missing_header_includes_www_authenticate_bearer(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    client = _client()
    try:
        resp = client.get("/incidents")
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"
    finally:
        _clear()


# ── Malformed Authorization header ───────────────────────────────────────────


def test_malformed_header_wrong_scheme_returns_401(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    client = _client()
    try:
        resp = client.get("/incidents", headers={"Authorization": f"Basic {TEST_KEY}"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Not authenticated."
    finally:
        _clear()


def test_malformed_header_bearer_with_no_token_returns_401(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    client = _client()
    try:
        resp = client.get("/incidents", headers={"Authorization": "Bearer"})
        assert resp.status_code == 401
    finally:
        _clear()


def test_malformed_header_garbage_returns_401(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    client = _client()
    try:
        resp = client.get("/incidents", headers={"Authorization": "not-a-valid-header"})
        assert resp.status_code == 401
    finally:
        _clear()


# ── Invalid API key ───────────────────────────────────────────────────────────


def test_invalid_key_returns_401_same_message_as_missing(monkeypatch) -> None:
    """Same status code and same detail string as the missing-header case —
    the response must not reveal that a (wrong) key was even present.
    """
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    client = _client()
    try:
        resp = client.get("/incidents", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Not authenticated."
    finally:
        _clear()


def test_unconfigured_api_key_fails_closed(monkeypatch) -> None:
    """If Settings.api_key is unset, no key can ever match — every
    protected request is rejected, never silently allowed through.
    """
    monkeypatch.setattr(settings, "api_key", None)
    client = _client()
    try:
        resp = client.get("/incidents", headers={"Authorization": "Bearer anything-at-all"})
        assert resp.status_code == 401
    finally:
        _clear()


# ── Valid API key ──────────────────────────────────────────────────────────────


def test_valid_key_reaches_incidents_route(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    db = MagicMock()
    db.scalars.return_value = []
    app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/incidents", headers={"Authorization": f"Bearer {TEST_KEY}"})
        assert resp.status_code == 200
    finally:
        _clear()


def test_valid_key_reaches_ingestion_route(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    monkeypatch.setattr(
        "app.api.routes.ingestion.IncidentIngestionService.ingest_github_repo",
        lambda self, *a, **kw: {
            "source": "github", "fetched": 0, "inserted": 0, "updated": 0, "skipped": 0,
        },
    )
    client = _client()
    try:
        resp = client.post(
            "/ingestion/github",
            json={"owner": "apache", "repo": "airflow"},
            headers={"Authorization": f"Bearer {TEST_KEY}"},
        )
        assert resp.status_code == 200
    finally:
        _clear()


def test_valid_key_reaches_search_route(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    fake_service = MagicMock()
    fake_service.search.return_value = []
    monkeypatch.setattr(
        "app.api.routes.search.build_routed_search_service", lambda db, **kw: fake_service
    )
    client = _client()
    try:
        resp = client.post(
            "/search/incidents",
            json={"query": "database timeout"},
            headers={"Authorization": f"Bearer {TEST_KEY}"},
        )
        assert resp.status_code == 200
    finally:
        _clear()


def test_valid_key_reaches_agent_route(monkeypatch) -> None:
    from tests.api.test_agent_api import _fake_session

    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    fake_orchestrator = MagicMock()
    fake_orchestrator.investigate.return_value = _fake_session()
    monkeypatch.setattr(
        "app.api.routes.agent.MultiAgentInvestigationOrchestrator", lambda db: fake_orchestrator
    )
    client = _client()
    try:
        resp = client.post(
            "/agent/investigate",
            json={"problem": "users cannot log in"},
            headers={"Authorization": f"Bearer {TEST_KEY}"},
        )
        assert resp.status_code == 200
    finally:
        _clear()


def test_valid_key_reaches_evaluation_route(monkeypatch) -> None:
    from tests.api.test_evaluation_api import FakeExperimentRepo, _get_repo

    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    app.dependency_overrides[_get_repo] = lambda: FakeExperimentRepo()
    client = _client()
    try:
        resp = client.get("/evaluation/stats", headers={"Authorization": f"Bearer {TEST_KEY}"})
        assert resp.status_code == 200
    finally:
        _clear()


def test_valid_key_reaches_evaluation_interactive_route(monkeypatch) -> None:
    """A bogus session_id with a VALID key must 404 (business logic ran),
    not 401 (auth blocked it) — proving the dependency let it through.
    """
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    client = _client()
    try:
        resp = client.get(
            "/evaluation/query/does-not-exist",
            headers={"Authorization": f"Bearer {TEST_KEY}"},
        )
        assert resp.status_code == 404
    finally:
        _clear()


# ── Public endpoints remain accessible ───────────────────────────────────────


def test_health_accessible_without_any_header() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/health")
        assert resp.status_code == 200
    finally:
        _clear()


def test_health_ready_accessible_without_any_header() -> None:
    app.dependency_overrides[get_db] = lambda: MagicMock()
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/health/ready")
        assert resp.status_code in (200, 503)  # never 401
    finally:
        _clear()


def test_docs_accessible_without_any_header() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/docs")
        assert resp.status_code == 200
    finally:
        _clear()


def test_redoc_accessible_without_any_header() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/redoc")
        assert resp.status_code == 200
    finally:
        _clear()


def test_openapi_json_accessible_without_any_header() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
    finally:
        _clear()


# ── Swagger / OpenAPI security scheme ────────────────────────────────────────


def test_openapi_registers_http_bearer_security_scheme() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    try:
        schema = client.get("/openapi.json").json()
        schemes = schema["components"]["securitySchemes"]
        assert "HTTPBearer" in schemes
        assert schemes["HTTPBearer"]["type"] == "http"
        assert schemes["HTTPBearer"]["scheme"] == "bearer"
    finally:
        _clear()


def test_openapi_marks_protected_routes_with_security_requirement() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    try:
        schema = client.get("/openapi.json").json()
        templated_paths = [
            "/incidents", "/incidents/{incident_id}", "/ingestion/github",
            "/search/incidents", "/agent/investigate", "/evaluation/query",
            "/evaluation/query/preview",
        ]
        methods = ["get", "get", "post", "post", "post", "post", "post"]
        for path, method in zip(templated_paths, methods):
            operation = schema["paths"][path][method]
            assert operation.get("security") == [{"HTTPBearer": []}], (
                f"{method.upper()} {path} missing the HTTPBearer security requirement"
            )
    finally:
        _clear()


def test_openapi_does_not_mark_health_as_protected() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    try:
        schema = client.get("/openapi.json").json()
        assert not schema["paths"]["/health"]["get"].get("security")
        assert not schema["paths"]["/health/ready"]["get"].get("security")
    finally:
        _clear()
