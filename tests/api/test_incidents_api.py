"""Phase 23 functional validation: ``/incidents`` had no dedicated test
coverage before this phase. No real database — ``db`` is a MagicMock whose
``.get``/``.scalars`` are configured per test.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.auth import require_api_key
from app.db.session import get_db
from app.main import app


def _incident(*, incident_id: uuid.UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=incident_id or uuid.uuid4(),
        source_type="github",
        source_external_id="123",
        source_url=None,
        owner="apache",
        repo="airflow",
        source="github",
        state="closed",
        title="pool exhausted",
        description="desc",
        severity="high",
        status="resolved",
        incident_type="bug",
        environment={},
        affected_components=[],
        tags=[],
        canonical_text="pool exhausted",
        created_at_source=datetime.now(UTC),
        updated_at_source=datetime.now(UTC),
    )


def _client(db: MagicMock) -> TestClient:
    """Phase 23B: also bypasses ``require_api_key`` — see
    tests/api/test_authentication.py for the real auth behavior.
    """
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[require_api_key] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


# ── GET /incidents ────────────────────────────────────────────────────────────


def test_list_incidents_default_limit() -> None:
    db = MagicMock()
    db.scalars.return_value = [_incident()]
    client = _client(db)
    try:
        resp = client.get("/incidents")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
    finally:
        app.dependency_overrides.clear()


def test_list_incidents_zero_limit_rejected() -> None:
    """Empty-input equivalent: limit=0 is not a valid page size."""
    db = MagicMock()
    client = _client(db)
    try:
        resp = client.get("/incidents", params={"limit": 0})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_list_incidents_negative_limit_rejected() -> None:
    db = MagicMock()
    client = _client(db)
    try:
        resp = client.get("/incidents", params={"limit": -5})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_list_incidents_oversized_limit_rejected() -> None:
    db = MagicMock()
    client = _client(db)
    try:
        resp = client.get("/incidents", params={"limit": 100_000})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_list_incidents_malformed_limit_rejected() -> None:
    db = MagicMock()
    client = _client(db)
    try:
        resp = client.get("/incidents", params={"limit": "not-a-number"})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


# ── GET /incidents/{incident_id} ──────────────────────────────────────────────


def test_get_incident_valid_uuid_found() -> None:
    incident_id = uuid.uuid4()
    db = MagicMock()
    db.get.return_value = _incident(incident_id=incident_id)
    client = _client(db)
    try:
        resp = client.get(f"/incidents/{incident_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == str(incident_id)
    finally:
        app.dependency_overrides.clear()


def test_get_incident_valid_uuid_missing_returns_404() -> None:
    db = MagicMock()
    db.get.return_value = None
    client = _client(db)
    try:
        resp = client.get(f"/incidents/{uuid.uuid4()}")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_incident_malformed_uuid_returns_422_not_404() -> None:
    """Before Phase 23, a non-UUID incident_id silently reached db.get() and
    came back as a plain 404. It should now be rejected as malformed input.
    """
    db = MagicMock()
    client = _client(db)
    try:
        resp = client.get("/incidents/not-a-uuid")
        assert resp.status_code == 422
        db.get.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_get_incident_empty_id_returns_404_route_not_matched() -> None:
    """GET /incidents/ (empty path segment) doesn't match {incident_id} at
    all — it resolves to the collection route instead. Documented here so
    the "empty input" case is explicit rather than assumed.
    """
    db = MagicMock()
    db.scalars.return_value = []
    client = _client(db)
    try:
        resp = client.get("/incidents/")
        assert resp.status_code in (200, 404, 307)
    finally:
        app.dependency_overrides.clear()


def test_get_incident_unicode_garbage_uuid_returns_422() -> None:
    db = MagicMock()
    client = _client(db)
    try:
        resp = client.get("/incidents/" + "🔥" * 10)
        assert resp.status_code == 422
        db.get.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_get_incident_oversized_id_returns_422_not_crash() -> None:
    db = MagicMock()
    client = _client(db)
    try:
        resp = client.get("/incidents/" + "a" * 5000)
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_get_incident_sql_injection_shaped_id_returns_422_never_reaches_db() -> None:
    """Confirms the UUID guard, not the ORM, is what stops this — db.get is
    never called with attacker-controlled SQL-shaped text.
    """
    db = MagicMock()
    client = _client(db)
    try:
        resp = client.get("/incidents/" + "'; DROP TABLE incidents;--")
        assert resp.status_code == 422
        db.get.assert_not_called()
    finally:
        app.dependency_overrides.clear()
