"""Phase 23: ``/health`` had no dedicated test coverage before this phase.
Also covers the new ``/health/ready`` readiness probe (Part 6: deployment
readiness).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.db.session import get_db
from app.main import app


def _client(db) -> TestClient:
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


def test_health_is_unconditionally_ok_even_with_no_db_override() -> None:
    """/health must never depend on the database — it's a liveness probe."""
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_ok_when_database_reachable() -> None:
    db = MagicMock()
    db.execute.return_value = None
    client = _client(db)
    try:
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "database": "reachable"}
    finally:
        app.dependency_overrides.clear()


def test_readiness_returns_503_when_database_unreachable() -> None:
    db = MagicMock()
    db.execute.side_effect = OperationalError("SELECT 1", {}, Exception("connection refused"))
    client = _client(db)
    try:
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"
        assert "connection refused" not in resp.text
    finally:
        app.dependency_overrides.clear()
