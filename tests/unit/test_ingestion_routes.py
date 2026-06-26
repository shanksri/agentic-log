from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app


def _override_db() -> Any:
    # The route never touches the real session in these tests — the service
    # method is stubbed — so any sentinel object is fine.
    yield object()


def _client(monkeypatch, capture: dict[str, Any]) -> TestClient:
    def fake_ingest_jira_project(self: Any, **kwargs: Any) -> dict[str, Any]:
        capture.update(kwargs)
        return {
            "source": f"jira:{kwargs['project_key']}",
            "fetched": 3,
            "inserted": 2,
            "updated": 1,
            "skipped": 0,
        }

    monkeypatch.setattr(
        "app.api.routes.ingestion.IncidentIngestionService.ingest_jira_project",
        fake_ingest_jira_project,
    )
    app.dependency_overrides[get_db] = _override_db
    return TestClient(app)


def test_jira_ingestion_success(monkeypatch) -> None:
    capture: dict[str, Any] = {}
    client = _client(monkeypatch, capture)
    try:
        response = client.post(
            "/ingestion/jira",
            json={
                "base_url": "https://issues.apache.org/jira",
                "project_key": "KAFKA",
                "limit": 50,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "source": "jira:KAFKA",
        "fetched": 3,
        "inserted": 2,
        "updated": 1,
        "skipped": 0,
    }


def test_jira_ingestion_forwards_parameters(monkeypatch) -> None:
    capture: dict[str, Any] = {}
    client = _client(monkeypatch, capture)
    try:
        client.post(
            "/ingestion/jira",
            json={
                "base_url": "https://issues.apache.org/jira",
                "project_key": "AIRFLOW",
                "limit": 120,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert capture == {
        "base_url": "https://issues.apache.org/jira",
        "project_key": "AIRFLOW",
        "limit": 120,
        "force_backfill": False,  # default applied
    }


def test_jira_ingestion_force_backfill_true(monkeypatch) -> None:
    capture: dict[str, Any] = {}
    client = _client(monkeypatch, capture)
    try:
        response = client.post(
            "/ingestion/jira",
            json={
                "base_url": "https://issues.apache.org/jira",
                "project_key": "KAFKA",
                "limit": 50,
                "force_backfill": True,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert capture["force_backfill"] is True


def test_jira_ingestion_rejects_out_of_range_limit(monkeypatch) -> None:
    capture: dict[str, Any] = {}
    client = _client(monkeypatch, capture)
    try:
        response = client.post(
            "/ingestion/jira",
            json={
                "base_url": "https://issues.apache.org/jira",
                "project_key": "KAFKA",
                "limit": 999,  # > 500
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422  # schema validation, service never called
    assert capture == {}
