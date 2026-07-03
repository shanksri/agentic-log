"""API tests for Phase 21H: Human-Friendly Interactive Evaluation API.

All evaluation components, retrieval, and DB are mocked — no OpenAI,
no database, no live retrieval.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.auth import require_api_key
from app.db.session import get_db
from app.main import app
from app.api.routes.evaluation_interactive import (
    PreviewSession,
    _SearchHit,
    _get_session_store,
    SESSION_TTL_SECONDS,
)


# ── Shared fixtures / fakes ───────────────────────────────────────────────────


def _make_incident(iid: uuid.UUID, title: str) -> MagicMock:
    inc = MagicMock()
    inc.id = iid
    inc.title = title
    inc.repo = "owner/repo"
    inc.source = "github"
    inc.source_type = "github"
    return inc


def _make_search_result(iid: uuid.UUID, title: str, score: float = 0.9) -> MagicMock:
    r = MagicMock()
    r.incident = _make_incident(iid, title)
    r.similarity_score = score
    return r


def _fake_db():
    yield object()


def _fresh_store() -> dict:
    return {}


def _client(store: dict | None = None) -> tuple[TestClient, dict]:
    """Return (client, session_store) with all real dependencies overridden.

    Phase 23B: also bypasses ``require_api_key`` — see
    tests/api/test_authentication.py for the real auth behavior.
    """
    s = store if store is not None else {}
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[_get_session_store] = lambda: s
    app.dependency_overrides[require_api_key] = lambda: None
    return TestClient(app, raise_server_exceptions=False), s


# ── POST /evaluation/query/preview ───────────────────────────────────────────


def test_preview_returns_session_id(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "OOM crash")],
            ),
        )
        resp = client.post("/evaluation/query/preview", json={"query": "memory leak", "k": 5})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "session_id" in data
        assert data["query"] == "memory leak"
        assert len(data["retrieved"]) == 1
    finally:
        app.dependency_overrides.clear()


def test_preview_stores_session(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "CPU spike")],
            ),
        )
        resp = client.post("/evaluation/query/preview", json={"query": "cpu", "k": 10})
        session_id = resp.json()["session_id"]
        assert session_id in store
        assert store[session_id].query == "cpu"
    finally:
        app.dependency_overrides.clear()


def test_preview_result_has_human_readable_fields(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [
                    _make_search_result(iid, "Kafka rebalance", score=0.85)
                ],
            ),
        )
        resp = client.post("/evaluation/query/preview", json={"query": "kafka lag", "k": 10})
        data = resp.json()
        hit = data["retrieved"][0]
        assert hit["incident_id"] == str(iid)
        assert hit["title"] == "Kafka rebalance"
        assert hit["similarity_score"] == pytest.approx(0.85)
        assert hit["rank"] == 1
        assert "repo" in hit
        assert "source" in hit
    finally:
        app.dependency_overrides.clear()


def test_preview_respects_k(monkeypatch) -> None:
    results = [_make_search_result(uuid.uuid4(), f"T{i}", 0.9 - i * 0.05) for i in range(3)]
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: results[:limit]),
        )
        resp = client.post("/evaluation/query/preview", json={"query": "q", "k": 2})
        data = resp.json()
        assert len(data["retrieved"]) == 2
        assert data["k"] == 2
    finally:
        app.dependency_overrides.clear()


def test_preview_includes_expires_at(monkeypatch) -> None:
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: []),
        )
        resp = client.post("/evaluation/query/preview", json={"query": "q", "k": 5})
        assert "expires_at" in resp.json()
    finally:
        app.dependency_overrides.clear()


def test_preview_retrieval_failure_returns_500(monkeypatch) -> None:
    client, _ = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: (_ for _ in ()).throw(
                RuntimeError("embeddings unavailable")
            )),
        )
        resp = client.post("/evaluation/query/preview", json={"query": "q", "k": 5})
        assert resp.status_code == 500
    finally:
        app.dependency_overrides.clear()


def test_preview_search_service_unavailable_returns_503(monkeypatch) -> None:
    from fastapi import HTTPException
    client, _ = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: (_ for _ in ()).throw(HTTPException(status_code=503, detail="no svc")),
        )
        resp = client.post("/evaluation/query/preview", json={"query": "q", "k": 5})
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_preview_multiple_results_ranked(monkeypatch) -> None:
    ids = [uuid.uuid4() for _ in range(3)]
    results = [_make_search_result(ids[i], f"T{i}", 0.95 - i * 0.1) for i in range(3)]
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: results),
        )
        resp = client.post("/evaluation/query/preview", json={"query": "q", "k": 10})
        data = resp.json()
        ranks = [h["rank"] for h in data["retrieved"]]
        assert ranks == [1, 2, 3]
    finally:
        app.dependency_overrides.clear()


# ── GET /evaluation/query/{session_id} ───────────────────────────────────────


def test_get_session_returns_data(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "DB crash")],
            ),
        )
        preview_resp = client.post("/evaluation/query/preview", json={"query": "db", "k": 5})
        sid = preview_resp.json()["session_id"]

        resp = client.get(f"/evaluation/query/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert data["query"] == "db"
        assert data["status"] == "pending"
        assert len(data["retrieved"]) == 1
    finally:
        app.dependency_overrides.clear()


def test_get_session_not_found() -> None:
    client, _ = _client()
    try:
        resp = client.get("/evaluation/query/no-such-session")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_session_shows_evaluated_status(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "OOM")],
            ),
        )
        # Step 1: preview
        prev = client.post("/evaluation/query/preview", json={"query": "oom", "k": 5})
        sid = prev.json()["session_id"]

        # Step 3: evaluate
        client.post(
            f"/evaluation/query/{sid}/evaluate",
            json={"selected_incident_ids": [str(iid)]},
        )

        # Status should now be "evaluated"
        resp = client.get(f"/evaluation/query/{sid}")
        assert resp.json()["status"] == "evaluated"
    finally:
        app.dependency_overrides.clear()


# ── POST /evaluation/query/{session_id}/evaluate ─────────────────────────────


def test_evaluate_session_computes_recall(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "X")],
            ),
        )
        prev = client.post("/evaluation/query/preview", json={"query": "q", "k": 10})
        sid = prev.json()["session_id"]

        resp = client.post(
            f"/evaluation/query/{sid}/evaluate",
            json={"selected_incident_ids": [str(iid)]},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["recall_at_k"] == pytest.approx(1.0)
        assert data["rank_of_first_expected"] == 1
    finally:
        app.dependency_overrides.clear()


def test_evaluate_session_does_not_rerun_retrieval(monkeypatch) -> None:
    iid = uuid.uuid4()
    call_count = {"n": 0}

    def fake_search(q, limit, call_site):
        call_count["n"] += 1
        return [_make_search_result(iid, "Y")]

    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=fake_search),
        )
        prev = client.post("/evaluation/query/preview", json={"query": "q", "k": 10})
        sid = prev.json()["session_id"]
        assert call_count["n"] == 1

        client.post(
            f"/evaluation/query/{sid}/evaluate",
            json={"selected_incident_ids": [str(iid)]},
        )
        # Retrieval must NOT have been called again
        assert call_count["n"] == 1
    finally:
        app.dependency_overrides.clear()


def test_evaluate_session_not_found() -> None:
    client, _ = _client()
    try:
        resp = client.post(
            "/evaluation/query/no-such-session/evaluate",
            json={"selected_incident_ids": []},
        )
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_evaluate_session_invalid_uuid_returns_422(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: [_make_search_result(iid, "Z")]),
        )
        prev = client.post("/evaluation/query/preview", json={"query": "q", "k": 5})
        sid = prev.json()["session_id"]
        resp = client.post(
            f"/evaluation/query/{sid}/evaluate",
            json={"selected_incident_ids": ["not-a-uuid"]},
        )
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_evaluate_session_empty_selection_no_match(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: [_make_search_result(iid, "A")]),
        )
        prev = client.post("/evaluation/query/preview", json={"query": "q", "k": 10})
        sid = prev.json()["session_id"]
        resp = client.post(
            f"/evaluation/query/{sid}/evaluate",
            json={"selected_incident_ids": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rank_of_first_expected"] is None
    finally:
        app.dependency_overrides.clear()


def test_evaluate_session_multiple_expected(monkeypatch) -> None:
    ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
    results = [_make_search_result(ids[i], f"T{i}", 0.9 - i * 0.1) for i in range(3)]
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: results),
        )
        prev = client.post("/evaluation/query/preview", json={"query": "q", "k": 10})
        sid = prev.json()["session_id"]
        resp = client.post(
            f"/evaluation/query/{sid}/evaluate",
            json={"selected_incident_ids": [str(ids[0]), str(ids[2])]},
        )
        data = resp.json()
        assert data["recall_at_k"] == pytest.approx(1.0)
        # Mark correct incidents in retrieved list
        is_expected = {h["incident_id"]: h["is_expected"] for h in data["retrieved"]}
        assert is_expected[str(ids[0])] is True
        assert is_expected[str(ids[1])] is False
        assert is_expected[str(ids[2])] is True
    finally:
        app.dependency_overrides.clear()


def test_evaluate_session_partial_recall(monkeypatch) -> None:
    ids = [uuid.uuid4(), uuid.uuid4()]
    # Only first incident is retrieved; second is expected but not found
    results = [_make_search_result(ids[0], "Found")]
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: results),
        )
        prev = client.post("/evaluation/query/preview", json={"query": "q", "k": 10})
        sid = prev.json()["session_id"]
        resp = client.post(
            f"/evaluation/query/{sid}/evaluate",
            json={"selected_incident_ids": [str(ids[0]), str(ids[1])]},
        )
        data = resp.json()
        # 1 of 2 expected found → recall = 0.5
        assert data["recall_at_k"] == pytest.approx(0.5)
    finally:
        app.dependency_overrides.clear()


def test_evaluate_session_mrr_rank_two(monkeypatch) -> None:
    target = uuid.uuid4()
    other = uuid.uuid4()
    results = [_make_search_result(other, "Wrong"), _make_search_result(target, "Right")]
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: results),
        )
        prev = client.post("/evaluation/query/preview", json={"query": "q", "k": 10})
        sid = prev.json()["session_id"]
        resp = client.post(
            f"/evaluation/query/{sid}/evaluate",
            json={"selected_incident_ids": [str(target)]},
        )
        data = resp.json()
        assert data["rank_of_first_expected"] == 2
        assert data["reciprocal_rank"] == pytest.approx(0.5)
    finally:
        app.dependency_overrides.clear()


def test_evaluate_session_marks_status_evaluated(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, store = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: [_make_search_result(iid, "T")]),
        )
        prev = client.post("/evaluation/query/preview", json={"query": "q", "k": 5})
        sid = prev.json()["session_id"]
        client.post(f"/evaluation/query/{sid}/evaluate", json={"selected_incident_ids": []})
        assert store[sid].status == "evaluated"
    finally:
        app.dependency_overrides.clear()


# ── Session expiry ────────────────────────────────────────────────────────────


def test_expired_session_returns_404() -> None:
    store: dict = {}
    past = datetime.now(UTC) - timedelta(seconds=10)
    store["expired-sid"] = PreviewSession(
        session_id="expired-sid",
        query="q",
        k=10,
        hits=[],
        created_at=(past - timedelta(hours=1)).isoformat(),
        expires_at=past.isoformat(),
    )
    client, _ = _client(store)
    try:
        resp = client.get("/evaluation/query/expired-sid")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_expired_session_pruned_on_access() -> None:
    store: dict = {}
    past = datetime.now(UTC) - timedelta(seconds=10)
    store["old"] = PreviewSession(
        session_id="old",
        query="q",
        k=5,
        hits=[],
        created_at=(past - timedelta(hours=1)).isoformat(),
        expires_at=past.isoformat(),
    )
    client, _ = _client(store)
    try:
        client.get("/evaluation/query/old")
        assert "old" not in store
    finally:
        app.dependency_overrides.clear()


def test_valid_session_not_pruned() -> None:
    store: dict = {}
    future = datetime.now(UTC) + timedelta(hours=1)
    store["live"] = PreviewSession(
        session_id="live",
        query="q",
        k=5,
        hits=[],
        created_at=datetime.now(UTC).isoformat(),
        expires_at=future.isoformat(),
    )
    client, _ = _client(store)
    try:
        # Access session — it should survive
        resp = client.get("/evaluation/query/live")
        assert resp.status_code == 200
        assert "live" in store
    finally:
        app.dependency_overrides.clear()


def test_session_ttl_is_30_minutes() -> None:
    assert SESSION_TTL_SECONDS == 1800


# ── POST /evaluation/query/by-title ──────────────────────────────────────────


def _fake_db_with_incidents(incidents: list[tuple[uuid.UUID, str]]):
    """Return a dependency override that yields a fake DB session."""
    class FakeQuery:
        def __init__(self, rows):
            self._rows = rows
            self._filtered = rows

        def filter(self, _cond):
            return self

        def all(self):
            return [MagicMock(id=iid, title=title) for iid, title in self._filtered]

    class FakeDB:
        def query(self, *cols):
            return FakeQuery(incidents)

    def override():
        yield FakeDB()

    return override


def test_by_title_resolves_and_scores(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, _ = _client()
    try:
        app.dependency_overrides[get_db] = _fake_db_with_incidents(
            [(iid, "Kafka consumer lag")]
        )
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "Kafka consumer lag")],
            ),
        )
        resp = client.post(
            "/evaluation/query/by-title",
            json={
                "query": "kafka lag",
                "expected_titles": ["Kafka consumer lag"],
                "k": 10,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["recall_at_k"] == pytest.approx(1.0)
        assert data["retrieved"][0]["is_expected"] is True
    finally:
        app.dependency_overrides.clear()


def test_by_title_case_insensitive(monkeypatch) -> None:
    iid = uuid.uuid4()
    client, _ = _client()
    try:
        # DB stores "Kafka Consumer Lag" but caller supplies lowercase
        app.dependency_overrides[get_db] = _fake_db_with_incidents(
            [(iid, "Kafka Consumer Lag")]
        )
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "Kafka Consumer Lag")],
            ),
        )
        resp = client.post(
            "/evaluation/query/by-title",
            json={
                "query": "kafka",
                "expected_titles": ["kafka consumer lag"],
                "k": 5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["recall_at_k"] == pytest.approx(1.0)
    finally:
        app.dependency_overrides.clear()


def test_by_title_unmatched_title_ignored(monkeypatch) -> None:
    client, _ = _client()
    iid = uuid.uuid4()
    try:
        app.dependency_overrides[get_db] = _fake_db_with_incidents(
            [(iid, "Real incident")]
        )
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "Real incident")],
            ),
        )
        resp = client.post(
            "/evaluation/query/by-title",
            json={
                "query": "q",
                "expected_titles": ["does not exist anywhere"],
                "k": 10,
            },
        )
        assert resp.status_code == 200
        # No expected incidents resolved → no-match-expected, recall undefined
        data = resp.json()
        assert data["rank_of_first_expected"] is None
    finally:
        app.dependency_overrides.clear()


def test_by_title_multiple_titles(monkeypatch) -> None:
    ids = [uuid.uuid4(), uuid.uuid4()]
    client, _ = _client()
    try:
        app.dependency_overrides[get_db] = _fake_db_with_incidents(
            [(ids[0], "T1"), (ids[1], "T2")]
        )
        results = [_make_search_result(ids[0], "T1"), _make_search_result(ids[1], "T2")]
        monkeypatch.setattr(
            "app.api.routes.evaluation_interactive._build_search_service",
            lambda db: MagicMock(search=lambda q, limit, call_site: results),
        )
        resp = client.post(
            "/evaluation/query/by-title",
            json={"query": "q", "expected_titles": ["T1", "T2"], "k": 10},
        )
        data = resp.json()
        assert data["recall_at_k"] == pytest.approx(1.0)
    finally:
        app.dependency_overrides.clear()


def test_by_title_empty_expected_titles_rejected() -> None:
    client, _ = _client()
    try:
        resp = client.post(
            "/evaluation/query/by-title",
            json={"query": "q", "expected_titles": [], "k": 10},
        )
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


# ── Backward compatibility: Phase 21G endpoints unaffected ───────────────────


def test_phase21g_query_endpoint_still_works(monkeypatch) -> None:
    """POST /evaluation/query must continue to function without changes."""
    import uuid as _uuid
    iid = _uuid.uuid4()

    from app.api.routes.evaluation import _build_search_service as _bss
    client, _ = _client()
    try:
        monkeypatch.setattr(
            "app.api.routes.evaluation._build_search_service",
            lambda db: MagicMock(
                search=lambda q, limit, call_site: [_make_search_result(iid, "Legacy")],
            ),
        )
        resp = client.post(
            "/evaluation/query",
            json={"query": "q", "expected_incident_ids": [str(iid)], "k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["recall_at_k"] == pytest.approx(1.0)
    finally:
        app.dependency_overrides.clear()


# ── Route registration ────────────────────────────────────────────────────────


def test_interactive_routes_in_openapi() -> None:
    client, _ = _client()
    try:
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/evaluation/query/preview" in paths
        assert "/evaluation/query/by-title" in paths
        assert "/evaluation/query/{session_id}/evaluate" in paths
        assert "/evaluation/query/{session_id}" in paths
    finally:
        app.dependency_overrides.clear()


def test_all_interactive_routes_tagged_evaluation() -> None:
    client, _ = _client()
    try:
        spec = client.get("/openapi.json").json()
        for path in [
            "/evaluation/query/preview",
            "/evaluation/query/by-title",
            "/evaluation/query/{session_id}/evaluate",
            "/evaluation/query/{session_id}",
        ]:
            for _method, op in spec["paths"].get(path, {}).items():
                assert "evaluation" in op.get("tags", []), (
                    f"{path} not tagged 'evaluation'"
                )
    finally:
        app.dependency_overrides.clear()
