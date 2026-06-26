from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from app.db.models import Incident, IncidentSource
from app.services.incident_ingestion import IncidentIngestionService

FIXED_NOW = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)


def _gh_payload(num: int) -> dict[str, Any]:
    return {
        "id": num,
        "number": num,
        "html_url": f"https://github.com/acme/api/issues/{num}",
        "title": f"Issue {num}",
        "body": "Something failed with a timeout.",
        "state": "closed",
        "created_at": "2026-05-01T10:00:00Z",
        "updated_at": "2026-05-01T11:00:00Z",
        "labels": [],
        "repository": {"owner": "acme", "name": "api", "full_name": "acme/api"},
        "comments_payload": [],
    }


class _FakeGitHubCollector:
    last: dict[str, Any] = {}

    def __init__(self, token: str | None = None) -> None:
        _FakeGitHubCollector.last["token"] = token

    def collect_issues(self, owner, repo, *, state, limit, include_comments, since=None):
        _FakeGitHubCollector.last.update(
            owner=owner, repo=repo, state=state, limit=limit,
            include_comments=include_comments, since=since,
        )
        return [_gh_payload(1), _gh_payload(2)]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _FakeEmbedding:
    model_name = "fake-model"

    def embed_text(self, _t: str) -> list[float]:
        return [0.0] * 384


class _FakeSession:
    def __init__(self, sources: dict[Any, IncidentSource]) -> None:
        self._sources = sources
        self.added: list[Any] = []
        self.commits = 0

    def get(self, model, ident):
        if model is IncidentSource:
            return self._sources.get(ident)
        return None

    def scalar(self, _stmt):
        return None

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.commits += 1


def _service(session, monkeypatch, *, token="env-token"):
    monkeypatch.setattr(
        "app.ingestion.adapters.github.GitHubCollector", _FakeGitHubCollector
    )
    monkeypatch.setattr(
        "app.services.incident_ingestion.settings.github_token", token, raising=False
    )
    return IncidentIngestionService(
        db=session,  # type: ignore[arg-type]
        embedding_service=_FakeEmbedding(),  # type: ignore[arg-type]
        now=lambda: FIXED_NOW,
    )


def test_ingest_source_onboards_github_repo_from_row_only(monkeypatch) -> None:
    sid = uuid.uuid4()
    source = IncidentSource(
        source_type="github",
        name="GitHub acme/api",
        config={"owner": "acme", "repo": "api", "state": "all", "limit": 50,
                "include_comments": False},
    )
    source.id = sid
    svc = _service(_FakeSession({sid: source}), monkeypatch)

    result = svc.ingest_source(sid)

    # adapter resolved via registry + config forwarded from the DB row
    assert _FakeGitHubCollector.last["owner"] == "acme"
    assert _FakeGitHubCollector.last["repo"] == "api"
    assert _FakeGitHubCollector.last["limit"] == 50
    assert _FakeGitHubCollector.last["include_comments"] is False
    # token inherited from environment since the row omitted it
    assert _FakeGitHubCollector.last["token"] == "env-token"
    # watermark-aware result
    assert result["source"] == "github:GitHub acme/api"
    assert result["mode"] == "backfill"
    assert result["fetched"] == 2
    assert result["inserted"] == 2
    assert source.last_ingested_at == FIXED_NOW


def test_ingest_source_passes_since_on_incremental(monkeypatch) -> None:
    sid = uuid.uuid4()
    previous = datetime(2026, 6, 19, 8, 30, tzinfo=timezone.utc)
    source = IncidentSource(
        source_type="github", name="GitHub acme/api",
        config={"owner": "acme", "repo": "api"},
    )
    source.id = sid
    source.last_ingested_at = previous
    svc = _service(_FakeSession({sid: source}), monkeypatch)

    result = svc.ingest_source(sid)

    assert result["mode"] == "incremental"
    assert _FakeGitHubCollector.last["since"] == previous


def test_ingest_source_explicit_token_in_config_wins(monkeypatch) -> None:
    sid = uuid.uuid4()
    source = IncidentSource(
        source_type="github", name="GitHub acme/api",
        config={"owner": "acme", "repo": "api", "token": "row-token"},
    )
    source.id = sid
    svc = _service(_FakeSession({sid: source}), monkeypatch)

    svc.ingest_source(sid)
    assert _FakeGitHubCollector.last["token"] == "row-token"


def test_ingest_source_unknown_id_raises(monkeypatch) -> None:
    svc = _service(_FakeSession({}), monkeypatch)
    with pytest.raises(ValueError, match="No incident_source"):
        svc.ingest_source(uuid.uuid4())


def test_ingest_source_unregistered_source_type_raises(monkeypatch) -> None:
    sid = uuid.uuid4()
    source = IncidentSource(source_type="mystery", name="Mystery", config={})
    source.id = sid
    svc = _service(_FakeSession({sid: source}), monkeypatch)
    with pytest.raises(KeyError, match="No adapter registered"):
        svc.ingest_source(sid)


def test_ingest_source_incidents_get_generic_source(monkeypatch) -> None:
    sid = uuid.uuid4()
    source = IncidentSource(
        source_type="github", name="GitHub acme/api",
        config={"owner": "acme", "repo": "api"},
    )
    source.id = sid
    session = _FakeSession({sid: source})
    svc = _service(session, monkeypatch)

    svc.ingest_source(sid)

    incidents = [o for o in session.added if isinstance(o, Incident)]
    assert incidents
    for inc in incidents:
        assert inc.source_type == "github"
        assert inc.source == "github"
