"""Phase 13B: both ad-hoc (Mode A) and stored-source (Mode B) ingestion must
flow through the single _dispatch core, resolving adapters via SourceRegistry.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from app.db.models import IncidentSource
from app.ingestion.adapters import SourceRegistry
from app.services.incident_ingestion import IncidentIngestionService

FIXED_NOW = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)


class _FakeSession:
    def __init__(self, sources: dict[Any, IncidentSource] | None = None) -> None:
        self._sources = sources or {}
        self.added: list[Any] = []
        self.commits = 0

    def get(self, model, ident):
        return self._sources.get(ident) if model is IncidentSource else None

    def scalar(self, _stmt):
        return None  # _get_or_create_source always misses → creates a row

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.commits += 1


class _FakeEmbedding:
    model_name = "fake-model"

    def embed_text(self, _t: str) -> list[float]:
        return [0.0] * 384


def _service(session) -> IncidentIngestionService:
    return IncidentIngestionService(
        db=session,  # type: ignore[arg-type]
        embedding_service=_FakeEmbedding(),  # type: ignore[arg-type]
        now=lambda: FIXED_NOW,
    )


def _spy_registry(monkeypatch) -> list[str]:
    """Record every source_type resolved through SourceRegistry.get."""
    seen: list[str] = []
    real_get = SourceRegistry.get

    def spy(source_type: str):
        seen.append(source_type)
        return real_get(source_type)

    monkeypatch.setattr(SourceRegistry, "get", spy)
    return seen


class _FakeGitHubCollector:
    def __init__(self, token=None):
        pass

    def collect_issues(self, owner, repo, *, state, limit, include_comments, since=None):
        return []  # empty corpus: exercises dispatch without normalization noise

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _FakeJiraCollector:
    def __init__(self, base_url, token=None):
        pass

    def collect_issues(self, project_key, *, limit, since=None, status_filter=None):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


@pytest.fixture(autouse=True)
def _patch_collectors(monkeypatch):
    monkeypatch.setattr(
        "app.ingestion.adapters.github.GitHubCollector", _FakeGitHubCollector
    )
    monkeypatch.setattr(
        "app.ingestion.adapters.jira.JiraCollector", _FakeJiraCollector
    )


# ── Mode A: ad-hoc payload ingestion uses SourceRegistry ─────────────────────

def test_manual_github_uses_source_registry(monkeypatch) -> None:
    seen = _spy_registry(monkeypatch)
    svc = _service(_FakeSession())

    result = svc.ingest_github_repo(
        "hashicorp", "terraform", state="closed", limit=500, include_comments=False
    )

    assert seen == ["github"]  # resolved via registry, not hardcoded adapter
    assert result["source"] == "github:hashicorp/terraform"
    assert result["mode"] == "backfill"


def test_manual_jira_uses_source_registry(monkeypatch) -> None:
    seen = _spy_registry(monkeypatch)
    svc = _service(_FakeSession())

    result = svc.ingest_jira_project("https://issues.apache.org/jira", "KAFKA", limit=500)

    assert seen == ["jira"]
    assert result["source"] == "jira:KAFKA"


def test_manual_ingestion_does_not_require_preexisting_row(monkeypatch) -> None:
    _spy_registry(monkeypatch)
    session = _FakeSession()  # no sources pre-inserted
    svc = _service(session)

    # Caller never inserted an incident_sources row; the call still succeeds.
    result = svc.ingest_github_repo("a", "b", state="all", limit=10, include_comments=True)
    assert result["fetched"] == 0
    # exactly one auto-managed source row was created (get-or-create)
    sources = [o for o in session.added if isinstance(o, IncidentSource)]
    assert len(sources) == 1
    assert sources[0].source_type == "github"


# ── Mode B: stored-source ingestion still works and uses the registry ────────

def test_stored_source_uses_registry_and_db_config(monkeypatch) -> None:
    seen = _spy_registry(monkeypatch)
    sid = uuid.uuid4()
    source = IncidentSource(
        source_type="github", name="GitHub acme/api",
        config={"owner": "acme", "repo": "api"},
    )
    source.id = sid
    svc = _service(_FakeSession({sid: source}))

    result = svc.ingest_source(sid)

    assert seen == ["github"]
    assert result["source"] == "github:GitHub acme/api"


# ── No duplicate logic: every entry point funnels through _dispatch ──────────

def test_all_entry_points_funnel_through_dispatch(monkeypatch) -> None:
    calls: list[str] = []
    real_dispatch = IncidentIngestionService._dispatch

    def spy(self, source, config, *, force_backfill=False):
        calls.append(source.source_type)
        return real_dispatch(self, source, config, force_backfill=force_backfill)

    monkeypatch.setattr(IncidentIngestionService, "_dispatch", spy)

    sid = uuid.uuid4()
    stored = IncidentSource(source_type="github", name="GitHub x/y", config={"owner": "x", "repo": "y"})
    stored.id = sid
    svc = _service(_FakeSession({sid: stored}))

    svc.ingest_github_repo("a", "b", state="all", limit=5, include_comments=True)
    svc.ingest_jira_project("https://j", "KAFKA", limit=5)
    svc.ingest_source(sid)

    # all three modes went through the single core helper
    assert calls == ["github", "jira", "github"]
