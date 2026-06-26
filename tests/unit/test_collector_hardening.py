"""Phase 14A-14C: GitHub collector hardening (diagnostics, low-yield abort,
scan budget, timeout handling) and result-payload surfacing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
import pytest

from app.db.models import IncidentSource
from app.ingestion.collectors.github_collector import (
    MAX_SCANNED_ITEMS,
    CollectionDiagnostics,
    GitHubCollector,
)
from app.services.incident_ingestion import IncidentIngestionService


def _pr() -> dict[str, Any]:
    return {"id": 1, "pull_request": {"url": "x"}, "comments": 0}


def _issue(n: int) -> dict[str, Any]:
    return {"number": n, "title": f"i{n}", "comments": 0}


def _collector(handler: Callable[[httpx.Request], httpx.Response]) -> GitHubCollector:
    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )
    return GitHubCollector(client=client)


def _page(request: httpx.Request) -> int:
    return int(request.url.params.get("page", "1"))


# ── Part 2: low-yield abort (pages>=MAX_PAGES, avg<threshold) ────────────────

def test_low_yield_abort_pr_heavy_repo() -> None:
    # Every page = 100 PRs, 0 issues (apache/kafka style).
    def handler(req):
        return httpx.Response(200, json=[_pr() for _ in range(100)])

    c = _collector(handler)
    issues = c.collect_issues("apache", "kafka", limit=500, include_comments=False)
    d = c.last_diagnostics

    assert issues == []
    assert d.exit_reason == "low_yield_abort"
    assert d.issues_collected == 0
    assert d.prs_filtered == 2000
    assert d.pages_traversed == 20  # capped, did not walk to page 100


def test_low_yield_abort_via_page_rule_small_batches() -> None:
    # 50 items/page, all PRs → at page 20 raw=1000 (<2000) but avg<0.5.
    def handler(req):
        return httpx.Response(200, json=[_pr() for _ in range(50)])

    c = _collector(handler)
    c.collect_issues("apache", "kafka", limit=500, include_comments=False)
    d = c.last_diagnostics

    assert d.exit_reason == "low_yield_abort"
    assert d.pages_traversed == 20
    assert d.raw_items_scanned == 1000  # Part-2 rule, not the scan budget


# ── Part 3: scan-budget abort ────────────────────────────────────────────────

def test_scan_budget_abort() -> None:
    # 60 issues + 40 PRs per page (avg yield 60 > 0.5, so Part-2 never fires),
    # very high limit → only the scan budget can stop it.
    def handler(req):
        p = _page(req)
        base = p * 1000
        items = [_issue(base + i) for i in range(60)] + [_pr() for _ in range(40)]
        return httpx.Response(200, json=items)

    c = _collector(handler)
    issues = c.collect_issues("acme", "big", limit=100000, include_comments=False)
    d = c.last_diagnostics

    assert d.exit_reason == "low_yield_abort"
    assert d.raw_items_scanned >= MAX_SCANNED_ITEMS
    assert d.issues_collected == 1200  # 60 * 20 pages
    assert d.effective_yield == 60.0


# ── Part 4: timeout_partial ──────────────────────────────────────────────────

def test_timeout_partial_preserves_collected_issues() -> None:
    def handler(req):
        if _page(req) >= 4:
            raise httpx.ReadTimeout("simulated", request=req)
        return httpx.Response(200, json=[_issue(_page(req) * 100 + i) for i in range(100)])

    c = _collector(handler)
    issues = c.collect_issues("acme", "api", limit=1000, include_comments=False)
    d = c.last_diagnostics

    assert d.exit_reason == "timeout_partial"
    assert len(issues) == 300  # pages 1-3 preserved
    assert d.pages_traversed == 3


def test_timeout_exception_subclass_also_handled() -> None:
    def handler(req):
        raise httpx.ConnectTimeout("simulated", request=req)  # TimeoutException subclass

    c = _collector(handler)
    issues = c.collect_issues("acme", "api", limit=10, include_comments=False)
    assert c.last_diagnostics.exit_reason == "timeout_partial"
    assert issues == []


# ── Normal exits still correct ───────────────────────────────────────────────

def test_limit_reached() -> None:
    def handler(req):
        return httpx.Response(200, json=[_issue(i) for i in range(100)])

    c = _collector(handler)
    issues = c.collect_issues("acme", "api", limit=10, include_comments=False)
    d = c.last_diagnostics
    assert len(issues) == 10
    assert d.exit_reason == "limit_reached"
    assert d.pages_traversed == 1


def test_empty_batch_exit() -> None:
    def handler(req):
        return httpx.Response(200, json=[] if _page(req) > 1 else [_issue(1)])

    c = _collector(handler)
    issues = c.collect_issues("acme", "api", limit=500, include_comments=False)
    assert len(issues) == 1
    assert c.last_diagnostics.exit_reason == "empty_batch"


# ── Part 5: diagnostics surfaced in the ingestion result payload ─────────────

class _FakeSession:
    def __init__(self, sources):
        self._s = sources
        self.added = []

    def get(self, model, ident):
        return self._s.get(ident) if model is IncidentSource else None

    def scalar(self, _):
        return None

    def add(self, o):
        self.added.append(o)

    def flush(self):
        pass

    def commit(self):
        pass


class _FakeEmbedding:
    model_name = "fake"

    def embed_text(self, _):
        return [0.0] * 384


class _FakeGitHubCollector:
    def __init__(self, token=None):
        self.last_diagnostics = None

    def collect_issues(self, owner, repo, *, state, limit, include_comments, since=None):
        self.last_diagnostics = CollectionDiagnostics(
            pages_traversed=20, raw_items_scanned=2000, issues_collected=0,
            prs_filtered=2000, effective_yield=0.0, exit_reason="low_yield_abort",
        )
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def test_result_payload_includes_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.ingestion.adapters.github.GitHubCollector", _FakeGitHubCollector
    )
    svc = IncidentIngestionService(
        db=_FakeSession({}),  # type: ignore[arg-type]
        embedding_service=_FakeEmbedding(),  # type: ignore[arg-type]
        now=lambda: datetime(2026, 6, 20, tzinfo=timezone.utc),
    )
    result = svc.ingest_github_repo(
        "apache", "kafka", state="closed", limit=500, include_comments=False
    )

    # new diagnostic fields
    assert result["exit_reason"] == "low_yield_abort"
    assert result["pages_traversed"] == 20
    assert result["raw_items_scanned"] == 2000
    assert result["effective_yield"] == 0.0
    # backward-compatible fields preserved
    assert result["source"] == "github:apache/kafka"
    assert result["fetched"] == 0
    assert {"inserted", "updated", "skipped"} <= result.keys()
