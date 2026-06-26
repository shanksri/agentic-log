from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.ingestion.adapters import SourceRegistry
from app.ingestion.adapters.jira import JiraAdapter
from app.ingestion.collectors.jira_collector import JiraCollector
from app.ingestion.normalizers.jira_normalizer import JiraNormalizer


# ── fixtures ─────────────────────────────────────────────────────────────────

def _issue(
    key: str = "KAFKA-123",
    *,
    status: str = "Resolved",
    summary: str = "Broker crashes on startup with timeout",
    description: str = "The broker fails with a connection timeout during boot. "
    "This is a long description well over one hundred characters so the "
    "confidence body bonus is triggered for the test.",
    labels: list[str] | None = None,
    comments: list[str] | None = None,
    priority: str | None = None,
    resolution: str | None = None,
    components: list[str] | None = None,
    base_url: str = "https://issues.example.com",
) -> dict[str, Any]:
    return {
        "key": key,
        "_base_url": base_url,
        "fields": {
            "project": {"key": key.split("-")[0]},
            "summary": summary,
            "description": description,
            "status": {"name": status},
            "labels": labels if labels is not None else ["major", "bug"],
            "priority": {"name": priority} if priority else None,
            "resolution": {"name": resolution} if resolution else None,
            "components": [{"name": c} for c in (components or [])],
            "created": "2026-05-01T10:00:00.000+0000",
            "updated": "2026-05-02T11:30:00.000+0000",
            "comment": {
                "comments": [{"body": body} for body in (comments or [])],
            },
        },
    }


# ── JiraNormalizer ───────────────────────────────────────────────────────────

def test_normalizer_extracts_core_fields() -> None:
    issue = _issue(comments=["Fixed by increasing the connection pool size."])
    incident = JiraNormalizer().normalize(issue)

    assert incident.source_type == "jira"
    assert incident.source_external_id == "KAFKA-123"
    assert incident.title == "Broker crashes on startup with timeout"
    assert incident.status == "resolved"
    assert incident.severity == "high"  # "major" label
    assert incident.incident_type == "bug"
    assert incident.source_url == "https://issues.example.com/browse/KAFKA-123"
    assert incident.resolution_summary is not None
    assert incident.is_gold_labeled is True
    assert incident.created_at_source == datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
    assert "Broker crashes on startup with timeout" in incident.canonical_text
    assert "KAFKA |" in incident.canonical_text


def test_normalizer_populates_source_metadata() -> None:
    incident = JiraNormalizer().normalize(
        _issue(
            labels=["minor", "performance"],
            priority="Minor",
            resolution="Fixed",
            components=["Broker", "Connect"],
        )
    )

    assert incident.source_metadata == {
        "project_key": "KAFKA",
        "issue_key": "KAFKA-123",
        "jira_status": "Resolved",
        "priority": "Minor",
        "resolution": "Fixed",
        "components": ["Broker", "Connect"],
        "labels": ["minor", "performance"],
        "incident_type": "performance",
    }
    assert incident.severity == "medium"  # "Minor" priority


# ── Phase 12B: priority → severity ───────────────────────────────────────────

@pytest.mark.parametrize(
    "priority,expected",
    [
        ("Blocker", "critical"),
        ("Critical", "critical"),
        ("Major", "high"),
        ("Minor", "medium"),
        ("Trivial", "low"),
        ("Highest", "critical"),
        ("Lowest", "low"),
    ],
)
def test_priority_maps_to_severity(priority: str, expected: str) -> None:
    incident = JiraNormalizer().normalize(_issue(priority=priority, labels=[]))
    assert incident.severity == expected


def test_priority_takes_precedence_over_labels() -> None:
    # label says "trivial" (low) but priority says "Blocker" (critical)
    incident = JiraNormalizer().normalize(_issue(priority="Blocker", labels=["trivial"]))
    assert incident.severity == "critical"


def test_severity_falls_back_to_labels_when_no_priority() -> None:
    incident = JiraNormalizer().normalize(_issue(priority=None, labels=["major"]))
    assert incident.severity == "high"


def test_unknown_priority_falls_through_to_unknown() -> None:
    incident = JiraNormalizer().normalize(_issue(priority="Whatever", labels=[]))
    assert incident.severity == "unknown"


# ── Phase 12B: structured resolution → gold-label ────────────────────────────

def test_structured_resolution_fixed_is_gold() -> None:
    incident = JiraNormalizer().normalize(
        _issue(status="Closed", resolution="Fixed", comments=[])
    )
    assert incident.resolution_summary == "Fixed"
    assert incident.is_gold_labeled is True


def test_structured_resolution_wont_fix_is_not_gold() -> None:
    incident = JiraNormalizer().normalize(
        _issue(status="Closed", resolution="Won't Fix", comments=[])
    )
    assert incident.resolution_summary == "Won't Fix"  # still recorded
    assert incident.is_gold_labeled is False  # but not gold


def test_structured_resolution_takes_precedence_over_comments() -> None:
    # comment also says "fixed" but the structured field is authoritative
    incident = JiraNormalizer().normalize(
        _issue(status="Resolved", resolution="Done", comments=["fixed in a comment"])
    )
    assert incident.resolution_summary == "Done"
    assert incident.is_gold_labeled is True


def test_resolution_falls_back_to_comments_when_field_missing() -> None:
    incident = JiraNormalizer().normalize(
        _issue(status="Resolved", resolution=None, comments=["Fixed by a rollback."])
    )
    assert incident.resolution_summary == "Fixed by a rollback."
    assert incident.is_gold_labeled is True


def test_unresolved_with_no_resolution_field_is_not_gold() -> None:
    incident = JiraNormalizer().normalize(
        _issue(status="In Progress", resolution=None, comments=["fixed maybe"])
    )
    assert incident.resolution_summary is None
    assert incident.is_gold_labeled is False


# ── Phase 12B: component extraction ──────────────────────────────────────────

def test_components_populate_affected_components() -> None:
    incident = JiraNormalizer().normalize(_issue(components=["Broker", "Connect"]))
    assert incident.affected_components == ["Broker", "Connect"]


def test_components_fall_back_to_project_key_when_empty() -> None:
    incident = JiraNormalizer().normalize(_issue(components=[]))
    assert incident.affected_components == ["KAFKA"]


def test_components_are_deduped_and_sorted() -> None:
    incident = JiraNormalizer().normalize(_issue(components=["Connect", "Broker", "Connect"]))
    assert incident.affected_components == ["Broker", "Connect"]


def test_normalizer_open_issue_is_not_gold_labeled() -> None:
    incident = JiraNormalizer().normalize(
        _issue(status="In Progress", comments=["Fixed it"])
    )
    assert incident.status == "open"
    assert incident.resolution_summary is None  # resolution only for resolved
    assert incident.is_gold_labeled is False


def test_normalizer_strips_jira_wiki_markup() -> None:
    issue = _issue(
        description="h2. Summary\n{code:java}stacktrace{code}\nA [link|http://x] failed "
        "with *bold* timeout error well over the twenty character excerpt threshold.",
    )
    incident = JiraNormalizer().normalize(issue)
    assert "{code" not in incident.canonical_text
    assert "h2." not in incident.canonical_text
    assert "|http" not in incident.canonical_text


# ── JiraAdapter ──────────────────────────────────────────────────────────────

def test_adapter_normalize_delegates_to_normalizer() -> None:
    incident = JiraAdapter().normalize(_issue())
    assert incident.source_type == "jira"
    assert incident.source_metadata["issue_key"] == "KAFKA-123"


def test_adapter_is_registered() -> None:
    assert "jira" in SourceRegistry.registered_types()
    assert isinstance(SourceRegistry.get("jira"), JiraAdapter)


# ── JiraCollector (httpx MockTransport, no network) ──────────────────────────

def _mock_client(pages: list[dict[str, Any]]) -> tuple[httpx.Client, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        start_at = int(request.url.params.get("startAt", "0"))
        # Serve the page whose startAt matches; default to empty.
        for page in pages:
            if page["startAt"] == start_at:
                return httpx.Response(200, json=page)
        return httpx.Response(200, json={"startAt": start_at, "total": 0, "issues": []})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://issues.example.com",
    )
    return client, requests


def test_collector_paginates_until_total_reached() -> None:
    pages = [
        {"startAt": 0, "total": 3, "issues": [_issue("KAFKA-1"), _issue("KAFKA-2")]},
        {"startAt": 2, "total": 3, "issues": [_issue("KAFKA-3")]},
    ]
    client, requests = _mock_client(pages)
    collector = JiraCollector("https://issues.example.com", client=client)

    issues = collector.collect_issues("KAFKA", limit=100)

    assert [i["key"] for i in issues] == ["KAFKA-1", "KAFKA-2", "KAFKA-3"]
    assert len(requests) == 2
    # base_url injected for the normalizer
    assert all(i["_base_url"] == "https://issues.example.com" for i in issues)


def test_collector_respects_limit() -> None:
    pages = [
        {"startAt": 0, "total": 10, "issues": [_issue(f"KAFKA-{n}") for n in range(100)]},
    ]
    client, _ = _mock_client(pages)
    collector = JiraCollector("https://issues.example.com", client=client)

    issues = collector.collect_issues("KAFKA", limit=2)
    assert len(issues) == 2


def test_collector_backfill_has_no_since_clause() -> None:
    client, requests = _mock_client([{"startAt": 0, "total": 0, "issues": []}])
    collector = JiraCollector("https://issues.example.com", client=client)

    collector.collect_issues("KAFKA", limit=50)

    jql = requests[0].url.params["jql"]
    assert "project = KAFKA" in jql
    assert "updated >=" not in jql
    assert "ORDER BY updated ASC" in jql


def test_collector_incremental_adds_since_clause() -> None:
    client, requests = _mock_client([{"startAt": 0, "total": 0, "issues": []}])
    collector = JiraCollector("https://issues.example.com", client=client)

    since = datetime(2026, 6, 19, 8, 30, tzinfo=timezone.utc)
    collector.collect_issues("KAFKA", limit=50, since=since)

    jql = requests[0].url.params["jql"]
    assert 'updated >= "2026-06-19 08:30"' in jql


def test_collector_status_filter_in_jql() -> None:
    client, requests = _mock_client([{"startAt": 0, "total": 0, "issues": []}])
    collector = JiraCollector("https://issues.example.com", client=client)

    collector.collect_issues("KAFKA", limit=50, status_filter=["Resolved", "Closed"])

    jql = requests[0].url.params["jql"]
    assert 'status in ("Resolved", "Closed")' in jql


# ── Ingestion service integration (no network, fake DB) ──────────────────────

class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0
        self.added: list[Any] = []

    def scalar(self, _stmt: Any) -> None:
        return None

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.commits += 1


class _FakeEmbedding:
    model_name = "fake-model"

    def embed_text(self, _text: str) -> list[float]:
        return [0.0] * 384


class _FakeJiraCollector:
    """Stands in for JiraCollector; records since, returns canned issues."""

    last_since: datetime | None = None

    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url

    def collect_issues(self, project_key: str, *, limit: int, since: datetime | None = None,
                       status_filter: Any = None) -> list[dict[str, Any]]:
        _FakeJiraCollector.last_since = since
        return [_issue("KAFKA-1"), _issue("KAFKA-2")]

    def __enter__(self) -> "_FakeJiraCollector":
        return self

    def __exit__(self, *args: object) -> None:
        pass


def test_ingest_jira_project_end_to_end(monkeypatch) -> None:
    from app.services.incident_ingestion import IncidentIngestionService

    monkeypatch.setattr(
        "app.ingestion.adapters.jira.JiraCollector", _FakeJiraCollector
    )
    fixed_now = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
    service = IncidentIngestionService(
        db=_FakeSession(),  # type: ignore[arg-type]
        embedding_service=_FakeEmbedding(),  # type: ignore[arg-type]
        now=lambda: fixed_now,
    )

    result = service.ingest_jira_project(
        "https://issues.example.com", "KAFKA", limit=50
    )

    assert result["source"] == "jira:KAFKA"
    assert result["mode"] == "backfill"
    assert result["fetched"] == 2
    assert result["inserted"] == 2
    assert result["new_watermark"] == fixed_now.isoformat()
    assert _FakeJiraCollector.last_since is None  # backfill → no since


def test_jira_incident_populates_source_and_keeps_github_columns_null(monkeypatch) -> None:
    from app.db.models import Incident
    from app.services.incident_ingestion import IncidentIngestionService

    monkeypatch.setattr(
        "app.ingestion.adapters.jira.JiraCollector", _FakeJiraCollector
    )
    session = _FakeSession()
    service = IncidentIngestionService(
        db=session,  # type: ignore[arg-type]
        embedding_service=_FakeEmbedding(),  # type: ignore[arg-type]
        now=lambda: datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc),
    )

    service.ingest_jira_project("https://issues.example.com", "KAFKA", limit=50)

    incidents = [obj for obj in session.added if isinstance(obj, Incident)]
    assert incidents, "expected at least one Incident to be added"
    for incident in incidents:
        assert incident.source_type == "jira"
        assert incident.source == "jira"  # generic source now populated
        # GitHub-only legacy columns stay unset for non-GitHub sources
        assert incident.owner is None
        assert incident.repo is None
        assert incident.state is None
