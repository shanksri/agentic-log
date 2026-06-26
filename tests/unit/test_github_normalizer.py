from __future__ import annotations

from app.ingestion.normalizers.github_normalizer import GitHubNormalizer


def test_github_normalizer_extracts_incident_fields() -> None:
    payload = {
        "id": 123,
        "number": 42,
        "html_url": "https://github.com/acme/api/issues/42",
        "title": "API timeout after deploy",
        "body": "Requests fail with sqlalchemy timeout after the latest deploy.",
        "state": "closed",
        "created_at": "2026-05-01T10:00:00Z",
        "updated_at": "2026-05-01T11:00:00Z",
        "labels": [{"name": "sev2"}, {"name": "deployment"}, {"name": "component:database"}],
        "repository": {"owner": "acme", "name": "api", "full_name": "acme/api"},
        "comments_payload": [
            {"body": "Investigating connection pool usage."},
            {"body": "Fixed by rolling back the deploy and increasing pool size."},
        ],
    }

    incident = GitHubNormalizer().normalize(payload)

    assert incident.source_type == "github"
    assert incident.source_external_id == "acme/api#42"
    assert incident.status == "resolved"
    assert incident.severity == "high"
    assert incident.incident_type == "deployment"
    assert incident.resolution_summary is not None
    assert incident.is_gold_labeled is True
    assert "database" in incident.affected_components
    assert "API timeout after deploy" in incident.symptoms
    assert "sqlalchemy timeout" in incident.canonical_text


def test_github_normalizer_keeps_open_issue_unlabeled_when_resolution_absent() -> None:
    payload = {
        "id": 456,
        "number": 7,
        "html_url": "https://github.com/acme/web/issues/7",
        "title": "Frontend crashes on startup",
        "body": "Unhandled exception during boot.",
        "state": "open",
        "created_at": "2026-05-01T10:00:00Z",
        "updated_at": "2026-05-01T11:00:00Z",
        "labels": [{"name": "bug"}],
        "repository": {"owner": "acme", "name": "web", "full_name": "acme/web"},
        "comments_payload": [],
    }

    incident = GitHubNormalizer().normalize(payload)

    assert incident.status == "open"
    assert incident.resolution_summary is None
    assert incident.is_gold_labeled is False
