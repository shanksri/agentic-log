from __future__ import annotations

from dataclasses import replace

from app.ingestion.normalizers.github_normalizer import GitHubNormalizer
from app.services.deduplication import DeduplicationService


def test_payload_hash_is_stable_for_key_order() -> None:
    service = DeduplicationService()

    first = {"b": 2, "a": {"d": 4, "c": 3}}
    second = {"a": {"c": 3, "d": 4}, "b": 2}

    assert service.payload_hash(first) == service.payload_hash(second)


def test_incident_key_uses_source_identity_not_mutable_content() -> None:
    payload = {
        "id": 123,
        "number": 42,
        "html_url": "https://github.com/acme/api/issues/42",
        "title": "Original title",
        "body": "Original body",
        "state": "open",
        "created_at": "2026-05-01T10:00:00Z",
        "updated_at": "2026-05-01T11:00:00Z",
        "labels": [],
        "repository": {"owner": "acme", "name": "api", "full_name": "acme/api"},
        "comments_payload": [],
    }
    incident = GitHubNormalizer().normalize(payload)
    changed_content = replace(incident, title="Updated title", description="Updated body")

    service = DeduplicationService()

    assert service.incident_key(incident) == service.incident_key(changed_content)
