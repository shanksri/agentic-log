"""Test 8 + 9 — sanitization at the persistence boundary.

Verifies _upsert_raw_document sanitizes before persistence and that the stored
payload and payload_hash derive from the SAME sanitized object.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.db.models import RawDocument
from app.services.deduplication import DeduplicationService
from app.services.incident_ingestion import IncidentIngestionService

NUL = chr(0)


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def scalar(self, _stmt):
        return None  # no existing raw_document → insert path

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def commit(self):
        pass


class _FakeEmbedding:
    model_name = "fake"

    def embed_text(self, _):
        return [0.0] * 384


def _service() -> IncidentIngestionService:
    return IncidentIngestionService(
        db=_FakeSession(),  # type: ignore[arg-type]
        embedding_service=_FakeEmbedding(),  # type: ignore[arg-type]
    )


def test_upsert_raw_document_sanitizes_and_hashes_consistently() -> None:
    svc = _service()
    source = SimpleNamespace(id="src-1")
    normalized = SimpleNamespace(
        source_external_id="golang/go#73820",
        source_url="https://github.com/golang/go/issues/73820",
    )
    payload = {"body": "panic" + NUL + "trace", "id": 73820}

    raw_doc = svc._upsert_raw_document(source, payload, normalized)  # type: ignore[arg-type]

    # Test 8: stored payload is sanitized (no NUL → PostgreSQL would accept it)
    assert isinstance(raw_doc, RawDocument)
    assert raw_doc.payload == {"body": "panictrace", "id": 73820}
    assert NUL not in raw_doc.payload["body"]
    # caller payload untouched
    assert payload["body"] == "panic" + NUL + "trace"

    # Test 9: payload_hash matches the hash of the actually-stored payload
    dedup = DeduplicationService()
    assert raw_doc.payload_hash == dedup.payload_hash(raw_doc.payload)


def test_clean_payload_is_unchanged_and_stable() -> None:
    svc = _service()
    source = SimpleNamespace(id="src-1")
    normalized = SimpleNamespace(source_external_id="acme/api#1", source_url=None)
    payload = {"body": "all good\nmultiline", "n": 1}

    raw_doc = svc._upsert_raw_document(source, payload, normalized)  # type: ignore[arg-type]

    assert raw_doc.payload == payload
    assert raw_doc.payload_hash == DeduplicationService().payload_hash(payload)
