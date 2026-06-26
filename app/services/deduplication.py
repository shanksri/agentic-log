from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol


class _HasSourceIdentity(Protocol):
    source_type: str
    source_external_id: str


class DeduplicationService:
    def payload_hash(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def incident_key(self, incident: _HasSourceIdentity) -> str:
        stable = {
            "source_type": incident.source_type,
            "source_external_id": incident.source_external_id,
        }
        canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
