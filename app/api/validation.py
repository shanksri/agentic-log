"""Phase 23: shared input-validation helpers for the API layer.

These are thin, dependency-free guards used by route handlers to reject
malformed path parameters *before* they reach a repository, database lookup,
or filesystem path join. They introduce no new behavior beyond turning a
would-be silent 404 (or, for filesystem-backed lookups, a potential path
escape) into an explicit ``422 Unprocessable Entity``.

Nothing here changes any existing validated (UUID, dataset, etc.) success
path — a well-formed value always passes through unchanged.
"""

from __future__ import annotations

import re
import uuid

from fastapi import HTTPException

# Identifiers that end up joined onto a filesystem path (run_id,
# experiment_name) or used as dict/lookup keys (session_id) must not contain
# path separators or ".." segments. This allow-list also bounds length,
# which doubles as oversized-payload protection for these fields.
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_ID_LENGTH = 200


def validate_safe_identifier(value: str, *, field_name: str) -> str:
    """Reject empty, oversized, or path-traversal-shaped identifiers.

    Used for any identifier that is joined onto a filesystem path (e.g.
    ``ExperimentRepository``'s ``{base_dir}/history/{run_id}``) so a
    request can never walk outside the intended directory regardless of
    what the underlying repository does with the value.
    """
    if not value or not value.strip():
        raise HTTPException(status_code=422, detail=f"{field_name} must not be empty.")
    if len(value) > _MAX_ID_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be at most {_MAX_ID_LENGTH} characters.",
        )
    if not _SAFE_ID_PATTERN.match(value):
        raise HTTPException(
            status_code=422,
            detail=(
                f"{field_name} may only contain letters, digits, '.', '_', "
                "and '-'."
            ),
        )
    return value


def validate_uuid(value: str, *, field_name: str) -> uuid.UUID:
    """Parse ``value`` as a UUID or raise a 422 naming the offending field."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} is not a valid UUID: {value!r}",
        ) from exc
