"""Phase 23B: centralized Bearer API-key authentication.

One dependency (`require_api_key`), wired onto every business router's
``APIRouter(dependencies=[...])`` — never repeated inside individual route
bodies. This is intentionally NOT a user-management system: no accounts,
passwords, sessions, JWTs, OAuth, refresh tokens, roles, or a login
endpoint. The platform is meant to run as an internal service (behind an
API gateway or inside a trusted network), so a single shared secret
compared on every request is the right amount of mechanism — see
``docs/README.md``'s Phase 23B note for the fuller rationale.

# Why HTTPBearer, not APIKeyHeader

``Authorization: Bearer <API_KEY>`` (rather than a custom header) is the
scheme the spec calls for, and it is what lets Swagger UI's stock
"Authorize" button work with zero extra configuration: FastAPI recognizes
``fastapi.security.HTTPBearer`` as a ``SecurityBase`` and auto-registers a
matching entry in the OpenAPI ``components.securitySchemes`` — the Swagger
page then shows one lock icon per protected route and a single "Authorize"
dialog; a user pastes the key once and every subsequent "Try it out" call
on a protected route carries the header automatically. This works even
though ``HTTPBearer`` is used as a *sub*-dependency of ``require_api_key``
(not directly as a route parameter) — FastAPI's dependant-tree walker
detects ``SecurityBase`` instances at any nesting depth, which is exactly
the pattern FastAPI's own docs use for this same reason.

# Why auto_error=False

``HTTPBearer``'s default (``auto_error=True``) raises ``403 Forbidden`` for
a missing/malformed header — not ``401 Unauthorized``. The spec requires
401 for every failure mode (missing header, malformed header, invalid
key), so ``auto_error`` is disabled and this module raises 401 itself,
uniformly, in every case — see ``_unauthorized()``.

# Why one generic error message

"Do not leak which part failed": a caller gets the same status code and
the same detail string whether the header was missing, malformed, or
carried a wrong key. Distinguishing those cases in the response would
hand an attacker a free oracle for guessing towards a valid key.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

_UNAUTHORIZED_DETAIL = "Not authenticated."
_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}

# auto_error=False: see module docstring. bearerFormat/description are
# purely cosmetic — they control what Swagger's "Authorize" dialog shows.
_bearer_scheme = HTTPBearer(
    bearerFormat="API key",
    description="Paste the platform API key (no 'Bearer ' prefix needed).",
    auto_error=False,
)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=_UNAUTHORIZED_DETAIL,
        headers=_UNAUTHORIZED_HEADERS,
    )


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency: raises 401 unless the request carries
    ``Authorization: Bearer <API_KEY>`` with a key matching
    ``Settings.api_key``.

    ``credentials`` is ``None`` for every malformed case ``HTTPBearer``
    itself can detect (missing header, non-Bearer scheme, empty token) —
    see the module docstring's "Why auto_error=False". A missing/unset
    ``settings.api_key`` fails every request closed rather than open: an
    unconfigured key can never match, by design (see ``Settings.api_key``'s
    docstring in ``app/core/config.py``).
    """
    if credentials is None:
        raise _unauthorized()

    configured_key = settings.api_key
    if not configured_key or not secrets.compare_digest(credentials.credentials, configured_key):
        raise _unauthorized()
