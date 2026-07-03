from __future__ import annotations

import logging

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.api.dependencies import DbSession
from app.api.schemas import HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is up and serving requests. Unconditional by
    design — a Kubernetes liveness probe should never depend on a
    downstream service, or a transient DB blip would trigger unnecessary
    container restarts. See ``/health/ready`` for a dependency check.
    """
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(db: DbSession, response: Response) -> ReadinessResponse:
    """Readiness: is the database reachable right now? Intended for a
    Kubernetes readiness probe / load-balancer health check — a pod that
    can't reach its database should stop receiving traffic without being
    killed and restarted (that's what ``/health`` is for). Returns 503
    (not 500) when the dependency is down, so a probe correctly reads this
    as "not ready" rather than "endpoint broken".
    """
    try:
        db.execute(text("SELECT 1"))
        return ReadinessResponse(status="ok", database="reachable")
    except Exception:  # noqa: BLE001 — readiness must report, never 500
        logger.exception("Readiness check: database unreachable")
        response.status_code = 503
        return ReadinessResponse(status="degraded", database="unreachable")
