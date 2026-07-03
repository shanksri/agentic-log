from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.api.routes import agent, evaluation, evaluation_interactive, health, incidents, ingestion, search
from app.core.logging import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: best-effort DB connectivity probe, logged either way — never
    blocks the app from starting (a DB that isn't up yet, e.g. during a
    rolling deploy, shouldn't prevent the container from becoming live;
    ``/health/ready`` is what a load balancer/orchestrator should gate
    traffic on). Shutdown: dispose the connection pool cleanly rather than
    letting connections leak on process exit.
    """
    from app.db.session import engine

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("startup: database connectivity OK")
    except Exception:  # noqa: BLE001 — startup must never crash on this
        logger.warning("startup: database not reachable yet (will retry per-request)")
    yield
    engine.dispose()
    logger.info("shutdown: database connection pool disposed")


app = FastAPI(
    title="Enterprise Incident Intelligence Platform",
    version="0.1.0",
    description="Phase 1 GitHub incident ingestion and semantic incident search.",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(agent.router)
app.include_router(incidents.router)
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(evaluation.router)
app.include_router(evaluation_interactive.router)


# ── Phase 23: platform-wide failure handling ─────────────────────────────────
#
# Neither handler changes any route's behavior on success. Both exist so that
# a failure the route itself didn't anticipate (a DB connection drop, an
# unguarded service exception) degrades to a clean, typed JSON error instead
# of an unhandled-exception traceback — the same "log full detail server-side,
# return a generic message client-side" discipline already used by the
# evaluation API's ``_build_search_service``/``_build_orchestrator`` helpers,
# applied platform-wide as a safety net.


@app.exception_handler(OperationalError)
async def database_unavailable_handler(request: Request, exc: OperationalError) -> JSONResponse:
    logger.exception("Database operation failed for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=503,
        content={"detail": "Database is temporarily unavailable. Please retry shortly."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred."},
    )
