from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import agent, health, incidents, ingestion, search
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="Enterprise Incident Intelligence Platform",
    version="0.1.0",
    description="Phase 1 GitHub incident ingestion and semantic incident search.",
)
app.include_router(health.router)
app.include_router(agent.router)
app.include_router(incidents.router)
app.include_router(ingestion.router)
app.include_router(search.router)
