from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/incidents"
    )
    github_token: str | None = None
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    api_key: str | None = Field(
        default=None,
        description=(
            "Shared secret for the platform's Bearer API-key authentication "
            "(Phase 23B) — required on every /ingestion, /search, /agent, "
            "/incidents, and /evaluation request as 'Authorization: Bearer "
            "<API_KEY>'. Unset means no key can ever match, so every "
            "protected request is rejected (fail-closed), not left open."
        ),
    )
    search_routing_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in switch for adaptive retrieval routing (Phase 18A-18C) in "
            "production /search and the investigation orchestrator. False "
            "preserves dense-only behavior exactly "
            "(see RoutedSearchConfig.routing_enabled)."
        ),
    )

    # ── Phase 23C: endpoint-aware rate limiting ─────────────────────────────
    #
    # All limits are "requests per 60-second fixed window" per caller
    # identity (see app/api/rate_limit.py). Values below the Search/Agent/
    # Evaluation-*/Interactive-Evaluation line are the Phase 23C spec's own
    # suggested defaults. `rate_limit_incidents_per_minute` and
    # `rate_limit_ingestion_per_minute` are NOT in the spec's suggested
    # list — added because leaving those two routers completely unlimited
    # would leave real abuse vectors unaddressed (ingestion triggers
    # external HTTP calls; see the Phase 23 production-readiness review's
    # SSRF/cost-exhaustion findings), which would contradict this phase's
    # own stated objective. `rate_limit_evaluation_runs_per_minute` covers
    # the read-only GET /evaluation/runs*, /evaluation/stats views, also
    # not named in the spec, for the same reason.
    rate_limit_enabled: bool = Field(
        default=True,
        description="Global kill switch — False disables all rate limiting (health is always unlimited regardless).",
    )
    rate_limit_search_per_minute: int = 100
    rate_limit_agent_per_minute: int = 20
    rate_limit_evaluation_query_per_minute: int = 20
    rate_limit_evaluation_retrieval_per_minute: int = 5
    rate_limit_evaluation_reasoning_per_minute: int = 5
    rate_limit_evaluation_full_per_minute: int = 2
    rate_limit_interactive_evaluation_per_minute: int = 20
    rate_limit_incidents_per_minute: int = 100
    rate_limit_ingestion_per_minute: int = 10
    rate_limit_evaluation_runs_per_minute: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
