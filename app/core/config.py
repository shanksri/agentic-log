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
    search_routing_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in switch for adaptive retrieval routing (Phase 18A-18C) in "
            "production /search and the investigation orchestrator. False "
            "preserves dense-only behavior exactly "
            "(see RoutedSearchConfig.routing_enabled)."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
