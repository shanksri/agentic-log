"""Tests for Phase 24A: ENVIRONMENT setting, production-only configuration
validation, and positive Field(gt=0) constraints on rate-limit/embedding-
dimension settings (app/core/config.py).

Every test constructs Settings directly with `_env_file=None` and clears
the relevant OS env vars first, so results never depend on this machine's
real .env file or shell environment -- explicit kwargs are the only input.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings

_ENV_VARS_TO_CLEAR = (
    "ENVIRONMENT", "DATABASE_URL", "API_KEY", "GITHUB_TOKEN", "OPENAI_API_KEY",
    "EMBEDDING_DIMENSIONS", "RATE_LIMIT_SEARCH_PER_MINUTE",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


# ── Defaults / development behavior is unchanged ────────────────────────────


def test_defaults_match_current_behavior() -> None:
    s = _settings()
    assert s.environment == "development"
    assert s.database_url == "postgresql+psycopg://postgres:postgres@localhost:5432/incidents"
    assert s.api_key is None
    assert s.embedding_dimensions == 384
    assert s.rate_limit_search_per_minute == 100
    assert s.rate_limit_agent_per_minute == 20
    assert s.rate_limit_evaluation_query_per_minute == 20
    assert s.rate_limit_evaluation_retrieval_per_minute == 5
    assert s.rate_limit_evaluation_reasoning_per_minute == 5
    assert s.rate_limit_evaluation_full_per_minute == 2
    assert s.rate_limit_interactive_evaluation_per_minute == 20
    assert s.rate_limit_incidents_per_minute == 100
    assert s.rate_limit_ingestion_per_minute == 10
    assert s.rate_limit_evaluation_runs_per_minute == 60


def test_development_construction_unaffected_by_missing_api_key_or_localhost_db() -> None:
    # The whole point of gating on ENVIRONMENT: this must keep working
    # exactly as it did before Phase 24A.
    s = _settings(environment="development", api_key=None)
    assert s.api_key is None
    s2 = _settings(
        environment="development",
        database_url="postgresql+psycopg://postgres:postgres@localhost:5432/incidents",
    )
    assert "localhost" in s2.database_url


def test_environment_defaults_to_development_when_unset() -> None:
    assert _settings().environment == "development"


# ── Production validation: API_KEY ──────────────────────────────────────────


def test_production_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="API_KEY"):
        _settings(
            environment="production",
            api_key=None,
            database_url="postgresql+psycopg://user:pass@db.internal:5432/incidents",
        )


def test_production_rejects_placeholder_api_key() -> None:
    with pytest.raises(ValidationError, match="API_KEY"):
        _settings(
            environment="production",
            api_key="changeme-generate-a-real-secret",
            database_url="postgresql+psycopg://user:pass@db.internal:5432/incidents",
        )


def test_production_accepts_real_api_key() -> None:
    s = _settings(
        environment="production",
        api_key="a-real-secret-key-value",
        database_url="postgresql+psycopg://user:pass@db.internal:5432/incidents",
    )
    assert s.api_key == "a-real-secret-key-value"


# ── Production validation: DATABASE_URL ─────────────────────────────────────


def test_production_rejects_localhost_database_url() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        _settings(
            environment="production",
            api_key="a-real-secret-key-value",
            database_url="postgresql+psycopg://postgres:postgres@localhost:5432/incidents",
        )


def test_production_rejects_127_0_0_1_database_url() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        _settings(
            environment="production",
            api_key="a-real-secret-key-value",
            database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:5432/incidents",
        )


def test_production_accepts_non_localhost_database_url() -> None:
    s = _settings(
        environment="production",
        api_key="a-real-secret-key-value",
        database_url="postgresql+psycopg://user:pass@db.internal.example.com:5432/incidents",
    )
    assert "localhost" not in s.database_url


def test_production_construction_succeeds_with_both_required_values() -> None:
    s = _settings(
        environment="production",
        api_key="a-real-secret-key-value",
        database_url="postgresql+psycopg://user:pass@db.internal:5432/incidents",
    )
    assert s.environment == "production"


# ── Positive Field(gt=0) constraints ────────────────────────────────────────


@pytest.mark.parametrize(
    "field_name",
    [
        "rate_limit_search_per_minute",
        "rate_limit_agent_per_minute",
        "rate_limit_evaluation_query_per_minute",
        "rate_limit_evaluation_retrieval_per_minute",
        "rate_limit_evaluation_reasoning_per_minute",
        "rate_limit_evaluation_full_per_minute",
        "rate_limit_interactive_evaluation_per_minute",
        "rate_limit_incidents_per_minute",
        "rate_limit_ingestion_per_minute",
        "rate_limit_evaluation_runs_per_minute",
        "embedding_dimensions",
    ],
)
@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_non_positive_values_rejected(field_name: str, bad_value: int) -> None:
    with pytest.raises(ValidationError):
        _settings(**{field_name: bad_value})


def test_positive_values_still_accepted() -> None:
    s = _settings(rate_limit_search_per_minute=1, embedding_dimensions=1)
    assert s.rate_limit_search_per_minute == 1
    assert s.embedding_dimensions == 1
