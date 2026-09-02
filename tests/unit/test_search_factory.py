"""Tests for Phase 24B: build_routed_search_service() wiring the
process-wide EmbeddingService cache (app/services/embedding_service.py)
into the dense IncidentSearchService it constructs, instead of a fresh
EmbeddingService() (and thus a fresh SentenceTransformer model load) on
every call -- the concrete fix for the "model reloaded on every /search/*
and /agent/investigate request" finding.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.embedding_service import get_embedding_service, reset_embedding_service_cache
from app.services.search_factory import build_routed_search_service, reset_bm25_cache


@pytest.fixture(autouse=True)
def _reset_caches_around_test():
    reset_embedding_service_cache()
    reset_bm25_cache()
    yield
    reset_embedding_service_cache()
    reset_bm25_cache()


def _fake_db() -> MagicMock:
    db = MagicMock()
    db.execute.return_value.all.return_value = []  # empty corpus for BM25 indexing
    return db


def test_build_routed_search_service_uses_the_cached_embedding_service() -> None:
    service = build_routed_search_service(_fake_db())

    assert service._dense.embedding_service is get_embedding_service()


def test_two_calls_share_the_same_embedding_service_instance() -> None:
    first = build_routed_search_service(_fake_db())
    second = build_routed_search_service(_fake_db())

    assert first._dense.embedding_service is second._dense.embedding_service


def test_reset_embedding_service_cache_forces_a_fresh_instance_on_next_call() -> None:
    first = build_routed_search_service(_fake_db())
    reset_embedding_service_cache()
    second = build_routed_search_service(_fake_db())

    assert first._dense.embedding_service is not second._dense.embedding_service
