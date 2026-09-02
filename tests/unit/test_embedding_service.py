from __future__ import annotations

import sys
import threading
import types

import pytest

from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    get_embedding_service,
    reset_embedding_service_cache,
)


class FakeVector:
    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, normalize_embeddings: bool) -> FakeVector:
        self.calls.append((text, normalize_embeddings))
        return FakeVector()


def test_embedding_service_returns_plain_float_list() -> None:
    service = EmbeddingService(model_name="fake-model")
    fake_model = FakeModel()
    service.__dict__["model"] = fake_model

    vector = service.embed_text("database timeout")

    assert vector == [0.1, 0.2, 0.3]
    assert fake_model.calls == [("database timeout", True)]


# ── Failure paths (Phase 24B) ────────────────────────────────────────────────


def _raising_sentence_transformers_module() -> types.ModuleType:
    # A fake `sentence_transformers` module injected into sys.modules so
    # `from sentence_transformers import SentenceTransformer` (the deferred
    # import inside EmbeddingService.model) resolves to it without
    # triggering a real (slow, torch-importing) import.
    def _raise(*args: object, **kwargs: object):
        raise RuntimeError("network unreachable")

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _raise  # type: ignore[attr-defined]
    return fake


def test_model_load_failure_raises_embedding_service_error(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", _raising_sentence_transformers_module()
    )
    service = EmbeddingService(model_name="fake-model")

    with pytest.raises(EmbeddingServiceError, match="Failed to load embedding model"):
        _ = service.model


def test_encode_failure_raises_embedding_service_error() -> None:
    class RaisingModel:
        def encode(self, text: str, normalize_embeddings: bool) -> None:
            raise RuntimeError("tokenizer exploded")

    service = EmbeddingService(model_name="fake-model")
    service.__dict__["model"] = RaisingModel()

    with pytest.raises(EmbeddingServiceError, match="Embedding failed"):
        service.embed_text("some text")


def test_embed_text_does_not_swallow_embedding_service_error_from_model_load(
    monkeypatch,
) -> None:
    # embed_text()'s own except clauses must re-raise an EmbeddingServiceError
    # from the lazy `.model` load unchanged, not double-wrap it.
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", _raising_sentence_transformers_module()
    )
    service = EmbeddingService(model_name="fake-model")

    with pytest.raises(EmbeddingServiceError, match="Failed to load embedding model"):
        service.embed_text("some text")


# ── Process-wide cache (Phase 24B) ───────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_embedding_service_cache_around_test():
    reset_embedding_service_cache()
    yield
    reset_embedding_service_cache()


def test_get_embedding_service_returns_same_instance_across_calls() -> None:
    first = get_embedding_service()
    second = get_embedding_service()
    assert first is second


def test_get_embedding_service_returns_an_embedding_service() -> None:
    assert isinstance(get_embedding_service(), EmbeddingService)


def test_reset_embedding_service_cache_forces_a_new_instance() -> None:
    first = get_embedding_service()
    reset_embedding_service_cache()
    second = get_embedding_service()
    assert first is not second


def test_get_embedding_service_is_thread_safe_under_concurrent_first_access() -> None:
    # Double-checked locking: many threads racing on the very first call
    # (cache empty) must all observe exactly one constructed instance, never
    # a handful of independently-constructed ones.
    instances: list[EmbeddingService] = []
    lock = threading.Lock()
    barrier = threading.Barrier(20)

    def _get() -> None:
        barrier.wait()  # maximize the chance every thread races the check together
        instance = get_embedding_service()
        with lock:
            instances.append(instance)

    threads = [threading.Thread(target=_get) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 20
    assert len({id(instance) for instance in instances}) == 1
