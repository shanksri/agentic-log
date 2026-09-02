from __future__ import annotations

import logging
import threading
from functools import cached_property
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding model fails to load or encode text (model
    download failure, out-of-memory, tokenizer error). Typed so callers can
    distinguish "embedding backend is broken" from an unrelated bug,
    matching ``LLMResponseError`` in ``llm_service.py``.
    """


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(self.model_name)
        except Exception as exc:  # noqa: BLE001 — model loading can fail in many SDK-internal ways
            raise EmbeddingServiceError(
                f"Failed to load embedding model {self.model_name!r}: {exc}"
            ) from exc

    def embed_text(self, text: str) -> list[float]:
        try:
            vector = self.model.encode(text, normalize_embeddings=True)
        except EmbeddingServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 — encode() failures are SDK-internal
            raise EmbeddingServiceError(f"Embedding failed: {exc}") from exc
        return [float(value) for value in vector.tolist()]


# Phase 24B: process-wide cache, mirroring app/services/search_factory.py's
# `_bm25_cache`/`get_bm25_retriever` pattern exactly (double-checked locking,
# lazy build on first use, reused for the process lifetime). Without this,
# every production caller that doesn't already hold an `EmbeddingService`
# instance (e.g. a fresh `build_routed_search_service()` per request)
# reloads the SentenceTransformer model from scratch on first use — this
# cache is what lets those callers share one already-loaded model instead.
# Same model_name/config, same `embed_text()` outputs — this changes only
# whether a new `SentenceTransformer` gets loaded, never what it computes.
_embedding_service_cache: EmbeddingService | None = None
_embedding_service_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    """Return the process-local ``EmbeddingService``, constructing it once
    (thread-safe, double-checked locking) and reusing it on every
    subsequent call. The underlying ``SentenceTransformer`` model is loaded
    lazily on first ``embed_text()`` call (unchanged — see
    ``EmbeddingService.model``), not eagerly here.
    """
    global _embedding_service_cache
    if _embedding_service_cache is None:
        with _embedding_service_lock:
            if _embedding_service_cache is None:
                logger.info("embedding_service.cache_initialized")
                _embedding_service_cache = EmbeddingService()
    return _embedding_service_cache


def reset_embedding_service_cache() -> None:
    """Drop the cached instance so the next call to ``get_embedding_service``
    constructs a fresh one. Exposed for tests; production code has no
    automatic invalidation trigger (same contract as
    ``search_factory.reset_bm25_cache``).
    """
    global _embedding_service_cache
    with _embedding_service_lock:
        _embedding_service_cache = None
