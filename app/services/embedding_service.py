from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


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
