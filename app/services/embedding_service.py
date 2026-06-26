from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]
