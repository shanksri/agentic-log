from __future__ import annotations

from app.services.embedding_service import EmbeddingService


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
