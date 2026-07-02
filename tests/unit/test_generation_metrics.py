"""Tests for Phase 22: Generation Metric Engine (BERTScore — semantic
answer similarity)."""
from __future__ import annotations

import dataclasses

import pytest

from app.evaluation.generation_metrics import (
    GenerationMetrics,
    aggregate_generation_metrics,
    bert_score,
    compute_generation_metrics,
    cosine_similarity,
    summarize_values,
    tokenize,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────


class OneHotTokenEmbedder:
    """Deterministic TokenEmbedder: each distinct word maps to a fixed
    one-hot vector, so identical words have cosine 1.0 and distinct words
    cosine 0.0 — makes BERTScore hand-computable.
    """

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    def embed_tokens(self, text: str):
        tokens = tokenize(text)
        vectors = []
        for token in tokens:
            if token not in self._vocabulary:
                self._vocabulary[token] = len(self._vocabulary)
        dims = max(len(self._vocabulary), 1)
        for token in tokens:
            vector = [0.0] * dims
            vector[self._vocabulary[token]] = 1.0
            vectors.append(vector)
        return vectors


class SynonymAwareTokenEmbedder(OneHotTokenEmbedder):
    """OneHot embedder that maps configured synonyms onto the SAME vector —
    simulates what a real contextual model does for semantically equivalent
    wording, which is exactly why Phase 22 chose BERTScore over n-gram
    metrics.
    """

    def __init__(self, synonyms: dict[str, str]) -> None:
        super().__init__()
        self._synonyms = synonyms

    def embed_tokens(self, text: str):
        canonical = " ".join(self._synonyms.get(t, t) for t in tokenize(text))
        return super().embed_tokens(canonical)


# ── Tokenization / cosine ─────────────────────────────────────────────────────


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Restart the Kafka-Broker!") == ["restart", "the", "kafka", "broker"]


def test_cosine_similarity_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── BERTScore ─────────────────────────────────────────────────────────────────


def test_bert_score_perfect_match_is_all_ones() -> None:
    embedder = OneHotTokenEmbedder()
    precision, recall, f1 = bert_score(
        "kafka broker crashed", "kafka broker crashed", token_embedder=embedder
    )
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_bert_score_semantically_equivalent_wording_scores_high() -> None:
    # "reboot" and "restart" mapped to the same embedding — a lexical metric
    # would punish this paraphrase; BERTScore must not.
    embedder = SynonymAwareTokenEmbedder({"reboot": "restart", "node": "broker"})
    _, _, f1 = bert_score(
        "reboot the kafka node", "restart the kafka broker", token_embedder=embedder
    )
    assert f1 == pytest.approx(1.0)


def test_bert_score_unrelated_answer_is_zero() -> None:
    embedder = OneHotTokenEmbedder()
    assert bert_score("apples oranges", "kafka broker", token_embedder=embedder) == (
        0.0, 0.0, 0.0,
    )


def test_bert_score_partial_overlap_hand_computed() -> None:
    # candidate "a b" vs reference "a c" with one-hot embeddings:
    # P = (1 + 0)/2 = 0.5, R = (1 + 0)/2 = 0.5, F1 = 0.5.
    embedder = OneHotTokenEmbedder()
    precision, recall, f1 = bert_score("a b", "a c", token_embedder=embedder)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_bert_score_empty_either_side_is_zero() -> None:
    embedder = OneHotTokenEmbedder()
    assert bert_score("", "kafka", token_embedder=embedder) == (0.0, 0.0, 0.0)
    assert bert_score("kafka", "", token_embedder=embedder) == (0.0, 0.0, 0.0)


# ── compute_generation_metrics ────────────────────────────────────────────────


def test_compute_without_embedder_is_undefined_not_zero() -> None:
    metrics = compute_generation_metrics("same text", "same text")
    assert metrics.bert_score_precision is None
    assert metrics.bert_score_recall is None
    assert metrics.bert_score_f1 is None


def test_compute_with_embedder_returns_defined_floats() -> None:
    metrics = compute_generation_metrics(
        "kafka broker crashed", "kafka broker crashed",
        token_embedder=OneHotTokenEmbedder(),
    )
    assert metrics.bert_score_f1 == pytest.approx(1.0)


def test_compute_is_deterministic() -> None:
    a = compute_generation_metrics(
        "kafka down", "the kafka broker is down", token_embedder=OneHotTokenEmbedder()
    )
    b = compute_generation_metrics(
        "kafka down", "the kafka broker is down", token_embedder=OneHotTokenEmbedder()
    )
    assert a == b


def test_generation_metrics_is_frozen() -> None:
    metrics = compute_generation_metrics("a", "a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.bert_score_f1 = 0.5  # type: ignore[misc]


# ── Aggregation ───────────────────────────────────────────────────────────────


def _metrics(f1: float | None) -> GenerationMetrics:
    return GenerationMetrics(
        bert_score_precision=f1, bert_score_recall=f1, bert_score_f1=f1
    )


def test_summarize_values_hand_computed() -> None:
    summary = summarize_values([0.2, 0.6, 0.4])
    assert summary.count == 3
    assert summary.mean == pytest.approx(0.4)
    assert summary.median == pytest.approx(0.4)
    assert summary.minimum == pytest.approx(0.2)
    assert summary.maximum == pytest.approx(0.6)


def test_summarize_values_empty_is_none() -> None:
    assert summarize_values([]) is None


def test_aggregate_skips_undefined_values() -> None:
    aggregate = aggregate_generation_metrics([_metrics(0.8), _metrics(None)])
    assert aggregate.num_scored == 2
    assert aggregate.bert_score_f1.count == 1
    assert aggregate.bert_score_f1.mean == pytest.approx(0.8)


def test_aggregate_all_undefined_is_none_not_zero() -> None:
    aggregate = aggregate_generation_metrics([_metrics(None)])
    assert aggregate.bert_score_f1 is None


def test_aggregate_empty_input() -> None:
    aggregate = aggregate_generation_metrics([])
    assert aggregate.num_scored == 0
    assert aggregate.bert_score_f1 is None
