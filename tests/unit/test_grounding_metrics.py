"""Tests for Phase 22: Grounding Metric Engine (RAGAS-style Faithfulness,
Answer Relevancy, Context Precision, Context Recall, Context Entity Recall).

Every test uses a scripted fake ``GroundingLLMClient`` (matching
``LLMService.generate_json``'s shape) and a deterministic fake
``SentenceEmbedder`` — no OpenAI, no model downloads, exact expected values.
"""
from __future__ import annotations

import pytest

from app.evaluation.grounding_metrics import (
    GroundingMetrics,
    GroundingResponseError,
    aggregate_grounding_metrics,
    answer_relevancy,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────


class ScriptedLLM:
    """Returns queued JSON responses in order; records every call."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self._responses.pop(0)


class KeywordEmbedder:
    """Embeds text as a bag-of-words one-hot-sum vector: texts sharing words
    have positive cosine; disjoint texts have cosine 0.
    """

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    def embed_text(self, text: str):
        words = text.lower().split()
        for word in words:
            if word not in self._vocabulary:
                self._vocabulary[word] = len(self._vocabulary)
        vector = [0.0] * max(len(self._vocabulary), 1)
        for word in words:
            vector[self._vocabulary[word]] += 1.0
        return vector


CONTEXTS = ["Kafka broker crash\nresolution: increase heap", "Unrelated incident"]


# ── Faithfulness ──────────────────────────────────────────────────────────────


def test_faithfulness_fully_supported_answer_is_one() -> None:
    llm = ScriptedLLM([
        {"claims": ["the broker crashed", "heap was increased"]},
        {"verdicts": [True, True]},
    ])
    assert faithfulness("answer text", CONTEXTS, llm=llm) == pytest.approx(1.0)


def test_faithfulness_hallucinated_answer_is_zero() -> None:
    llm = ScriptedLLM([
        {"claims": ["the moon caused the outage"]},
        {"verdicts": [False]},
    ])
    assert faithfulness("hallucination", CONTEXTS, llm=llm) == pytest.approx(0.0)


def test_faithfulness_partially_supported_hand_computed() -> None:
    llm = ScriptedLLM([
        {"claims": ["a", "b", "c", "d"]},
        {"verdicts": [True, True, True, False]},
    ])
    assert faithfulness("answer", CONTEXTS, llm=llm) == pytest.approx(0.75)


def test_faithfulness_no_contexts_is_none() -> None:
    llm = ScriptedLLM([])
    assert faithfulness("answer", [], llm=llm) is None
    assert llm.calls == []  # no LLM cost when uncomputable


def test_faithfulness_zero_claims_is_none() -> None:
    llm = ScriptedLLM([{"claims": []}])
    assert faithfulness("answer", CONTEXTS, llm=llm) is None


def test_faithfulness_malformed_claims_raises() -> None:
    llm = ScriptedLLM([{"claims": "not a list"}])
    with pytest.raises(GroundingResponseError, match="claims"):
        faithfulness("answer", CONTEXTS, llm=llm)


def test_faithfulness_verdict_length_mismatch_raises() -> None:
    llm = ScriptedLLM([
        {"claims": ["a", "b"]},
        {"verdicts": [True]},
    ])
    with pytest.raises(GroundingResponseError, match="exactly 2"):
        faithfulness("answer", CONTEXTS, llm=llm)


# ── Answer Relevancy ──────────────────────────────────────────────────────────


def test_answer_relevancy_identical_generated_questions_is_one() -> None:
    llm = ScriptedLLM([
        {"questions": ["why is the broker down", "why is the broker down"]},
    ])
    score = answer_relevancy(
        "why is the broker down", "answer", llm=llm, embedder=KeywordEmbedder()
    )
    assert score == pytest.approx(1.0)


def test_answer_relevancy_irrelevant_answer_is_zero() -> None:
    # Generated questions share NO words with the original question, so the
    # bag-of-words cosine is exactly zero.
    llm = ScriptedLLM([{"questions": ["which fruit looks yellow"]}])
    score = answer_relevancy(
        "why did the broker crash", "bananas are yellow",
        llm=llm, embedder=KeywordEmbedder(),
    )
    assert score == pytest.approx(0.0)


def test_answer_relevancy_partial_overlap_is_between() -> None:
    llm = ScriptedLLM([{"questions": ["why is the broker slow"]}])
    score = answer_relevancy(
        "why is the broker down", "answer", llm=llm, embedder=KeywordEmbedder()
    )
    assert 0.0 < score < 1.0


def test_answer_relevancy_empty_answer_is_none() -> None:
    llm = ScriptedLLM([])
    assert answer_relevancy("q", "   ", llm=llm, embedder=KeywordEmbedder()) is None


def test_answer_relevancy_zero_questions_is_none() -> None:
    llm = ScriptedLLM([{"questions": []}])
    assert answer_relevancy("q", "answer", llm=llm, embedder=KeywordEmbedder()) is None


# ── Context Precision ─────────────────────────────────────────────────────────


def test_context_precision_all_relevant_is_one() -> None:
    llm = ScriptedLLM([{"verdicts": [True, True]}])
    assert context_precision("q", CONTEXTS, "truth", llm=llm) == pytest.approx(1.0)


def test_context_precision_is_rank_sensitive_hand_computed() -> None:
    # Relevant chunk at rank 1 of [T, F]: (1/1) / 1 = 1.0
    llm_good = ScriptedLLM([{"verdicts": [True, False]}])
    good = context_precision("q", CONTEXTS, "truth", llm=llm_good)
    # Relevant chunk at rank 2 of [F, T]: (1/2) / 1 = 0.5
    llm_bad = ScriptedLLM([{"verdicts": [False, True]}])
    bad = context_precision("q", CONTEXTS, "truth", llm=llm_bad)
    assert good == pytest.approx(1.0)
    assert bad == pytest.approx(0.5)
    assert bad < good


def test_context_precision_nothing_relevant_is_zero() -> None:
    llm = ScriptedLLM([{"verdicts": [False, False]}])
    assert context_precision("q", CONTEXTS, "truth", llm=llm) == 0.0


def test_context_precision_no_contexts_is_none() -> None:
    llm = ScriptedLLM([])
    assert context_precision("q", [], "truth", llm=llm) is None


# ── Context Recall ────────────────────────────────────────────────────────────


def test_context_recall_fully_covered_reference_is_one() -> None:
    llm = ScriptedLLM([
        {"claims": ["broker crashed", "heap increased"], "verdicts": [True, True]},
    ])
    assert context_recall("reference", CONTEXTS, llm=llm) == pytest.approx(1.0)


def test_context_recall_partial_coverage_hand_computed() -> None:
    llm = ScriptedLLM([
        {"claims": ["a", "b", "c"], "verdicts": [True, False, False]},
    ])
    assert context_recall("reference", CONTEXTS, llm=llm) == pytest.approx(1 / 3)


def test_context_recall_empty_reference_is_none() -> None:
    llm = ScriptedLLM([])
    assert context_recall("  ", CONTEXTS, llm=llm) is None


def test_context_recall_no_contexts_is_none() -> None:
    llm = ScriptedLLM([])
    assert context_recall("reference", [], llm=llm) is None


# ── Context Entity Recall ─────────────────────────────────────────────────────


def test_context_entity_recall_hand_computed() -> None:
    llm = ScriptedLLM([
        {"entities": ["Kafka", "ZooKeeper", "JVM"]},   # from reference
        {"entities": ["kafka", "jvm", "Postgres"]},    # from contexts (case-insensitive)
    ])
    score = context_entity_recall("reference", CONTEXTS, llm=llm)
    assert score == pytest.approx(2 / 3)


def test_context_entity_recall_no_reference_entities_is_none() -> None:
    llm = ScriptedLLM([{"entities": []}])
    assert context_entity_recall("reference", CONTEXTS, llm=llm) is None


def test_context_entity_recall_no_contexts_is_none() -> None:
    llm = ScriptedLLM([])
    assert context_entity_recall("reference", [], llm=llm) is None


# ── Aggregation ───────────────────────────────────────────────────────────────


def _grounding(
    *, faith: float | None = None, relevancy: float | None = None
) -> GroundingMetrics:
    return GroundingMetrics(
        faithfulness=faith,
        answer_relevancy=relevancy,
        context_precision=None,
        context_recall=None,
        context_entity_recall=None,
    )


def test_aggregate_grounding_over_defined_values_only() -> None:
    aggregate = aggregate_grounding_metrics([
        _grounding(faith=1.0, relevancy=0.8),
        _grounding(faith=0.5, relevancy=None),
    ])
    assert aggregate.num_scored == 2
    assert aggregate.faithfulness.count == 2
    assert aggregate.faithfulness.mean == pytest.approx(0.75)
    assert aggregate.answer_relevancy.count == 1
    assert aggregate.answer_relevancy.mean == pytest.approx(0.8)
    assert aggregate.context_precision is None  # zero defined values


def test_aggregate_grounding_empty_input() -> None:
    aggregate = aggregate_grounding_metrics([])
    assert aggregate.num_scored == 0
    assert aggregate.faithfulness is None
    assert aggregate.context_entity_recall is None
