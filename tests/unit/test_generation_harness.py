"""Tests for Phase 22: Generation Evaluation Harness (generation +
grounding stages, per-metric skip semantics, per-query isolation)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evaluation.generation_harness import (
    GenerationEvaluationConfig,
    GenerationEvaluationMode,
    LLMServiceAnswerGenerator,
    evaluate_generation,
    render_incident_context,
)

FULL = GenerationEvaluationConfig(mode=GenerationEvaluationMode.FULL)
from app.evaluation.gold_dataset import ExpectedIncident, GoldDataset, GoldQuery
from tests.unit.test_generation_metrics import OneHotTokenEmbedder
from tests.unit.test_grounding_metrics import KeywordEmbedder, ScriptedLLM


# ── Fakes / builders ──────────────────────────────────────────────────────────


def _incident(title: str, *, resolution: str = "restarted the broker") -> SimpleNamespace:
    return SimpleNamespace(title=title, resolution_summary=resolution, status="resolved")


def _result(title: str) -> SimpleNamespace:
    return SimpleNamespace(incident=_incident(title))


class FakeSearchService:
    def __init__(self, responses: dict[str, list] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[dict] = []

    def search(self, query, *, limit=10, call_site=None):
        self.calls.append({"query": query, "limit": limit, "call_site": call_site})
        return self._responses.get(query, [])


class FakeAnswerGenerator:
    def __init__(self, answers: dict[str, str] | None = None, *, raises: bool = False) -> None:
        self._answers = answers or {}
        self._raises = raises
        self.calls: list[dict] = []

    def generate_answer(self, query: str, context: str) -> str:
        self.calls.append({"query": query, "context": context})
        if self._raises:
            raise RuntimeError("LLM unavailable")
        return self._answers.get(query, "generated answer")


def _gold_query(
    *, id: str = "q-1", query: str = "broker down", reference: str | None = None
) -> GoldQuery:
    return GoldQuery(
        id=id,
        query=query,
        category="lexical-overlap",
        difficulty="easy",
        expected_incidents=(
            ExpectedIncident(source_type="github", source_external_id="a/b#1", relevance=3),
        ),
        reference_answer=reference,
    )


def _dataset(queries: tuple[GoldQuery, ...]) -> GoldDataset:
    return GoldDataset(
        version="2.1.0",
        description="generation test dataset",
        created_at="2026-07-02T00:00:00Z",
        queries=queries,
    )


def _grounding_llm_for_one_query() -> ScriptedLLM:
    """Scripted responses for one fully-grounded query, in call order:
    faithfulness (claims, verdicts), answer relevancy (questions),
    context precision (verdicts), context recall (claims+verdicts),
    entity recall (reference entities, context entities).
    """
    return ScriptedLLM([
        {"claims": ["broker crashed"]},
        {"verdicts": [True]},
        {"questions": ["broker down"]},
        {"verdicts": [True]},
        {"claims": ["restart fixes it"], "verdicts": [True]},
        {"entities": ["kafka"]},
        {"entities": ["kafka"]},
    ])


# ── Skip semantics ────────────────────────────────────────────────────────────


def test_nothing_computable_skips_query_before_any_cost() -> None:
    # No reference AND no grounding backend: skipped, zero retrieval/LLM cost.
    search = FakeSearchService()
    generator = FakeAnswerGenerator()
    dataset = _dataset((_gold_query(reference=None),))

    report = evaluate_generation(dataset, search, generator)

    assert report.num_skipped == 1
    assert report.num_answered == 0
    assert report.results[0].skipped is True
    assert "nothing to evaluate" in report.results[0].skip_reason
    assert search.calls == []
    assert generator.calls == []


def test_missing_reference_skips_bertscore_but_grounding_still_runs() -> None:
    search = FakeSearchService({"broker down": [_result("Kafka broker crash")]})
    generator = FakeAnswerGenerator()
    # Grounding scripted for: faithfulness x2, relevancy. Context precision
    # is reference-dependent since Phase 22B (no circular fallback) and must
    # NOT consume calls; recall/entity recall likewise.
    llm = ScriptedLLM([
        {"claims": ["broker crashed"]},
        {"verdicts": [True]},
        {"questions": ["broker down"]},
    ])
    dataset = _dataset((_gold_query(reference=None),))

    report = evaluate_generation(
        dataset, search, generator,
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
        config=FULL,
    )

    result = report.results[0]
    assert result.skipped is False
    assert result.generation is None  # BERTScore skipped: no reference
    assert result.grounding is not None
    assert result.grounding.faithfulness == pytest.approx(1.0)
    assert result.grounding.answer_relevancy == pytest.approx(1.0)
    # Phase 22B: no reference -> precision SKIPPED, never judged against the
    # generated answer (circular), never fabricated.
    assert result.grounding.context_precision is None
    assert result.grounding.context_recall is None
    assert result.grounding.context_entity_recall is None
    assert any("bert_score skipped" in n for n in result.notes)
    assert any(
        "context_precision skipped" in n and "circular" in n for n in result.notes
    )
    assert any("context_recall skipped" in n for n in result.notes)
    assert llm._responses == []  # exactly the scripted calls were made


def test_missing_retrieved_context_skips_grounding_but_bertscore_still_runs() -> None:
    search = FakeSearchService()  # returns [] for every query
    generator = FakeAnswerGenerator({"broker down": "restart the kafka broker"})
    llm = ScriptedLLM([])  # must never be consulted
    dataset = _dataset((_gold_query(reference="restart the kafka broker"),))

    report = evaluate_generation(
        dataset, search, generator,
        token_embedder=OneHotTokenEmbedder(),
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
    )

    result = report.results[0]
    assert result.num_contexts == 0
    # BERTScore computed despite no context.
    assert result.generation.bert_score_f1 == pytest.approx(1.0)
    # Every grounding metric undefined; no LLM call burned.
    grounding = result.grounding
    assert grounding.faithfulness is None
    assert grounding.answer_relevancy is None
    assert grounding.context_precision is None
    assert grounding.context_recall is None
    assert grounding.context_entity_recall is None
    assert any("no retrieved context" in n for n in result.notes)
    assert llm.calls == []


def test_fully_configured_query_scores_both_halves() -> None:
    search = FakeSearchService({"broker down": [_result("Kafka broker crash")]})
    generator = FakeAnswerGenerator({"broker down": "restart the kafka broker"})
    dataset = _dataset((_gold_query(reference="restart the kafka broker"),))

    report = evaluate_generation(
        dataset, search, generator,
        token_embedder=OneHotTokenEmbedder(),
        grounding_llm=_grounding_llm_for_one_query(),
        sentence_embedder=KeywordEmbedder(),
        config=FULL,
    )

    assert report.num_answered == 1
    assert report.num_generation_scored == 1
    assert report.num_grounding_scored == 1
    result = report.results[0]
    assert result.generation.bert_score_f1 == pytest.approx(1.0)
    assert result.grounding.faithfulness == pytest.approx(1.0)
    assert result.grounding.context_entity_recall == pytest.approx(1.0)
    # Aggregates populated on both halves.
    assert report.generation_aggregate.bert_score_f1.mean == pytest.approx(1.0)
    assert report.grounding_aggregate.faithfulness.mean == pytest.approx(1.0)


def test_generation_failure_is_recorded_not_raised() -> None:
    search = FakeSearchService()
    generator = FakeAnswerGenerator(raises=True)
    dataset = _dataset((_gold_query(reference="some reference"),))

    report = evaluate_generation(dataset, search, generator)

    assert report.num_failed == 1
    assert report.num_answered == 0
    result = report.results[0]
    assert result.skipped is True
    assert "generation failed" in result.skip_reason


def test_malformed_grounding_response_downgrades_one_metric_only() -> None:
    search = FakeSearchService({"broker down": [_result("Kafka broker crash")]})
    generator = FakeAnswerGenerator()
    # Faithfulness claims malformed -> that metric None; the rest proceed.
    llm = ScriptedLLM([
        {"claims": "NOT A LIST"},          # faithfulness -> error
        {"questions": ["broker down"]},     # relevancy OK
        {"verdicts": [True]},               # precision OK
        {"claims": ["x"], "verdicts": [True]},   # recall OK
        {"entities": ["kafka"]},
        {"entities": ["kafka"]},
    ])
    dataset = _dataset((_gold_query(reference="restart the kafka broker"),))

    report = evaluate_generation(
        dataset, search, generator,
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
        config=FULL,
    )

    grounding = report.results[0].grounding
    assert grounding.faithfulness is None
    assert grounding.answer_relevancy == pytest.approx(1.0)
    assert grounding.context_precision == pytest.approx(1.0)
    assert grounding.context_recall == pytest.approx(1.0)
    assert any("faithfulness failed" in n for n in report.results[0].notes)


def test_no_sentence_embedder_downgrades_answer_relevancy_only() -> None:
    search = FakeSearchService({"broker down": [_result("Kafka broker crash")]})
    llm = ScriptedLLM([
        {"claims": ["broker crashed"]},
        {"verdicts": [True]},
        # (no relevancy call — skipped without embedder)
        {"verdicts": [True]},
        {"claims": ["x"], "verdicts": [True]},
        {"entities": ["kafka"]},
        {"entities": ["kafka"]},
    ])
    dataset = _dataset((_gold_query(reference="ref"),))

    report = evaluate_generation(
        dataset, FakeSearchService({"broker down": [_result("Kafka broker crash")]}),
        FakeAnswerGenerator(), grounding_llm=llm,
        config=FULL,
    )

    grounding = report.results[0].grounding
    assert grounding.answer_relevancy is None
    assert grounding.faithfulness == pytest.approx(1.0)
    assert any("no sentence embedder" in n for n in report.results[0].notes)


def test_mixed_dataset_counts() -> None:
    generator = FakeAnswerGenerator({"q-a": "exact reference text"})
    dataset = _dataset((
        _gold_query(id="q-1", query="q-a", reference="exact reference text"),
        _gold_query(id="q-2", query="q-b", reference=None),   # nothing computable
    ))

    report = evaluate_generation(
        dataset, FakeSearchService(), generator, token_embedder=OneHotTokenEmbedder()
    )

    assert report.num_answered == 1
    assert report.num_generation_scored == 1
    assert report.num_grounding_scored == 0
    assert report.num_skipped == 1
    assert len(report.results) == 2


def test_search_called_with_configured_k_and_call_site() -> None:
    search = FakeSearchService()
    dataset = _dataset((_gold_query(reference="ref"),))

    evaluate_generation(dataset, search, FakeAnswerGenerator(), k=7)

    assert search.calls == [
        {"query": "broker down", "limit": 7, "call_site": "generation_evaluation"}
    ]


# ── Context rendering ─────────────────────────────────────────────────────────


def test_render_incident_context_includes_title_status_resolution() -> None:
    text = render_incident_context(_result("Kafka broker crash"))
    assert "Kafka broker crash" in text
    assert "resolved" in text
    assert "restarted the broker" in text


# ── LLMServiceAnswerGenerator adapter ─────────────────────────────────────────


def test_llm_service_adapter_delegates_to_generate_investigation() -> None:
    calls: list[dict] = []

    class FakeLLMService:
        def generate_investigation(self, *, problem: str, context: str) -> str:
            calls.append({"problem": problem, "context": context})
            return "adapter answer"

    adapter = LLMServiceAnswerGenerator(FakeLLMService())
    answer = adapter.generate_answer("why is the broker down", "evidence block")

    assert answer == "adapter answer"
    assert calls == [{"problem": "why is the broker down", "context": "evidence block"}]
