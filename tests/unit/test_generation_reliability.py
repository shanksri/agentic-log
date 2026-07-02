"""Tests for Phase 22B: Grounding Reliability & Production Evaluation.

Covers evaluation modes (FAST/STANDARD/FULL), repeated execution with
variance + confidence bands, the removal of circular context precision,
cost reduction through disabled metrics, serialization, and persistence —
all with deterministic fakes (no OpenAI, no models).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.evaluation_pipeline import PipelineInputs
from app.evaluation.experiment_tracking import ExperimentRepository
from app.evaluation.generation_benchmark import (
    FileGenerationBenchmarkRepository,
    create_generation_benchmark_run,
)
from app.evaluation.generation_harness import (
    GenerationEvaluationConfig,
    GenerationEvaluationMode,
    evaluate_generation,
)
from app.evaluation.grounding_metrics import (
    EvaluatorConfidence,
    classify_evaluator_confidence,
    measure_stability,
)
from tests.unit.test_generation_harness import (
    FakeAnswerGenerator,
    FakeSearchService,
    _dataset,
    _gold_query,
    _result,
)
from tests.unit.test_generation_pipeline_integration import (
    FakeGroundingLLM,
    FakeSentenceEmbedder,
    _gold_dataset,
    _pipeline,
)
from tests.unit.test_grounding_metrics import KeywordEmbedder, ScriptedLLM

FAST = GenerationEvaluationConfig(mode=GenerationEvaluationMode.FAST)
STANDARD = GenerationEvaluationConfig(mode=GenerationEvaluationMode.STANDARD)
FULL = GenerationEvaluationConfig(mode=GenerationEvaluationMode.FULL)


def _search() -> FakeSearchService:
    return FakeSearchService({"broker down": [_result("Kafka broker crash")]})


def _referenced_dataset():
    return _dataset((_gold_query(reference="restart the kafka broker"),))


# ── Config validation ─────────────────────────────────────────────────────────


def test_config_defaults_are_conservative() -> None:
    config = GenerationEvaluationConfig()
    assert config.mode is GenerationEvaluationMode.FAST
    assert config.evaluation_repetitions == 1


def test_config_rejects_non_positive_repetitions() -> None:
    with pytest.raises(ValueError, match="evaluation_repetitions"):
        GenerationEvaluationConfig(evaluation_repetitions=0)


def test_mode_metric_sets() -> None:
    assert FAST.enabled_grounding_metrics == {"faithfulness"}
    assert STANDARD.enabled_grounding_metrics == {"faithfulness", "answer_relevancy"}
    assert FULL.enabled_grounding_metrics == {
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "context_entity_recall",
    }


# ── Confidence bands (Part 2) ─────────────────────────────────────────────────


def test_confidence_band_boundaries() -> None:
    assert classify_evaluator_confidence(0.0) is EvaluatorConfidence.HIGH
    assert classify_evaluator_confidence(0.049) is EvaluatorConfidence.HIGH
    assert classify_evaluator_confidence(0.05) is EvaluatorConfidence.MEDIUM
    assert classify_evaluator_confidence(0.10) is EvaluatorConfidence.MEDIUM
    assert classify_evaluator_confidence(0.101) is EvaluatorConfidence.LOW


def test_measure_stability_hand_computed() -> None:
    # stdev([1.0, 0.5, 1.0]) = sqrt(((1/6)^2*2 + (1/3)^2) / 2) ≈ 0.2887
    stability = measure_stability([1.0, 0.5, 1.0])
    assert stability.count == 3
    assert stability.mean == pytest.approx(0.8333, abs=1e-4)
    assert stability.std_dev == pytest.approx(0.2887, abs=1e-4)
    assert stability.minimum == 0.5
    assert stability.maximum == 1.0
    assert stability.confidence is EvaluatorConfidence.LOW


def test_measure_stability_bands_via_two_samples() -> None:
    # stdev of two values = |a-b| / sqrt(2)
    assert measure_stability([0.5, 0.56]).confidence is EvaluatorConfidence.HIGH
    assert measure_stability([0.5, 0.60]).confidence is EvaluatorConfidence.MEDIUM
    assert measure_stability([0.5, 0.80]).confidence is EvaluatorConfidence.LOW


def test_measure_stability_requires_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        measure_stability([1.0])


# ── Evaluation modes (Part 1) ─────────────────────────────────────────────────


def test_fast_mode_runs_faithfulness_only_and_costs_two_calls() -> None:
    llm = ScriptedLLM([
        {"claims": ["broker crashed"]},
        {"verdicts": [True]},
    ])
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
        config=FAST,
    )

    grounding = report.results[0].grounding
    assert grounding.faithfulness == pytest.approx(1.0)
    # Disabled metrics are skipped (None with a note), never zero.
    assert grounding.answer_relevancy is None
    assert grounding.context_precision is None
    assert grounding.context_recall is None
    assert grounding.context_entity_recall is None
    notes = report.results[0].notes
    assert any("answer_relevancy skipped: disabled in 'fast' mode" in n for n in notes)
    assert any("context_precision skipped: disabled in 'fast' mode" in n for n in notes)
    # Cost: exactly 2 grounding LLM calls.
    assert len(llm.calls) == 2
    assert report.evaluation_mode == "fast"


def test_standard_mode_adds_answer_relevancy_three_calls() -> None:
    llm = ScriptedLLM([
        {"claims": ["broker crashed"]},
        {"verdicts": [True]},
        {"questions": ["broker down"]},
    ])
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
        config=STANDARD,
    )

    grounding = report.results[0].grounding
    assert grounding.faithfulness == pytest.approx(1.0)
    assert grounding.answer_relevancy == pytest.approx(1.0)
    assert grounding.context_precision is None
    assert len(llm.calls) == 3
    assert report.evaluation_mode == "standard"


def test_full_mode_runs_all_five_metrics_seven_calls() -> None:
    llm = ScriptedLLM([
        {"claims": ["broker crashed"]},
        {"verdicts": [True]},
        {"questions": ["broker down"]},
        {"verdicts": [True]},
        {"claims": ["restart fixes it"], "verdicts": [True]},
        {"entities": ["kafka"]},
        {"entities": ["kafka"]},
    ])
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
        config=FULL,
    )

    grounding = report.results[0].grounding
    assert grounding.faithfulness == pytest.approx(1.0)
    assert grounding.answer_relevancy == pytest.approx(1.0)
    assert grounding.context_precision == pytest.approx(1.0)
    assert grounding.context_recall == pytest.approx(1.0)
    assert grounding.context_entity_recall == pytest.approx(1.0)
    assert len(llm.calls) == 7
    assert report.evaluation_mode == "full"


def test_fast_mode_is_cheaper_than_full_mode() -> None:
    def _count(config) -> int:
        llm = FakeGroundingLLMCounting()
        evaluate_generation(
            _referenced_dataset(), _search(), FakeAnswerGenerator(),
            grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
            config=config,
        )
        return llm.call_count

    fast_calls = _count(FAST)
    full_calls = _count(FULL)
    assert fast_calls == 2
    assert full_calls == 7
    assert fast_calls < full_calls


class FakeGroundingLLMCounting(FakeGroundingLLM):
    def __init__(self) -> None:
        self.call_count = 0

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.call_count += 1
        return super().generate_json(
            system_prompt=system_prompt, user_prompt=user_prompt
        )


# ── Circular context precision removed (Part 3) ───────────────────────────────


def test_context_precision_skipped_without_reference_no_fallback_call() -> None:
    # FULL mode, no reference: precision must be skipped with a reason — the
    # scripted LLM has NO precision-verdict response queued, so a fallback
    # call would fail loudly.
    llm = ScriptedLLM([
        {"claims": ["broker crashed"]},
        {"verdicts": [True]},
        {"questions": ["broker down"]},
    ])
    report = evaluate_generation(
        _dataset((_gold_query(reference=None),)), _search(), FakeAnswerGenerator(),
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
        config=FULL,
    )

    result = report.results[0]
    assert result.grounding.context_precision is None
    assert any(
        "context_precision skipped: no reference_answer" in n and "circular" in n
        for n in result.notes
    )
    assert llm._responses == []  # exactly 3 calls — no fallback happened


def test_context_precision_still_computes_with_reference() -> None:
    llm = ScriptedLLM([
        {"claims": ["broker crashed"]},
        {"verdicts": [True]},
        {"questions": ["broker down"]},
        {"verdicts": [True]},
        {"claims": ["x"], "verdicts": [True]},
        {"entities": ["kafka"]},
        {"entities": ["kafka"]},
    ])
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
        config=FULL,
    )
    assert report.results[0].grounding.context_precision == pytest.approx(1.0)


# ── Repeatability (Part 2) ────────────────────────────────────────────────────


def _varying_faithfulness_llm() -> ScriptedLLM:
    """Three FAST-mode repetitions with varying verdicts: samples
    [1.0, 0.5, 1.0]."""
    return ScriptedLLM([
        {"claims": ["a", "b"]}, {"verdicts": [True, True]},    # rep 1 -> 1.0
        {"claims": ["a", "b"]}, {"verdicts": [True, False]},   # rep 2 -> 0.5
        {"claims": ["a", "b"]}, {"verdicts": [True, True]},    # rep 3 -> 1.0
    ])


def test_repetition_reports_mean_and_preserves_variance() -> None:
    config = GenerationEvaluationConfig(
        mode=GenerationEvaluationMode.FAST, evaluation_repetitions=3
    )
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=_varying_faithfulness_llm(),
        config=config,
    )

    result = report.results[0]
    # Reported metric is the MEAN of the three samples.
    assert result.grounding.faithfulness == pytest.approx(0.8333, abs=1e-4)
    # Per-query stability preserved.
    stability = result.grounding_stability.faithfulness
    assert stability.count == 3
    assert stability.std_dev == pytest.approx(0.2887, abs=1e-4)
    assert stability.minimum == 0.5
    assert stability.maximum == 1.0
    assert stability.confidence is EvaluatorConfidence.LOW
    # Report-level roll-up.
    assert report.repetitions == 3
    assert report.metric_variance["faithfulness"] == pytest.approx(0.2887, abs=1e-4)
    assert report.metric_confidence["faithfulness"] == "low"


def test_deterministic_judge_yields_zero_variance_high_confidence() -> None:
    config = GenerationEvaluationConfig(
        mode=GenerationEvaluationMode.FAST, evaluation_repetitions=3
    )
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=FakeGroundingLLM(),   # same verdicts every repetition
        config=config,
    )

    stability = report.results[0].grounding_stability.faithfulness
    assert stability.std_dev == pytest.approx(0.0)
    assert stability.confidence is EvaluatorConfidence.HIGH
    assert report.metric_confidence["faithfulness"] == "high"


def test_single_repetition_records_no_stability() -> None:
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=FakeGroundingLLM(),
        config=FAST,
    )
    assert report.results[0].grounding_stability is None
    assert report.repetitions == 1
    assert report.metric_variance is None
    assert report.metric_confidence is None


def test_repetition_failures_are_isolated_and_noted() -> None:
    # Rep 1 succeeds (1.0), rep 2's claims are malformed (fails), rep 3
    # succeeds (1.0): value = mean of 2 samples, stability recorded, note
    # names the failure count.
    llm = ScriptedLLM([
        {"claims": ["a"]}, {"verdicts": [True]},
        {"claims": "MALFORMED"},
        {"claims": ["a"]}, {"verdicts": [True]},
    ])
    config = GenerationEvaluationConfig(
        mode=GenerationEvaluationMode.FAST, evaluation_repetitions=3
    )
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=llm, config=config,
    )

    result = report.results[0]
    assert result.grounding.faithfulness == pytest.approx(1.0)
    assert result.grounding_stability.faithfulness.count == 2
    assert any("faithfulness failed on 1/3 repetition" in n for n in result.notes)


def test_single_surviving_sample_yields_no_stability_with_note() -> None:
    # Two of three repetitions fail: mean over 1 sample, stability None.
    llm = ScriptedLLM([
        {"claims": ["a"]}, {"verdicts": [True]},
        {"claims": "MALFORMED"},
        {"claims": "MALFORMED"},
    ])
    config = GenerationEvaluationConfig(
        mode=GenerationEvaluationMode.FAST, evaluation_repetitions=3
    )
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=llm, config=config,
    )

    result = report.results[0]
    assert result.grounding.faithfulness == pytest.approx(1.0)
    assert result.grounding_stability is None
    assert any("stability unavailable" in n for n in result.notes)


# ── Serialization / persistence / benchmark integration ─────────────────────


def test_benchmark_round_trip_preserves_stability_and_mode(tmp_path: Path) -> None:
    config = GenerationEvaluationConfig(
        mode=GenerationEvaluationMode.FAST, evaluation_repetitions=3
    )
    report = evaluate_generation(
        _referenced_dataset(), _search(), FakeAnswerGenerator(),
        grounding_llm=_varying_faithfulness_llm(), config=config,
    )
    repo = FileGenerationBenchmarkRepository(tmp_path)
    run = create_generation_benchmark_run(experiment_name="exp", report=report)
    repo.save(run)

    loaded = repo.get(run.run_id)

    assert loaded.report.evaluation_mode == "fast"
    assert loaded.report.repetitions == 3
    assert loaded.report.metric_variance["faithfulness"] == pytest.approx(0.2887, abs=1e-4)
    assert loaded.report.metric_confidence["faithfulness"] == "low"
    stability = loaded.report.results[0].grounding_stability.faithfulness
    assert stability.count == 3
    assert stability.confidence is EvaluatorConfidence.LOW


def test_experiment_repository_persists_mode_and_variance(tmp_path: Path) -> None:
    pipeline = _pipeline(
        run_generation=True, generation_mode="fast", generation_repetitions=2
    )
    result = pipeline.run(
        PipelineInputs(
            gold_dataset=_gold_dataset(),
            search_service=FakeSearchService({"broker down": [_result("Kafka broker crash")]}),
            answer_generator=FakeAnswerGenerator(),
            grounding_llm=FakeGroundingLLM(),
            sentence_embedder=FakeSentenceEmbedder(),
        )
    )
    repo = ExperimentRepository(tmp_path)
    run_id = repo.save(result, experiment_name="rel", git_commit="")

    loaded = repo.load(run_id)
    assert loaded.generation_report["evaluation_mode"] == "fast"
    assert loaded.generation_report["repetitions"] == 2
    assert loaded.generation_report["metric_variance"]["faithfulness"] == pytest.approx(0.0)
    assert loaded.generation_report["metric_confidence"]["faithfulness"] == "high"
