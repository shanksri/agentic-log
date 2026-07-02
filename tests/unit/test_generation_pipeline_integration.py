"""Tests for Phase 22A: generation-stage integration into the Phase 21E
pipeline and Phase 21F experiment tracking.

Backward compatibility is the headline concern: generation is the only
opt-in stage (default off), pre-22A pipeline construction must behave
identically, and runs persisted before generation existed must still load.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
    EvaluationPipelineConfig,
    PipelineInputs,
    PipelineRepositories,
)
from app.evaluation.experiment_tracking import ExperimentRepository
from app.evaluation.generation_benchmark import InMemoryGenerationBenchmarkRepository
from app.evaluation.gold_dataset import ExpectedIncident, GoldDataset, GoldQuery


# ── Fakes ─────────────────────────────────────────────────────────────────────


def _incident(title: str) -> SimpleNamespace:
    return SimpleNamespace(title=title, resolution_summary="restarted", status="resolved")


class FakeSearchService:
    def search(self, query, *, limit=10, call_site=None):
        return [SimpleNamespace(incident=_incident("Kafka broker crash"))]


class FakeAnswerGenerator:
    def generate_answer(self, query: str, context: str) -> str:
        return "restart the kafka broker"


class FakeGroundingLLM:
    """Answers every grounding prompt with a fully-supportive verdict shape,
    keyed off the system prompt's distinctive wording — enough to drive the
    real grounding metrics deterministically through the pipeline.
    """

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        if "decompose an answer" in system_prompt:
            return {"claims": ["restart the kafka broker"]}
        if "verify claims" in system_prompt:
            return {"verdicts": [True]}
        if "questions" in system_prompt:
            return {"questions": ["broker down"]}
        if "useful" in system_prompt:
            return {"verdicts": [True]}
        if "reference answer into atomic" in system_prompt:
            return {"claims": ["restart fixes it"], "verdicts": [True]}
        if "extract named entities" in system_prompt:
            return {"entities": ["kafka"]}
        raise AssertionError(f"unexpected grounding prompt: {system_prompt[:60]}")


class FakeSentenceEmbedder:
    def embed_text(self, text: str):
        words = text.lower().split()
        return [1.0 if w in words else 0.0 for w in ("broker", "down", "restart", "kafka")]


def _gold_dataset(*, with_reference: bool = True) -> GoldDataset:
    return GoldDataset(
        version="2.1.0",
        description="gen pipeline test",
        created_at="2026-07-02T00:00:00Z",
        queries=(
            GoldQuery(
                id="q-1",
                query="broker down",
                category="lexical-overlap",
                difficulty="easy",
                expected_incidents=(
                    ExpectedIncident(
                        source_type="github", source_external_id="a/b#1", relevance=3
                    ),
                ),
                reference_answer="restart the kafka broker" if with_reference else None,
            ),
        ),
    )


def _pipeline(
    *, run_generation: bool, generation_repo=None,
    generation_mode: str = "full", generation_repetitions: int = 1,
) -> EvaluationPipeline:
    return EvaluationPipeline(
        config=EvaluationPipelineConfig(
            # Isolate the generation stage: every other stage off.
            run_retrieval=False,
            run_reasoning=False,
            run_judge=False,
            run_failure_analysis=False,
            run_validation=False,
            run_generation=run_generation,
            generation_mode=generation_mode,
            generation_repetitions=generation_repetitions,
        ),
        repositories=PipelineRepositories(generation_repo=generation_repo),
    )


# ── Pipeline stage behavior ───────────────────────────────────────────────────


def test_generation_stage_off_by_default_no_report_no_warning() -> None:
    # Default config: run_generation must be False (opt-in, unlike every
    # other stage) and its absence must add no warning noise.
    config = EvaluationPipelineConfig()
    assert config.run_generation is False

    pipeline = EvaluationPipeline(
        config=EvaluationPipelineConfig(
            run_retrieval=False, run_reasoning=False, run_judge=False,
            run_failure_analysis=False, run_validation=False,
        )
    )
    result = pipeline.run(PipelineInputs(gold_dataset=_gold_dataset()))

    assert result.generation_report is None
    assert result.generation_benchmark is None
    assert not any("eneration" in w for w in result.execution_summary.warnings)
    assert result.execution_summary.generation_queries == 0


def test_generation_stage_runs_and_persists_when_enabled() -> None:
    repo = InMemoryGenerationBenchmarkRepository()
    pipeline = _pipeline(run_generation=True, generation_repo=repo)

    result = pipeline.run(
        PipelineInputs(
            gold_dataset=_gold_dataset(),
            search_service=FakeSearchService(),
            answer_generator=FakeAnswerGenerator(),
        )
    )

    assert result.generation_report is not None
    assert result.generation_report.num_answered == 1
    assert result.generation_report.num_generation_scored == 1
    # BERTScore fields undefined without a token embedder (never fabricated).
    assert result.generation_report.generation_aggregate.bert_score_f1 is None
    assert result.execution_summary.generation_queries == 1
    # Persisted to the generation repository.
    assert result.generation_benchmark is not None
    assert repo.latest().run_id == result.generation_benchmark.run_id


def test_generation_skipped_with_warning_when_generator_missing() -> None:
    pipeline = _pipeline(run_generation=True)
    result = pipeline.run(
        PipelineInputs(gold_dataset=_gold_dataset(), search_service=FakeSearchService())
    )
    assert result.generation_report is None
    assert any("no answer_generator" in w for w in result.execution_summary.warnings)


def test_generation_skipped_with_warning_when_gold_dataset_missing() -> None:
    pipeline = _pipeline(run_generation=True)
    result = pipeline.run(
        PipelineInputs(
            search_service=FakeSearchService(), answer_generator=FakeAnswerGenerator()
        )
    )
    assert result.generation_report is None
    assert any("no gold_dataset" in w for w in result.execution_summary.warnings)


def test_generation_dataset_without_references_scores_nothing() -> None:
    pipeline = _pipeline(run_generation=True)
    result = pipeline.run(
        PipelineInputs(
            gold_dataset=_gold_dataset(with_reference=False),
            search_service=FakeSearchService(),
            answer_generator=FakeAnswerGenerator(),
        )
    )
    assert result.generation_report is not None
    # No reference and no grounding backend: nothing computable, skipped.
    assert result.generation_report.num_answered == 0
    assert result.generation_report.num_skipped == 1
    assert result.execution_summary.generation_queries == 0


def test_generation_stage_error_is_recorded_not_raised() -> None:
    class ExplodingSearch:
        def search(self, *a, **k):
            raise RuntimeError("db down")

    # Per-query failures are isolated inside the harness (num_failed), so
    # the pipeline-level report still exists.
    pipeline = _pipeline(run_generation=True)
    result = pipeline.run(
        PipelineInputs(
            gold_dataset=_gold_dataset(),
            search_service=ExplodingSearch(),
            answer_generator=FakeAnswerGenerator(),
        )
    )
    assert result.generation_report is not None
    assert result.generation_report.num_failed == 1
    assert result.generation_report.num_answered == 0


def test_grounding_backends_flow_through_pipeline_end_to_end() -> None:
    """With grounding_llm + sentence_embedder supplied, the real RAGAS-style
    metrics compute inside the pipeline stage and land in the aggregate.
    """
    pipeline = _pipeline(run_generation=True)
    result = pipeline.run(
        PipelineInputs(
            gold_dataset=_gold_dataset(),
            search_service=FakeSearchService(),
            answer_generator=FakeAnswerGenerator(),
            grounding_llm=FakeGroundingLLM(),
            sentence_embedder=FakeSentenceEmbedder(),
        )
    )

    report = result.generation_report
    assert report is not None
    assert report.num_grounding_scored == 1
    assert report.grounding_aggregate.faithfulness.mean == pytest.approx(1.0)
    assert report.grounding_aggregate.answer_relevancy.mean == pytest.approx(1.0)
    assert report.grounding_aggregate.context_precision.mean == pytest.approx(1.0)
    assert report.grounding_aggregate.context_recall.mean == pytest.approx(1.0)
    assert report.grounding_aggregate.context_entity_recall.mean == pytest.approx(1.0)


# ── Experiment tracking (Phase 21F) integration ───────────────────────────────


def _run_pipeline_result(*, generation: bool):
    pipeline = _pipeline(
        run_generation=generation,
        generation_repo=InMemoryGenerationBenchmarkRepository(),
    )
    return pipeline.run(
        PipelineInputs(
            gold_dataset=_gold_dataset(),
            search_service=FakeSearchService(),
            answer_generator=FakeAnswerGenerator(),
            grounding_llm=FakeGroundingLLM(),
            sentence_embedder=FakeSentenceEmbedder(),
        )
    )


def test_experiment_repository_persists_and_loads_generation_report(tmp_path: Path) -> None:
    repo = ExperimentRepository(tmp_path)
    result = _run_pipeline_result(generation=True)

    run_id = repo.save(result, experiment_name="gen", git_commit="")

    assert (tmp_path / "history" / run_id / "generation_report.json").exists()
    loaded = repo.load(run_id)
    assert loaded.generation_report is not None
    assert loaded.generation_report["num_answered"] == 1
    assert loaded.generation_report["grounding_aggregate"]["faithfulness"]["mean"] == pytest.approx(1.0)
    # latest/ mirror carries it too.
    assert repo.latest().generation_report is not None


def test_experiment_repository_loads_pre_generation_runs_as_none(tmp_path: Path) -> None:
    # A run persisted WITHOUT generation (the pre-22A shape: no
    # generation_report.json on disk) must load with generation_report=None.
    repo = ExperimentRepository(tmp_path)
    result = _run_pipeline_result(generation=False)

    run_id = repo.save(result, experiment_name="old-shape", git_commit="")

    assert not (tmp_path / "history" / run_id / "generation_report.json").exists()
    loaded = repo.load(run_id)
    assert loaded.generation_report is None
