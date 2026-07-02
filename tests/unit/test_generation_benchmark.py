"""Tests for Phase 22: Generation Benchmark Integration (persistence +
serialization round-trip, including grounding metrics)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.generation_benchmark import (
    FileGenerationBenchmarkRepository,
    InMemoryGenerationBenchmarkRepository,
    create_generation_benchmark_run,
)
from app.evaluation.generation_harness import (
    GenerationEvaluationReport,
    GenerationQueryResult,
)
from app.evaluation.generation_metrics import (
    GenerationMetrics,
    aggregate_generation_metrics,
)
from app.evaluation.grounding_metrics import (
    GroundingMetrics,
    aggregate_grounding_metrics,
)


def _generation_metrics() -> GenerationMetrics:
    return GenerationMetrics(
        bert_score_precision=0.9, bert_score_recall=0.7, bert_score_f1=0.7875
    )


def _grounding_metrics() -> GroundingMetrics:
    return GroundingMetrics(
        faithfulness=1.0,
        answer_relevancy=0.85,
        context_precision=0.5,
        context_recall=None,          # reference-dependent metric undefined
        context_entity_recall=None,
    )


def _report() -> GenerationEvaluationReport:
    generation = _generation_metrics()
    grounding = _grounding_metrics()
    return GenerationEvaluationReport(
        dataset_version="2.1.0",
        dataset_description="d",
        k=5,
        num_answered=1,
        num_generation_scored=1,
        num_grounding_scored=1,
        num_skipped=1,
        num_failed=0,
        results=(
            GenerationQueryResult(
                query_id="q-1",
                query="broker down",
                generated_answer="restart it",
                reference_answer="restart the broker",
                num_contexts=2,
                generation=generation,
                grounding=grounding,
                skipped=False,
                skip_reason=None,
                notes=("context_recall skipped: no reference_answer",),
            ),
            GenerationQueryResult(
                query_id="q-2",
                query="nothing computable here",
                generated_answer=None,
                reference_answer=None,
                num_contexts=0,
                generation=None,
                grounding=None,
                skipped=True,
                skip_reason="no reference_answer and no grounding backend — nothing to evaluate",
                notes=(),
            ),
        ),
        generation_aggregate=aggregate_generation_metrics([generation]),
        grounding_aggregate=aggregate_grounding_metrics([grounding]),
        started_at="2026-07-02T00:00:00+00:00",
        finished_at="2026-07-02T00:00:01+00:00",
        duration_seconds=1.0,
    )


def test_create_generation_benchmark_run_assigns_id_and_timestamp() -> None:
    run = create_generation_benchmark_run(experiment_name="exp", report=_report())
    assert run.run_id
    assert run.timestamp
    assert run.experiment_name == "exp"
    assert run.report.num_answered == 1


def test_in_memory_repository_save_get_list_latest_delete() -> None:
    repo = InMemoryGenerationBenchmarkRepository()
    run_a = create_generation_benchmark_run(
        experiment_name="exp", report=_report(), timestamp="2026-07-02T01:00:00"
    )
    run_b = create_generation_benchmark_run(
        experiment_name="exp", report=_report(), timestamp="2026-07-02T02:00:00"
    )
    repo.save(run_a)
    repo.save(run_b)

    assert repo.get(run_a.run_id) is run_a
    assert [r.run_id for r in repo.list_runs()] == [run_a.run_id, run_b.run_id]
    assert repo.latest().run_id == run_b.run_id
    assert repo.delete(run_a.run_id) is True
    assert repo.get(run_a.run_id) is None


def test_in_memory_repository_rejects_duplicate_run_id() -> None:
    repo = InMemoryGenerationBenchmarkRepository()
    run = create_generation_benchmark_run(experiment_name="exp", report=_report())
    repo.save(run)
    with pytest.raises(ValueError):
        repo.save(run)


def test_file_repository_round_trips_a_run(tmp_path: Path) -> None:
    repo = FileGenerationBenchmarkRepository(tmp_path)
    run = create_generation_benchmark_run(experiment_name="exp", report=_report())
    repo.save(run)

    loaded = repo.get(run.run_id)

    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.report.dataset_version == "2.1.0"
    assert loaded.report.num_skipped == 1
    # Scored result: both halves survive, including None grounding fields.
    scored = loaded.report.results[0]
    assert scored.generation.bert_score_f1 == pytest.approx(0.7875)
    assert scored.grounding.faithfulness == pytest.approx(1.0)
    assert scored.grounding.context_recall is None
    assert scored.notes == ("context_recall skipped: no reference_answer",)
    # Skipped result: None generation/grounding survive.
    skipped = loaded.report.results[1]
    assert skipped.generation is None
    assert skipped.grounding is None
    assert skipped.skipped is True
    # Both aggregates survive.
    assert loaded.report.generation_aggregate.bert_score_f1.mean == pytest.approx(0.7875)
    assert loaded.report.grounding_aggregate.faithfulness.mean == pytest.approx(1.0)
    assert loaded.report.grounding_aggregate.context_recall is None


def test_file_repository_list_and_latest_ordering(tmp_path: Path) -> None:
    repo = FileGenerationBenchmarkRepository(tmp_path)
    older = create_generation_benchmark_run(
        experiment_name="exp", report=_report(), timestamp="2026-07-02T01:00:00"
    )
    newer = create_generation_benchmark_run(
        experiment_name="exp", report=_report(), timestamp="2026-07-02T02:00:00"
    )
    repo.save(newer)
    repo.save(older)

    assert [r.run_id for r in repo.list_runs()] == [older.run_id, newer.run_id]
    assert repo.latest().run_id == newer.run_id
