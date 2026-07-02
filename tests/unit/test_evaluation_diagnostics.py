"""Tests for Phase 22C: Evaluation Diagnostics & Quality Insights.

All inputs are plain report dicts — built either by hand or by running the
real Phase 22 harness with deterministic fakes and converting via the real
``to_jsonable`` (which also proves the dict shapes diagnostics consume are
exactly what serialization produces).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.evaluation_diagnostics import (
    DEFAULT_TOP_N,
    HealthStatus,
    build_health_report,
    compute_evaluation_trends,
    detect_outliers,
    diagnose_cost,
    diagnose_skips,
    diagnose_stability,
)
from app.evaluation.evaluation_pipeline import PipelineInputs
from app.evaluation.experiment_tracking import ExperimentRepository
from app.evaluation.generation_harness import (
    GenerationEvaluationConfig,
    GenerationEvaluationMode,
    evaluate_generation,
)
from app.evaluation.serialization import to_jsonable
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


# ── Dict-report builders ──────────────────────────────────────────────────────


def _retrieval_dict(per_query: list[dict]) -> dict:
    return {
        "per_query": per_query,
        "aggregate_metrics": {"mean_recall_at_k": 0.8, "mean_ndcg_at_k": 0.7},
    }


def _retrieval_outcome(qid: str, *, recall: float, mrr: float = 0.5, ndcg: float = 0.5) -> dict:
    return {
        "query_id": qid,
        "skipped": False,
        "skip_reason": None,
        "metric": {"recall_at_k": recall, "reciprocal_rank": mrr, "ndcg_at_k": ndcg},
    }


def _generation_result_dict(
    qid: str,
    *,
    faithfulness: float | None = None,
    bert: float | None = None,
    stability: dict | None = None,
    notes: list[str] | None = None,
    answered: bool = True,
    relevancy: float | None = None,
) -> dict:
    return {
        "query_id": qid,
        "query": f"query text for {qid}",
        "generated_answer": "an answer" if answered else None,
        "reference_answer": "a reference",
        "num_contexts": 1,
        "generation": {"bert_score_f1": bert} if bert is not None else None,
        "grounding": {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": None,
            "context_recall": None,
            "context_entity_recall": None,
        },
        "skipped": False,
        "skip_reason": None,
        "notes": notes or [],
        "grounding_stability": stability,
    }


def _generation_dict(results: list[dict], *, repetitions: int = 1) -> dict:
    return {
        "results": results,
        "num_answered": sum(1 for r in results if r["generated_answer"]),
        "num_grounding_scored": sum(1 for r in results if r["grounding"]),
        "repetitions": repetitions,
        "evaluation_mode": "fast",
        "metric_variance": None,
        "metric_confidence": None,
        "generation_aggregate": {"bert_score_f1": {"mean": 0.9}},
        "grounding_aggregate": {"faithfulness": {"mean": 0.8}},
    }


def _stability_entry(std: float, confidence: str) -> dict:
    return {
        "count": 3, "mean": 0.8, "std_dev": std,
        "minimum": 0.5, "maximum": 1.0, "confidence": confidence,
    }


# ── Part 1: Outlier detection ─────────────────────────────────────────────────


def test_outliers_rank_worst_first_across_multiple() -> None:
    report = _retrieval_dict([
        _retrieval_outcome("q-good", recall=1.0),
        _retrieval_outcome("q-bad", recall=0.2),
        _retrieval_outcome("q-mid", recall=0.5),
    ])
    sections = detect_outliers(retrieval_report=report, run_id="run-1")
    recall_section = next(s for s in sections if s.metric == "recall_at_k")
    assert [e.subject_id for e in recall_section.entries] == ["q-bad", "q-mid", "q-good"]
    assert recall_section.entries[0].score == 0.2
    assert recall_section.entries[0].run_id == "run-1"
    # Failing queries link to failure analysis; perfect ones don't.
    assert recall_section.entries[0].failure_reference is not None
    assert recall_section.entries[2].failure_reference is None


def test_outliers_ties_broken_deterministically_by_subject_id() -> None:
    report = _retrieval_dict([
        _retrieval_outcome("q-z", recall=0.5),
        _retrieval_outcome("q-a", recall=0.5),
    ])
    sections = detect_outliers(retrieval_report=report)
    recall_section = next(s for s in sections if s.metric == "recall_at_k")
    assert [e.subject_id for e in recall_section.entries] == ["q-a", "q-z"]


def test_outliers_respect_top_n() -> None:
    report = _retrieval_dict(
        [_retrieval_outcome(f"q-{i}", recall=i / 10) for i in range(10)]
    )
    sections = detect_outliers(retrieval_report=report, top_n=3)
    recall_section = next(s for s in sections if s.metric == "recall_at_k")
    assert len(recall_section.entries) == 3


def test_outliers_skip_undefined_values() -> None:
    outcome_without_metric = {"query_id": "q-skipped", "skipped": True,
                              "skip_reason": "search failed", "metric": None}
    report = _retrieval_dict([outcome_without_metric, _retrieval_outcome("q-1", recall=0.9)])
    sections = detect_outliers(retrieval_report=report)
    recall_section = next(s for s in sections if s.metric == "recall_at_k")
    assert [e.subject_id for e in recall_section.entries] == ["q-1"]


def test_outliers_generation_grounding_and_reasoning_sections() -> None:
    generation = _generation_dict([
        _generation_result_dict("q-hallucinated", faithfulness=0.1, bert=0.4),
        _generation_result_dict("q-faithful", faithfulness=1.0, bert=0.95),
    ])
    reasoning = {
        "results": [
            {"scenario_id": "s-1", "problem": "p1", "planner_correct": False,
             "decision_correct": True, "explanation": ["planner picked NETWORK"]},
            {"scenario_id": "s-2", "problem": "p2", "planner_correct": True,
             "decision_correct": False, "explanation": []},
        ],
    }
    judge = {
        "judge_evaluations": [
            {"stage": "session", "score": {"value": 3.0, "band": "Weak"},
             "explanation": "weak reasoning"},
            {"stage": "session", "score": {"value": 8.0, "band": "Excellent"},
             "explanation": "solid"},
        ],
    }
    sections = detect_outliers(
        generation_report=generation, reasoning_report=reasoning, judge_report=judge
    )
    by_metric = {s.metric: s for s in sections}
    # All twelve-minus-retrieval sections present for the supplied reports.
    assert set(by_metric) >= {
        "bert_score_f1", "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "context_entity_recall", "judge_score",
        "planner_accuracy", "decision_accuracy",
    }
    assert by_metric["faithfulness"].entries[0].subject_id == "q-hallucinated"
    assert by_metric["bert_score_f1"].entries[0].score == 0.4
    # Judge outliers map index-wise onto reasoning scenario ids.
    assert by_metric["judge_score"].entries[0].subject_id == "s-1"
    assert by_metric["judge_score"].entries[0].reason == "weak reasoning"
    # Planner/decision incorrectness ranks first with reason + failure link.
    assert by_metric["planner_accuracy"].entries[0].subject_id == "s-1"
    assert by_metric["planner_accuracy"].entries[0].reason == "planner picked NETWORK"
    assert by_metric["decision_accuracy"].entries[0].subject_id == "s-2"
    assert by_metric["decision_accuracy"].entries[0].failure_reference is not None


# ── Part 2: Stability diagnostics ─────────────────────────────────────────────


def test_stability_ranking_and_distribution() -> None:
    generation = _generation_dict(
        [
            _generation_result_dict(
                "q-stable", faithfulness=0.9,
                stability={"faithfulness": _stability_entry(0.01, "high")},
            ),
            _generation_result_dict(
                "q-unstable", faithfulness=0.5,
                stability={"faithfulness": _stability_entry(0.3, "low")},
            ),
            _generation_result_dict(
                "q-mid", faithfulness=0.7,
                stability={"faithfulness": _stability_entry(0.07, "medium")},
            ),
        ],
        repetitions=3,
    )
    diagnostics = diagnose_stability(generation)
    assert diagnostics.repetitions == 3
    assert diagnostics.num_measured == 3
    assert diagnostics.mean_std_dev == pytest.approx((0.01 + 0.3 + 0.07) / 3)
    assert diagnostics.confidence_distribution == {"high": 1, "medium": 1, "low": 1}
    # Worst-first, not hidden in averages.
    assert diagnostics.most_unstable[0].subject_id == "q-unstable"
    assert diagnostics.most_unstable[0].std_dev == pytest.approx(0.3)
    assert diagnostics.most_unstable[0].confidence == "low"


def test_stability_none_without_generation_report() -> None:
    assert diagnose_stability(None) is None


def test_stability_empty_for_single_repetition_run() -> None:
    generation = _generation_dict(
        [_generation_result_dict("q-1", faithfulness=0.9)], repetitions=1
    )
    diagnostics = diagnose_stability(generation)
    assert diagnostics.num_measured == 0
    assert diagnostics.mean_std_dev is None
    assert diagnostics.most_unstable == ()


# ── Part 3: Cost diagnostics ──────────────────────────────────────────────────


def test_cost_aggregation_hand_computed_full_mode() -> None:
    # One query, FULL grounding, 1 repetition, all metrics defined, judged:
    # LLM: 1 answer + 2 faithfulness + 1 relevancy + 1 precision + 1 recall
    #      + 2 entity = 8; + 1 judge = 9.
    # Embedding: 2 BERTScore + 4 relevancy = 6.
    result = _generation_result_dict("q-1", faithfulness=1.0, bert=0.9, relevancy=0.8)
    result["grounding"]["context_precision"] = 1.0
    result["grounding"]["context_recall"] = 1.0
    result["grounding"]["context_entity_recall"] = 1.0
    generation = _generation_dict([result])
    judge = {"judge_evaluations": [{"stage": "session", "score": {"value": 8.0}}]}

    cost = diagnose_cost(generation_report=generation, judge_report=judge,
                         skipped_evaluations=2)

    assert cost.total_llm_calls == 9
    assert cost.total_embedding_calls == 6
    assert cost.llm_calls_by_metric == {
        "answer_generation": 1, "faithfulness": 2, "answer_relevancy": 1,
        "context_precision": 1, "context_recall": 1, "context_entity_recall": 2,
        "judge": 1,
    }
    assert cost.per_query[0].subject_id == "q-1"
    assert cost.per_query[0].llm_calls == 8
    assert cost.per_query[0].embedding_calls == 6
    assert cost.judge_evaluations == 1
    assert cost.skipped_evaluations == 2
    assert "token usage" in cost.note  # never fabricated


def test_cost_scales_with_repetitions() -> None:
    result = _generation_result_dict("q-1", faithfulness=1.0)
    generation = _generation_dict([result], repetitions=3)
    cost = diagnose_cost(generation_report=generation)
    # 1 answer + 2 faithfulness x 3 reps = 7.
    assert cost.total_llm_calls == 7
    assert cost.llm_calls_by_metric["faithfulness"] == 6


def test_cost_empty_reports() -> None:
    cost = diagnose_cost()
    assert cost.total_llm_calls == 0
    assert cost.total_embedding_calls == 0
    assert cost.per_query == ()


# ── Part 4: Skip diagnostics ──────────────────────────────────────────────────


def test_skip_aggregation_with_percentages() -> None:
    generation = _generation_dict([
        _generation_result_dict("q-1", notes=[
            "bert_score skipped: no reference_answer",
            "context_recall skipped: no reference_answer",
            "answer_relevancy skipped: disabled in 'fast' mode",
            "grounding skipped: no retrieved context",
        ]),
        _generation_result_dict("q-2", notes=[
            "answer_relevancy skipped: no sentence embedder configured",
            "faithfulness failed on 1/3 repetition(s): GroundingResponseError",
            "grounding skipped: no grounding LLM configured",
        ]),
    ])
    skips = diagnose_skips(generation)
    assert skips.total_skips == 7
    assert skips.by_reason["no_reference_answer"] == 2
    assert skips.by_reason["metric_disabled_by_mode"] == 1
    assert skips.by_reason["no_retrieved_context"] == 1
    assert skips.by_reason["missing_embeddings"] == 1
    assert skips.by_reason["llm_failures"] == 1
    assert skips.by_reason["grounding_unavailable"] == 1
    assert skips.percentages["no_reference_answer"] == pytest.approx(28.57, abs=0.01)
    assert sum(skips.percentages.values()) == pytest.approx(100.0, abs=0.1)


def test_skip_diagnostics_empty() -> None:
    skips = diagnose_skips(None)
    assert skips.total_skips == 0
    assert all(v == 0.0 for v in skips.percentages.values())


# ── Part 5: Health dashboard ──────────────────────────────────────────────────


def test_perfect_evaluation_is_healthy() -> None:
    retrieval = _retrieval_dict([_retrieval_outcome("q-1", recall=1.0, mrr=1.0, ndcg=1.0)])
    generation = _generation_dict(
        [_generation_result_dict("q-1", faithfulness=1.0, bert=1.0)]
    )
    health = build_health_report(
        retrieval_report=retrieval, generation_report=generation, run_id="run-ok"
    )
    assert health.overall_health is HealthStatus.HEALTHY
    assert health.critical_findings == ()
    assert health.warnings == ()
    assert health.worst_retrieval_query.score == 1.0
    assert health.worst_hallucination.score == 1.0
    assert health.run_id == "run-ok"


def test_hallucination_and_zero_recall_are_critical() -> None:
    retrieval = _retrieval_dict([_retrieval_outcome("q-nothing", recall=0.0)])
    generation = _generation_dict(
        [_generation_result_dict("q-halluc", faithfulness=0.2, bert=0.3)]
    )
    health = build_health_report(
        retrieval_report=retrieval, generation_report=generation
    )
    assert health.overall_health is HealthStatus.CRITICAL
    assert any("hallucination" in f for f in health.critical_findings)
    assert any("recall 0.00" in f for f in health.critical_findings)
    assert health.worst_hallucination.subject_id == "q-halluc"
    assert health.worst_retrieval_query.subject_id == "q-nothing"


def test_warnings_only_yield_degraded() -> None:
    reasoning = {
        "results": [
            {"scenario_id": "s-1", "problem": "p", "planner_correct": False,
             "decision_correct": True, "explanation": ["wrong strategy"]},
        ],
    }
    health = build_health_report(reasoning_report=reasoning)
    assert health.overall_health is HealthStatus.DEGRADED
    assert any("planner_accuracy 0.0" in w for w in health.warnings)
    assert health.critical_findings == ()


def test_dashboard_pulls_recommendations_and_stability() -> None:
    generation = _generation_dict(
        [
            _generation_result_dict(
                "q-1", faithfulness=0.9,
                stability={"faithfulness": _stability_entry(0.2, "low")},
            ),
        ],
        repetitions=3,
    )
    quality = {"recommendations": [
        {"problem": "p1", "recommended_action": "fix retrieval"},
        {"problem": "p2", "recommended_action": "fix judge"},
    ]}
    health = build_health_report(
        generation_report=generation, quality_report=quality
    )
    assert health.most_unstable_query.subject_id == "q-1"
    assert any("evaluator unstable" in w for w in health.warnings)
    assert health.top_recommendations[0]["recommended_action"] == "fix retrieval"
    assert health.total_skipped_metrics == health.skips.total_skips
    assert health.estimated_llm_calls == health.cost.total_llm_calls


def test_dashboard_serializes_to_json() -> None:
    health = build_health_report(
        retrieval_report=_retrieval_dict([_retrieval_outcome("q-1", recall=0.5)]),
        generation_report=_generation_dict(
            [_generation_result_dict("q-1", faithfulness=0.9, bert=0.8)]
        ),
    )
    payload = to_jsonable(health)
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["overall_health"] == "degraded" or round_tripped["overall_health"] in (
        "healthy", "critical",
    )
    assert round_tripped["cost"]["note"]
    assert isinstance(round_tripped["outliers"], list)


# ── End-to-end against REAL harness output ────────────────────────────────────


def test_diagnostics_on_real_harness_report_via_to_jsonable() -> None:
    """Prove the dict shapes diagnostics consume are exactly what the real
    Phase 22 harness + serialization produce."""
    llm = ScriptedLLM([
        {"claims": ["a", "b"]}, {"verdicts": [True, False]},   # faithfulness 0.5
        {"questions": ["broker down"]},                          # relevancy 1.0
    ])
    config = GenerationEvaluationConfig(mode=GenerationEvaluationMode.STANDARD)
    report = evaluate_generation(
        _dataset((_gold_query(reference="restart the kafka broker"),)),
        FakeSearchService({"broker down": [_result("Kafka broker crash")]}),
        FakeAnswerGenerator(),
        grounding_llm=llm, sentence_embedder=KeywordEmbedder(),
        config=config,
    )
    generation_dict = to_jsonable(report)

    health = build_health_report(generation_report=generation_dict)
    assert health.worst_hallucination.score == pytest.approx(0.5)
    assert health.skips.by_reason["metric_disabled_by_mode"] == 3  # precision/recall/entity
    # STANDARD, 1 rep: 1 answer + 2 faithfulness + 1 relevancy = 4 LLM calls.
    assert health.cost.total_llm_calls == 4


# ── Historical trends + persistence ───────────────────────────────────────────


def _persist_two_runs(tmp_path: Path) -> ExperimentRepository:
    repo = ExperimentRepository(tmp_path)
    for run_index in range(2):
        pipeline = _pipeline(
            run_generation=True, generation_mode="fast", generation_repetitions=2
        )
        result = pipeline.run(
            PipelineInputs(
                gold_dataset=_gold_dataset(),
                search_service=FakeSearchService(
                    {"broker down": [_result("Kafka broker crash")]}
                ),
                answer_generator=FakeAnswerGenerator(),
                grounding_llm=FakeGroundingLLM(),
                sentence_embedder=FakeSentenceEmbedder(),
            )
        )
        repo.save(result, experiment_name=f"trend-{run_index}", git_commit="")
    return repo


def test_trends_across_persisted_history(tmp_path: Path) -> None:
    repo = _persist_two_runs(tmp_path)
    trends = compute_evaluation_trends(repo)

    assert len(trends.faithfulness) == 2
    assert all(p.value == pytest.approx(1.0) for p in trends.faithfulness)
    # No token embedder in these runs -> BERTScore undefined, never fabricated.
    assert all(p.value is None for p in trends.bert_score)
    # No retrieval stage ran -> retrieval trend points exist with None values.
    assert all(p.value is None for p in trends.retrieval_recall)
    assert len(trends.estimated_llm_calls) == 2
    assert all(p.value > 0 for p in trends.estimated_llm_calls)
    # Deterministic grounding fake -> zero variance stability trend.
    assert all(p.value == pytest.approx(0.0) for p in trends.evaluator_stability)
    # Points carry run identity for plotting.
    assert trends.faithfulness[0].run_id != trends.faithfulness[1].run_id
    assert trends.faithfulness[0].timestamp <= trends.faithfulness[1].timestamp


def test_trends_empty_repository(tmp_path: Path) -> None:
    trends = compute_evaluation_trends(ExperimentRepository(tmp_path))
    assert trends.faithfulness == ()
    assert trends.estimated_llm_calls == ()


def test_diagnostics_on_persisted_run(tmp_path: Path) -> None:
    """Benchmark persistence integration: the dashboard builds directly from
    a run loaded off disk."""
    repo = _persist_two_runs(tmp_path)
    run_id = repo.list_runs()[-1]
    run = repo.load(run_id)

    health = build_health_report(
        retrieval_report=run.retrieval_report,
        generation_report=run.generation_report,
        reasoning_report=run.reasoning_report,
        judge_report=run.judge_report,
        quality_report=run.quality_report,
        run_id=run_id,
    )
    assert health.run_id == run_id
    assert health.overall_health is HealthStatus.HEALTHY
    assert health.cost.total_llm_calls > 0
