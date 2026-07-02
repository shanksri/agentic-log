"""Evaluation Diagnostics & Quality Insights (Phase 22C).

A PURE analysis layer over already-computed evaluation reports. This module
introduces NO new evaluation metrics, changes NO metric formulas, and never
re-runs retrieval, generation, grounding, reasoning, or judging — it only
reads report data and surfaces the engineering questions aggregate numbers
can't answer:

- Which queries are failing?                → Part 1: outlier sections
- Which answers hallucinate the most?      → Part 1 (faithfulness) + Part 5
- Which evaluations are unstable?          → Part 2: stability diagnostics
- Where is evaluation cost going?          → Part 3: cost diagnostics
- Which metrics are skipped, and why?      → Part 4: skip diagnostics
- What deserves investigation first?       → Part 5: EvaluationHealthReport

# Why everything here operates on plain dicts

Every function consumes the JSON-dict view of a report — the exact shape
``app.evaluation.serialization.to_jsonable`` produces and
``ExperimentRepository`` (Phase 21F) round-trips. This is the same
convention Phase 21F's failure filters already use, and it means ONE
implementation serves both a fresh in-memory ``EvaluationPipelineResult``
(caller applies ``to_jsonable`` first) and a persisted run loaded from disk
(already dicts) — no typed/dict adapter duplication.

# Cost estimation contract (Part 3)

The existing services do not track token usage (only the Phase 18D
benchmark script's wrapper does), so per the brief, cost is estimated as
CALL COUNTS ONLY — token values are never fabricated. Counts derive from
the documented per-metric call constants of the Phase 22 implementations
(faithfulness = 2 LLM calls, entity recall = 2, the rest 1 each; answer
relevancy additionally embeds the question plus its 3 generated questions),
multiplied by the run's ``repetitions``. This is an ESTIMATE: a metric that
failed on some repetitions still consumed calls the estimate attributes to
it, and a metric that short-circuited (zero claims) consumed fewer.

# Health classification (Part 5)

``overall_health`` derives from documented thresholds over EXISTING scores
(no new metric): any faithfulness < 0.5 (majority of the answer unsupported)
or any recall_at_k == 0.0 (retrieval found nothing relevant) is a critical
finding → CRITICAL; otherwise any warning (incorrect planner/decision
scenarios, LOW evaluator confidence, LLM-failure skips) → DEGRADED;
otherwise HEALTHY.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

DEFAULT_TOP_N = 5

# Documented call-count constants mirroring the Phase 22 implementations —
# see app.evaluation.grounding_metrics (LLM calls per metric execution) and
# answer_relevancy's question-embedding behavior.
_GROUNDING_LLM_CALLS_PER_EXECUTION = {
    "faithfulness": 2,
    "answer_relevancy": 1,
    "context_precision": 1,
    "context_recall": 1,
    "context_entity_recall": 2,
}
# 1 original question + DEFAULT_RELEVANCY_QUESTION_COUNT generated questions.
_RELEVANCY_EMBEDDING_CALLS_PER_EXECUTION = 4
# BERTScore embeds candidate + reference token sequences.
_BERTSCORE_EMBEDDING_CALLS = 2

_GROUNDING_METRIC_KEYS = tuple(_GROUNDING_LLM_CALLS_PER_EXECUTION)


def _get(data: Mapping[str, Any] | None, *keys: str) -> Any:
    """Nested, None-tolerant dict access."""
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


# ── Part 1: Outlier detection ─────────────────────────────────────────────────


@dataclass(frozen=True)
class OutlierEntry:
    """One worst-performer for one metric."""

    subject_id: str
    query: str | None
    score: float
    reason: str | None
    run_id: str | None
    failure_reference: str | None


@dataclass(frozen=True)
class OutlierSection:
    """The worst-N entries for one metric, ascending by score (worst
    first); ties broken deterministically by ``subject_id``."""

    metric: str
    entries: tuple[OutlierEntry, ...]


def _rank(
    candidates: list[tuple[str, str | None, float, str | None, str | None]],
    *,
    metric: str,
    run_id: str | None,
    top_n: int,
) -> OutlierSection:
    ordered = sorted(candidates, key=lambda item: (item[2], item[0]))
    return OutlierSection(
        metric=metric,
        entries=tuple(
            OutlierEntry(
                subject_id=subject_id,
                query=query,
                score=score,
                reason=reason,
                run_id=run_id,
                failure_reference=failure_reference,
            )
            for subject_id, query, score, reason, failure_reference in ordered[:top_n]
        ),
    )


def detect_outliers(
    *,
    retrieval_report: Mapping[str, Any] | None = None,
    generation_report: Mapping[str, Any] | None = None,
    reasoning_report: Mapping[str, Any] | None = None,
    judge_report: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[OutlierSection, ...]:
    """Rank the worst-performing subjects per metric. Sections are emitted
    only for reports that were supplied; entries only for subjects whose
    metric value is defined (skipped/None values never rank).
    """
    sections: list[OutlierSection] = []

    if retrieval_report is not None:
        per_query = retrieval_report.get("per_query") or []
        for metric_key in ("recall_at_k", "reciprocal_rank", "ndcg_at_k"):
            candidates = []
            for outcome in per_query:
                value = _get(outcome, "metric", metric_key)
                if value is None:
                    continue
                qid = outcome.get("query_id") or "?"
                failure_ref = (
                    f"failure_analysis: subject_id={qid!r}" if value < 1.0 else None
                )
                candidates.append(
                    (qid, None, float(value), outcome.get("skip_reason"), failure_ref)
                )
            sections.append(
                _rank(candidates, metric=metric_key, run_id=run_id, top_n=top_n)
            )

    if generation_report is not None:
        results = generation_report.get("results") or []
        gen_metrics: list[tuple[str, tuple[str, ...]]] = [
            ("bert_score_f1", ("generation", "bert_score_f1")),
        ]
        gen_metrics += [(name, ("grounding", name)) for name in _GROUNDING_METRIC_KEYS]
        for metric_key, path in gen_metrics:
            candidates = []
            for result in results:
                value = _get(result, *path)
                if value is None:
                    continue
                qid = result.get("query_id") or "?"
                reason = None
                notes = result.get("notes") or []
                relevant_notes = [n for n in notes if metric_key in n]
                if relevant_notes:
                    reason = relevant_notes[0]
                failure_ref = (
                    f"failure_analysis: subject_id={qid!r}" if value < 1.0 else None
                )
                candidates.append(
                    (qid, result.get("query"), float(value), reason, failure_ref)
                )
            sections.append(
                _rank(candidates, metric=metric_key, run_id=run_id, top_n=top_n)
            )

    if judge_report is not None:
        evaluations = judge_report.get("judge_evaluations") or []
        # Judge evaluations align index-wise with the reasoning results the
        # pipeline judged (one evaluate_session per InvestigationResult).
        reasoning_results = (
            (reasoning_report or {}).get("results") or [] if reasoning_report else []
        )
        candidates = []
        for index, evaluation in enumerate(evaluations):
            value = _get(evaluation, "score", "value")
            if value is None:
                continue
            if index < len(reasoning_results):
                subject = reasoning_results[index].get("scenario_id") or f"evaluation[{index}]"
                query = reasoning_results[index].get("problem")
            else:
                subject = f"evaluation[{index}]"
                query = None
            band = _get(evaluation, "score", "band")
            failure_ref = (
                f"failure_analysis: subject_id={subject!r}"
                if band in ("Poor", "Weak")
                else None
            )
            candidates.append(
                (subject, query, float(value), evaluation.get("explanation"), failure_ref)
            )
        sections.append(
            _rank(candidates, metric="judge_score", run_id=run_id, top_n=top_n)
        )

    if reasoning_report is not None:
        results = reasoning_report.get("results") or []
        for metric_key, field in (
            ("planner_accuracy", "planner_correct"),
            ("decision_accuracy", "decision_correct"),
        ):
            candidates = []
            for result in results:
                correct = result.get(field)
                if correct is None:
                    continue
                score = 1.0 if correct else 0.0
                subject = result.get("scenario_id") or "?"
                explanation = result.get("explanation") or []
                reason = explanation[0] if explanation else None
                failure_ref = (
                    f"failure_analysis: subject_id={subject!r}" if not correct else None
                )
                candidates.append(
                    (subject, result.get("problem"), score, reason, failure_ref)
                )
            sections.append(
                _rank(candidates, metric=metric_key, run_id=run_id, top_n=top_n)
            )

    return tuple(sections)


# ── Part 2: Evaluation stability diagnostics ──────────────────────────────────


@dataclass(frozen=True)
class UnstableQuery:
    """One (query, metric) stability record, for worst-first ranking."""

    subject_id: str
    query: str | None
    metric: str
    std_dev: float
    confidence: str


@dataclass(frozen=True)
class StabilityDiagnostics:
    """Phase 22B stability information surfaced instead of hidden in
    averages — see module docstring."""

    repetitions: int
    num_measured: int
    mean_std_dev: float | None
    confidence_distribution: dict[str, int]
    most_unstable: tuple[UnstableQuery, ...]


def diagnose_stability(
    generation_report: Mapping[str, Any] | None,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> StabilityDiagnostics | None:
    """Collect every per-query ``MetricStability`` record from the report.
    ``None`` when no generation report exists. All fields empty/None when
    the run used ``repetitions == 1`` (no stability was measured).
    """
    if generation_report is None:
        return None

    records: list[UnstableQuery] = []
    distribution: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for result in generation_report.get("results") or []:
        stability = result.get("grounding_stability")
        if not stability:
            continue
        for metric in _GROUNDING_METRIC_KEYS:
            entry = stability.get(metric)
            if not entry:
                continue
            confidence = str(entry.get("confidence"))
            distribution[confidence] = distribution.get(confidence, 0) + 1
            records.append(
                UnstableQuery(
                    subject_id=result.get("query_id") or "?",
                    query=result.get("query"),
                    metric=metric,
                    std_dev=float(entry.get("std_dev", 0.0)),
                    confidence=confidence,
                )
            )

    # Worst first: descending std_dev, ties broken by subject then metric.
    ranked = sorted(records, key=lambda r: (-r.std_dev, r.subject_id, r.metric))
    return StabilityDiagnostics(
        repetitions=int(generation_report.get("repetitions", 1)),
        num_measured=len(records),
        mean_std_dev=(
            statistics.mean(r.std_dev for r in records) if records else None
        ),
        confidence_distribution=distribution,
        most_unstable=tuple(ranked[:top_n]),
    )


# ── Part 3: Cost diagnostics ──────────────────────────────────────────────────


@dataclass(frozen=True)
class QueryCost:
    subject_id: str
    llm_calls: int
    embedding_calls: int


@dataclass(frozen=True)
class CostDiagnostics:
    """Call-count cost estimate — see module docstring's "Cost estimation
    contract". Token usage is not tracked by the existing services and is
    therefore never reported here.
    """

    total_llm_calls: int
    total_embedding_calls: int
    generation_evaluations: int
    grounding_evaluations: int
    judge_evaluations: int
    skipped_evaluations: int
    llm_calls_by_metric: dict[str, int]
    per_query: tuple[QueryCost, ...]
    note: str


_COST_NOTE = (
    "Estimated from the Phase 22 implementations' documented call-count "
    "constants x repetitions; token usage is not tracked by the existing "
    "services and is not fabricated."
)


def diagnose_cost(
    *,
    generation_report: Mapping[str, Any] | None = None,
    judge_report: Mapping[str, Any] | None = None,
    skipped_evaluations: int = 0,
) -> CostDiagnostics:
    llm_by_metric: dict[str, int] = {}
    per_query: list[QueryCost] = []
    total_llm = 0
    total_embedding = 0

    repetitions = int((generation_report or {}).get("repetitions", 1))
    for result in (generation_report or {}).get("results") or []:
        query_llm = 0
        query_embedding = 0
        answered = result.get("generated_answer") is not None
        failed = "generation failed" in (result.get("skip_reason") or "")
        if answered or failed:
            query_llm += 1
            llm_by_metric["answer_generation"] = (
                llm_by_metric.get("answer_generation", 0) + 1
            )
        if _get(result, "generation", "bert_score_f1") is not None:
            query_embedding += _BERTSCORE_EMBEDDING_CALLS
        grounding = result.get("grounding")
        if grounding:
            for metric, calls in _GROUNDING_LLM_CALLS_PER_EXECUTION.items():
                if grounding.get(metric) is None:
                    continue
                metric_calls = calls * repetitions
                query_llm += metric_calls
                llm_by_metric[metric] = llm_by_metric.get(metric, 0) + metric_calls
                if metric == "answer_relevancy":
                    query_embedding += (
                        _RELEVANCY_EMBEDDING_CALLS_PER_EXECUTION * repetitions
                    )
        total_llm += query_llm
        total_embedding += query_embedding
        per_query.append(
            QueryCost(
                subject_id=result.get("query_id") or "?",
                llm_calls=query_llm,
                embedding_calls=query_embedding,
            )
        )

    judge_evaluations = len((judge_report or {}).get("judge_evaluations") or [])
    if judge_evaluations:
        llm_by_metric["judge"] = judge_evaluations
        total_llm += judge_evaluations

    return CostDiagnostics(
        total_llm_calls=total_llm,
        total_embedding_calls=total_embedding,
        generation_evaluations=int((generation_report or {}).get("num_answered", 0)),
        grounding_evaluations=int(
            (generation_report or {}).get("num_grounding_scored", 0)
        ),
        judge_evaluations=judge_evaluations,
        skipped_evaluations=skipped_evaluations,
        llm_calls_by_metric=llm_by_metric,
        per_query=tuple(per_query),
        note=_COST_NOTE,
    )


# ── Part 4: Skip diagnostics ──────────────────────────────────────────────────

SKIP_REASON_CATEGORIES = (
    "no_reference_answer",
    "no_retrieved_context",
    "metric_disabled_by_mode",
    "grounding_unavailable",
    "missing_embeddings",
    "llm_failures",
)


def _classify_skip(note: str) -> str | None:
    lowered = note.lower()
    if "disabled in" in lowered:
        return "metric_disabled_by_mode"
    if "no grounding llm" in lowered or "no grounding backend" in lowered:
        return "grounding_unavailable"
    if "no sentence embedder" in lowered or "no token embedder" in lowered:
        return "missing_embeddings"
    if "no retrieved context" in lowered:
        return "no_retrieved_context"
    if "no reference_answer" in lowered:
        return "no_reference_answer"
    if "failed" in lowered:
        return "llm_failures"
    return None


@dataclass(frozen=True)
class SkipDiagnostics:
    total_skips: int
    by_reason: dict[str, int]
    percentages: dict[str, float]


def diagnose_skips(
    generation_report: Mapping[str, Any] | None,
) -> SkipDiagnostics:
    """Classify every recorded skip/failure note by reason category and
    report counts plus percentages (of total skips)."""
    by_reason = {category: 0 for category in SKIP_REASON_CATEGORIES}
    total = 0
    for result in (generation_report or {}).get("results") or []:
        notes = list(result.get("notes") or [])
        skip_reason = result.get("skip_reason")
        if skip_reason:
            notes.append(skip_reason)
        for note in notes:
            category = _classify_skip(str(note))
            if category is not None:
                by_reason[category] += 1
                total += 1
    percentages = {
        category: (round(100.0 * count / total, 2) if total else 0.0)
        for category, count in by_reason.items()
    }
    return SkipDiagnostics(
        total_skips=total, by_reason=by_reason, percentages=percentages
    )


# ── Part 5: Health dashboard ──────────────────────────────────────────────────


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


# Documented interpretation thresholds over EXISTING scores (not new
# metrics): a majority-unsupported answer, and a retrieval that found
# nothing relevant.
FAITHFULNESS_CRITICAL_BELOW = 0.5
RECALL_CRITICAL_AT = 0.0


@dataclass(frozen=True)
class EvaluationHealthReport:
    """The Phase 22C dashboard: consumes existing reports, recomputes
    nothing — see module docstring."""

    run_id: str | None
    overall_health: HealthStatus
    critical_findings: tuple[str, ...]
    warnings: tuple[str, ...]
    most_unstable_query: UnstableQuery | None
    worst_hallucination: OutlierEntry | None
    worst_retrieval_query: OutlierEntry | None
    worst_reasoning_scenario: OutlierEntry | None
    total_skipped_metrics: int
    estimated_llm_calls: int
    estimated_embedding_calls: int
    top_recommendations: tuple[dict[str, Any], ...]
    outliers: tuple[OutlierSection, ...]
    stability: StabilityDiagnostics | None
    cost: CostDiagnostics
    skips: SkipDiagnostics


def _section(
    sections: Sequence[OutlierSection], metric: str
) -> OutlierSection | None:
    for section in sections:
        if section.metric == metric:
            return section
    return None


def _worst(sections: Sequence[OutlierSection], metric: str) -> OutlierEntry | None:
    section = _section(sections, metric)
    if section is None or not section.entries:
        return None
    return section.entries[0]


def build_health_report(
    *,
    retrieval_report: Mapping[str, Any] | None = None,
    generation_report: Mapping[str, Any] | None = None,
    reasoning_report: Mapping[str, Any] | None = None,
    judge_report: Mapping[str, Any] | None = None,
    quality_report: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> EvaluationHealthReport:
    """Assemble the full diagnostics dashboard from existing report dicts."""
    outliers = detect_outliers(
        retrieval_report=retrieval_report,
        generation_report=generation_report,
        reasoning_report=reasoning_report,
        judge_report=judge_report,
        run_id=run_id,
        top_n=top_n,
    )
    stability = diagnose_stability(generation_report, top_n=top_n)
    skips = diagnose_skips(generation_report)
    cost = diagnose_cost(
        generation_report=generation_report,
        judge_report=judge_report,
        skipped_evaluations=skips.total_skips,
    )

    critical: list[str] = []
    warnings: list[str] = []

    faithfulness_section = _section(outliers, "faithfulness")
    if faithfulness_section:
        for entry in faithfulness_section.entries:
            if entry.score < FAITHFULNESS_CRITICAL_BELOW:
                critical.append(
                    f"query {entry.subject_id!r}: faithfulness "
                    f"{entry.score:.2f} — majority of the answer is not "
                    "supported by the retrieved context (probable "
                    "hallucination)"
                )

    recall_section = _section(outliers, "recall_at_k")
    if recall_section:
        for entry in recall_section.entries:
            if entry.score == RECALL_CRITICAL_AT:
                critical.append(
                    f"query {entry.subject_id!r}: recall 0.00 — retrieval "
                    "found no relevant incident"
                )

    for metric_key in ("planner_accuracy", "decision_accuracy"):
        section = _section(outliers, metric_key)
        if section:
            for entry in section.entries:
                if entry.score == 0.0:
                    warnings.append(
                        f"scenario {entry.subject_id!r}: {metric_key} 0.0"
                        + (f" — {entry.reason}" if entry.reason else "")
                    )

    if stability is not None:
        for record in stability.most_unstable:
            if record.confidence == "low":
                warnings.append(
                    f"query {record.subject_id!r}: {record.metric} evaluator "
                    f"unstable (std_dev {record.std_dev:.3f}, LOW confidence)"
                )

    if skips.by_reason.get("llm_failures"):
        warnings.append(
            f"{skips.by_reason['llm_failures']} metric execution(s) lost to "
            "LLM failures — see per-query notes"
        )

    if critical:
        overall = HealthStatus.CRITICAL
    elif warnings:
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.HEALTHY

    recommendations = tuple(
        dict(entry) for entry in ((quality_report or {}).get("recommendations") or [])[:top_n]
    )

    return EvaluationHealthReport(
        run_id=run_id,
        overall_health=overall,
        critical_findings=tuple(critical),
        warnings=tuple(warnings),
        most_unstable_query=(
            stability.most_unstable[0]
            if stability and stability.most_unstable
            else None
        ),
        worst_hallucination=_worst(outliers, "faithfulness"),
        worst_retrieval_query=_worst(outliers, "recall_at_k"),
        worst_reasoning_scenario=_worst(outliers, "decision_accuracy"),
        total_skipped_metrics=skips.total_skips,
        estimated_llm_calls=cost.total_llm_calls,
        estimated_embedding_calls=cost.total_embedding_calls,
        top_recommendations=recommendations,
        outliers=outliers,
        stability=stability,
        cost=cost,
        skips=skips,
    )


# ── Benchmark integration: historical trends ──────────────────────────────────


@dataclass(frozen=True)
class TrendPoint:
    run_id: str
    timestamp: str
    value: float | None


@dataclass(frozen=True)
class EvaluationTrends:
    """One point per persisted run (oldest first) per tracked signal;
    ``value`` is ``None`` for runs where that report/section was absent —
    never fabricated."""

    faithfulness: tuple[TrendPoint, ...]
    bert_score: tuple[TrendPoint, ...]
    retrieval_recall: tuple[TrendPoint, ...]
    retrieval_ndcg: tuple[TrendPoint, ...]
    judge_score: tuple[TrendPoint, ...]
    skipped_metrics: tuple[TrendPoint, ...]
    estimated_llm_calls: tuple[TrendPoint, ...]
    evaluator_stability: tuple[TrendPoint, ...]


def compute_evaluation_trends(repository) -> EvaluationTrends:
    """Trend series across an ``ExperimentRepository``'s history (duck-typed:
    needs ``list_runs()`` and ``load(run_id)``). Reads only persisted report
    dicts; recomputes nothing beyond the Phase 22C diagnostics themselves.
    """
    faithfulness: list[TrendPoint] = []
    bert: list[TrendPoint] = []
    recall: list[TrendPoint] = []
    ndcg: list[TrendPoint] = []
    judge: list[TrendPoint] = []
    skipped: list[TrendPoint] = []
    llm_calls: list[TrendPoint] = []
    stability: list[TrendPoint] = []

    for run_identifier in repository.list_runs():
        run = repository.load(run_identifier)
        if run is None:
            continue
        rid = run.metadata.run_id
        timestamp = run.metadata.timestamp
        generation = getattr(run, "generation_report", None)
        retrieval = run.retrieval_report
        judge_report = run.judge_report

        def _point(value: Any) -> TrendPoint:
            return TrendPoint(
                run_id=rid,
                timestamp=timestamp,
                value=float(value) if value is not None else None,
            )

        faithfulness.append(
            _point(_get(generation, "grounding_aggregate", "faithfulness", "mean"))
        )
        bert.append(
            _point(_get(generation, "generation_aggregate", "bert_score_f1", "mean"))
        )
        recall.append(
            _point(_get(retrieval, "aggregate_metrics", "mean_recall_at_k"))
        )
        ndcg.append(_point(_get(retrieval, "aggregate_metrics", "mean_ndcg_at_k")))
        judge.append(
            _point(_get(judge_report, "judge_aggregate", "mean_session_score"))
        )
        skip_diag = diagnose_skips(generation)
        skipped.append(_point(skip_diag.total_skips if generation else None))
        cost = diagnose_cost(
            generation_report=generation,
            judge_report=judge_report,
            skipped_evaluations=skip_diag.total_skips,
        )
        llm_calls.append(
            _point(cost.total_llm_calls if (generation or judge_report) else None)
        )
        variance = _get(generation, "metric_variance") or None
        stability.append(
            _point(
                statistics.mean(variance.values())
                if isinstance(variance, Mapping) and variance
                else None
            )
        )

    return EvaluationTrends(
        faithfulness=tuple(faithfulness),
        bert_score=tuple(bert),
        retrieval_recall=tuple(recall),
        retrieval_ndcg=tuple(ndcg),
        judge_score=tuple(judge),
        skipped_metrics=tuple(skipped),
        estimated_llm_calls=tuple(llm_calls),
        evaluator_stability=tuple(stability),
    )
