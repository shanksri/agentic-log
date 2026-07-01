"""AI Quality Intelligence — AI Quality Report & Benchmark Integration
(Phase 21A).

Assembles ``app.evaluation.failure_analysis``'s failures/clusters and
``app.evaluation.recommendation_engine``'s recommendations into one
immutable ``AIQualityReport``, and reuses (never duplicates) Phase
16F/20A/20B's existing benchmark repositories to build a report from
stored runs rather than freshly-supplied report objects.

# Updated architecture (final stage)

```
analyze_retrieval_failures() / analyze_reasoning_failures() /
analyze_judge_failures()                       [failure_analysis.py]
            │
            ▼
cluster_failures() -> tuple[FailureCluster, ...]   [failure_analysis.py]
            │
            ▼
generate_recommendations() -> tuple[Recommendation, ...]
                                                [recommendation_engine.py]
            │
            ▼
build_quality_report(...) -> AIQualityReport        (THIS MODULE)
```

# Benchmark integration

``build_quality_report_from_benchmarks`` accepts the SAME repository
interfaces Phase 16F (``BenchmarkRepository``)/20A
(``ReasoningBenchmarkRepository``)/20B
(``JudgedReasoningBenchmarkRepository``) already define — it calls only
their existing public methods (``.latest()``/``.list_runs()``), never
introduces a new storage mechanism, and never mutates any stored run.
Three usage shapes are supported, all via the same function:

- **single benchmark** — pass only ``retrieval_repo`` (or only
  ``reasoning_repo``/``judged_repo``) with no ``history`` flag; the latest
  run's report is analyzed alone.
- **multiple benchmark runs** — pass ``include_history=True``; every
  stored run's report (oldest to newest, via ``.list_runs()``) is
  analyzed, and per-run failure counts feed ``TrendSummary`` (see below).
- **regression history** — if the latest retrieval/reasoning run carries
  an embedded ``RegressionReport``/``ReasoningRegressionReport``
  (Phase 16E/20A, already computed by whatever produced that run), its
  verdict is folded directly into ``TrendSummary.regression_verdict``
  rather than this module re-deriving a verdict of its own.

# Trend summary

``TrendSummary`` is populated ONLY when at least two historical runs are
available (``include_history=True`` with >= 2 stored runs) — with a
single run there is no trend to report, and this module returns
``trend_summary=None`` rather than fabricating a one-point "trend."
``failure_count_trend`` is the literal sequence of total failure counts
across the analyzed historical runs, oldest first — a reader can see
whether the count is rising, falling, or flat without this module judging
which direction is "good" (fewer failures is presumably better, but this
module does not embed that value judgment as a verdict; the literal
sequence is the evidence).

# Explainability

Every field on ``AIQualityReport`` traces back to a ``FailureRecord`` (via
``failure_clusters``) or a ``Recommendation`` (itself built from a
cluster) - "nothing should appear without evidence" is satisfied
structurally: ``build_quality_report`` never adds a component summary,
cluster, or recommendation that doesn't correspond to at least one
``FailureRecord`` produced by ``failure_analysis``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from app.evaluation.benchmark import BenchmarkRepository
from app.evaluation.failure_analysis import (
    Component,
    FailureCluster,
    FailureRecord,
    FailureSummary,
    SeverityFailureCount,
    analyze_judge_failures,
    analyze_reasoning_failures,
    analyze_retrieval_failures,
    cluster_failures,
    summarize_failures,
)
from app.evaluation.judge import JudgeEvaluation
from app.evaluation.judge_benchmark import JudgedReasoningBenchmarkRepository
from app.evaluation.reasoning_benchmark import ReasoningBenchmarkRepository
from app.evaluation.recommendation_engine import Recommendation, generate_recommendations


# ── Report data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComponentSummary:
    component: Component
    total_failures: int
    severity_breakdown: tuple[SeverityFailureCount, ...]


@dataclass(frozen=True)
class TrendSummary:
    """Populated only with >= 2 historical runs - see module docstring's
    "Trend summary". ``regression_verdict`` is carried through verbatim
    from an already-computed ``RegressionReport``/``ReasoningRegressionReport``
    when one was supplied, never re-derived.
    """

    failure_count_trend: tuple[int, ...]
    regression_verdict: str | None


@dataclass(frozen=True)
class AIQualityReport:
    generated_at: str
    overall_summary: str
    failure_summary: FailureSummary
    component_summaries: tuple[ComponentSummary, ...]
    failure_clusters: tuple[FailureCluster, ...]
    recommendations: tuple[Recommendation, ...]
    trend_summary: TrendSummary | None


# ── Component summaries ──────────────────────────────────────────────────────────


def _component_summaries(failures: Sequence[FailureRecord]) -> tuple[ComponentSummary, ...]:
    by_component: dict[Component, list[FailureRecord]] = {}
    order: list[Component] = []
    for failure in failures:
        if failure.component not in by_component:
            by_component[failure.component] = []
            order.append(failure.component)
        by_component[failure.component].append(failure)

    summaries = []
    for component in order:
        members = by_component[component]
        summary = summarize_failures(members)
        summaries.append(
            ComponentSummary(
                component=component, total_failures=len(members),
                severity_breakdown=summary.by_severity,
            )
        )
    return tuple(summaries)


def _overall_summary(failures: Sequence[FailureRecord], clusters: Sequence[FailureCluster]) -> str:
    if not failures:
        return "No failures detected across the analyzed evaluation artifacts."
    components = sorted({failure.component.value for failure in failures})
    return (
        f"{len(failures)} failure(s) detected across {len(components)} component(s) "
        f"({', '.join(components)}), grouped into {len(clusters)} cluster(s)."
    )


# ── Report assembly ──────────────────────────────────────────────────────────────


def build_quality_report(
    *,
    retrieval_reports: Sequence = (),
    reasoning_reports: Sequence = (),
    judge_evaluations: Sequence[JudgeEvaluation] = (),
    judge_errors: Sequence[str] = (),
    regression_verdict: str | None = None,
) -> AIQualityReport:
    """Analyze already-built evaluation artifacts and assemble an
    ``AIQualityReport``. Never reruns retrieval/reasoning/judging; never
    makes an LLM call.

    ``retrieval_reports``/``reasoning_reports`` accept zero or more
    reports so a caller can analyze either a single report or a full
    history (oldest first) in one call - history feeds
    ``TrendSummary.failure_count_trend`` when 2+ reports are supplied.
    """
    failures: list[FailureRecord] = []
    failure_count_trend: list[int] = []

    for report in retrieval_reports:
        report_failures = analyze_retrieval_failures(report)
        failures.extend(report_failures)
        failure_count_trend.append(len(report_failures))

    for report in reasoning_reports:
        report_failures = analyze_reasoning_failures(report)
        failures.extend(report_failures)
        failure_count_trend.append(len(report_failures))

    judge_failures = analyze_judge_failures(judge_evaluations, judge_errors=judge_errors)
    failures.extend(judge_failures)

    clusters = cluster_failures(failures)
    recommendations = generate_recommendations(clusters)
    summary = summarize_failures(failures)
    component_summaries = _component_summaries(failures)

    trend = None
    if len(failure_count_trend) >= 2 or regression_verdict is not None:
        trend = TrendSummary(
            failure_count_trend=tuple(failure_count_trend), regression_verdict=regression_verdict,
        )

    return AIQualityReport(
        generated_at=datetime.now(UTC).isoformat(),
        overall_summary=_overall_summary(failures, clusters),
        failure_summary=summary,
        component_summaries=component_summaries,
        failure_clusters=clusters,
        recommendations=recommendations,
        trend_summary=trend,
    )


# ── Benchmark integration ────────────────────────────────────────────────────────


def build_quality_report_from_benchmarks(
    *,
    retrieval_repo: BenchmarkRepository | None = None,
    reasoning_repo: ReasoningBenchmarkRepository | None = None,
    judged_repo: JudgedReasoningBenchmarkRepository | None = None,
    experiment_name: str | None = None,
    include_history: bool = False,
) -> AIQualityReport:
    """Build an ``AIQualityReport`` directly from existing benchmark
    repositories - see module docstring's "Benchmark integration". Calls
    only each repository's already-existing public methods; never
    introduces new storage.
    """
    retrieval_reports = []
    reasoning_reports = []
    judge_evaluations: list[JudgeEvaluation] = []
    regression_verdict: str | None = None

    if retrieval_repo is not None:
        runs = (
            retrieval_repo.list_runs(experiment_name=experiment_name) if include_history
            else _as_tuple(retrieval_repo.latest(experiment_name=experiment_name))
        )
        retrieval_reports = [run.report for run in runs]
        if runs and runs[-1].regression is not None:
            regression_verdict = runs[-1].regression.verdict.value

    if reasoning_repo is not None:
        runs = (
            reasoning_repo.list_runs(experiment_name=experiment_name) if include_history
            else _as_tuple(reasoning_repo.latest(experiment_name=experiment_name))
        )
        reasoning_reports = [run.report for run in runs]
        if runs and runs[-1].regression is not None and regression_verdict is None:
            regression_verdict = runs[-1].regression.verdict.value

    if judged_repo is not None:
        runs = (
            judged_repo.list_runs(experiment_name=experiment_name) if include_history
            else _as_tuple(judged_repo.latest(experiment_name=experiment_name))
        )
        for run in runs:
            judge_evaluations.extend(run.judge_evaluations)
            if run.reasoning_run.report not in reasoning_reports:
                reasoning_reports.append(run.reasoning_run.report)

    return build_quality_report(
        retrieval_reports=retrieval_reports, reasoning_reports=reasoning_reports,
        judge_evaluations=judge_evaluations, regression_verdict=regression_verdict,
    )


def _as_tuple(value) -> tuple:
    return (value,) if value is not None else ()
