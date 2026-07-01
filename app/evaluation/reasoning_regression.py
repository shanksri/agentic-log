"""Reasoning Regression Runner (Phase 20A).

Compares two completed ``InvestigationEvaluationReport``s (this phase's own
harness) and produces a ``ReasoningRegressionReport`` — the reasoning-layer
analogue of Phase 16E's ``compare()``. Pure comparison only: never re-runs
an investigation, never calls an orchestrator/agent, never makes an LLM
call. It reads already-computed ``ReasoningMetrics`` values and
arithmetically diffs them.

This module does NOT import Phase 16E's private (underscore-prefixed)
helpers (``_classify``/``_metric_delta``/``_verdict_from_classifications``)
even though the comparison rule is identical — importing another module's
underscore-prefixed internals is not a stable contract to depend on.
Instead it reimplements the same small, already-documented rule locally
(see ``EPSILON``/``_classify`` below, byte-for-byte the same logic as
Phase 16E's, on different inputs).

# Regression workflow

```
compare_reasoning(baseline, candidate) -> ReasoningRegressionReport
  1. _check_compatibility(baseline, candidate)
       - incompatible -> return immediately, verdict=INCOMPATIBLE
  2. five per-category deltas, each independently classified
       - planner:    planner_accuracy
       - hypothesis: hypothesis_recall + hypothesis_precision (both must
                      not regress for the category to read IMPROVED/
                      UNCHANGED - see "Per-category verdicts")
       - decision:   decision_accuracy
       - critic:     critic_accuracy
       - iteration:  mean_iteration_count (lower is better) +
                      convergence_rate + stopping_accuracy (higher is
                      better) - orchestrator EFFICIENCY/behavior, not
                      "is the answer right"
  3. overall verdict, derived ONLY from planner/hypothesis/decision/critic
     (the four "is the reasoning correct" categories) - mirrors Phase
     16E's choice to exclude diagnostic-only metrics from the headline
     verdict; iteration/stopping efficiency is reported but does not
     drive "did reasoning quality improve"
  4. assemble and return an immutable ReasoningRegressionReport
```

# Per-category verdicts

Each category's verdict reuses the exact four-way rule Phase 16E already
established for its own bucket verdicts (IMPROVED / REGRESSED / UNCHANGED
/ MIXED), applied to that category's one-or-more underlying metric
classifications: if every classification considered is UNCHANGED (or
UNDEFINED) -> UNCHANGED; at least one IMPROVED and none REGRESSED ->
IMPROVED; at least one REGRESSED and none IMPROVED -> REGRESSED; one of
each -> MIXED. The "hypothesis" and "iteration" categories each fold
together two-to-three underlying metrics through this same rule rather
than inventing a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.evaluation.reasoning_harness import InvestigationEvaluationReport, ReasoningMetrics

EPSILON = 1e-9


class DeltaClassification(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    UNDEFINED = "undefined"


class Verdict(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    MIXED = "mixed"
    INCOMPATIBLE = "incompatible"


# ── Report data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompatibilityCheck:
    compatible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MetricDelta:
    baseline: float | None
    candidate: float | None
    delta: float | None
    classification: DeltaClassification


@dataclass(frozen=True)
class CategoryDelta:
    """One reasoning category's verdict, derived from one or more
    underlying ``MetricDelta``s — see module docstring's "Per-category
    verdicts".
    """

    category: str
    metrics: dict[str, MetricDelta]
    verdict: Verdict


@dataclass(frozen=True)
class ReasoningRegressionReport:
    """The complete, immutable result of comparing two
    ``InvestigationEvaluationReport``s.
    """

    baseline: InvestigationEvaluationReport
    candidate: InvestigationEvaluationReport
    compatibility: CompatibilityCheck
    verdict: Verdict
    planner: CategoryDelta | None
    hypothesis: CategoryDelta | None
    decision: CategoryDelta | None
    critic: CategoryDelta | None
    iteration: CategoryDelta | None
    summary: str


# ── Comparison ─────────────────────────────────────────────────────────────────


def compare_reasoning(
    baseline: InvestigationEvaluationReport, candidate: InvestigationEvaluationReport
) -> ReasoningRegressionReport:
    """Compare ``baseline`` against ``candidate``. Never raises for an
    incompatible pair; returns a report with ``verdict=INCOMPATIBLE`` and
    the reasons listed instead.
    """
    compatibility = _check_compatibility(baseline, candidate)
    if not compatibility.compatible:
        return ReasoningRegressionReport(
            baseline=baseline, candidate=candidate, compatibility=compatibility,
            verdict=Verdict.INCOMPATIBLE, planner=None, hypothesis=None, decision=None,
            critic=None, iteration=None,
            summary=f"Reports are not comparable: {'; '.join(compatibility.reasons)}",
        )

    b, c = baseline.metrics, candidate.metrics

    planner = _category(
        "planner", {"planner_accuracy": _delta(b.planner_accuracy, c.planner_accuracy, True)}
    )
    hypothesis = _category(
        "hypothesis",
        {
            "hypothesis_recall": _delta(b.hypothesis_recall, c.hypothesis_recall, True),
            "hypothesis_precision": _delta(b.hypothesis_precision, c.hypothesis_precision, True),
        },
    )
    decision = _category(
        "decision", {"decision_accuracy": _delta(b.decision_accuracy, c.decision_accuracy, True)}
    )
    critic = _category(
        "critic", {"critic_accuracy": _delta(b.critic_accuracy, c.critic_accuracy, True)}
    )
    iteration = _category(
        "iteration",
        {
            "mean_iteration_count": _delta(
                b.mean_iteration_count, c.mean_iteration_count, False
            ),
            "convergence_rate": _delta(b.convergence_rate, c.convergence_rate, True),
            "stopping_accuracy": _delta(b.stopping_accuracy, c.stopping_accuracy, True),
        },
    )

    verdict = _overall_verdict([planner, hypothesis, decision, critic])
    summary = _build_summary(verdict, planner, hypothesis, decision, critic, iteration)

    return ReasoningRegressionReport(
        baseline=baseline, candidate=candidate, compatibility=compatibility, verdict=verdict,
        planner=planner, hypothesis=hypothesis, decision=decision, critic=critic,
        iteration=iteration, summary=summary,
    )


def _check_compatibility(
    baseline: InvestigationEvaluationReport, candidate: InvestigationEvaluationReport
) -> CompatibilityCheck:
    reasons: list[str] = []
    if baseline.dataset_version != candidate.dataset_version:
        reasons.append(
            "reasoning dataset version differs: "
            f"baseline={baseline.dataset_version!r} candidate={candidate.dataset_version!r}"
        )
    baseline_ids = {result.scenario_id for result in baseline.results}
    candidate_ids = {result.scenario_id for result in candidate.results}
    if baseline_ids != candidate_ids:
        reasons.append(
            "scenario coverage differs: "
            f"missing_in_candidate={sorted(baseline_ids - candidate_ids)} "
            f"missing_in_baseline={sorted(candidate_ids - baseline_ids)}"
        )
    return CompatibilityCheck(compatible=not reasons, reasons=tuple(reasons))


# ── Delta computation (reimplemented locally; see module docstring) ────────────


def _classify(
    baseline: float | None, candidate: float | None, *, higher_is_better: bool
) -> DeltaClassification:
    if baseline is None and candidate is None:
        return DeltaClassification.UNCHANGED
    if baseline is None or candidate is None:
        return DeltaClassification.UNDEFINED
    delta = candidate - baseline
    if not higher_is_better:
        delta = -delta
    if abs(delta) <= EPSILON:
        return DeltaClassification.UNCHANGED
    return DeltaClassification.IMPROVED if delta > 0 else DeltaClassification.REGRESSED


def _delta(baseline: float | None, candidate: float | None, higher_is_better: bool) -> MetricDelta:
    classification = _classify(baseline, candidate, higher_is_better=higher_is_better)
    delta = candidate - baseline if (baseline is not None and candidate is not None) else None
    return MetricDelta(
        baseline=baseline, candidate=candidate, delta=delta, classification=classification
    )


def _verdict_from_classifications(classifications: list[DeltaClassification]) -> Verdict:
    meaningful = [c for c in classifications if c != DeltaClassification.UNDEFINED]
    if not meaningful:
        return Verdict.UNCHANGED
    improved = DeltaClassification.IMPROVED in meaningful
    regressed = DeltaClassification.REGRESSED in meaningful
    if improved and regressed:
        return Verdict.MIXED
    if improved:
        return Verdict.IMPROVED
    if regressed:
        return Verdict.REGRESSED
    return Verdict.UNCHANGED


def _category(name: str, metrics: dict[str, MetricDelta]) -> CategoryDelta:
    verdict = _verdict_from_classifications([delta.classification for delta in metrics.values()])
    return CategoryDelta(category=name, metrics=metrics, verdict=verdict)


def _overall_verdict(categories: list[CategoryDelta]) -> Verdict:
    """Combine several already-classified ``CategoryDelta`` verdicts into
    one overall verdict, using the same improved/regressed/mixed logic as
    ``_verdict_from_classifications``, but over category-level ``Verdict``s
    (which may themselves already be ``MIXED``) rather than single metric
    classifications.
    """
    verdicts = [category.verdict for category in categories]
    improved = Verdict.IMPROVED in verdicts or Verdict.MIXED in verdicts
    regressed = Verdict.REGRESSED in verdicts or Verdict.MIXED in verdicts
    if improved and regressed:
        return Verdict.MIXED
    if improved:
        return Verdict.IMPROVED
    if regressed:
        return Verdict.REGRESSED
    return Verdict.UNCHANGED


# ── Summary ────────────────────────────────────────────────────────────────────


def _build_summary(
    verdict: Verdict,
    planner: CategoryDelta,
    hypothesis: CategoryDelta,
    decision: CategoryDelta,
    critic: CategoryDelta,
    iteration: CategoryDelta,
) -> str:
    parts = [f"Overall reasoning verdict: {verdict.value}."]
    for category in (planner, hypothesis, decision, critic):
        parts.append(f"{category.category} {category.verdict.value}.")
    parts.append(f"iteration (diagnostic, not counted in overall) {iteration.verdict.value}.")
    return " ".join(parts)
