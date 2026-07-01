"""AI Quality Intelligence — Failure Analysis (Phase 21A).

A PURE analysis layer over evaluation artifacts already produced by every
earlier phase (16D/16E/16F retrieval; 20A reasoning; 20B judge). This
module never reruns retrieval, never reruns an investigation, never makes
an LLM call, and never imports ``IncidentSearchService``,
``MultiAgentInvestigationOrchestrator``, ``LLMService``, or any agent
class — it only reads already-built, immutable report objects
(``EvaluationReport``, ``InvestigationEvaluationReport``,
``JudgeEvaluation``) and derives structured failure intelligence from
them.

This phase does NOT modify any prior phase. Every type imported from
``app.evaluation.harness``/``reasoning_harness``/``judge`` here is
read-only.

# Updated architecture

```
EvaluationReport (16D)         InvestigationEvaluationReport (20A)
        │                                    │
        ▼                                    ▼
analyze_retrieval_failures()      analyze_reasoning_failures()
        │                                    │            JudgeEvaluation (20B)
        │                                    │                   │
        │                                    │                   ▼
        │                                    │      analyze_judge_failures()
        └────────────────┬───────────────────┴──────────┬────────┘
                          ▼                              ▼
                  tuple[FailureRecord, ...]   (THIS MODULE — every record
                          │                     already carries its own
                          ▼                     cause_chain; see "Root
                  cluster_failures()             cause methodology")
                          │
                          ▼
                  tuple[FailureCluster, ...]
                          │
                          ▼
                  summarize_failures() -> FailureSummary
```

(Root Cause Analysis is not a separate later pass — every ``FailureRecord``
already carries its own ``cause_chain`` at construction time, computed by
the same ``analyze_*`` function that detected the failure, since the cause
chain depends on data only that function has in scope. The Recommendation
Engine and AI Quality Report sit downstream, in
``app.evaluation.recommendation_engine``/``ai_quality_report``.)

# Failure analysis workflow

```
analyze_retrieval_failures(report: EvaluationReport)
  for each QueryEvaluationOutcome in report.per_query:
    - skipped -> FailureRecord(component=RETRIEVAL, category=SEARCH_FAILURE)
    - recall_at_k < 1.0 (and defined) -> FailureRecord(category=INCOMPLETE_RECALL)
    - num_unresolved_expected > 0 -> FailureRecord(category=UNRESOLVED_GOLD_ENTRY)
  (a query with recall_at_k == 1.0, not skipped, no unresolved entries
   produces NO failure record - only deviations are recorded, per
   "nothing should appear without evidence")

analyze_reasoning_failures(report: InvestigationEvaluationReport)
  for each InvestigationResult in report.results:
    - not planner_correct      -> FailureRecord(component=PLANNER)
    - not hypothesis_recall_hit -> FailureRecord(component=HYPOTHESIS_GENERATOR)
    - not decision_correct     -> FailureRecord(component=DECISION)
    - not critic_correct       -> FailureRecord(component=CRITIC)
    - not stopping_correct     -> FailureRecord(component=ORCHESTRATOR)
  (one InvestigationResult can produce MULTIPLE FailureRecords, one per
   failing check - each check is evaluated independently because each is
   evidence Phase 20A already computed; this module does not re-derive
   any of them)

analyze_judge_failures(evaluations, *, judge_errors=())
  for each JudgeEvaluation: score.band in {"Poor", "Weak"}
    -> FailureRecord(component=JUDGE, category=LOW_CONFIDENCE)
  for each pre-caught parse-failure message in judge_errors (the caller's
  own already-caught ``JudgeResponseError`` text - this module never
  catches an exception itself, since doing so would require calling the
  judge, which it never does)
    -> FailureRecord(component=JUDGE, category=MALFORMED_EVALUATION)
```

# Root cause methodology

Every ``FailureRecord.cause_chain`` is a ``tuple[CauseStep, ...]`` ordered
``IMMEDIATE -> UNDERLYING -> SYSTEMIC`` (never more, never reordered) -
"the framework should distinguish Immediate Cause, Underlying Cause,
Systemic Cause." For reasoning failures specifically, the chain follows
the SAME most-upstream-first logic the brief's own example demonstrates
(planner -> hypotheses -> decision -> critic -> stopping): if an
``InvestigationResult`` fails MULTIPLE checks, the cause chain is anchored
on the MOST UPSTREAM failing stage (planner first, then hypotheses, then
decision, then critic, then stopping), because a downstream failure is
frequently a consequence of an upstream one (e.g. a wrong planner strategy
plausibly explains a wrong decision, but not vice versa) - this is a
documented HEURISTIC, not a verified causal proof; see "Risks discovered"
in this phase's documentation for exactly what that heuristic does not
capture (a downstream failure with a fully correct upstream chain still
needs its own, shallower cause chain - see ``_trace_reasoning_failure``).

For retrieval failures, the chain checks whether the SAME category's mean
recall across the whole report is also below the dataset-wide mean -
if so, the systemic cause names the category; if the query is an isolated
underperformer within an otherwise-healthy category, the systemic cause
says so explicitly rather than overgeneralizing from one query.

For judge failures, the chain is shallow by necessity (a Judge's score is
already a holistic semantic judgment with no further breakdown available
to this module without re-invoking the judge, which it must not do) -
immediate and underlying are populated from the evaluation's own
``weaknesses``; systemic names the rubric criterion most often implicated
across that stage's failing evaluations.

# Severity methodology

Severity is derived from IMPACT, never assigned arbitrarily, via ONE
shared function, ``classify_severity(deviation)``, where ``deviation`` is
a ``[0.0, 1.0]`` measure of "how far this single failure is from the
ideal outcome" - 0.0 is no failure at all (never produced, since a record
with zero deviation would not be a failure), 1.0 is the worst possible
outcome for that failure type:

```
deviation >= 0.75  -> CRITICAL
deviation >= 0.50  -> HIGH
deviation >= 0.25  -> MEDIUM
deviation <  0.25  -> LOW
```

Per-analyzer deviation computation (each is the natural "distance from
ideal" for that data, never an arbitrary constant):

- **Retrieval, incomplete recall**: ``1.0 - recall_at_k`` (a complete miss
  -> 1.0 -> CRITICAL; a near-complete recall -> small deviation -> LOW).
- **Retrieval, search failure (skipped)**: fixed ``1.0`` (a query that
  could not be evaluated at all is the worst outcome retrieval can have
  for that query - there is no partial-credit number to compute).
- **Reasoning, any failing check**: fixed ``0.5`` (HIGH) for a single
  failing check, ``min(1.0, 0.5 + 0.15 * (extra_failing_checks))`` when
  the SAME ``InvestigationResult`` fails more than one check - more
  simultaneous failures on one investigation is evidence of a deeper
  problem than one isolated failing check, without inventing a new
  per-check weighting scheme.
- **Judge, low score**: ``(SCORE_MAX - score.value) / (SCORE_MAX -
  SCORE_MIN)`` - the score's own distance from the top of its own already-
  defined rubric scale (Phase 20B), never re-derived.
- **Judge, malformed evaluation**: fixed ``1.0`` (no score exists at all -
  the worst case for that stage's evaluation).

Cluster-level severity (``FailureCluster.severity``) is the MAXIMUM
severity among its member ``FailureRecord``s - "if any record in this
cluster is CRITICAL, the cluster is CRITICAL" - the simplest aggregation
rule that never understates a cluster's worst member.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from app.evaluation.harness import EvaluationReport, QueryEvaluationOutcome
from app.evaluation.judge import SCORE_MAX, SCORE_MIN, JudgeEvaluation
from app.evaluation.reasoning_harness import InvestigationEvaluationReport, InvestigationResult

# ── Enums ──────────────────────────────────────────────────────────────────────


class Component(str, Enum):
    RETRIEVAL = "retrieval"
    PLANNER = "planner"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    EVIDENCE_EVALUATOR = "evidence_evaluator"
    DECISION = "decision"
    CRITIC = "critic"
    ORCHESTRATOR = "orchestrator"
    JUDGE = "judge"


class FailureCategory(str, Enum):
    # Retrieval
    SEARCH_FAILURE = "search_failure"
    INCOMPLETE_RECALL = "incomplete_recall"
    UNRESOLVED_GOLD_ENTRY = "unresolved_gold_entry"
    # Reasoning
    STRATEGY_MISMATCH = "strategy_mismatch"
    MISSING_HYPOTHESIS = "missing_hypothesis"
    DUPLICATE_HYPOTHESIS = "duplicate_hypothesis"
    INCORRECT_DECISION = "incorrect_decision"
    INCORRECT_CRITIQUE = "incorrect_critique"
    NO_CONVERGENCE = "no_convergence"
    # Judge
    LOW_CONFIDENCE = "low_confidence"
    MALFORMED_EVALUATION = "malformed_evaluation"
    RULE_DISAGREEMENT = "rule_disagreement"


class CauseLevel(str, Enum):
    IMMEDIATE = "immediate"
    UNDERLYING = "underlying"
    SYSTEMIC = "systemic"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3,
}


def classify_severity(deviation: float) -> Severity:
    """Map a ``[0.0, 1.0]`` impact deviation to a fixed severity band - see
    module docstring's "Severity methodology". Clamped, never raises.
    """
    clamped = max(0.0, min(1.0, deviation))
    if clamped >= 0.75:
        return Severity.CRITICAL
    if clamped >= 0.50:
        return Severity.HIGH
    if clamped >= 0.25:
        return Severity.MEDIUM
    return Severity.LOW


def _max_severity(severities: Sequence[Severity]) -> Severity:
    return max(severities, key=lambda s: _SEVERITY_ORDER[s])


# ── Report data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CauseStep:
    level: CauseLevel
    description: str


@dataclass(frozen=True)
class FailureRecord:
    """One detected failure, fully evidence-backed - see module docstring.
    ``subject_id`` is the affected query id (retrieval) or scenario id
    (reasoning/judge).
    """

    component: Component
    stage: str
    category: FailureCategory
    severity: Severity
    subject_id: str
    description: str
    evidence: tuple[str, ...]
    metrics_involved: tuple[str, ...]
    cause_chain: tuple[CauseStep, ...]


@dataclass(frozen=True)
class ComponentFailureCount:
    component: Component
    count: int


@dataclass(frozen=True)
class CategoryFailureCount:
    category: FailureCategory
    count: int


@dataclass(frozen=True)
class SeverityFailureCount:
    severity: Severity
    count: int


@dataclass(frozen=True)
class FailureCluster:
    """Failures grouped by ``(component, category)`` - see module
    docstring's "Failure clustering". ``severity`` is the max severity
    among ``failures`` (see "Severity methodology").
    """

    component: Component
    category: FailureCategory
    failures: tuple[FailureRecord, ...]
    severity: Severity
    common_cause: str


@dataclass(frozen=True)
class FailureSummary:
    total_failures: int
    by_component: tuple[ComponentFailureCount, ...]
    by_category: tuple[CategoryFailureCount, ...]
    by_severity: tuple[SeverityFailureCount, ...]


# ── Retrieval failure analysis ──────────────────────────────────────────────────


def analyze_retrieval_failures(report: EvaluationReport) -> tuple[FailureRecord, ...]:
    category_means = {
        name: agg.mean_recall_at_k for name, agg in report.category_breakdown.items()
    }
    overall_mean = report.aggregate_metrics.mean_recall_at_k

    records: list[FailureRecord] = []
    for outcome in report.per_query:
        records.extend(_retrieval_outcome_failures(outcome, category_means, overall_mean))
    return tuple(records)


def _retrieval_outcome_failures(
    outcome: QueryEvaluationOutcome,
    category_means: Mapping[str, float | None],
    overall_mean: float | None,
) -> list[FailureRecord]:
    records: list[FailureRecord] = []

    if outcome.skipped:
        records.append(
            FailureRecord(
                component=Component.RETRIEVAL, stage="retrieve",
                category=FailureCategory.SEARCH_FAILURE,
                severity=classify_severity(1.0), subject_id=outcome.query_id,
                description=f"query {outcome.query_id!r} was skipped: {outcome.skip_reason}",
                evidence=(str(outcome.skip_reason),), metrics_involved=("recall_at_k",),
                cause_chain=(
                    CauseStep(CauseLevel.IMMEDIATE, f"search call failed: {outcome.skip_reason}"),
                    CauseStep(
                        CauseLevel.UNDERLYING, "retrieval raised an exception for this query"
                    ),
                    CauseStep(CauseLevel.SYSTEMIC, "infrastructure/connectivity instability"),
                ),
            )
        )
    elif outcome.metric is not None and outcome.metric.recall_at_k is not None:
        recall = outcome.metric.recall_at_k
        if recall < 1.0:
            deviation = 1.0 - recall
            category_mean = category_means.get(outcome.category)
            systemic = (
                f"category {outcome.category!r} underperforms the dataset overall "
                f"(category mean={category_mean:.2f} < overall mean={overall_mean:.2f})"
                if category_mean is not None and overall_mean is not None
                and category_mean < overall_mean
                else f"isolated query-level miss within an otherwise-healthy "
                f"{outcome.category!r} category"
            )
            records.append(
                FailureRecord(
                    component=Component.RETRIEVAL, stage="retrieve",
                    category=FailureCategory.INCOMPLETE_RECALL,
                    severity=classify_severity(deviation), subject_id=outcome.query_id,
                    description=f"query {outcome.query_id!r} recall@k={recall:.2f}",
                    evidence=(f"recall_at_k={recall:.2f}", f"category={outcome.category}"),
                    metrics_involved=("recall_at_k",),
                    cause_chain=(
                        CauseStep(CauseLevel.IMMEDIATE, f"recall@k={recall:.2f} < 1.0"),
                        CauseStep(
                            CauseLevel.UNDERLYING,
                            f"not every gold-relevant incident was retrieved for category "
                            f"{outcome.category!r}",
                        ),
                        CauseStep(CauseLevel.SYSTEMIC, systemic),
                    ),
                )
            )

    if outcome.num_unresolved_expected > 0:
        records.append(
            FailureRecord(
                component=Component.RETRIEVAL, stage="resolve",
                category=FailureCategory.UNRESOLVED_GOLD_ENTRY,
                severity=classify_severity(0.5), subject_id=outcome.query_id,
                description=(
                    f"query {outcome.query_id!r} has {outcome.num_unresolved_expected} "
                    "unresolved expected incident(s)"
                ),
                evidence=(f"num_unresolved_expected={outcome.num_unresolved_expected}",),
                metrics_involved=("resolution_coverage",),
                cause_chain=(
                    CauseStep(
                        CauseLevel.IMMEDIATE,
                        f"{outcome.num_unresolved_expected} expected incident(s) did not resolve",
                    ),
                    CauseStep(
                        CauseLevel.UNDERLYING,
                        "gold dataset references a stable identity no longer present in the "
                        "corpus",
                    ),
                    CauseStep(
                        CauseLevel.SYSTEMIC, "gold dataset drift relative to the live corpus"
                    ),
                ),
            )
        )
    return records


# ── Reasoning failure analysis ──────────────────────────────────────────────────


def analyze_reasoning_failures(report: InvestigationEvaluationReport) -> tuple[FailureRecord, ...]:
    records: list[FailureRecord] = []
    for result in report.results:
        records.extend(_reasoning_result_failures(result))
    return tuple(records)


def _failing_checks(result: InvestigationResult) -> list[str]:
    checks = []
    if not result.planner_correct:
        checks.append("planner")
    if not result.hypothesis_recall_hit:
        checks.append("hypotheses")
    if not result.decision_correct:
        checks.append("decision")
    if not result.critic_correct:
        checks.append("critic")
    if not result.stopping_correct:
        checks.append("stopping")
    return checks


def _reasoning_result_failures(result: InvestigationResult) -> list[FailureRecord]:
    failing = _failing_checks(result)
    if not failing:
        return []
    extra = max(0, len(failing) - 1)
    deviation = min(1.0, 0.5 + 0.15 * extra)
    severity = classify_severity(deviation)
    evidence = result.explanation or (f"failing checks: {failing}",)

    records: list[FailureRecord] = []
    if "planner" in failing:
        records.append(_planner_failure(result, severity, evidence))
    if "hypotheses" in failing:
        records.append(_hypothesis_failure(result, severity, evidence))
    if "decision" in failing:
        records.append(
            _decision_failure(result, severity, evidence, planner_failed="planner" in failing)
        )
    if "critic" in failing:
        records.append(_critic_failure(result, severity, evidence))
    if "stopping" in failing:
        records.append(_orchestrator_failure(result, severity, evidence))
    return records


def _planner_failure(result, severity, evidence) -> FailureRecord:
    return FailureRecord(
        component=Component.PLANNER, stage="plan", category=FailureCategory.STRATEGY_MISMATCH,
        severity=severity, subject_id=result.scenario_id,
        description=(
            f"planner selected {result.actual_strategy!r}, expected {result.expected_strategy!r}"
        ),
        evidence=evidence, metrics_involved=("planner_accuracy",),
        cause_chain=(
            CauseStep(
                CauseLevel.IMMEDIATE,
                f"planner selected {result.actual_strategy!r} instead of "
                f"{result.expected_strategy!r}",
            ),
            CauseStep(
                CauseLevel.UNDERLYING,
                "a different strategy's keyword(s) matched before the correct strategy's",
            ),
            CauseStep(CauseLevel.SYSTEMIC, "planner rule priority ordering"),
        ),
    )


def _hypothesis_failure(result, severity, evidence) -> FailureRecord:
    return FailureRecord(
        component=Component.HYPOTHESIS_GENERATOR, stage="generate",
        category=FailureCategory.MISSING_HYPOTHESIS, severity=severity,
        subject_id=result.scenario_id,
        description=(
            f"no generated hypothesis matched any expected root cause in "
            f"{list(result.expected_root_causes)}"
        ),
        evidence=evidence, metrics_involved=("hypothesis_recall",),
        cause_chain=(
            CauseStep(
                CauseLevel.IMMEDIATE,
                f"generated root causes {list(result.actual_root_causes)} missed every "
                "expected one",
            ),
            CauseStep(
                CauseLevel.UNDERLYING,
                "hypothesis generation did not cover the expected explanation space",
            ),
            CauseStep(
                CauseLevel.SYSTEMIC,
                "hypothesis generation diversity/coverage for this scenario type",
            ),
        ),
    )


def _decision_failure(result, severity, evidence, *, planner_failed: bool) -> FailureRecord:
    underlying = (
        "the planner's wrong strategy shaped hypothesis generation toward the wrong explanation"
        if planner_failed
        else "evidence supporting the correct hypothesis was insufficient relative to the "
        "acceptance threshold"
    )
    return FailureRecord(
        component=Component.DECISION, stage="decide", category=FailureCategory.INCORRECT_DECISION,
        severity=severity, subject_id=result.scenario_id,
        description=(
            f"decision accuracy failure: actual_verdict={result.actual_verdict!r}, "
            f"expected_root_causes={list(result.expected_root_causes)}"
        ),
        evidence=evidence, metrics_involved=("decision_accuracy",),
        cause_chain=(
            CauseStep(
                CauseLevel.IMMEDIATE, "decision stage did not accept the correct hypothesis"
            ),
            CauseStep(CauseLevel.UNDERLYING, underlying),
            CauseStep(
                CauseLevel.SYSTEMIC,
                "planner rule priority ordering" if planner_failed
                else "evidence evaluation / composite scoring threshold calibration",
            ),
        ),
    )


def _critic_failure(result, severity, evidence) -> FailureRecord:
    return FailureRecord(
        component=Component.CRITIC, stage="critique", category=FailureCategory.INCORRECT_CRITIQUE,
        severity=severity, subject_id=result.scenario_id,
        description=(
            f"critic verdict {result.actual_verdict!r} did not match expected "
            f"{result.expected_verdict!r}"
        ),
        evidence=evidence, metrics_involved=("critic_accuracy",),
        cause_chain=(
            CauseStep(CauseLevel.IMMEDIATE, "critic verdict mismatch"),
            CauseStep(
                CauseLevel.UNDERLYING,
                "critic heuristics (contradiction ratio / margin / missing evidence) did not "
                "align with this case",
            ),
            CauseStep(CauseLevel.SYSTEMIC, "critic threshold calibration for this scenario type"),
        ),
    )


def _orchestrator_failure(result, severity, evidence) -> FailureRecord:
    return FailureRecord(
        component=Component.ORCHESTRATOR, stage="orchestrate",
        category=FailureCategory.NO_CONVERGENCE, severity=severity, subject_id=result.scenario_id,
        description=(
            f"stopping reason {result.actual_stopping_reason!r} did not match expected "
            f"{result.expected_stopping_reason!r}"
        ),
        evidence=evidence, metrics_involved=("stopping_accuracy", "convergence_rate"),
        cause_chain=(
            CauseStep(CauseLevel.IMMEDIATE, "orchestrator stopped for an unexpected reason"),
            CauseStep(
                CauseLevel.UNDERLYING,
                "stopping condition priority/order produced an unintended outcome for this case",
            ),
            CauseStep(CauseLevel.SYSTEMIC, "orchestrator stopping policy for this scenario type"),
        ),
    )


# ── Judge failure analysis ──────────────────────────────────────────────────────


def analyze_judge_failures(
    evaluations: Sequence[JudgeEvaluation], *, judge_errors: Sequence[str] = ()
) -> tuple[FailureRecord, ...]:
    records: list[FailureRecord] = []
    for evaluation in evaluations:
        if evaluation.score.band in {"Poor", "Weak"}:
            deviation = (SCORE_MAX - evaluation.score.value) / (SCORE_MAX - SCORE_MIN)
            weakness_criteria = [w.criterion for w in evaluation.weaknesses] or ["unspecified"]
            records.append(
                FailureRecord(
                    component=Component.JUDGE, stage=evaluation.stage,
                    category=FailureCategory.LOW_CONFIDENCE, severity=classify_severity(deviation),
                    subject_id=evaluation.stage,
                    description=(
                        f"{evaluation.stage} judged {evaluation.score.band!r} "
                        f"({evaluation.score.value:.1f}/10)"
                    ),
                    evidence=(evaluation.explanation,),
                    metrics_involved=(f"{evaluation.stage}_score",),
                    cause_chain=(
                        CauseStep(
                            CauseLevel.IMMEDIATE,
                            f"score {evaluation.score.value:.1f} in band "
                            f"{evaluation.score.band!r}",
                        ),
                        CauseStep(
                            CauseLevel.UNDERLYING, f"weaknesses identified: {weakness_criteria}"
                        ),
                        CauseStep(
                            CauseLevel.SYSTEMIC,
                            f"rubric criterion most implicated: {weakness_criteria[0]}",
                        ),
                    ),
                )
            )
    for index, error in enumerate(judge_errors):
        records.append(
            FailureRecord(
                component=Component.JUDGE, stage="unknown",
                category=FailureCategory.MALFORMED_EVALUATION, severity=classify_severity(1.0),
                subject_id=f"judge_error_{index}",
                description=f"judge response failed to parse: {error}",
                evidence=(error,), metrics_involved=(),
                cause_chain=(
                    CauseStep(CauseLevel.IMMEDIATE, f"parse failure: {error}"),
                    CauseStep(
                        CauseLevel.UNDERLYING,
                        "response did not conform to the expected JSON contract",
                    ),
                    CauseStep(
                        CauseLevel.SYSTEMIC,
                        "LLM judge prompt does not reliably elicit conforming output",
                    ),
                ),
            )
        )
    return tuple(records)


# ── Clustering and summary ──────────────────────────────────────────────────────


def cluster_failures(failures: Sequence[FailureRecord]) -> tuple[FailureCluster, ...]:
    """Group failures by ``(component, category)`` - see module
    docstring's "Failure clustering". Deterministic: iteration order
    follows first-seen ``(component, category)`` pair order, then within
    each cluster, the input order of ``failures`` is preserved.
    """
    order: list[tuple[Component, FailureCategory]] = []
    grouped: dict[tuple[Component, FailureCategory], list[FailureRecord]] = {}
    for failure in failures:
        key = (failure.component, failure.category)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(failure)

    clusters = []
    for component, category in order:
        members = tuple(grouped[(component, category)])
        systemic_causes = [
            step.description
            for member in members
            for step in member.cause_chain
            if step.level == CauseLevel.SYSTEMIC
        ]
        common_cause = (
            max(set(systemic_causes), key=systemic_causes.count) if systemic_causes
            else "no common systemic cause identified"
        )
        clusters.append(
            FailureCluster(
                component=component, category=category, failures=members,
                severity=_max_severity([m.severity for m in members]), common_cause=common_cause,
            )
        )
    return tuple(clusters)


def summarize_failures(failures: Sequence[FailureRecord]) -> FailureSummary:
    by_component: dict[Component, int] = {}
    by_category: dict[FailureCategory, int] = {}
    by_severity: dict[Severity, int] = {}
    for failure in failures:
        by_component[failure.component] = by_component.get(failure.component, 0) + 1
        by_category[failure.category] = by_category.get(failure.category, 0) + 1
        by_severity[failure.severity] = by_severity.get(failure.severity, 0) + 1

    return FailureSummary(
        total_failures=len(failures),
        by_component=tuple(
            ComponentFailureCount(component=c, count=n) for c, n in by_component.items()
        ),
        by_category=tuple(
            CategoryFailureCount(category=c, count=n) for c, n in by_category.items()
        ),
        by_severity=tuple(
            SeverityFailureCount(severity=s, count=n) for s, n in by_severity.items()
        ),
    )
