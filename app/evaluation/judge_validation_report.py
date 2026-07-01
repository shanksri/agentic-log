"""Judge Validation & Meta-Evaluation — Final Report (Phase 21B).

Assembles ``app.evaluation.judge_agreement``'s agreement/consistency/
prompt-sensitivity/bias analyses and ``app.evaluation.judge_calibration``'s
calibration/correlation analyses into one immutable
``JudgeValidationReport`` answering the brief's single engineering
question: "can this judge be trusted?" Composes Phase 20A/20B/21A's
existing benchmark repositories and report types by reference - never
introduces new benchmark storage, never modifies any Judge implementation,
never reruns reasoning or retrieval.

# Updated architecture (final stage)

```
compute_agreement() / analyze_bias()        [judge_agreement.py]
analyze_consistency() / analyze_prompt_sensitivity()
analyze_calibration() / analyze_correlation()  [judge_calibration.py]
            │
            ▼
   assemble_validation_report(...)              (THIS MODULE)
            │
            ▼
   JudgeValidationReport
   (agreement, consistency, prompt_sensitivity, calibration, bias,
    correlation, overall_trustworthiness, recommended_production_usage,
    confidence_level)
```

# Trustworthiness methodology

``overall_trustworthiness`` starts at a perfect score of ``1.0`` and is
PENALIZED by evidence, never assigned by guesswork:

```
-0.2  for each (pair, stage) BiasFinding present (a detected, threshold-
       crossing systematic tendency)
-0.2  for each AgreementResult whose agreement_within_tolerance < 0.6
       (fewer than 60% of paired scores agree within tolerance - the
       same "majority" cut point Phase 19C's CONTRADICTION_RATIO_THRESHOLD
       already uses for "is this the dominant signal")
-0.2  for each ConsistencyResult whose std_dev > 1.0 (a full point of
       scale on this framework's own 1-10 rubric - the same "one point on
       this established scale" magnitude this module's own
       BIAS_THRESHOLD uses elsewhere)
-0.2  for each CalibrationResult/CorrelationResult whose direction is
       "negative" (higher judge scores correspond to WORSE measured
       quality - the most concerning calibration failure mode)
```

The penalized score is clamped to ``[0.0, 1.0]`` then classified into a
``Trustworthiness`` band:

```
>= 0.75  HIGH
>= 0.5   MEDIUM
>= 0.25  LOW
<  0.25  VERY_LOW
```

When NO analyses were supplied at all (every input tuple empty),
``overall_trustworthiness`` is reported as ``Trustworthiness.
INSUFFICIENT_DATA`` rather than a fabricated score of any kind - "do not
invent calibration curves (or trust verdicts) beyond available data."

``recommended_production_usage`` is a fixed, documented sentence per band
(see ``_RECOMMENDATION_BY_TRUSTWORTHINESS``) - never a free-form
generated string, so the mapping from evidence to recommendation is
auditable and stable.

``confidence_level`` (``[0.0, 1.0]``) measures how much DATA backed the
verdict above, independent of what that verdict says: ``min(1.0,
total_n / CONFIDENCE_BASELINE_N)`` where ``total_n`` sums every
analysis's own ``n``/sample-count field and ``CONFIDENCE_BASELINE_N =
20`` (a round, documented "enough data points to draw a basic conclusion"
threshold - not a statistically rigorous power calculation, explicitly
flagged as such in "Risks discovered").

# Benchmark integration

``build_validation_report_from_benchmarks`` reuses Phase 20A's
``ReasoningBenchmarkRepository``, Phase 20B's
``JudgedReasoningBenchmarkRepository``, and Phase 21A's
``build_quality_report_from_benchmarks`` directly - calling only their
already-existing public methods (``.list_runs()``), never introducing new
storage. It extracts, across a judged repository's run history: the mean
session-stage judge score per run (for correlation against that run's
embedded reasoning ``ReasoningMetrics.decision_accuracy``) and, when a
``ReasoningRegressionReport`` is embedded on a run, its verdict (via
``regression_verdict_to_number``) for correlation against the SAME run's
mean judge score - directly answering "does the judge correlate with
system quality" and "judge score vs regression verdicts" using data this
phase's benchmark integration already has in scope, without recomputing
anything Phase 20A/20B/21A already computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from app.evaluation.judge_agreement import AgreementResult, BiasFinding, ConsistencyResult
from app.evaluation.judge_calibration import (
    CalibrationResult,
    CorrelationResult,
    analyze_correlation,
    regression_verdict_to_number,
)
from app.evaluation.judge_benchmark import JudgedReasoningBenchmarkRepository

CONFIDENCE_BASELINE_N = 20
PENALTY_PER_FINDING = 0.2
LOW_AGREEMENT_THRESHOLD = 0.6
HIGH_STD_DEV_THRESHOLD = 1.0


class Trustworthiness(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    INSUFFICIENT_DATA = "insufficient_data"


_TRUSTWORTHINESS_ORDER: dict[Trustworthiness, int] = {
    Trustworthiness.VERY_LOW: 0, Trustworthiness.LOW: 1, Trustworthiness.MEDIUM: 2,
    Trustworthiness.HIGH: 3,
}

_RECOMMENDATION_BY_TRUSTWORTHINESS: dict[Trustworthiness, str] = {
    Trustworthiness.HIGH: "Safe for production use as a primary evaluator.",
    Trustworthiness.MEDIUM: "Usable in production with periodic human spot-checking.",
    Trustworthiness.LOW: "Not recommended without human review of every result.",
    Trustworthiness.VERY_LOW: "Do not use in production; investigate root causes first.",
    Trustworthiness.INSUFFICIENT_DATA: (
        "Insufficient validation data to make a production recommendation."
    ),
}


@dataclass(frozen=True)
class JudgeValidationReport:
    generated_at: str
    agreement: tuple[AgreementResult, ...]
    consistency: tuple[ConsistencyResult, ...]
    calibration: tuple[CalibrationResult, ...]
    bias: tuple[BiasFinding, ...]
    correlation: tuple[CorrelationResult, ...]
    overall_trustworthiness: Trustworthiness
    recommended_production_usage: str
    confidence_level: float


def _classify_trustworthiness(score: float) -> Trustworthiness:
    if score >= 0.75:
        return Trustworthiness.HIGH
    if score >= 0.5:
        return Trustworthiness.MEDIUM
    if score >= 0.25:
        return Trustworthiness.LOW
    return Trustworthiness.VERY_LOW


def assemble_validation_report(
    *,
    agreement: Sequence[AgreementResult] = (),
    consistency: Sequence[ConsistencyResult] = (),
    calibration: Sequence[CalibrationResult] = (),
    bias: Sequence[BiasFinding] = (),
    correlation: Sequence[CorrelationResult] = (),
) -> JudgeValidationReport:
    """See module docstring's "Trustworthiness methodology"."""
    total_n = (
        sum(a.n for a in agreement) + sum(c.n for c in consistency)
        + sum(c.n for c in calibration) + sum(b.n for b in bias)
        + sum(c.n for c in correlation)
    )
    if total_n == 0:
        return JudgeValidationReport(
            generated_at=datetime.now(UTC).isoformat(), agreement=tuple(agreement),
            consistency=tuple(consistency), calibration=tuple(calibration), bias=tuple(bias),
            correlation=tuple(correlation),
            overall_trustworthiness=Trustworthiness.INSUFFICIENT_DATA,
            recommended_production_usage=_RECOMMENDATION_BY_TRUSTWORTHINESS[
                Trustworthiness.INSUFFICIENT_DATA
            ],
            confidence_level=0.0,
        )

    score = 1.0
    score -= PENALTY_PER_FINDING * len(bias)
    score -= PENALTY_PER_FINDING * sum(
        1 for a in agreement
        if a.agreement_within_tolerance is not None
        and a.agreement_within_tolerance < LOW_AGREEMENT_THRESHOLD
    )
    score -= PENALTY_PER_FINDING * sum(
        1 for c in consistency if c.std_dev > HIGH_STD_DEV_THRESHOLD
    )
    score -= PENALTY_PER_FINDING * sum(1 for c in calibration if c.direction == "negative")
    score -= PENALTY_PER_FINDING * sum(1 for c in correlation if c.direction == "negative")
    score = max(0.0, min(1.0, score))

    trustworthiness = _classify_trustworthiness(score)
    confidence_level = min(1.0, total_n / CONFIDENCE_BASELINE_N)

    return JudgeValidationReport(
        generated_at=datetime.now(UTC).isoformat(), agreement=tuple(agreement),
        consistency=tuple(consistency), calibration=tuple(calibration), bias=tuple(bias),
        correlation=tuple(correlation), overall_trustworthiness=trustworthiness,
        recommended_production_usage=_RECOMMENDATION_BY_TRUSTWORTHINESS[trustworthiness],
        confidence_level=confidence_level,
    )


# ── Benchmark integration ────────────────────────────────────────────────────────


def build_validation_report_from_benchmarks(
    *,
    judged_repo: JudgedReasoningBenchmarkRepository,
    experiment_name: str | None = None,
    existing_agreement: Sequence[AgreementResult] = (),
    existing_consistency: Sequence[ConsistencyResult] = (),
    existing_calibration: Sequence[CalibrationResult] = (),
    existing_bias: Sequence[BiasFinding] = (),
) -> JudgeValidationReport:
    """Build a ``JudgeValidationReport`` whose correlation analyses are
    derived directly from a ``JudgedReasoningBenchmarkRepository``'s run
    history (Phase 20B, unmodified) - see module docstring's "Benchmark
    integration". ``existing_*`` lets a caller fold in agreement/
    consistency/calibration/bias analyses computed elsewhere (e.g. against
    a human evaluation dataset) into the SAME final report.
    """
    runs = judged_repo.list_runs(experiment_name=experiment_name)

    judge_scores: list[float] = []
    decision_accuracies: list[float] = []
    regression_numbers: list[float] = []
    regression_paired_scores: list[float] = []

    for run in runs:
        if run.judge_aggregate is None or run.judge_aggregate.mean_session_score is None:
            continue
        mean_session = run.judge_aggregate.mean_session_score
        decision_accuracy = run.reasoning_run.report.metrics.decision_accuracy
        if decision_accuracy is not None:
            judge_scores.append(mean_session)
            decision_accuracies.append(decision_accuracy)

        if run.reasoning_run.regression is not None:
            number = regression_verdict_to_number(run.reasoning_run.regression.verdict.value)
            if number is not None:
                regression_paired_scores.append(mean_session)
                regression_numbers.append(number)

    correlation_results: list[CorrelationResult] = []
    if judge_scores:
        correlation_results.append(
            analyze_correlation(
                "mean_session_judge_score", judge_scores, "decision_accuracy", decision_accuracies
            )
        )
    if regression_paired_scores:
        correlation_results.append(
            analyze_correlation(
                "mean_session_judge_score", regression_paired_scores,
                "regression_verdict", regression_numbers,
            )
        )

    return assemble_validation_report(
        agreement=existing_agreement, consistency=existing_consistency,
        calibration=existing_calibration, bias=existing_bias, correlation=correlation_results,
    )
