"""Judge Validation — Calibration and Correlation Analysis (Phase 21B).

A pure statistics layer: Pearson correlation over already-paired numeric
series. Never calls a Judge, never reruns retrieval/reasoning. "Do not
invent calibration curves beyond available data" - this module computes
exactly one number (Pearson's r) per comparison and reports ``None`` (not
a guessed value) whenever fewer than two points exist or either series has
zero variance, rather than fitting any curve.

# Calibration methodology

``analyze_calibration(metric_name, points)`` answers "do higher judge
scores correspond to genuinely stronger systems" for ONE named external
quality metric (e.g. "reasoning_decision_accuracy", "retrieval_recall_at_k")
by correlating ``CalibrationPoint.judge_score`` against
``CalibrationPoint.quality_metric`` across every supplied point for that
metric. A ``CalibrationResult.correlation`` of ``None`` means "insufficient
or degenerate data," never "no calibration" - the report says so via
``direction = "undefined"`` rather than implying zero correlation was
computed.

# Correlation methodology

``analyze_correlation(series_a_name, series_a, series_b_name, series_b)``
is the same Pearson computation applied generically to any two named,
equal-length numeric series (e.g. a benchmark history's mean judge score
per run vs. that run's reasoning decision_accuracy, or vs. retrieval
mean_recall_at_k, or vs. an ``AIQualityReport.failure_summary.
total_failures`` count per run, or vs. a regression verdict numerically
encoded via ``regression_verdict_to_number``) - "analyze relationships
between judge scores, retrieval quality, reasoning quality, quality
intelligence reports, regression history," all expressed as the same
"two numeric series in, one correlation result out" shape, never five
bespoke comparison functions.

``regression_verdict_to_number`` maps a verdict STRING (this module never
imports ``app.evaluation.regression.Verdict``/``app.evaluation.
reasoning_regression.Verdict`` to avoid coupling to either concrete
enum - both already serialize to the same four/five lowercase strings)
deterministically: ``"improved" -> 1.0``, ``"unchanged" -> 0.0``,
``"mixed" -> 0.0`` (a mixed verdict is neither net-better nor net-worse on
this single axis), ``"regressed" -> -1.0``, ``"incompatible" -> None``
(not comparable at all, never coerced to a number).

# Direction classification

Both analyses classify a defined correlation ``r`` the same way:
``|r| < 0.2`` -> ``"weak"``, ``r >= 0.2`` -> ``"positive"``, ``r <= -0.2``
-> ``"negative"``. ``0.2`` is the conventional "small effect" threshold
from Cohen's effect-size guidelines for correlation coefficients - the
least arbitrary, most commonly-cited cut point available for this purpose,
not a value invented for this codebase.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

CORRELATION_WEAK_THRESHOLD = 0.2

_REGRESSION_VERDICT_TO_NUMBER: dict[str, float | None] = {
    "improved": 1.0, "unchanged": 0.0, "mixed": 0.0, "regressed": -1.0, "incompatible": None,
}


def regression_verdict_to_number(verdict: str) -> float | None:
    """See module docstring's "Correlation methodology". Returns ``None``
    for ``"incompatible"`` or any unrecognized verdict string.
    """
    return _REGRESSION_VERDICT_TO_NUMBER.get(verdict)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson's r over two equal-length series. ``None`` when fewer than
    two points exist or either series has zero variance (undefined, not
    coerced to 0.0).
    """
    if len(xs) != len(ys):
        raise ValueError(f"series must be equal length, got {len(xs)} and {len(ys)}")
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def classify_direction(correlation: float | None) -> str:
    if correlation is None:
        return "undefined"
    if correlation >= CORRELATION_WEAK_THRESHOLD:
        return "positive"
    if correlation <= -CORRELATION_WEAK_THRESHOLD:
        return "negative"
    return "weak"


# ── Calibration ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationPoint:
    subject_id: str
    judge_score: float
    quality_metric: float


@dataclass(frozen=True)
class CalibrationResult:
    metric_name: str
    n: int
    correlation: float | None
    direction: str
    points: tuple[CalibrationPoint, ...]


def analyze_calibration(
    metric_name: str, points: Sequence[CalibrationPoint]
) -> CalibrationResult:
    correlation = pearson_correlation(
        [p.judge_score for p in points], [p.quality_metric for p in points]
    )
    return CalibrationResult(
        metric_name=metric_name, n=len(points), correlation=correlation,
        direction=classify_direction(correlation), points=tuple(points),
    )


# ── Correlation ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorrelationResult:
    series_a_name: str
    series_b_name: str
    n: int
    correlation: float | None
    direction: str


def analyze_correlation(
    series_a_name: str, series_a: Sequence[float], series_b_name: str, series_b: Sequence[float]
) -> CorrelationResult:
    correlation = pearson_correlation(series_a, series_b)
    return CorrelationResult(
        series_a_name=series_a_name, series_b_name=series_b_name, n=len(series_a),
        correlation=correlation, direction=classify_direction(correlation),
    )
