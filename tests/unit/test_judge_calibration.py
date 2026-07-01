from __future__ import annotations

import pytest

from app.evaluation.judge_calibration import (
    CalibrationPoint,
    analyze_calibration,
    analyze_correlation,
    classify_direction,
    pearson_correlation,
    regression_verdict_to_number,
)


def test_pearson_correlation_perfect_positive() -> None:
    r = pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert r == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative() -> None:
    r = pearson_correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])
    assert r == pytest.approx(-1.0)


def test_pearson_correlation_none_when_fewer_than_two_points() -> None:
    assert pearson_correlation([1.0], [2.0]) is None
    assert pearson_correlation([], []) is None


def test_pearson_correlation_none_when_zero_variance() -> None:
    assert pearson_correlation([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None
    assert pearson_correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None


def test_pearson_correlation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        pearson_correlation([1.0, 2.0], [1.0])


def test_classify_direction_bands() -> None:
    assert classify_direction(None) == "undefined"
    assert classify_direction(0.5) == "positive"
    assert classify_direction(-0.5) == "negative"
    assert classify_direction(0.1) == "weak"
    assert classify_direction(-0.1) == "weak"
    assert classify_direction(0.2) == "positive"
    assert classify_direction(-0.2) == "negative"


def test_regression_verdict_to_number_mapping() -> None:
    assert regression_verdict_to_number("improved") == 1.0
    assert regression_verdict_to_number("unchanged") == 0.0
    assert regression_verdict_to_number("mixed") == 0.0
    assert regression_verdict_to_number("regressed") == -1.0
    assert regression_verdict_to_number("incompatible") is None
    assert regression_verdict_to_number("not_a_real_verdict") is None


# ── Calibration ────────────────────────────────────────────────────────────────


def test_analyze_calibration_positive_correlation() -> None:
    points = (
        CalibrationPoint("s1", judge_score=9.0, quality_metric=1.0),
        CalibrationPoint("s2", judge_score=5.0, quality_metric=0.5),
        CalibrationPoint("s3", judge_score=2.0, quality_metric=0.0),
    )
    result = analyze_calibration("decision_accuracy", points)
    assert result.n == 3
    assert result.correlation == pytest.approx(0.9966, abs=1e-3)
    assert result.direction == "positive"
    assert result.points == points


def test_analyze_calibration_undefined_with_insufficient_data() -> None:
    result = analyze_calibration("decision_accuracy", (CalibrationPoint("s1", 9.0, 1.0),))
    assert result.correlation is None
    assert result.direction == "undefined"


def test_analyze_calibration_empty_input() -> None:
    result = analyze_calibration("decision_accuracy", ())
    assert result.n == 0
    assert result.correlation is None


# ── Correlation ────────────────────────────────────────────────────────────────


def test_analyze_correlation_reports_both_series_names() -> None:
    result = analyze_correlation(
        "judge_score", [9.0, 5.0, 2.0], "recall_at_k", [1.0, 0.5, 0.0]
    )
    assert result.series_a_name == "judge_score"
    assert result.series_b_name == "recall_at_k"
    assert result.n == 3
    assert result.direction == "positive"


def test_analyze_correlation_is_deterministic() -> None:
    first = analyze_correlation("a", [1.0, 2.0, 3.0, 4.0], "b", [5.0, 1.0, 8.0, 2.0])
    second = analyze_correlation("a", [1.0, 2.0, 3.0, 4.0], "b", [5.0, 1.0, 8.0, 2.0])
    assert first == second
