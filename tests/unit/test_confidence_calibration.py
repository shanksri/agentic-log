"""Tests for Phase 22F: confidence-calibration reporting extensions
(app/evaluation/confidence_calibration.py). Pure functions, no DB -- verifies
signal computation, group statistics, and group-scoped confusion-matrix
diagnostics without touching `classify_confidence` or its thresholds."""
from __future__ import annotations

import pytest

from app.evaluation.confidence_calibration import (
    CalibrationGroup,
    build_calibration_row,
    compute_retrieval_signals,
    confusion_matrix_report,
    group_statistics,
)
from app.services.confidence import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM


# ── compute_retrieval_signals ────────────────────────────────────────────────


def test_signals_empty_scores() -> None:
    signals = compute_retrieval_signals([], [])
    assert signals.top1 is None
    assert signals.top2 is None
    assert signals.top1_minus_top2 is None
    assert signals.topk_mean is None
    assert signals.topk_min is None
    assert signals.num_results == 0
    assert signals.num_distinct_sources == 0


def test_signals_single_result_has_no_top2() -> None:
    signals = compute_retrieval_signals([0.5], ["github:a/b"])
    assert signals.top1 == 0.5
    assert signals.top2 is None
    assert signals.top1_minus_top2 is None
    assert signals.topk_mean == 0.5
    assert signals.topk_min == 0.5
    assert signals.top1_minus_topk_min == 0.0
    assert signals.num_results == 1


def test_signals_multiple_results_computed_correctly() -> None:
    scores = [0.8, 0.6, 0.4, 0.2]
    sources = ["github:a/b", "github:a/b", "github:c/d", "jira:E"]
    signals = compute_retrieval_signals(scores, sources)
    assert signals.top1 == 0.8
    assert signals.top2 == 0.6
    assert signals.top1_minus_top2 == pytest.approx(0.2)
    assert signals.topk_mean == pytest.approx(0.5)
    assert signals.topk_min == 0.2
    assert signals.top1_minus_topk_min == pytest.approx(0.6)
    assert signals.num_results == 4
    assert signals.num_distinct_sources == 3  # a/b counted once


def test_signals_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        compute_retrieval_signals([0.5, 0.4], ["only-one"])


# ── build_calibration_row / classify_confidence integration ────────────────


def test_build_row_uses_real_classify_confidence_unmodified() -> None:
    low = build_calibration_row(
        id="q1", group=CalibrationGroup.HARD_NEGATIVE, query_type="hard-negative-expected",
        scores=[0.30], source_keys=["github:a/b"],
    )
    medium = build_calibration_row(
        id="q2", group=CalibrationGroup.HARD_NEGATIVE, query_type="hard-negative-expected",
        scores=[0.45], source_keys=["github:a/b"],
    )
    high = build_calibration_row(
        id="q3", group=CalibrationGroup.POSITIVE, query_type="lexical-overlap",
        scores=[0.80], source_keys=["github:a/b"],
    )
    assert low.confidence_level == CONFIDENCE_LOW
    assert medium.confidence_level == CONFIDENCE_MEDIUM
    assert high.confidence_level == CONFIDENCE_HIGH


def test_build_row_no_results_is_low_confidence() -> None:
    row = build_calibration_row(
        id="q1", group=CalibrationGroup.EASY_NEGATIVE, query_type="no-match-expected",
        scores=[], source_keys=[],
    )
    assert row.top1_score is None
    assert row.confidence_level == CONFIDENCE_LOW


def test_calibration_row_is_match_matches_group() -> None:
    positive = build_calibration_row(
        id="p", group=CalibrationGroup.POSITIVE, query_type="lexical-overlap",
        scores=[0.9], source_keys=["a"],
    )
    easy_neg = build_calibration_row(
        id="e", group=CalibrationGroup.EASY_NEGATIVE, query_type="no-match-expected",
        scores=[0.2], source_keys=["a"],
    )
    hard_neg = build_calibration_row(
        id="h", group=CalibrationGroup.HARD_NEGATIVE, query_type="hard-negative-expected",
        scores=[0.6], source_keys=["a"],
    )
    assert positive.is_match is True
    assert easy_neg.is_match is False
    assert hard_neg.is_match is False


# ── group_statistics ──────────────────────────────────────────────────────


def test_group_statistics_computed_per_group() -> None:
    rows = [
        build_calibration_row(id="p1", group=CalibrationGroup.POSITIVE, query_type="t", scores=[0.8], source_keys=["a"]),
        build_calibration_row(id="p2", group=CalibrationGroup.POSITIVE, query_type="t", scores=[0.6], source_keys=["a"]),
        build_calibration_row(id="h1", group=CalibrationGroup.HARD_NEGATIVE, query_type="t", scores=[0.5], source_keys=["a"]),
    ]
    stats = group_statistics(rows)

    assert stats["POSITIVE"].count == 2
    assert stats["POSITIVE"].min_top1 == 0.6
    assert stats["POSITIVE"].max_top1 == 0.8
    assert stats["POSITIVE"].mean_top1 == pytest.approx(0.7)
    assert stats["POSITIVE"].median_top1 == pytest.approx(0.7)
    assert stats["POSITIVE"].level_counts == {CONFIDENCE_HIGH: 2}

    assert stats["HARD_NEGATIVE"].count == 1
    assert stats["HARD_NEGATIVE"].level_counts == {CONFIDENCE_MEDIUM: 1}
    assert "EASY_NEGATIVE" not in stats  # no rows in that group


def test_group_statistics_excludes_none_scores_from_numeric_stats() -> None:
    rows = [
        build_calibration_row(id="e1", group=CalibrationGroup.EASY_NEGATIVE, query_type="t", scores=[], source_keys=[]),
    ]
    stats = group_statistics(rows)
    assert stats["EASY_NEGATIVE"].count == 1
    assert stats["EASY_NEGATIVE"].min_top1 is None
    assert stats["EASY_NEGATIVE"].mean_top1 is None
    assert stats["EASY_NEGATIVE"].level_counts == {CONFIDENCE_LOW: 1}


# ── confusion_matrix_report ──────────────────────────────────────────────


def _row(id_: str, group: CalibrationGroup, score: float) -> object:
    return build_calibration_row(id=id_, group=group, query_type="t", scores=[score], source_keys=["a"])


def test_confusion_matrix_matches_existing_script_semantics() -> None:
    # Mirrors tests/eval/run_confidence_eval.py's confusion_matrix() shape:
    # predicted MATCH if top1_score >= threshold.
    rows = [
        _row("p1", CalibrationGroup.POSITIVE, 0.60),   # TP at 0.40
        _row("p2", CalibrationGroup.POSITIVE, 0.30),   # FN at 0.40
        _row("n1", CalibrationGroup.EASY_NEGATIVE, 0.20),  # TN at 0.40
        _row("n2", CalibrationGroup.HARD_NEGATIVE, 0.50),  # FP at 0.40
    ]
    result = confusion_matrix_report(rows, threshold=0.40)

    assert result.tp == 1
    assert result.fn == 1
    assert result.tn == 1
    assert result.fp == 1
    assert result.false_positive_ids == ("n2",)
    assert result.false_negative_ids == ("p2",)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)


def test_confusion_matrix_can_scope_to_hard_negatives_only() -> None:
    rows = [
        _row("p1", CalibrationGroup.POSITIVE, 0.60),
        _row("easy1", CalibrationGroup.EASY_NEGATIVE, 0.90),  # would be a FP if included
        _row("hard1", CalibrationGroup.HARD_NEGATIVE, 0.20),  # TN
    ]
    hard_only = confusion_matrix_report(
        rows, threshold=0.40, negative_groups=(CalibrationGroup.HARD_NEGATIVE,)
    )
    # easy1 excluded entirely -- only p1 (TP) and hard1 (TN) count.
    assert hard_only.tp == 1
    assert hard_only.tn == 1
    assert hard_only.fp == 0
    assert hard_only.negative_groups == ("HARD_NEGATIVE",)


def test_confusion_matrix_default_scope_combines_both_negative_groups() -> None:
    rows = [
        _row("p1", CalibrationGroup.POSITIVE, 0.60),
        _row("easy1", CalibrationGroup.EASY_NEGATIVE, 0.10),
        _row("hard1", CalibrationGroup.HARD_NEGATIVE, 0.20),
    ]
    result = confusion_matrix_report(rows, threshold=0.40)
    assert result.tp == 1
    assert result.tn == 2  # both negatives correctly below threshold
    assert set(result.negative_groups) == {"EASY_NEGATIVE", "HARD_NEGATIVE"}


def test_confusion_matrix_none_top1_treated_as_zero_for_thresholding() -> None:
    row = build_calibration_row(
        id="e1", group=CalibrationGroup.EASY_NEGATIVE, query_type="t", scores=[], source_keys=[]
    )
    result = confusion_matrix_report([row], threshold=0.40)
    assert result.tn == 1  # None -> 0.0 -> correctly below any positive threshold


def test_confusion_matrix_is_deterministic() -> None:
    rows = [
        _row("p1", CalibrationGroup.POSITIVE, 0.55),
        _row("h1", CalibrationGroup.HARD_NEGATIVE, 0.55),  # exact tie at threshold
    ]
    first = confusion_matrix_report(rows, threshold=0.55)
    second = confusion_matrix_report(rows, threshold=0.55)
    assert first == second
    assert first.tp == 1
    assert first.fp == 1  # >= threshold counts as predicted MATCH, same as the existing script
