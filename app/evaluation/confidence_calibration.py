"""Phase 22F: confidence-calibration reporting extensions.

Analysis/reporting layer only. Does NOT change `classify_confidence` or its
thresholds (`app/services/confidence.py`, unmodified, imported here exactly
as every other consumer imports it), does not change retrieval, generation,
or routing behavior, and introduces no production gating.

# Why this module exists

Phase 22E found that a real negative-control query's top1_score (0.5204)
sits in the calibrated MEDIUM band, not LOW -- and traced the calibration
itself (PHASE_4_CONFIDENCE_CALIBRATION.md) to only 4 negative-control
queries, none topically close to any ingested repo's own product surface.
The calibration doc's own authors flagged this gap in writing at the time
("MEDIUM is currently un-validated against negatives"). This module adds
the reporting machinery to test that gap empirically: group-aware
(POSITIVE / EASY_NEGATIVE / HARD_NEGATIVE) score-distribution statistics,
additional per-query retrieval signals beyond top1_score alone (top1-top2
gap, top-k mean/min, source diversity), and group-scoped confusion-matrix
diagnostics against the *existing*, unmodified thresholds --
`tests/eval/run_phase22f_confidence_calibration.py` is the live-DB runner
that wires this to real retrieval; this module contains only pure functions
operating on already-computed score lists, so it is unit-testable without a
database (mirrors the project's existing split between `app/evaluation/*`
library code and `tests/eval/run_*.py` live-wiring scripts).

# What this module does NOT do

- Does not change LOW_CONFIDENCE_THRESHOLD / HIGH_CONFIDENCE_THRESHOLD.
- Does not call classify_confidence with anything other than top1_score,
  exactly as every existing call site does.
- Does not decide or recommend a new threshold -- `confusion_matrix_report`
  reports candidate-threshold diagnostics for a human to read; it never
  selects or applies one.
- Does not query the database or call any retrieval service itself.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum

from app.services.confidence import classify_confidence


class CalibrationGroup(str, Enum):
    """Which population a calibration row belongs to. POSITIVE rows are
    treated as `is_match=True` for confusion-matrix purposes; both negative
    groups are `is_match=False` -- EASY vs HARD only changes which group a
    row is reported under, never how it's scored.
    """

    POSITIVE = "POSITIVE"
    EASY_NEGATIVE = "EASY_NEGATIVE"
    HARD_NEGATIVE = "HARD_NEGATIVE"

    @property
    def is_match(self) -> bool:
        return self is CalibrationGroup.POSITIVE


@dataclass(frozen=True)
class RetrievalSignals:
    """Per-query retrieval signals beyond top1_score alone -- the
    "additional signal" candidates Phase 22E identified. Computed from an
    already-fetched, rank-ordered score list; never re-queries anything.

    All fields are `None` when there are zero results (nothing to compute).
    `top2`/`top1_minus_top2` are `None` when there is only one result.
    """

    top1: float | None
    top2: float | None
    top1_minus_top2: float | None
    topk_mean: float | None
    topk_min: float | None
    top1_minus_topk_min: float | None
    num_results: int
    num_distinct_sources: int


def compute_retrieval_signals(
    scores: list[float], source_keys: list[str]
) -> RetrievalSignals:
    """Build `RetrievalSignals` from a rank-ordered (descending) similarity
    score list and a parallel list of source identifiers (e.g.
    ``f"{source_type}:{owner}/{repo}"``) used only to count distinct sources
    -- a coarse proxy for "is this corroborated by more than one place in
    the corpus, or just one narrow cluster."
    """
    if len(scores) != len(source_keys):
        raise ValueError(
            f"scores and source_keys must be the same length, got "
            f"{len(scores)} and {len(source_keys)}"
        )
    if not scores:
        return RetrievalSignals(
            top1=None,
            top2=None,
            top1_minus_top2=None,
            topk_mean=None,
            topk_min=None,
            top1_minus_topk_min=None,
            num_results=0,
            num_distinct_sources=0,
        )

    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else None
    topk_mean = statistics.mean(scores)
    topk_min = min(scores)
    return RetrievalSignals(
        top1=top1,
        top2=top2,
        top1_minus_top2=(top1 - top2) if top2 is not None else None,
        topk_mean=topk_mean,
        topk_min=topk_min,
        top1_minus_topk_min=top1 - topk_min,
        num_results=len(scores),
        num_distinct_sources=len(set(source_keys)),
    )


@dataclass(frozen=True)
class CalibrationRow:
    """One query's calibration record: which group it belongs to, its
    computed confidence level (via the real, unmodified
    ``classify_confidence``), and its full retrieval-signal breakdown.
    """

    id: str
    group: CalibrationGroup
    query_type: str
    confidence_level: str
    signals: RetrievalSignals

    @property
    def is_match(self) -> bool:
        return self.group.is_match

    @property
    def top1_score(self) -> float | None:
        return self.signals.top1


def build_calibration_row(
    *,
    id: str,  # noqa: A002 - matches the existing script's field name
    group: CalibrationGroup,
    query_type: str,
    scores: list[float],
    source_keys: list[str],
) -> CalibrationRow:
    """Build one `CalibrationRow`, computing confidence via the real
    `classify_confidence(top1_score)` -- identical call shape to every
    other consumer in the codebase.
    """
    signals = compute_retrieval_signals(scores, source_keys)
    confidence_level = classify_confidence(signals.top1)
    return CalibrationRow(
        id=id, group=group, query_type=query_type,
        confidence_level=confidence_level, signals=signals,
    )


@dataclass(frozen=True)
class GroupStatistics:
    group: CalibrationGroup
    count: int
    min_top1: float | None
    max_top1: float | None
    mean_top1: float | None
    median_top1: float | None
    level_counts: dict[str, int]


def group_statistics(rows: list[CalibrationRow]) -> dict[str, GroupStatistics]:
    """Per-group count/min/max/mean/median of top1_score, plus LOW/MEDIUM/
    HIGH breakdown. Rows with no results (``top1_score is None``) are
    counted in ``level_counts`` (always LOW, per ``classify_confidence``'s
    own contract) but excluded from the numeric score statistics.
    """
    by_group: dict[CalibrationGroup, list[CalibrationRow]] = {}
    for row in rows:
        by_group.setdefault(row.group, []).append(row)

    result: dict[str, GroupStatistics] = {}
    for group, group_rows in by_group.items():
        scores = [r.top1_score for r in group_rows if r.top1_score is not None]
        level_counts: dict[str, int] = {}
        for row in group_rows:
            level_counts[row.confidence_level] = level_counts.get(row.confidence_level, 0) + 1
        result[group.value] = GroupStatistics(
            group=group,
            count=len(group_rows),
            min_top1=min(scores) if scores else None,
            max_top1=max(scores) if scores else None,
            mean_top1=statistics.mean(scores) if scores else None,
            median_top1=statistics.median(scores) if scores else None,
            level_counts=level_counts,
        )
    return result


@dataclass(frozen=True)
class ConfusionMatrixResult:
    threshold: float
    negative_groups: tuple[str, ...]
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    false_positive_ids: tuple[str, ...]
    false_negative_ids: tuple[str, ...]


def confusion_matrix_report(
    rows: list[CalibrationRow],
    threshold: float,
    *,
    negative_groups: tuple[CalibrationGroup, ...] = (
        CalibrationGroup.EASY_NEGATIVE,
        CalibrationGroup.HARD_NEGATIVE,
    ),
) -> ConfusionMatrixResult:
    """Same confusion-matrix definition as the existing
    ``tests/eval/run_confidence_eval.py::confusion_matrix`` ("predicted
    MATCH if top1_score >= threshold"), generalized to scope the negative
    class to one or more `CalibrationGroup`s -- e.g. pass only
    ``(HARD_NEGATIVE,)`` to see how the threshold performs against hard
    negatives in isolation, ignoring easy ones entirely. POSITIVE rows are
    always the positive class.
    """
    scoped = [
        row for row in rows
        if row.group is CalibrationGroup.POSITIVE or row.group in negative_groups
    ]
    tp = fp = tn = fn = 0
    false_positive_ids: list[str] = []
    false_negative_ids: list[str] = []
    for row in scoped:
        predicted_match = (row.top1_score or 0.0) >= threshold
        actual_match = row.is_match
        if predicted_match and actual_match:
            tp += 1
        elif predicted_match and not actual_match:
            fp += 1
            false_positive_ids.append(row.id)
        elif not predicted_match and actual_match:
            fn += 1
            false_negative_ids.append(row.id)
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None

    return ConfusionMatrixResult(
        threshold=threshold,
        negative_groups=tuple(g.value for g in negative_groups),
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=precision, recall=recall,
        false_positive_rate=fpr, false_negative_rate=fnr,
        false_positive_ids=tuple(false_positive_ids),
        false_negative_ids=tuple(false_negative_ids),
    )
