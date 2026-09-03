"""Phase 22F: confidence calibration expansion — hard-negative diagnostics.

Positive/easy-negative source: tests/eval/gold/phase17c_benchmark_v1.json
(28 positive + 8 easy-negative queries), NOT the older gold_queries.json.
Reason (a Phase 22F finding in its own right, see the report): gold_queries.json
stores raw incident UUIDs as its match targets, which are NOT stable identity
(re-ingestion regenerates UUIDs -- source_external_id is the stable key, per
docs/architecture/01_system_overview.md's "stable identity, volatile rows"
principle). After this session's corpus restoration, ALL 10 of its
lexical-overlap queries' target UUIDs were verified missing from the live DB
(see report) -- so its measured top1 scores today reflect "whatever's
nearest to a now-absent target," not genuine match confidence, and running
it would silently contaminate the calibration. phase17c_benchmark_v1.json
uses (source_type, source_external_id) and was verified 34/34 resolved
against the current corpus in the same session (see
docs/evaluation_summary.md's hybrid-search section) -- its positives are
known to still exist.

Runs phase17c_benchmark_v1.json's 28 positives + 8 easy negatives, plus the
NEW gold_queries_hard_negatives_v1.json (20 hard negatives), against
IncidentSearchService.search() (dense-only, no expansion/reranking --
identical retrieval protocol to tests/eval/run_confidence_eval.py), then
reports group-aware (POSITIVE / EASY_NEGATIVE / HARD_NEGATIVE) score
distributions, additional retrieval signals, and group-scoped
confusion-matrix diagnostics against the EXISTING, unmodified confidence
thresholds.

Analysis/reporting only -- does not change classify_confidence, its
thresholds, retrieval, generation, or routing. See
app/evaluation/confidence_calibration.py for the pure reporting logic this
script wires to a live database.

Usage:
    python -m tests.eval.run_phase22f_confidence_calibration
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.db.session import SessionLocal
from app.evaluation.confidence_calibration import (
    CalibrationGroup,
    CalibrationRow,
    build_calibration_row,
    confusion_matrix_report,
    group_statistics,
)
from app.services.confidence import HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
from app.services.search import IncidentSearchService

POSITIVE_SOURCE_PATH = Path(__file__).parent / "gold" / "phase17c_benchmark_v1.json"
HARD_NEGATIVES_PATH = Path(__file__).parent / "gold_queries_hard_negatives_v1.json"
OUTPUT_PATH = Path(__file__).parent / "results" / "phase22f_confidence_calibration.json"

# Kept only for the staleness comparison documented in the report -- NOT used
# as this run's positive/easy-negative source (see module docstring).
STALE_GOLD_QUERIES_PATH = Path(__file__).parent / "gold_queries.json"

RETRIEVAL_LIMIT = 10  # top-k for the additional score-distribution signals

# Candidate thresholds spanning the observed range -- reported only, never applied.
CANDIDATE_THRESHOLDS = [0.30, 0.35, LOW_CONFIDENCE_THRESHOLD, 0.42, 0.45, 0.48, 0.50, HIGH_CONFIDENCE_THRESHOLD, 0.60, 0.65, 0.70]


def _source_key(incident) -> str:
    owner = getattr(incident, "owner", None)
    repo = getattr(incident, "repo", None)
    if owner and repo:
        return f"{incident.source_type}:{owner}/{repo}"
    return f"{incident.source_type}:{getattr(incident, 'source_external_id', incident.id)}"


def _build_row(search: IncidentSearchService, *, id_: str, group: CalibrationGroup, query_type: str, query: str) -> CalibrationRow:
    results = search.search(query, limit=RETRIEVAL_LIMIT)
    scores = [r.similarity_score for r in results]
    source_keys = [_source_key(r.incident) for r in results]
    return build_calibration_row(id=id_, group=group, query_type=query_type, scores=scores, source_keys=source_keys)


def main() -> None:
    positives_and_easy = json.loads(POSITIVE_SOURCE_PATH.read_text(encoding="utf-8"))["queries"]
    hard_negatives = json.loads(HARD_NEGATIVES_PATH.read_text(encoding="utf-8"))["queries"]

    db = SessionLocal()
    try:
        search = IncidentSearchService(db)
        rows: list[CalibrationRow] = []

        for entry in positives_and_easy:
            is_positive = len(entry["expected_incidents"]) > 0
            group = CalibrationGroup.POSITIVE if is_positive else CalibrationGroup.EASY_NEGATIVE
            rows.append(
                _build_row(search, id_=entry["id"], group=group, query_type=entry["category"], query=entry["query"])
            )

        for entry in hard_negatives:
            rows.append(
                _build_row(
                    search, id_=entry["id"], group=CalibrationGroup.HARD_NEGATIVE,
                    query_type=entry["query_type"], query=entry["query"],
                )
            )
    finally:
        db.close()

    stats = group_statistics(rows)

    combined_matrices = [confusion_matrix_report(rows, t) for t in CANDIDATE_THRESHOLDS]
    hard_only_matrices = [
        confusion_matrix_report(rows, t, negative_groups=(CalibrationGroup.HARD_NEGATIVE,))
        for t in CANDIDATE_THRESHOLDS
    ]
    easy_only_matrices = [
        confusion_matrix_report(rows, t, negative_groups=(CalibrationGroup.EASY_NEGATIVE,))
        for t in CANDIDATE_THRESHOLDS
    ]

    def _row_to_dict(row: CalibrationRow) -> dict:
        d = asdict(row)
        d["group"] = row.group.value
        d["is_match"] = row.is_match
        d["top1_score"] = row.top1_score
        return d

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "22F",
        "protocol_note": (
            "Same protocol as tests/eval/run_confidence_eval.py: dense-only "
            "IncidentSearchService.search(), no expansion/reranking. "
            "classify_confidence and its 0.40/0.55 thresholds are imported "
            "unmodified."
        ),
        "thresholds": {
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        },
        "group_statistics": {
            name: {
                "count": s.count, "min_top1": s.min_top1, "max_top1": s.max_top1,
                "mean_top1": s.mean_top1, "median_top1": s.median_top1,
                "level_counts": s.level_counts,
            }
            for name, s in stats.items()
        },
        "confusion_matrices": {
            "combined_negatives (easy+hard)": [asdict(m) for m in combined_matrices],
            "hard_negatives_only": [asdict(m) for m in hard_only_matrices],
            "easy_negatives_only": [asdict(m) for m in easy_only_matrices],
        },
        "rows": [_row_to_dict(r) for r in rows],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")

    print("\n=== Group statistics ===")
    for name, s in stats.items():
        print(f"  {name}: n={s.count} min={s.min_top1} max={s.max_top1} mean={s.mean_top1} median={s.median_top1}")
        print(f"    level_counts={s.level_counts}")

    print("\n=== Combined-negatives confusion matrix (candidate thresholds) ===")
    for m in combined_matrices:
        print(f"  t={m.threshold:.2f} TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn} precision={m.precision} recall={m.recall}")

    print("\n=== Hard-negatives-ONLY confusion matrix (candidate thresholds) ===")
    for m in hard_only_matrices:
        print(f"  t={m.threshold:.2f} TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn} precision={m.precision} recall={m.recall} FP_ids={m.false_positive_ids}")


if __name__ == "__main__":
    main()
