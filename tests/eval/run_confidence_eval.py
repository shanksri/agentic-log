"""Phase 4: confidence calibration evaluation.

Runs the gold query set against IncidentSearchService.search() (dense-only,
no expansion/reranking) and evaluates how well top1_score separates queries
that have a genuine expected match (MATCH) from no-match-expected queries
(NO_MATCH).

For a set of candidate thresholds, reports a confusion matrix (treating
"top1_score >= threshold" as a predicted MATCH) plus precision, recall, false
positive rate, and false negative rate.

Usage:
    python -m tests.eval.run_confidence_eval
    python -m tests.eval.run_confidence_eval --output tests/eval/results/confidence_v4.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.db.session import SessionLocal
from app.services.confidence import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    classify_confidence,
)
from app.services.search import IncidentSearchService

GOLD_QUERIES_PATH = Path(__file__).parent / "gold_queries.json"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "confidence_v4.json"

# Candidate thresholds to evaluate, spanning the observed score range.
CANDIDATE_THRESHOLDS = [0.30, 0.35, LOW_CONFIDENCE_THRESHOLD, 0.42, 0.45, HIGH_CONFIDENCE_THRESHOLD, 0.60]


def confusion_matrix(rows: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted_match = row["top1_score"] >= threshold
        actual_match = row["is_match"]
        if predicted_match and actual_match:
            tp += 1
        elif predicted_match and not actual_match:
            fp += 1
        elif not predicted_match and actual_match:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the confidence calibration evaluation")
    parser.add_argument("--gold-queries", type=Path, default=GOLD_QUERIES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    gold = json.loads(args.gold_queries.read_text(encoding="utf-8"))
    queries = gold["queries"]

    db = SessionLocal()
    try:
        search_service = IncidentSearchService(db)

        rows = []
        for entry in queries:
            results = search_service.search(entry["query"], limit=5)
            top1_score, top2_score = None, None
            if results:
                top1_score = results[0].similarity_score
            if len(results) > 1:
                top2_score = results[1].similarity_score
            top5_mean = (
                sum(r.similarity_score for r in results[:5]) / len(results[:5]) if results else None
            )
            is_match = len(entry["expected_incident_ids"]) > 0
            confidence_level = classify_confidence(top1_score)
            rows.append(
                {
                    "id": entry["id"],
                    "query_type": entry["query_type"],
                    "is_match": is_match,
                    "top1_score": top1_score if top1_score is not None else 0.0,
                    "top2_score": top2_score,
                    "top5_mean_score": top5_mean,
                    "top1_minus_top2": (
                        top1_score - top2_score
                        if top1_score is not None and top2_score is not None
                        else None
                    ),
                    "confidence_level": confidence_level,
                }
            )
    finally:
        db.close()

    match_scores = [row["top1_score"] for row in rows if row["is_match"]]
    no_match_scores = [row["top1_score"] for row in rows if not row["is_match"]]

    confidence_counts: dict[str, dict[str, int]] = {"MATCH": {}, "NO_MATCH": {}}
    for row in rows:
        bucket = "MATCH" if row["is_match"] else "NO_MATCH"
        level = row["confidence_level"]
        confidence_counts[bucket][level] = confidence_counts[bucket].get(level, 0) + 1

    matrices = [confusion_matrix(rows, threshold) for threshold in CANDIDATE_THRESHOLDS]

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "thresholds": {
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        },
        "score_distributions": {
            "match": {
                "count": len(match_scores),
                "min": min(match_scores) if match_scores else None,
                "max": max(match_scores) if match_scores else None,
            },
            "no_match": {
                "count": len(no_match_scores),
                "min": min(no_match_scores) if no_match_scores else None,
                "max": max(no_match_scores) if no_match_scores else None,
            },
        },
        "confidence_level_breakdown": confidence_counts,
        "candidate_threshold_confusion_matrices": matrices,
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print("\nScore distributions:")
    print(f"  MATCH:    {report['score_distributions']['match']}")
    print(f"  NO_MATCH: {report['score_distributions']['no_match']}")
    print("\nConfidence level breakdown:")
    print(f"  MATCH:    {confidence_counts['MATCH']}")
    print(f"  NO_MATCH: {confidence_counts['NO_MATCH']}")
    print("\nCandidate threshold confusion matrices (predicted MATCH if top1_score >= threshold):")
    for matrix in matrices:
        print(
            f"  threshold={matrix['threshold']:.2f}: "
            f"TP={matrix['tp']} FP={matrix['fp']} TN={matrix['tn']} FN={matrix['fn']} "
            f"precision={matrix['precision']} recall={matrix['recall']} "
            f"FPR={matrix['false_positive_rate']} FNR={matrix['false_negative_rate']}"
        )


if __name__ == "__main__":
    main()
