"""Calibrate a cross-encoder retrieval confidence gate against gold data.

# What this measures

Phase 22F established that top-1 cosine similarity cannot separate genuine
matches from topical neighbours: 18 of 20 hard negatives cleared the 0.40
LOW_CONFIDENCE_THRESHOLD, a ~90% false-positive rate. This asks whether a
local cross-encoder can, scoring the query against each retrieved incident
and taking the best score as the gate's confidence signal.

Test set (56 queries):
  - 28 positives      tests/eval/gold/phase17c_benchmark_v1.json
  -  8 easy negatives inline in that same file as "v2-neg-*"
  - 20 hard negatives tests/eval/gold_queries_hard_negatives_v1.json

# Methodology

Retrieval uses the production path, ``retrieve(expand=True, rerank=True)``,
because that is what a real gate would sit in front of. That path makes two
LLM calls per query, so this run is NOT deterministic; raw per-query scores
are written to JSON so the threshold sweep can be re-run offline for free.

A positive only counts toward recall when retrieval actually surfaced one of
its expected incidents. If retrieval missed, the gate rejecting that query is
correct behavior, not a false negative -- the same accounting rule used by
``probe_llm_relevance_gate_stability._rates``. Those queries are reported
separately as ``retrieval_miss`` rather than silently inflating or deflating
recall.

The cosine baseline at 0.40 is computed over the identical retrieved sets so
the comparison is like-for-like.

Read-only against the corpus. Writes .benchmarks/retrieval_gate_calibration.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.db.session import SessionLocal
from app.services.confidence import LOW_CONFIDENCE_THRESHOLD
from app.services.relevance_scorer import CrossEncoderRelevanceScorer
from app.services.search import IncidentSearchService

POSITIVES = Path("tests/eval/gold/phase17c_benchmark_v1.json")
HARD_NEGATIVES = Path("tests/eval/gold_queries_hard_negatives_v1.json")
OUT = Path(".benchmarks/retrieval_gate_calibration.json")
LIMIT = 5
THRESHOLDS = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]


def _rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"] if isinstance(data, dict) and "queries" in data else data


def _passage(result) -> str:
    incident = result.incident
    symptoms = " ".join(symptom.text for symptom in incident.symptoms)
    return f"{incident.title} {symptoms}".strip()


def _expected_ids(row: dict) -> set[str]:
    return {
        str(item.get("source_external_id"))
        for item in row.get("expected_incidents", ())
        if item.get("source_external_id")
    }


def collect() -> list[dict]:
    db = SessionLocal()
    search = IncidentSearchService(db)
    scorer = CrossEncoderRelevanceScorer()
    records: list[dict] = []

    # phase17c_benchmark_v1.json carries 8 easy negatives inline as "v2-neg-*".
    # They are negatives, not positives -- grouping them by filename alone
    # silently drops them from the false-positive denominator.
    work = [
        ("EASY_NEGATIVE" if r["id"].startswith("v2-neg-") else "POSITIVE", r)
        for r in _rows(POSITIVES)
    ]
    work += [("HARD_NEGATIVE", r) for r in _rows(HARD_NEGATIVES)]

    for index, (group, row) in enumerate(work, 1):
        query = row["query"]
        started = time.perf_counter()
        try:
            results = search.retrieve(
                query, limit=LIMIT, expand=True, rerank=True,
                call_site="calibrate_retrieval_gate",
            )
        except Exception as exc:  # noqa: BLE001 — per-query isolation
            records.append({"id": row["id"], "group": group, "query": query,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue

        scores = scorer.score(query, [_passage(r) for r in results]) if results else []
        expected = _expected_ids(row)
        retrieved_ids = [str(r.incident.source_external_id) for r in results]

        records.append({
            "id": row["id"],
            "group": group,
            "query": query,
            "error": None,
            "n_results": len(results),
            "top1_cosine": max((r.similarity_score for r in results), default=None),
            "top1_relevance": max(scores, default=None),
            "all_relevance": scores,
            "retrieval_hit": bool(expected & set(retrieved_ids)) if expected else None,
            "seconds": round(time.perf_counter() - started, 2),
        })
        print(f"  [{index:2d}/{len(work)}] {group:14s} {row['id']:26s} "
              f"cos={records[-1]['top1_cosine']} rel={records[-1]['top1_relevance']}")

    return records


def sweep(records: list[dict]) -> list[dict]:
    """Precision/recall/FPR per threshold. Positives whose retrieval missed
    the expected incident are excluded from recall entirely.
    """
    usable = [r for r in records if not r.get("error") and r.get("top1_relevance") is not None]
    positives = [r for r in usable if r["group"] == "POSITIVE" and r.get("retrieval_hit")]
    negatives = [r for r in usable if r["group"].endswith("NEGATIVE")]

    out = []
    for threshold in THRESHOLDS:
        tp = sum(1 for r in positives if r["top1_relevance"] >= threshold)
        fn = len(positives) - tp
        fp = sum(1 for r in negatives if r["top1_relevance"] >= threshold)
        tn = len(negatives) - fp
        out.append({
            "threshold": threshold,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "fpr": round(fp / (fp + tn), 4) if (fp + tn) else None,
        })
    return out


def cosine_baseline(records: list[dict]) -> dict:
    usable = [r for r in records if not r.get("error") and r.get("top1_cosine") is not None]
    positives = [r for r in usable if r["group"] == "POSITIVE" and r.get("retrieval_hit")]
    negatives = [r for r in usable if r["group"].endswith("NEGATIVE")]
    tp = sum(1 for r in positives if r["top1_cosine"] >= LOW_CONFIDENCE_THRESHOLD)
    fp = sum(1 for r in negatives if r["top1_cosine"] >= LOW_CONFIDENCE_THRESHOLD)
    fn, tn = len(positives) - tp, len(negatives) - fp
    return {
        "threshold": LOW_CONFIDENCE_THRESHOLD,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) else None,
    }


def main() -> None:
    print("Collecting retrievals and relevance scores (production path)...")
    records = collect()

    positives = [r for r in records if r["group"] == "POSITIVE" and not r.get("error")]
    misses = [r for r in positives if r.get("retrieval_hit") is False]
    report = {
        "test_set": {
            "positives": len(positives),
            "positives_with_retrieval_hit": sum(1 for r in positives if r.get("retrieval_hit")),
            "positives_retrieval_miss": len(misses),
            "hard_negatives": sum(1 for r in records if r["group"] == "HARD_NEGATIVE"),
            "easy_negatives": sum(1 for r in records if r["group"] == "EASY_NEGATIVE"),
            "errors": sum(1 for r in records if r.get("error")),
        },
        "cosine_baseline_at_0.40": cosine_baseline(records),
        "cross_encoder_sweep": sweep(records),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\ntest set: {report['test_set']}")
    b = report["cosine_baseline_at_0.40"]
    print(f"\ncosine baseline @0.40   recall={b['recall']}  fpr={b['fpr']}  "
          f"precision={b['precision']}")
    print("\ncross-encoder sweep")
    print(f"  {'thr':>6} {'tp':>3} {'fn':>3} {'fp':>3} {'tn':>3} "
          f"{'recall':>7} {'fpr':>7} {'prec':>7}")
    for row in report["cross_encoder_sweep"]:
        print(f"  {row['threshold']:6.1f} {row['tp']:3d} {row['fn']:3d} {row['fp']:3d} "
              f"{row['tn']:3d} {str(row['recall']):>7} {str(row['fpr']):>7} "
              f"{str(row['precision']):>7}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
