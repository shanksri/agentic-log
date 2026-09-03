"""Experiment: can a single cheap LLM relevance call separate genuine
matches from topically-adjacent hard negatives in the HIGH confidence band?

Context. Phase 22F established that top-1 dense similarity alone does not
separate the two classes, and that two cheap non-LLM signals (rank-1/2 gap,
source diversity) don't either. The orchestrator probe then showed the
composite-confidence floor already abstains on everything in the MEDIUM and
LOW bands -- so the gap is bounded to the HIGH band (top-1 >= 0.55), where
all 11 HIGH hard negatives got a confident, critic-approved root cause.

This measures the one untested candidate signal: one `generate_json` call
asking whether the top-1 retrieved incident describes the *same problem* as
the query. That targets exactly what embeddings cannot see -- same topic,
different problem.

Test set (HIGH band only, from tests/eval/results/phase22f_confidence_calibration.json):
  - 11 hard negatives  -> the gate SHOULD reject
  - 24 genuine matches -> the gate SHOULD keep

Critical distinction this script makes: for a positive query, the top-1
retrieved incident is not necessarily the *expected* one. If retrieval
already surfaced the wrong incident, the gate answering "not the same
problem" is correct behavior, not a false rejection. Positives are therefore
split into `top1_is_expected` True/False, and the headline recall cost is
measured only on the True subset.

Read-only against the corpus. Makes one real LLM call per query (35 total).
Writes .benchmarks/llm_relevance_gate_probe.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.db.session import SessionLocal
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchService

CALIBRATION = Path("tests/eval/results/phase22f_confidence_calibration.json")
HARD_NEGATIVES = Path("tests/eval/gold_queries_hard_negatives_v1.json")
PHASE17C = Path("tests/eval/gold/phase17c_benchmark_v1.json")
OUT = Path(".benchmarks/llm_relevance_gate_probe.json")

_SYSTEM = (
    "You decide whether a retrieved past incident describes the SAME underlying "
    "problem as an engineer's current problem statement.\n\n"
    "Same problem means: the same failure, in the same component, with the same "
    "mechanism. Merely sharing a topic, product area, subsystem, or vocabulary is "
    "NOT the same problem -- a different bug in the same subsystem must be judged "
    "not-a-match.\n\n"
    'Respond as JSON: {"same_problem": true|false, "reason": "<one short sentence>"}'
)


def _user_prompt(query: str, incident) -> str:
    return (
        f"Current problem statement:\n{query}\n\n"
        f"Retrieved past incident:\n"
        f"Title: {incident.title}\n"
        f"Status: {incident.status}\n"
        f"Resolution: {(incident.resolution_summary or '(none recorded)')[:600]}\n\n"
        "Does the retrieved incident describe the same underlying problem?"
    )


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))["rows"]
    high = [r for r in calibration if r["confidence_level"] == "HIGH"]

    hard_text = {
        q["id"]: q["query"]
        for q in json.loads(HARD_NEGATIVES.read_text(encoding="utf-8"))["queries"]
    }
    p17 = json.loads(PHASE17C.read_text(encoding="utf-8"))["queries"]
    pos_text = {q["id"]: q["query"] for q in p17}
    expected_ids = {
        q["id"]: {e["source_external_id"] for e in q["expected_incidents"]} for q in p17
    }

    db = SessionLocal()
    search = IncidentSearchService(db)
    llm = LLMService()

    rows: list[dict] = []
    for entry in high:
        qid, group = entry["id"], entry["group"]
        query = hard_text.get(qid) or pos_text.get(qid)
        if query is None:
            continue

        results = search.search(query, limit=1)
        if not results:
            continue
        incident = results[0].incident
        top1_external = getattr(incident, "source_external_id", None)

        row = {
            "id": qid,
            "group": group,
            "query": query,
            "top1_score": entry["top1_score"],
            "top1_incident": top1_external,
            "top1_title": incident.title,
            # For positives: did retrieval actually surface the expected incident?
            "top1_is_expected": (
                top1_external in expected_ids.get(qid, set()) if group == "POSITIVE" else None
            ),
        }

        started = time.perf_counter()
        try:
            verdict = llm.generate_json(
                system_prompt=_SYSTEM, user_prompt=_user_prompt(query, incident)
            )
            row["same_problem"] = bool(verdict.get("same_problem"))
            row["reason"] = str(verdict.get("reason", ""))[:200]
            row["error"] = None
        except Exception as exc:  # noqa: BLE001 -- per-query isolation
            row.update({"same_problem": None, "reason": None, "error": repr(exc)})
        row["seconds"] = round(time.perf_counter() - started, 1)

        rows.append(row)
        exp = "" if group != "POSITIVE" else f" expected_top1={row['top1_is_expected']}"
        print(
            f"[{group:13s}] {qid:26s} top1={row['top1_score']:.3f}{exp} "
            f"same_problem={row['same_problem']} ({row['seconds']}s) :: {row['reason']}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # ── Analysis ────────────────────────────────────────────────────────────
    ok = [r for r in rows if r["error"] is None]
    hard = [r for r in ok if r["group"] == "HARD_NEGATIVE"]
    pos_expected = [r for r in ok if r["group"] == "POSITIVE" and r["top1_is_expected"]]
    pos_other = [r for r in ok if r["group"] == "POSITIVE" and not r["top1_is_expected"]]

    rejected_hard = sum(1 for r in hard if r["same_problem"] is False)
    kept_pos = sum(1 for r in pos_expected if r["same_problem"] is True)

    print("\n=== RESULT (HIGH band only) ===")
    print(f"errors: {len(rows) - len(ok)}")
    print(
        f"hard negatives correctly REJECTED : {rejected_hard}/{len(hard)}"
        f"  (baseline today: 0/{len(hard)} -- all answered confidently)"
    )
    print(
        f"true positives correctly KEPT     : {kept_pos}/{len(pos_expected)}"
        f"  (top-1 was the expected incident)"
    )
    print(
        f"positives where top-1 was NOT the expected incident: {len(pos_other)}"
        f" -- gate said same_problem=True for "
        f"{sum(1 for r in pos_other if r['same_problem'] is True)} of them"
    )
    if hard and pos_expected:
        tp, fn = kept_pos, len(pos_expected) - kept_pos
        fp = len(hard) - rejected_hard
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        print(f"\nprecision={precision:.3f}  recall={recall:.3f}  FPR={fp / len(hard):.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
