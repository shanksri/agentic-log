"""Probe: does the production investigation orchestrator abstain on hard
negatives that the top-1 similarity threshold misses?

Phase 22F showed 18/20 hard negatives score >= 0.40 top-1 similarity, so
the generation-path decline gate (LOW only) never fires for them. The
production path (/agent/investigate -> MultiAgentInvestigationOrchestrator)
has mechanisms the simple generation path lacks: a 0.60 composite
acceptance floor, a rule-based critic, and an `is_uncertain` flag. Nobody
has measured whether those catch what the threshold misses. This runs the
orchestrator exactly as the route constructs it (default construction,
n_hypotheses=3 = InvestigationRequest's default) against:

  - the 20 hard negatives (tests/eval/gold_queries_hard_negatives_v1.json)
  - the 8 easy negatives from phase17c_benchmark_v1.json (baseline: these
    should abstain)
  - 6 verified genuine positives from phase17c_benchmark_v1.json
    (positive control: these should NOT abstain -- without this the
    negative results can't be interpreted)

Read-only against the corpus. Makes real OpenAI calls (hypothesis
generation, ~1-3 per investigation). Per-query isolation: an exception in
one investigation is recorded, not fatal. Writes
.benchmarks/orchestrator_hard_negative_probe.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.db.session import SessionLocal
from app.services.confidence import classify_confidence
from app.services.investigation_orchestrator import MultiAgentInvestigationOrchestrator
from app.services.search import IncidentSearchService

HARD_NEGATIVES = Path("tests/eval/gold_queries_hard_negatives_v1.json")
PHASE17C = Path("tests/eval/gold/phase17c_benchmark_v1.json")
OUT = Path(".benchmarks/orchestrator_hard_negative_probe.json")
N_HYPOTHESES = 3  # InvestigationRequest default -- matches production

# Positive controls: 2 lexical-overlap, 2 paraphrase, 2 multi-concept, all
# verified resolvable against the current corpus earlier this session.
POSITIVE_IDS = ["v2-lex-02", "v2-lex-08", "v2-para-04", "v2-para-08", "v2-multi-03", "v2-multi-06"]


def _load_queries() -> list[dict]:
    hard = json.loads(HARD_NEGATIVES.read_text(encoding="utf-8"))["queries"]
    p17 = json.loads(PHASE17C.read_text(encoding="utf-8"))["queries"]
    out = [{"group": "HARD_NEGATIVE", "id": q["id"], "query": q["query"]} for q in hard]
    out += [
        {"group": "EASY_NEGATIVE", "id": q["id"], "query": q["query"]}
        for q in p17 if q["category"] == "no-match-expected"
    ]
    out += [
        {"group": "POSITIVE", "id": q["id"], "query": q["query"]}
        for q in p17 if q["id"] in POSITIVE_IDS
    ]
    return out


def main() -> None:
    db = SessionLocal()
    search = IncidentSearchService(db)
    orchestrator = MultiAgentInvestigationOrchestrator(db)  # default construction, as the route does

    rows: list[dict] = []
    for entry in _load_queries():
        row = dict(entry)
        # Raw top-1 similarity, independent of the orchestrator, for correlation.
        top = search.search(entry["query"], limit=1)
        row["top1_score"] = top[0].similarity_score if top else None
        row["top1_level"] = classify_confidence(row["top1_score"])

        started = time.perf_counter()
        try:
            session = orchestrator.investigate(entry["query"], n_hypotheses=N_HYPOTHESES)
            inv = session.final_report.investigation
            crit = session.final_report.critique
            row.update(
                {
                    "error": None,
                    "is_uncertain": inv.is_uncertain,
                    "confidence": inv.confidence,
                    "confidence_level": inv.confidence_level,
                    "selected_root_cause": (
                        inv.selected_hypothesis.root_cause[:160]
                        if inv.selected_hypothesis else None
                    ),
                    "critic_verdict": crit.verdict.value,
                    "critic_confidence": crit.confidence,
                    "stopping_reason": session.stopping_reason.value,
                    "total_iterations": session.total_iterations,
                    "num_rejected": len(inv.rejected_hypotheses),
                    "num_supporting": len(inv.supporting_evidence),
                    "num_contradicting": len(inv.contradicting_evidence),
                }
            )
        except Exception as exc:  # noqa: BLE001 -- per-query isolation
            row.update({"error": repr(exc)})
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(
            f"[{row['group']:13s}] {row['id']:26s} top1={row['top1_score']:.3f}/{row['top1_level']:6s} "
            f"uncertain={row.get('is_uncertain')} conf={row.get('confidence')} "
            f"critic={row.get('critic_verdict')} stop={row.get('stopping_reason')} "
            f"({row['seconds']}s){' ERROR ' + row['error'] if row.get('error') else ''}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
