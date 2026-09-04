"""Trace ONE investigation end-to-end, printing the real value at each stage.

Diagnostic/explanatory tool -- runs the production orchestrator unmodified
and reports what it produced at every step, so the architecture can be read
against actual numbers instead of a diagram. Read-only; changes nothing.

Usage:
    python scripts/trace_investigation.py "your problem statement"
"""

from __future__ import annotations

import sys
import time

from app.db.session import SessionLocal
from app.services.investigation_orchestrator import MultiAgentInvestigationOrchestrator
from app.services.search import IncidentSearchService

DEFAULT_PROBLEM = (
    "MemoryQoS does not set memory.high for BestEffort pods on cgroup v2"
)


def main() -> None:
    problem = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROBLEM
    db = SessionLocal()

    print("=" * 78)
    print(f"PROBLEM: {problem}")
    print("=" * 78)

    # ── Stage 1: what the initial retrieval actually returns ────────────────
    search = IncidentSearchService(db)
    raw = search.search(problem, limit=5)
    top1, level = IncidentSearchService.confidence_for(raw)
    print(f"\n[1] INITIAL RETRIEVAL (raw dense, no expand/rerank -- shown for reference)")
    print(f"    top1_score={top1:.4f}  ->  confidence_level={level}")
    for i, r in enumerate(raw, 1):
        print(f"      {i}. {r.similarity_score:.4f}  {r.incident.source_external_id}")
        print(f"         {r.incident.title[:88]}")

    # ── Stage 2: the real orchestrator run ──────────────────────────────────
    print(f"\n[2] ORCHESTRATOR (production path: retrieve(expand=True, rerank=True) + loop)")
    started = time.perf_counter()
    orchestrator = MultiAgentInvestigationOrchestrator(db)
    session = orchestrator.investigate(problem, n_hypotheses=3)
    elapsed = time.perf_counter() - started
    print(f"    completed in {elapsed:.1f}s")
    print(f"    total_iterations={session.total_iterations}")
    print(f"    stopping_reason={session.stopping_reason.value}")
    print(f"    stop_explanation={session.stop_explanation}")

    # ── Stage 3: per-iteration internals ────────────────────────────────────
    for it in session.iterations:
        print(f"\n--- ITERATION {it.iteration_number} ---")
        print(f"  PLANNER (rule-based, 0 LLM calls)")
        print(f"    strategy={it.plan.strategy.value}")
        print(f"    rationale={it.plan.strategy_rationale[:120]}")

        print(f"  HYPOTHESES (1 LLM call, n=3)")
        for h in it.hypotheses:
            ev = it.evaluations[h.id]
            print(f"    - [{h.id}] raw_confidence={h.raw_confidence}")
            print(f"      root_cause : {h.root_cause[:100]}")
            print(f"      keywords   : {list(h.validation_keywords)}")
            print(f"      EVIDENCE SEARCH (rule-based, 0 LLM calls)")
            print(f"        query      : {ev.query[:90]!r}")
            print(f"        supporting : {len(ev.supporting_evidence)}  "
                  f"(similarity >= 0.40)")
            print(f"        contradict : {len(ev.contradicting_evidence)} (below 0.40)")
            print(f"        missing    : {len(ev.missing_evidence)}")
            for s in list(ev.supporting_evidence)[:2]:
                print(f"          + {s[:92]}")

        d = it.decision
        print(f"  DECISION (composite score, floor=0.60)")
        print(f"    is_uncertain={d.is_uncertain}  accepted_score={d.accepted_score}")
        accepted = getattr(d, "accepted", None)
        print(f"    accepted={'None (abstained)' if accepted is None else accepted.root_cause[:80]}")

        c = it.critique
        print(f"  CRITIC (rule-based, 0 LLM calls)")
        print(f"    verdict={c.verdict.value}  confidence={c.confidence}")
        print(f"    explanation={c.explanation[:140]}")
        print(f"  progress_note={it.progress_note[:120]}")

    # ── Stage 4: what the API would return ──────────────────────────────────
    inv = session.final_report.investigation
    print(f"\n[3] FINAL RESPONSE (what POST /agent/investigate returns)")
    print(f"    selected_root_cause = "
          f"{inv.selected_hypothesis.root_cause[:100] if inv.selected_hypothesis else None}")
    print(f"    confidence          = {inv.confidence}")
    print(f"    confidence_level    = {inv.confidence_level}")
    print(f"    is_uncertain        = {inv.is_uncertain}")
    print(f"    supporting_evidence = {len(inv.supporting_evidence)} items")
    print(f"    rejected_hypotheses = {len(inv.rejected_hypotheses)}")
    print(f"    critique.verdict    = {session.final_report.critique.verdict.value}")


if __name__ == "__main__":
    main()
