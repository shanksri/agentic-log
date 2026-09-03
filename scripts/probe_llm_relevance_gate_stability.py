"""Stability experiment for the LLM relevance gate (validation, not integration).

`scripts/probe_llm_relevance_gate.py` measured the gate once over the 35
HIGH-band calibration queries: 11/11 hard negatives rejected, 20/21 genuine
matches kept, precision 1.00 / recall 0.95. That was a SINGLE run at
non-zero temperature, so it carries no variance information -- exactly the
gap `app/evaluation/grounding_metrics.py`'s evaluator-stability tooling
exists to close (Phase 22B: "stability needs >= 2 successful samples; with
fewer it is None, never a fabricated std of 0").

This re-runs the identical experiment N times and reports run-to-run
variance using that same tooling (`measure_stability`,
`classify_evaluator_confidence`), so the gate is held to the same standard
as every other evaluator in this project.

Identical-by-construction: the system prompt and user-prompt builder are
IMPORTED from the original probe rather than copied, so they cannot drift.
Query selection (HIGH band from the Phase 22F calibration), labels, and the
gate criterion (`same_problem` true/false) are unchanged. The dataset is
NOT expanded -- this measures the stability of an already-measured result.

Retrieval is re-run inside every repetition (not hoisted) so that any
retrieval non-determinism would show up as a changed top-1 rather than
being silently assumed away.

Read-only against the corpus. Makes N x 35 real LLM calls. Wires up the
same token-usage tracking pattern used by scripts/run_phase18d_benchmark.py
so cost is measured rather than guessed. Writes
.benchmarks/llm_relevance_gate_stability.json.

Usage:
    python scripts/probe_llm_relevance_gate_stability.py [--repetitions 5]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from app.db.session import SessionLocal
from app.evaluation.grounding_metrics import classify_evaluator_confidence, measure_stability
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchService

CALIBRATION = Path("tests/eval/results/phase22f_confidence_calibration.json")
HARD_NEGATIVES = Path("tests/eval/gold_queries_hard_negatives_v1.json")
PHASE17C = Path("tests/eval/gold/phase17c_benchmark_v1.json")
ORIGINAL_PROBE = Path("scripts/probe_llm_relevance_gate.py")
OUT = Path(".benchmarks/llm_relevance_gate_stability.json")

# Illustrative published gpt-4o-mini rates, same constants and same caveat as
# scripts/run_phase18d_benchmark.py -- order-of-magnitude only, NOT guaranteed
# current.
PROMPT_COST_PER_1K = 0.00015
COMPLETION_COST_PER_1K = 0.0006


def _load_original_probe():
    """Import the original probe module so the prompt and gate logic are the
    same objects, not a copy that could silently diverge."""
    spec = importlib.util.spec_from_file_location("_original_probe", ORIGINAL_PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class _UsageTrackingLLM:
    """Wraps LLMService, monkeypatching its OpenAI client's create() in place
    to record token usage per call -- same pattern as Phase 18D's benchmark.
    """

    def __init__(self, inner: LLMService) -> None:
        self._inner = inner
        self.calls: list[dict] = []
        original_create = inner.client.chat.completions.create

        def tracked_create(*args, **kwargs):
            response = original_create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            self.calls.append(
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", None) or 0,
                }
            )
            return response

        inner.client.chat.completions.create = tracked_create

    def generate_json(self, *, system_prompt: str, user_prompt: str):
        return self._inner.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)

    def summary(self) -> dict:
        prompt = sum(c["prompt_tokens"] for c in self.calls)
        completion = sum(c["completion_tokens"] for c in self.calls)
        return {
            "num_calls": len(self.calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "estimated_cost_usd": round(
                prompt / 1000 * PROMPT_COST_PER_1K + completion / 1000 * COMPLETION_COST_PER_1K, 4
            ),
        }


def _build_test_set() -> list[dict]:
    """HIGH-band queries from the Phase 22F calibration -- identical selection
    to the original probe."""
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))["rows"]
    high = [r for r in calibration if r["confidence_level"] == "HIGH"]

    hard_text = {
        q["id"]: q["query"]
        for q in json.loads(HARD_NEGATIVES.read_text(encoding="utf-8"))["queries"]
    }
    p17 = json.loads(PHASE17C.read_text(encoding="utf-8"))["queries"]
    pos_text = {q["id"]: q["query"] for q in p17}
    expected = {
        q["id"]: {e["source_external_id"] for e in q["expected_incidents"]} for q in p17
    }

    out = []
    for row in high:
        qid = row["id"]
        query = hard_text.get(qid) or pos_text.get(qid)
        if query is None:
            continue
        out.append(
            {
                "id": qid,
                "group": row["group"],
                "query": query,
                "top1_score": row["top1_score"],
                "expected_external_ids": sorted(expected.get(qid, set())),
            }
        )
    return out


def _rates(decisions: dict[str, bool | None], test_set: list[dict], expected_map: dict[str, bool]):
    """precision / recall / FPR for ONE repetition, using the original probe's
    definitions: positives counted only where top-1 was the expected incident.
    """
    tp = fn = fp = tn = 0
    for entry in test_set:
        decision = decisions.get(entry["id"])
        if decision is None:
            continue
        if entry["group"] == "HARD_NEGATIVE":
            if decision:
                fp += 1
            else:
                tn += 1
        elif expected_map.get(entry["id"]):  # positive whose top-1 was the expected incident
            if decision:
                tp += 1
            else:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": precision, "recall": recall, "fpr": fpr}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    reps = args.repetitions

    probe = _load_original_probe()
    system_prompt = probe._SYSTEM
    build_user_prompt = probe._user_prompt

    test_set = _build_test_set()
    db = SessionLocal()
    search = IncidentSearchService(db)
    llm = _UsageTrackingLLM(LLMService())

    # per_query[qid] = {"decisions": [...], "reasons": [...], "top1": [...]}
    per_query: dict[str, dict] = {
        e["id"]: {"decisions": [], "reasons": [], "top1_incidents": [], "top1_is_expected": []}
        for e in test_set
    }
    per_rep_decisions: list[dict[str, bool | None]] = []

    started_at = time.time()
    for rep in range(1, reps + 1):
        rep_decisions: dict[str, bool | None] = {}
        rep_started = time.perf_counter()
        for entry in test_set:
            results = search.search(entry["query"], limit=1)  # re-run per rep, not hoisted
            if not results:
                rep_decisions[entry["id"]] = None
                continue
            incident = results[0].incident
            external = getattr(incident, "source_external_id", None)
            is_expected = external in set(entry["expected_external_ids"])

            try:
                verdict = llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=build_user_prompt(entry["query"], incident),
                )
                decision = bool(verdict.get("same_problem"))
                reason = str(verdict.get("reason", ""))[:200]
            except Exception as exc:  # noqa: BLE001 -- per-query isolation
                decision, reason = None, f"ERROR: {exc!r}"

            rep_decisions[entry["id"]] = decision
            per_query[entry["id"]]["decisions"].append(decision)
            per_query[entry["id"]]["reasons"].append(reason)
            per_query[entry["id"]]["top1_incidents"].append(external)
            per_query[entry["id"]]["top1_is_expected"].append(is_expected)

        per_rep_decisions.append(rep_decisions)
        print(f"repetition {rep}/{reps} done in {time.perf_counter() - rep_started:.1f}s", flush=True)

    total_runtime = time.time() - started_at

    # `top1_is_expected` is a property of retrieval; use the modal value (it is
    # expected to be constant across repetitions -- verified below).
    expected_map = {
        qid: (Counter(v["top1_is_expected"]).most_common(1)[0][0] if v["top1_is_expected"] else False)
        for qid, v in per_query.items()
    }

    rep_rates = [_rates(d, test_set, expected_map) for d in per_rep_decisions]

    # ── Per-query stability, using the project's own evaluator-stability tooling ──
    query_reports = []
    for entry in test_set:
        qid = entry["id"]
        decisions = [d for d in per_query[qid]["decisions"] if d is not None]
        samples = [1.0 if d else 0.0 for d in decisions]
        stability = measure_stability(samples) if len(samples) >= 2 else None
        unanimous = len(set(decisions)) == 1 if decisions else False
        majority = Counter(decisions).most_common(1)[0][0] if decisions else None
        agreement = (
            Counter(decisions).most_common(1)[0][1] / len(decisions) if decisions else None
        )
        retrieval_stable = len(set(per_query[qid]["top1_incidents"])) <= 1
        query_reports.append(
            {
                "id": qid,
                "group": entry["group"],
                "top1_score": entry["top1_score"],
                "top1_is_expected": expected_map[qid],
                "retrieval_stable": retrieval_stable,
                "top1_incidents": sorted(set(per_query[qid]["top1_incidents"])),
                "decisions": decisions,
                "unanimous": unanimous,
                "majority_decision": majority,
                "agreement": agreement,
                "mean": stability.mean if stability else None,
                "std_dev": stability.std_dev if stability else None,
                "evaluator_confidence": stability.confidence.value if stability else None,
                "reasons": per_query[qid]["reasons"],
            }
        )

    def _agg(key: str):
        vals = [r[key] for r in rep_rates if r[key] is not None]
        if len(vals) < 2:
            return {"values": vals, "mean": vals[0] if vals else None, "std_dev": None, "confidence": None}
        st = measure_stability(vals)
        return {
            "values": vals,
            "mean": st.mean,
            "std_dev": st.std_dev,
            "min": st.minimum,
            "max": st.maximum,
            "confidence": st.confidence.value,
        }

    unstable = [q for q in query_reports if not q["unanimous"]]
    hard = [q for q in query_reports if q["group"] == "HARD_NEGATIVE"]
    pos_expected = [q for q in query_reports if q["group"] == "POSITIVE" and q["top1_is_expected"]]
    pos_other = [q for q in query_reports if q["group"] == "POSITIVE" and not q["top1_is_expected"]]

    report = {
        "repetitions": reps,
        "model": llm._inner.model,
        "temperature": 0.2,  # LLMService.generate_json's fixed setting
        "num_queries": len(test_set),
        "prompt_source": str(ORIGINAL_PROBE),
        "runtime_seconds": round(total_runtime, 1),
        "usage": llm.summary(),
        "per_repetition_rates": rep_rates,
        "aggregate": {
            "precision": _agg("precision"),
            "recall": _agg("recall"),
            "fpr": _agg("fpr"),
        },
        "queries": query_reports,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\n=== STABILITY over {reps} repetitions, {len(test_set)} queries ===")
    for name in ("precision", "recall", "fpr"):
        a = report["aggregate"][name]
        sd = f"{a['std_dev']:.4f}" if a["std_dev"] is not None else "n/a"
        print(f"  {name:9s} mean={a['mean']:.4f} std={sd} values={[round(v,3) for v in a['values']]} conf={a['confidence']}")

    print(f"\nqueries with a changed decision across repetitions: {len(unstable)}/{len(query_reports)}")
    for q in unstable:
        print(f"  {q['id']:26s} ({q['group']}) decisions={q['decisions']} agreement={q['agreement']:.2f} std={q['std_dev']:.3f} conf={q['evaluator_confidence']}")

    print(f"\nhard negatives  : {sum(1 for q in hard if q['unanimous'] and q['majority_decision'] is False)}/{len(hard)} unanimously REJECTED across all {reps} reps")
    print(f"genuine matches : {sum(1 for q in pos_expected if q['unanimous'] and q['majority_decision'] is True)}/{len(pos_expected)} unanimously KEPT across all {reps} reps")
    print(f"positives whose top-1 was NOT the expected incident: {len(pos_other)}")
    for q in pos_other:
        print(f"  {q['id']:26s} decisions={q['decisions']} (gate rejecting these is arguably correct)")

    retrieval_unstable = [q for q in query_reports if not q["retrieval_stable"]]
    print(f"\nretrieval top-1 changed across repetitions: {len(retrieval_unstable)} queries")

    print("\n--- v2-multi-04 (the single false rejection in the original run) ---")
    for q in query_reports:
        if q["id"] == "v2-multi-04":
            print(f"  decisions={q['decisions']} unanimous={q['unanimous']} std={q['std_dev']} conf={q['evaluator_confidence']}")
            for i, r in enumerate(q["reasons"], 1):
                print(f"    rep{i}: {r}")

    u = report["usage"]
    print(f"\ncalls={u['num_calls']} prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']} est_cost=${u['estimated_cost_usd']}")
    print(f"runtime={report['runtime_seconds']}s")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
