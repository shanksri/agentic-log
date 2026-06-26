"""
Phase 7 supplemental analysis: does hypothesis representation quality
(mechanism-only vs. mechanism+symptom) predict evidence-selection accuracy,
keyword recall, and root-cause correctness?

No production code is changed. Analysis only.

Steps
-----
1. Load hypothesis_v7.json (35 hypotheses across 7 positive cases).
2. For each hypothesis, classify it via LLM into:
     mechanism_only      – describes the root action/trigger, no observable outcome
     symptom_only        – describes the observable failure, no underlying cause
     mechanism_symptom   – names both the root action AND the observable consequence
     neither             – too vague to classify
3. For each hypothesis, compute cosine(root_cause, title) for every retrieved
   incident in that case's pool → determine which incident C_hyp would select,
   compare to the expected incident, record selection accuracy.
4. Aggregate stats by category and run Fisher's exact test on the 2x2
   (mechanism_symptom vs. other) × (C selects correctly vs. not).
5. Print all sibling-mismatch failures with full detail.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from app.db.session import SessionLocal
from app.db.models import Incident
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

V7_PATH   = Path("tests/eval/results/hypothesis_v7.json")
GOLD_PATH = Path("tests/eval/hypothesis_gold.json")

CLASSES = ("mechanism_only", "symptom_only", "mechanism_symptom", "neither")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def classify_hypothesis(root_cause: str, llm: LLMService) -> tuple[str, str]:
    """Return (class, rationale) via LLM."""
    result = llm.generate_json(
        system_prompt=(
            "You classify incident-investigation hypothesis statements. "
            "Return JSON with keys 'classification' and 'rationale'. "
            "'classification' must be exactly one of: "
            "mechanism_only, symptom_only, mechanism_symptom, neither.\n\n"
            "Definitions:\n"
            "  mechanism_only     – names the root action, trigger, or code-level cause "
            "but does NOT state what the observable user-facing failure is "
            "(e.g., crash, error message, wrong output).\n"
            "  symptom_only       – names the observable failure/error but does NOT "
            "name a specific code-level cause or trigger.\n"
            "  mechanism_symptom  – names BOTH a specific code-level root cause AND "
            "the observable outcome or failure symptom.\n"
            "  neither            – too vague or generic to determine either."
        ),
        user_prompt=f"Hypothesis: {root_cause}",
    )
    return str(result.get("classification", "neither")), str(result.get("rationale", ""))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p-value for [[a,b],[c,d]]."""
    def log_fact(n: int) -> float:
        return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0

    total = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d

    def cell_log_prob(a_: int) -> float:
        b_ = r1 - a_
        c_ = c1 - a_
        d_ = r2 - c_
        if b_ < 0 or c_ < 0 or d_ < 0:
            return float("-inf")
        return (
            log_fact(r1) + log_fact(r2) + log_fact(c1) + log_fact(c2)
            - log_fact(total)
            - log_fact(a_) - log_fact(b_) - log_fact(c_) - log_fact(d_)
        )

    observed_lp = cell_log_prob(a)
    a_min = max(0, r1 - r2)
    a_max = min(r1, c1)

    p = sum(
        math.exp(cell_log_prob(k) - observed_lp)
        for k in range(a_min, a_max + 1)
        if cell_log_prob(k) <= observed_lp + 1e-10
    )
    return min(1.0, p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    v7   = json.loads(V7_PATH.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    gold_by_id = {c["id"]: c for c in gold["cases"]}

    db  = SessionLocal()
    emb = EmbeddingService()
    llm = LLMService()

    # Pre-load all retrieved incidents per case
    case_retrieved: dict[str, list[tuple[str, Incident, list[float]]]] = {}
    for case in v7["cases"]:
        if case["is_negative_case"]:
            continue
        pool: list[tuple[str, Incident, list[float]]] = []
        for rid in case["retrieved_top5_ids"]:
            inc = db.get(Incident, rid)
            if inc:
                tvec = emb.embed_text(inc.title)
                pool.append((rid, inc, tvec))
        case_retrieved[case["id"]] = pool

    # Analyse each hypothesis
    rows: list[dict] = []
    total = sum(
        len(case["hypotheses"])
        for case in v7["cases"]
        if not case["is_negative_case"]
    )
    done = 0

    for case in v7["cases"]:
        if case["is_negative_case"]:
            continue
        cid        = case["id"]
        gold_case  = gold_by_id[cid]
        exp_ids    = set(gold_case["expected_incident_ids"])
        exp_id     = gold_case["expected_incident_ids"][0]
        pool       = case_retrieved[cid]

        for h in case["hypotheses"]:
            done += 1
            rc = h["root_cause"]
            print(f"  [{done}/{total}] classifying: {rc[:60]}...", flush=True)

            cls, rationale = classify_hypothesis(rc, llm)
            if cls not in CLASSES:
                cls = "neither"

            # Cosine against all retrieved titles
            rc_vec = emb.embed_text(rc)
            scores = [(cosine(rc_vec, tvec), rid, inc.title)
                      for rid, inc, tvec in pool]
            scores.sort(reverse=True)

            chosen_rid   = scores[0][1]
            chosen_score = scores[0][0]
            gold_score   = next((s for s, rid, _ in scores if rid == exp_id), 0.0)
            gap          = chosen_score - gold_score
            chose_gold   = chosen_rid in exp_ids

            rows.append({
                "case_id":           cid,
                "rank":              h["rank"],
                "root_cause":        rc,
                "classification":    cls,
                "rationale":         rationale,
                "is_match":          h["is_match"],
                "kw_recall_a":       h["validation_keyword_recall_ok_a"],
                "kw_recall_c":       h["validation_keyword_recall_ok_c"],
                "chose_gold":        chose_gold,
                "chosen_rid":        chosen_rid,
                "chosen_title":      scores[0][2],
                "chosen_score":      chosen_score,
                "gold_score":        gold_score,
                "gap":               gap,
                "all_scores":        [(s, rid, t) for s, rid, t in scores],
            })

    db.close()

    # ---------------------------------------------------------------------------
    # Aggregate stats
    # ---------------------------------------------------------------------------
    from collections import defaultdict

    cat_stats: dict[str, dict] = {c: {
        "n": 0, "correct_rc": 0, "kw_ok_a": 0, "kw_ok_c": 0,
        "chose_gold": 0, "kw_a_denom": 0, "kw_c_denom": 0,
    } for c in CLASSES}

    sibling_mismatches: list[dict] = []

    for r in rows:
        cat = r["classification"]
        st  = cat_stats[cat]
        st["n"]          += 1
        st["correct_rc"] += int(r["is_match"])
        if r["kw_recall_a"] is not None:
            st["kw_a_denom"] += 1
            st["kw_ok_a"]    += int(r["kw_recall_a"])
        if r["kw_recall_c"] is not None:
            st["kw_c_denom"] += 1
            st["kw_ok_c"]    += int(r["kw_recall_c"])
        st["chose_gold"] += int(r["chose_gold"])
        if not r["chose_gold"]:
            sibling_mismatches.append(r)

    # ---------------------------------------------------------------------------
    # Fisher's exact: mechanism_symptom vs. rest × chose_gold vs. not
    # ---------------------------------------------------------------------------
    ms_gold     = sum(1 for r in rows if r["classification"] == "mechanism_symptom" and r["chose_gold"])
    ms_not      = sum(1 for r in rows if r["classification"] == "mechanism_symptom" and not r["chose_gold"])
    other_gold  = sum(1 for r in rows if r["classification"] != "mechanism_symptom" and r["chose_gold"])
    other_not   = sum(1 for r in rows if r["classification"] != "mechanism_symptom" and not r["chose_gold"])
    fisher_p    = fisher_exact_2x2(ms_gold, ms_not, other_gold, other_not)

    # Same test for root-cause correctness
    ms_rc_ok    = sum(1 for r in rows if r["classification"] == "mechanism_symptom" and r["is_match"])
    ms_rc_no    = sum(1 for r in rows if r["classification"] == "mechanism_symptom" and not r["is_match"])
    ot_rc_ok    = sum(1 for r in rows if r["classification"] != "mechanism_symptom" and r["is_match"])
    ot_rc_no    = sum(1 for r in rows if r["classification"] != "mechanism_symptom" and not r["is_match"])
    fisher_rc_p = fisher_exact_2x2(ms_rc_ok, ms_rc_no, ot_rc_ok, ot_rc_no)

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("CLASSIFICATION DISTRIBUTION")
    print("=" * 80)
    for cat in CLASSES:
        st = cat_stats[cat]
        print(f"  {cat:<22} n={st['n']}")

    print("\n\n" + "=" * 80)
    print("STATS BY CATEGORY")
    print("=" * 80)
    header = f"{'Category':<22} {'n':>3}  {'RC%':>6}  {'KwA%':>6}  {'KwC%':>6}  {'GoldSel%':>9}"
    print(header)
    print("-" * 60)
    for cat in CLASSES:
        st = cat_stats[cat]
        n  = st["n"]
        if n == 0:
            continue
        rc_pct  = 100 * st["correct_rc"] / n
        kwa_pct = 100 * st["kw_ok_a"] / st["kw_a_denom"] if st["kw_a_denom"] else float("nan")
        kwc_pct = 100 * st["kw_ok_c"] / st["kw_c_denom"] if st["kw_c_denom"] else float("nan")
        gs_pct  = 100 * st["chose_gold"] / n
        print(f"  {cat:<22} {n:>3}  {rc_pct:>6.1f}  {kwa_pct:>6.1f}  {kwc_pct:>6.1f}  {gs_pct:>9.1f}")

    print(f"\n  Fisher's exact (mechanism_symptom vs. rest × gold-selection): p={fisher_p:.4f}")
    print(f"  Fisher's exact (mechanism_symptom vs. rest × root-cause correct): p={fisher_rc_p:.4f}")
    print(f"\n  Contingency (gold-selection):")
    print(f"    mechanism_symptom: {ms_gold} correct, {ms_not} wrong")
    print(f"    other:             {other_gold} correct, {other_not} wrong")

    print("\n\n" + "=" * 80)
    print(f"SIBLING-MISMATCH FAILURES ({len(sibling_mismatches)} total)")
    print("=" * 80)
    for r in sibling_mismatches:
        print(f"\n  {r['case_id']} rank{r['rank']}  class={r['classification']}  is_match={r['is_match']}")
        print(f"  Hypothesis: {r['root_cause']}")
        print(f"  Chosen:   [{r['chosen_score']:.4f}] {r['chosen_title'][:70]}")
        print(f"  Gold:     [{r['gold_score']:.4f}] (expected incident)")
        print(f"  Gap: {r['gap']:+.4f}")
        print(f"  All scores:")
        for s, rid, title in r["all_scores"]:
            marker = " <- GOLD" if rid in {r['chosen_rid']} and rid != r['chosen_rid'] else ""
            print(f"    {s:.4f}  {title[:65]}")
        print(f"  Rationale: {r['rationale'][:120]}")

    print("\n\n" + "=" * 80)
    print("ALL HYPOTHESES — CLASSIFICATION + EVIDENCE SELECTION")
    print("=" * 80)
    for r in rows:
        ok = "✓" if r["chose_gold"] else "✗"
        print(f"  {r['case_id']} r{r['rank']}  {r['classification']:<22} rc={int(r['is_match'])}  gold={ok} (gap={r['gap']:+.4f})  kw_A={r['kw_recall_a']}  kw_C={r['kw_recall_c']}")
        print(f"    {r['root_cause'][:90]}")

    # Dump JSON for the markdown report
    Path("tests/eval/results/hypothesis_structure_v7.json").write_text(
        json.dumps({"rows": rows, "cat_stats": cat_stats,
                    "fisher_gold_p": fisher_p, "fisher_rc_p": fisher_rc_p,
                    "contingency_gold": {"ms_gold": ms_gold, "ms_not": ms_not,
                                         "other_gold": other_gold, "other_not": other_not},
                    "contingency_rc": {"ms_ok": ms_rc_ok, "ms_no": ms_rc_no,
                                       "ot_ok": ot_rc_ok, "ot_no": ot_rc_no},
                    "sibling_mismatches": sibling_mismatches},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print("\nWrote tests/eval/results/hypothesis_structure_v7.json")


if __name__ == "__main__":
    main()
