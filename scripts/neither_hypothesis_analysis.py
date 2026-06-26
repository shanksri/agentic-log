"""
Gather all data needed to analyse the 7 "neither" hypotheses.
Prints a structured dump for each: problem, retrieval context, hypothesis,
all sibling hypotheses from the same case, and cosine scores.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from app.db.session import SessionLocal
from app.db.models import Incident
from app.services.embedding_service import EmbeddingService

V7   = json.loads(Path("tests/eval/results/hypothesis_v7.json").read_text())
GOLD = json.loads(Path("tests/eval/hypothesis_gold.json").read_text())
STRU = json.loads(Path("tests/eval/results/hypothesis_structure_v7.json").read_text())

gold_by_id = {c["id"]: c for c in GOLD["cases"]}

# Build a map: (case_id, rank) -> classification
cls_map = {(r["case_id"], r["rank"]): r["classification"] for r in STRU["rows"]}

db  = SessionLocal()
emb = EmbeddingService()

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

for case in V7["cases"]:
    cid = case["id"]
    if case["is_negative_case"]:
        continue
    gc = gold_by_id[cid]
    exp_id = gc["expected_incident_ids"][0] if gc["expected_incident_ids"] else None

    # Check if any hypothesis in this case is "neither"
    neither_ranks = [h["rank"] for h in case["hypotheses"]
                     if cls_map.get((cid, h["rank"])) == "neither"]
    if not neither_ranks:
        continue

    # Fetch retrieved incidents
    pool = []
    for rid in case["retrieved_top5_ids"]:
        inc = db.get(Incident, rid)
        if inc:
            symptoms = [s.text for s in inc.symptoms]
            pool.append((rid, inc, symptoms))

    print(f"\n{'='*80}")
    print(f"CASE: {cid}  |  neither ranks: {neither_ranks}")
    print(f"Problem: {gc['problem']}")
    print(f"Retrieval confidence: {case['initial_confidence_level']}  top1={case['initial_top1_score']:.4f}")
    print(f"Expected: {exp_id}")
    print()

    print("RETRIEVED POOL (top 5):")
    for i, (rid, inc, symptoms) in enumerate(pool, 1):
        gold_marker = " <-- GOLD" if rid == exp_id else ""
        print(f"  {i}. [{rid[:8]}] {inc.title}{gold_marker}")
        for s in symptoms[:2]:
            print(f"       S: {s[:100]}")
    print()

    print("ALL HYPOTHESES FOR THIS CASE:")
    for h in case["hypotheses"]:
        cls = cls_map.get((cid, h["rank"]), "?")
        marker = " <-- NEITHER" if cls == "neither" else ""
        print(f"  rank{h['rank']} [{cls}] match={h['is_match']} kw_A={h['validation_keyword_recall_ok_a']} kw_C={h['validation_keyword_recall_ok_c']}{marker}")
        print(f"    rc: {h['root_cause']}")
        print(f"    A kws: {h['validation_keywords_a']}")

    print()
    print("COSINE SCORES FOR NEITHER HYPOTHESES:")
    for h in case["hypotheses"]:
        if cls_map.get((cid, h["rank"])) != "neither":
            continue
        rc_vec = emb.embed_text(h["root_cause"])
        scores = []
        for rid, inc, _ in pool:
            tv = emb.embed_text(inc.title)
            scores.append((cosine(rc_vec, tv), rid, inc.title))
        scores.sort(reverse=True)
        print(f"  rank{h['rank']}: {h['root_cause'][:80]}")
        for s, rid, t in scores:
            g = " <-- GOLD" if rid == exp_id else ""
            print(f"    {s:.4f}  {t[:65]}{g}")
    print()

    # Most specific correct hypothesis (for comparison)
    correct_mechs = [h for h in case["hypotheses"]
                     if h["is_match"] and cls_map.get((cid, h["rank"])) == "mechanism_only"]
    correct_ms    = [h for h in case["hypotheses"]
                     if h["is_match"] and cls_map.get((cid, h["rank"])) == "mechanism_symptom"]
    best_correct  = (correct_ms + correct_mechs)
    if best_correct:
        best = best_correct[0]
        print(f"BEST CORRECT SPECIFIC HYPOTHESIS (rank{best['rank']}, {cls_map.get((cid, best['rank']))}):")
        print(f"  {best['root_cause']}")
        print(f"  A kws: {best['validation_keywords_a']}")

db.close()
