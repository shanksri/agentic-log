"""Analysis-only: measure Jira-ingestion impact on retrieval.

Re-runs the dense retrieval eval (search(), no rerank — identical config to
tests/eval/results/canonical_v3a_dense.json) against the current corpus and
diffs every case against that pre-Jira baseline. Read-only; writes one report.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from app.db.models import Incident
from app.db.session import SessionLocal
from app.services.embedding_service import EmbeddingService
from app.services.search import IncidentSearchService

GOLD = Path("tests/eval/gold_queries.json")
BASELINE = Path("tests/eval/results/canonical_v3a_dense.json")


def recall_at_k(retrieved, expected, k):
    if not expected:
        return None
    return len(set(retrieved[:k]) & expected) / len(expected)


def mrr(retrieved, expected):
    for rank, iid in enumerate(retrieved, start=1):
        if iid in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, expected, k):
    dcg = sum(1.0 / math.log2(r + 1) for r, iid in enumerate(retrieved[:k], 1) if iid in expected)
    ideal = min(len(expected), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def entropy(labels):
    n = len(labels)
    if not n:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def label_for(inc):
    if inc.source_type == "github":
        return f"github:{inc.owner}/{inc.repo}"
    if inc.source_type == "jira":
        pk = (inc.source_metadata or {}).get("project_key", "?")
        return f"jira:{pk}"
    return inc.source_type


def main():
    gold = json.loads(GOLD.read_text())["queries"]
    base = {q["id"]: q for q in json.loads(BASELINE.read_text())["queries"]}

    db = SessionLocal()
    try:
        # ---- Phase 1: corpus audit ----
        comp = dict(db.execute(
            select(Incident.source_type, __import__("sqlalchemy").func.count())
            .group_by(Incident.source_type)
        ).all())

        svc = IncidentSearchService(db, embedding_service=EmbeddingService())

        rows = []
        all_after_top5_sources = []
        all_after_top5_labels = []
        all_base_top5_labels = []
        jira_retrievals = []  # (qid, rank, score, displaced?)

        for entry in gold:
            qid = entry["id"]
            expected = set(entry["expected_incident_ids"])
            results = svc.search(entry["query"], limit=10)
            retrieved = [str(r.incident.id) for r in results]
            incs = {str(r.incident.id): r.incident for r in results}
            scores = {str(r.incident.id): r.similarity_score for r in results}

            base_q = base.get(qid, {})
            base_retrieved = base_q.get("retrieved_incident_ids", [])

            # source types / labels for current top-5
            top5 = retrieved[:5]
            top5_src = [incs[i].source_type for i in top5]
            top5_lbl = [label_for(incs[i]) for i in top5]
            all_after_top5_sources += top5_src
            all_after_top5_labels += top5_lbl
            all_base_top5_labels += base_retrieved[:5]  # baseline labels are ids; use as-is for entropy of ids

            # Jira incidents in current top-10
            for rank, iid in enumerate(retrieved, 1):
                if incs[iid].source_type == "jira":
                    displaced = (
                        len(base_retrieved) >= rank
                        and base_retrieved[rank - 1] not in retrieved[:rank]
                    )
                    jira_retrievals.append({
                        "qid": qid, "rank": rank, "score": round(scores[iid], 4),
                        "in_top5": rank <= 5, "displaced_base_id": (
                            base_retrieved[rank - 1] if len(base_retrieved) >= rank else None
                        ),
                    })

            exp_rank_now = next((r for r, i in enumerate(retrieved, 1) if i in expected), None)
            exp_rank_base = next(
                (r for r, i in enumerate(base_retrieved, 1) if i in expected), None
            )

            rows.append({
                "id": qid, "type": entry["query_type"],
                "r1": recall_at_k(retrieved, expected, 1),
                "r5": recall_at_k(retrieved, expected, 5),
                "r10": recall_at_k(retrieved, expected, 10),
                "mrr": mrr(retrieved, expected),
                "ndcg": ndcg_at_k(retrieved, expected, 10),
                "exp_rank_now": exp_rank_now, "exp_rank_base": exp_rank_base,
                "now_top5": top5, "base_top5": base_retrieved[:5],
                "now_top5_src": top5_src,
                "jira_in_top5": any(s == "jira" for s in top5_src),
            })

        # ---- Phase 1 extras: Jira gold rate / type dist ----
        jira_incs = db.execute(
            select(Incident.is_gold_labeled, Incident.incident_type)
            .where(Incident.source_type == "jira")
        ).all()
        jira_gold = sum(1 for g, _ in jira_incs if g)
        jira_types = Counter(t for _, t in jira_incs)
    finally:
        db.close()

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals)

    # baseline overall from file
    bo = json.loads(BASELINE.read_text())["overall"]
    base_r1 = sum(
        1 for q in base.values()
        if q.get("retrieved_incident_ids") and
        q["retrieved_incident_ids"][0] in set(q["expected_incident_ids"])
    ) / len(base)

    print("=" * 70)
    print("PHASE 1 — CORPUS AUDIT")
    print("=" * 70)
    for st, c in sorted(comp.items(), key=lambda x: -x[1]):
        print(f"  {st:10s} {c}")
    print(f"  TOTAL      {sum(comp.values())}")
    print(f"  Jira gold-label rate: {jira_gold}/{len(jira_incs)} = {jira_gold/len(jira_incs):.1%}")
    print(f"  Jira incident types: {dict(jira_types)}")

    print("\n" + "=" * 70)
    print("PHASE 2 — RETRIEVAL BENCHMARK (dense search(), N=%d)" % len(rows))
    print("=" * 70)
    print(f"  {'Metric':10s} {'Before':>8s} {'After':>8s} {'Delta':>8s}")
    for name, after, before in [
        ("Recall@1", avg("r1"), base_r1),
        ("Recall@5", avg("r5"), bo["recall_at_5"]),
        ("Recall@10", avg("r10"), bo["recall_at_10"]),
        ("MRR", avg("mrr"), bo["mrr"]),
        ("NDCG@10", avg("ndcg"), bo["ndcg_at_10"]),
    ]:
        print(f"  {name:10s} {before:8.4f} {after:8.4f} {after-before:+8.4f}")

    print("\n" + "=" * 70)
    print("PHASE 3 — PER-CASE DIFF")
    print("=" * 70)
    improved = regressed = unchanged = jira_entered = 0
    for r in rows:
        changed = r["now_top5"] != r["base_top5"]
        moved = ""
        if r["exp_rank_now"] != r["exp_rank_base"]:
            if r["exp_rank_now"] and r["exp_rank_base"]:
                moved = f"expected {r['exp_rank_base']}→{r['exp_rank_now']}"
        if r["jira_in_top5"]:
            jira_entered += 1
        if r["exp_rank_now"] and r["exp_rank_base"]:
            if r["exp_rank_now"] < r["exp_rank_base"]:
                improved += 1; tag = "IMPROVED"
            elif r["exp_rank_now"] > r["exp_rank_base"]:
                regressed += 1; tag = "REGRESSED"
            else:
                unchanged += 1; tag = "unchanged"
        else:
            tag = "check"
        flag = " [JIRA in top5]" if r["jira_in_top5"] else ""
        if changed or moved or flag:
            print(f"  {r['id']:8s} {tag:9s} {moved:20s}{flag}")
            if changed:
                print(f"      base top5: {r['base_top5']}")
                print(f"      now  top5: {r['now_top5']}")
                print(f"      now  src : {r['now_top5_src']}")
    print(f"\n  improved={improved} regressed={regressed} unchanged={unchanged} "
          f"cases_with_jira_in_top5={jira_entered}")

    print("\n" + "=" * 70)
    print("PHASE 4 — JIRA CONTRIBUTION")
    print("=" * 70)
    if not jira_retrievals:
        print("  No Jira incidents retrieved in any gold query's top-10.")
    for j in jira_retrievals:
        print(f"  {j['qid']:8s} rank={j['rank']} score={j['score']} "
              f"top5={j['in_top5']} displaced_base={j['displaced_base_id']}")

    print("\n" + "=" * 70)
    print("PHASE 5 — RETRIEVAL DIVERSITY (top-5 across all queries)")
    print("=" * 70)
    print(f"  unique source_types after: {sorted(set(all_after_top5_sources))}")
    print(f"  source_type counts after : {dict(Counter(all_after_top5_sources))}")
    print(f"  unique labels after      : {len(set(all_after_top5_labels))}")
    print(f"  unique ids base          : {len(set(all_base_top5_labels))}")
    print(f"  source_type entropy after: {entropy(all_after_top5_sources):.4f} bits")
    print(f"  label entropy after      : {entropy(all_after_top5_labels):.4f} bits")


if __name__ == "__main__":
    main()
