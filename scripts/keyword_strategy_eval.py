"""
Design and evaluate three keyword generation strategies for validation-keyword recall@5.

  A. Current:           LLM-generated keywords (validation_keyword_eval query from v6a)
  B. Literal full:      Top-5 terms from title + all symptoms (including log lines)
  C. Evidence-oriented: Top-5 terms from title only of the supporting retrieved incident

Strategy C rationale
--------------------
The generated keywords (A) paraphrase the root cause at one level of abstraction too high —
they omit the literal vocabulary (crash type, flags, type-system operation names) that would
distinguish this incident from near-duplicates.

Extracting from title + ALL symptoms (B) over-corrects: log lines (GC output, stack frames,
numeric timings) produce noisy tokens that hurt retrieval.

Strategy C uses the title of the retrieved incident that best supports the hypothesis as the
keyword source.  The title is always human-written, specific, and free of log noise.  In this
evaluation we use the *expected* incident's title as a proxy (since retrieval@5 = 1.0 for all
positive cases, the expected incident is always in the initial candidate set and would be
identifiable by a hypothesis-to-title similarity step in production).

Per-hypothesis variant (C_hyp)
--------------------------------
In addition to the case-level oracle, we also run the per-hypothesis version of C:
for each hypothesis, take the title of the retrieved incident whose embedding is most similar
to the hypothesis root_cause embedding. This is the realistic production signal.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from app.db.session import SessionLocal
from app.db.models import Incident
from app.services.search import IncidentSearchService, IncidentSearchResult
from app.services.embedding_service import EmbeddingService

GOLD_PATH   = Path("tests/eval/hypothesis_gold.json")
V6A_PATH    = Path("tests/eval/results/hypothesis_v6a.json")
K           = 5

STOP = {
    "the","a","an","is","in","of","to","and","or","for","with","when","was","it",
    "this","that","be","on","at","by","from","as","are","not","but","have","has",
    "had","its","i","we","they","he","she","you","my","no","do","can","will","if",
    "so","up","out","also","more","than","then","may","get","after","any","all",
    "should","would","could","been","were","did","does","via","into","just","there",
    "their","which","while","during","where","how","what","work","works","doesn",
    "don","t","s","isn","doesn","doesn't","doesn",
}


# ---------------------------------------------------------------------------
# Strategy B / C shared: literal term extraction
# ---------------------------------------------------------------------------

def _is_log_line(text: str) -> bool:
    """Heuristic: reject lines that look like raw log/GC output."""
    # Has many numbers, colons, or looks like a memory address or GC log
    if re.search(r'\d{3,}\.\d+\s*\(\d', text):    # GC: "1400.1 (1465.9)"
        return True
    if re.search(r'0x[0-9a-f]{6,}', text):         # hex address
        return True
    if re.search(r'^\s*\d+:\s', text):              # stack frame "  3: v8::..."
        return True
    if len(re.findall(r'\d+', text)) > 6:           # too many numbers
        return True
    return False


def extract_title_terms(title: str, n: int = 5) -> list[str]:
    """Strategy C: extract top-n discriminating terms from a title string only."""
    candidates: list[tuple[str, float]] = []

    # CLI flags
    for m in re.finditer(r'--?[\w\-]+', title):
        t = m.group()
        candidates.append((t, 4.0 + len(t) * 0.1))

    # Backtick-quoted identifiers
    for m in re.finditer(r'`([^`]+)`', title):
        t = m.group(1).strip()
        if t:
            candidates.append((t, 3.5 + len(t) * 0.05))

    # Version / release tags like [2.8.0-rc]
    for m in re.finditer(r'\[[\d\w\.\-]+\]', title):
        candidates.append((m.group(), 3.0))

    # ALLCAPS identifiers
    for m in re.finditer(r'\b[A-Z][A-Z_]{3,}\b', title):
        t = m.group()
        candidates.append((t, 2.5))

    # PascalCase / camelCase identifiers
    for m in re.finditer(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', title):
        candidates.append((m.group(), 2.0))

    # All remaining words length >= 5, not stop-words
    for word in re.findall(r'\b[\w][\w\-\.]+\b', title):
        if len(word) >= 5 and word.lower() not in STOP:
            candidates.append((word, 1.0 + len(word) * 0.05))

    # Deduplicate (keep highest score), then sort
    seen: dict[str, float] = {}
    for t, s in candidates:
        seen[t] = max(seen.get(t, -99.0), s)
    ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)

    # Remove substrings already covered by a higher-ranked longer term
    selected: list[str] = []
    for term, _ in ranked:
        lower = term.lower()
        if not any(lower in s.lower() or s.lower() in lower for s in selected):
            selected.append(term)
        if len(selected) == n:
            break
    return selected


def extract_terms_full(title: str, symptoms: list[str], n: int = 5) -> list[str]:
    """Strategy B: title + all symptoms, filtering log lines."""
    prose_symptoms = [s for s in symptoms if not _is_log_line(s)]
    # Use symptom text only for additional ALLCAPS / flags not in title
    extra: list[str] = []
    for s in prose_symptoms:
        for m in re.finditer(r'--?[\w\-]+', s):
            extra.append(m.group())
        for m in re.finditer(r'`([^`]+)`', s):
            extra.append(m.group(1).strip())
        for m in re.finditer(r'\b[A-Z][A-Z_]{3,}\b', s):
            extra.append(m.group())
    title_terms = extract_title_terms(title, n=n + len(extra))
    combined = title_terms + [e for e in extra if e not in title_terms]
    return combined[:n]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def recall_at_k(query: str, expected_id: str, svc: IncidentSearchService) -> float:
    if not query:
        return 0.0
    results = svc.search(query, limit=K)
    return 1.0 if expected_id in {str(r.incident.id) for r in results} else 0.0


def main() -> None:
    gold   = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    v6a    = json.loads(V6A_PATH.read_text(encoding="utf-8"))
    v6a_by = {c["id"]: c for c in v6a["cases"]}

    db  = SessionLocal()
    emb = EmbeddingService()
    svc = IncidentSearchService(db, embedding_service=emb)

    results_table = []

    for case in gold["cases"]:
        cid = case["id"]
        if not case["expected_incident_ids"]:
            continue

        exp_id  = case["expected_incident_ids"][0]
        inc     = db.get(Incident, exp_id)
        symptoms = [s.text for s in inc.symptoms]

        # ---- Strategy A: current generated keywords ----
        kw_eval  = (v6a_by[cid].get("validation_keyword_eval") or {})
        query_a  = kw_eval.get("query", "")
        recall_a = recall_at_k(query_a, exp_id, svc)

        # ---- Strategy B: title + prose symptoms ----
        terms_b  = extract_terms_full(inc.title, symptoms, n=5)
        query_b  = " ".join(terms_b)
        recall_b = recall_at_k(query_b, exp_id, svc)

        # ---- Strategy C: title only of supporting incident (oracle) ----
        terms_c  = extract_title_terms(inc.title, n=5)
        query_c  = " ".join(terms_c)
        recall_c = recall_at_k(query_c, exp_id, svc)

        # ---- Strategy C_hyp: pick supporting incident by hypothesis similarity ----
        # For each hypothesis, embed root_cause → find retrieved incident whose
        # title embedding is most similar → use that title for keywords.
        # Then compute recall@5 at the *case* level: did any hypothesis's C_hyp
        # query retrieve the expected incident?
        retrieved_ids = v6a_by[cid]["retrieved_top5_ids"]
        retrieved_incs = [db.get(Incident, rid) for rid in retrieved_ids if db.get(Incident, rid)]

        best_recall_chyp = 0.0
        best_query_chyp  = ""
        best_hyp_rc      = ""
        best_inc_title   = ""

        for hyp in v6a_by[cid]["hypotheses"]:
            rc_text = hyp["root_cause"]
            rc_vec  = emb.embed_text(rc_text)

            # Score each retrieved incident by cosine(root_cause, title)
            best_sim   = -1.0
            best_rinc  = None
            for rinc in retrieved_incs:
                title_vec = emb.embed_text(rinc.title)
                sim = cosine(rc_vec, title_vec)
                if sim > best_sim:
                    best_sim  = sim
                    best_rinc = rinc

            if best_rinc is None:
                continue

            terms_chyp = extract_title_terms(best_rinc.title, n=5)
            query_chyp = " ".join(terms_chyp)
            r = recall_at_k(query_chyp, exp_id, svc)
            if r > best_recall_chyp:
                best_recall_chyp = r
                best_query_chyp  = query_chyp
                best_hyp_rc      = rc_text
                best_inc_title   = best_rinc.title

        results_table.append({
            "id":          cid,
            "title":       inc.title[:55],
            "query_a":     query_a,
            "query_b":     query_b,
            "query_c":     query_c,
            "query_chyp":  best_query_chyp,
            "recall_a":    recall_a,
            "recall_b":    recall_b,
            "recall_c":    recall_c,
            "recall_chyp": best_recall_chyp,
            "terms_c":     terms_c,
            "best_hyp_rc": best_hyp_rc,
            "best_rinc_title": best_inc_title,
        })

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    header = f"{'Case':<8} {'A':>6} {'B':>6} {'C':>6} {'C_hyp':>7}"
    print(header)
    print("-" * 45)
    sums = {"a": 0.0, "b": 0.0, "c": 0.0, "chyp": 0.0}
    n = len(results_table)

    for r in results_table:
        flag_c    = "✓" if r["recall_c"]    > r["recall_a"] else ("✗" if r["recall_c"]    < r["recall_a"] else " ")
        flag_chyp = "✓" if r["recall_chyp"] > r["recall_a"] else ("✗" if r["recall_chyp"] < r["recall_a"] else " ")
        print(f"{r['id']:<8} {r['recall_a']:>6.1f} {r['recall_b']:>6.1f} {r['recall_c']:>5.1f}{flag_c} {r['recall_chyp']:>5.1f}{flag_chyp}")
        sums["a"]    += r["recall_a"]
        sums["b"]    += r["recall_b"]
        sums["c"]    += r["recall_c"]
        sums["chyp"] += r["recall_chyp"]

    print("-" * 45)
    print(f"{'Recall@5':<8} {sums['a']/n:>6.3f} {sums['b']/n:>6.3f} {sums['c']/n:>5.3f}  {sums['chyp']/n:>5.3f}")

    print("\n\nPer-case detail")
    print("=" * 90)
    for r in results_table:
        print(f"\n{r['id']}  {r['title']}")
        print(f"  A (generated): {r['query_a']}")
        print(f"  B (full lit):  {r['query_b']}")
        print(f"  C (title):     {r['terms_c']}  →  {r['query_c']}")
        print(f"  C_hyp support: [{r['best_rinc_title'][:60]}]")
        print(f"         rc:     {r['best_hyp_rc'][:70]}")
        print(f"         query:  {r['query_chyp']}")
        print(f"  Recall  A={r['recall_a']}  B={r['recall_b']}  C={r['recall_c']}  C_hyp={r['recall_chyp']}")


if __name__ == "__main__":
    main()
