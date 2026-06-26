"""
Compare keyword recall@5 between:
  A. Current generated keywords (from hypothesis_v6a.json, validation_keyword_eval)
  B. Top-5 literal terms extracted from expected incident title + symptoms
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from app.db.session import SessionLocal
from app.db.models import Incident
from app.services.search import IncidentSearchService
from app.services.embedding_service import EmbeddingService

GOLD_PATH = Path("tests/eval/hypothesis_gold.json")
V6A_PATH = Path("tests/eval/results/hypothesis_v6a.json")
K = 5

# Stop-words to skip when scoring literal terms
STOP = {
    "the","a","an","is","in","of","to","and","or","for","with","when","was",
    "it","this","that","be","on","at","by","from","as","are","not","but",
    "have","has","had","its","i","we","they","he","she","you","my","no","do",
    "can","will","if","so","up","out","also","more","than","then","may","get",
    "after","error","issue","bug","problem","used","using","after","any","all",
    "should","would","could","been","were","did","does","via","into","just",
    "there","their","which","while","during","when","where","how","what",
}


def score_term(t: str) -> float:
    """Higher = more specific/discriminating."""
    t_lower = t.lower()
    words = t_lower.split()
    # Penalise pure stop-words
    if all(w in STOP for w in words):
        return -1.0
    score = 0.0
    # Prefer longer terms (more specific)
    score += len(words) * 0.3
    # Prefer uppercase / camelCase / symbols (error strings, flags, identifiers)
    if re.search(r'[A-Z_\-\.\[\]:]', t):
        score += 2.0
    # Prefer terms that look like CLI flags
    if t.startswith('--') or t.startswith('-'):
        score += 3.0
    # Prefer terms containing digits (version numbers, sizes)
    if re.search(r'\d', t):
        score += 1.5
    # Prefer terms in backtick or code-like position (heuristic: was quoted)
    # Penalise very short / very generic terms
    if len(t) < 4:
        score -= 1.0
    return score


def extract_literal_terms(title: str, symptoms: list[str], n: int = 5) -> list[str]:
    """Extract the n most discriminating literal terms from title + symptoms."""
    text_blocks = [title] + symptoms

    candidates: dict[str, float] = {}

    for block in text_blocks:
        # Extract tokens: words, compound identifiers, flags, error strings
        # 1. Multi-word error-like phrases in CAPS_WITH_UNDERSCORES
        for m in re.finditer(r'[A-Z][A-Z_]{3,}(?:\s+[A-Z_]+)*', block):
            t = m.group().strip()
            if len(t) >= 4:
                candidates[t] = max(candidates.get(t, -99), score_term(t))

        # 2. CLI flags  --foo or -f
        for m in re.finditer(r'--?[\w\-]+', block):
            t = m.group()
            candidates[t] = max(candidates.get(t, -99), score_term(t))

        # 3. Backtick-quoted identifiers
        for m in re.finditer(r'`([^`]+)`', block):
            t = m.group(1).strip()
            if t and len(t) >= 3:
                candidates[t] = max(candidates.get(t, -99), score_term(t))

        # 4. camelCase / PascalCase identifiers
        for m in re.finditer(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', block):
            t = m.group()
            candidates[t] = max(candidates.get(t, -99), score_term(t))

        # 5. Numbers with units (e.g. "10s of MB", "1400.1")
        for m in re.finditer(r'\d[\d\.,]*\s*(?:ms|MB|GB|KB|s\b)', block):
            t = m.group().strip()
            candidates[t] = max(candidates.get(t, -99), score_term(t))

        # 6. Exact title words (non-stop, length >= 5)
        if block == title:
            for word in re.findall(r'\b[\w\-\.]+\b', block):
                if len(word) >= 5 and word.lower() not in STOP:
                    candidates[word] = max(candidates.get(word, -99), score_term(word) + 0.5)

    # Sort and return top n
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    # Deduplicate substrings
    selected: list[str] = []
    for term, _ in ranked:
        if not any(term.lower() in s.lower() or s.lower() in term.lower() for s in selected):
            selected.append(term)
        if len(selected) == n:
            break
    return selected


def recall_at_k(query: str, expected_id: str, svc: IncidentSearchService, k: int = 5) -> float:
    results = svc.search(query, limit=k)
    retrieved = {str(r.incident.id) for r in results}
    return 1.0 if expected_id in retrieved else 0.0


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    v6a = json.loads(V6A_PATH.read_text(encoding="utf-8"))
    v6a_by_id = {c["id"]: c for c in v6a["cases"]}

    db = SessionLocal()
    emb = EmbeddingService()
    svc = IncidentSearchService(db, embedding_service=emb)

    print(f"{'Case':<8} {'Expected incident title':<52} {'A recall':<10} {'B recall':<10} {'Delta'}")
    print("-" * 100)

    total_a = total_b = fixed = 0
    n = 0

    for case in gold["cases"]:
        cid = case["id"]
        if not case["expected_incident_ids"]:
            continue
        exp_id = case["expected_incident_ids"][0]
        inc = db.get(Incident, exp_id)
        symptoms = [s.text for s in inc.symptoms]

        # A: current generated keywords (validation_keyword_eval query from v6a)
        v6a_case = v6a_by_id[cid]
        kw_eval = v6a_case.get("validation_keyword_eval") or {}
        query_a = kw_eval.get("query", "")
        recall_a = recall_at_k(query_a, exp_id, svc) if query_a else 0.0

        # B: top-5 literal terms from title + symptoms
        literal_terms = extract_literal_terms(inc.title, symptoms, n=5)
        query_b = " ".join(literal_terms)
        recall_b = recall_at_k(query_b, exp_id, svc) if query_b else 0.0

        delta = recall_b - recall_a
        if recall_a == 0.0 and recall_b == 1.0:
            fixed += 1
        total_a += recall_a
        total_b += recall_b
        n += 1

        title_short = inc.title[:50]
        print(f"{cid:<8} {title_short:<52} {recall_a:<10.1f} {recall_b:<10.1f} {delta:+.1f}")
        print(f"         A query: {query_a}")
        print(f"         B terms: {literal_terms}")
        print(f"         B query: {query_b}")
        print()

    print("-" * 100)
    print(f"{'TOTAL':<8} {'':52} {total_a/n:<10.3f} {total_b/n:<10.3f} {(total_b-total_a)/n:+.3f}")
    print(f"Failures fixed by B: {fixed}/{n - int(total_a)}")


if __name__ == "__main__":
    main()
