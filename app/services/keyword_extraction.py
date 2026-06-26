"""Evidence-oriented keyword extraction for hypothesis validation.

Strategy C_hyp: for each hypothesis, find the retrieved incident whose title
embedding is most similar to the hypothesis's root_cause, then extract literal
terms from that incident's title.  This grounds validation_keywords in
vocabulary that actually appears in the incident corpus rather than in
paraphrased root-cause text.

Controlled by the USE_EVIDENCE_KEYWORDS environment variable (default: false).
When false, the agent continues to use the LLM-generated validation_keywords
unchanged.

Why title only (not symptoms):
  Symptoms can contain raw log lines (GC output, stack traces, numeric
  timings) that fragment into noisy tokens on tokenisation and hurt retrieval.
  Titles are human-written, specific, and log-noise-free.  On the Phase 5
  gold set, title-only extraction achieves recall@5 = 1.0 vs 0.857 for
  LLM-generated keywords, with no regression on any passing case.
"""
from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.search import IncidentSearchResult

USE_EVIDENCE_KEYWORDS: bool = os.getenv("USE_EVIDENCE_KEYWORDS", "false").lower() == "true"

_STOP = {
    "the","a","an","is","in","of","to","and","or","for","with","when","was","it",
    "this","that","be","on","at","by","from","as","are","not","but","have","has",
    "had","its","i","we","they","he","she","you","my","no","do","can","will","if",
    "so","up","out","also","more","than","then","may","get","after","any","all",
    "should","would","could","been","were","did","does","via","into","just","there",
    "their","which","while","during","where","how","what","work","works","t","s",
}


def extract_title_terms(title: str, n: int = 5) -> list[str]:
    """Extract the n most discriminating literal terms from an incident title.

    Priority order:
      1. CLI flags (--watch, --d)          — highly specific, rarely shared
      2. Backtick-quoted identifiers       — code names from issue text
      3. Version / bracket tags [2.8.0-rc] — pinpoint-specific
      4. ALLCAPS identifiers               — error constant names
      5. PascalCase identifiers            — class / component names
      6. Remaining words >= 5 chars, non-stop
    """
    candidates: dict[str, float] = {}

    def add(term: str, score: float) -> None:
        candidates[term] = max(candidates.get(term, -99.0), score)

    for m in re.finditer(r'--?[\w\-]+', title):
        add(m.group(), 4.0 + len(m.group()) * 0.1)

    for m in re.finditer(r'`([^`]+)`', title):
        t = m.group(1).strip()
        if t:
            add(t, 3.5 + len(t) * 0.05)

    for m in re.finditer(r'\[[\d\w\.\-]+\]', title):
        add(m.group(), 3.0)

    for m in re.finditer(r'\b[A-Z][A-Z_]{3,}\b', title):
        add(m.group(), 2.5)

    for m in re.finditer(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', title):
        add(m.group(), 2.0)

    for word in re.findall(r'\b[\w][\w\-\.]+\b', title):
        if len(word) >= 5 and word.lower() not in _STOP:
            add(word, 1.0 + len(word) * 0.05)

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    # Remove terms that are substrings of a higher-ranked term (dedup)
    selected: list[str] = []
    for term, _ in ranked:
        lower = term.lower()
        if not any(lower in s.lower() or s.lower() in lower for s in selected):
            selected.append(term)
        if len(selected) == n:
            break
    return selected


def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def derive_evidence_keywords(
    hypotheses: list[dict[str, Any]],
    retrieved_results: list[IncidentSearchResult],
    embedding_service: Any,
    *,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Replace each hypothesis's validation_keywords with evidence-oriented terms.

    For each hypothesis:
      1. Embed the root_cause text.
      2. Find the retrieved incident whose title embedding is most similar.
      3. Extract literal terms from that incident's title.
      4. Replace validation_keywords in a shallow copy of the hypothesis dict.

    Retrieved incident title embeddings are computed once and reused across all
    hypotheses to minimise embedding calls (n_retrieved embeds, not
    n_hypotheses × n_retrieved).

    Returns a new list of hypothesis dicts; the originals are not mutated.
    """
    if not retrieved_results or not hypotheses:
        return hypotheses

    # Pre-compute title embeddings for retrieved incidents (shared across hyps)
    titled: list[tuple[IncidentSearchResult, list[float]]] = []
    for result in retrieved_results:
        vec = embedding_service.embed_text(result.incident.title)
        titled.append((result, vec))

    updated: list[dict[str, Any]] = []
    for hyp in hypotheses:
        rc_vec = embedding_service.embed_text(hyp["root_cause"])

        best_sim    = -1.0
        best_result = retrieved_results[0]
        for result, title_vec in titled:
            sim = _cosine(rc_vec, title_vec)
            if sim > best_sim:
                best_sim    = sim
                best_result = result

        terms = extract_title_terms(best_result.incident.title, n=n)
        updated.append({**hyp, "validation_keywords": terms})

    return updated
