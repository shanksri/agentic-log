# 11 — Query Expansion

# Purpose

To broaden the candidate pool by generating semantically related phrasings of the query and
searching each, so relevant incidents phrased differently from the user's query can surface.

# Problem Statement

A single embedding of a short or idiosyncratic query may sit far from the right incident in
vector space (terse queries like "memory", domain jargon like "ISR shrink"). One probe from one
phrasing under-explores the neighborhood.

# High-Level Architecture

```
query ─► LLMService.expand_search_query(query) ─► [phrase1, phrase2, ...]
   [query, phrase1, phrase2, ...] ─► each → dense search ─► merge (doc 12)
```

# Detailed Flow

When `expand=True`, `_expand_query` asks the LLM for related phrasings. The original query plus
each phrase is searched at the wider candidate limit (25); results feed the candidate merge,
which keeps the best distance per incident. Enters: query string. Leaves: extra candidate
phrases (empty list if no LLM configured).

# Design Decisions

- **Why expand before rerank.** Expansion is the *recovery* mechanism — it adds candidates the
  base probe missed. Reranking can only reorder what expansion produced (doc 10).
- **Why LLM-generated phrases (not synonym tables).** Incident language is open-ended; an LLM
  produces context-appropriate paraphrases ("background scheduler" ↔ "triggerer") that static
  synonym lists cannot.
- **Original query always included.** Expansion augments, never replaces, the user's phrasing.
- **Best-distance merge.** An incident found by multiple phrases keeps its closest match, so
  expansion can only improve or maintain a candidate's standing.

# Tradeoffs

- **Advantage:** empirically the most reliable quality lever — consistent top-1 lifts (e.g.
  "panic: runtime error" 0.62→0.83, "memory" 0.44→0.61); recovers candidates dense missed.
- **Disadvantage:** one LLM call + P extra ANN probes per query (latency, cost,
  non-determinism); occasionally introduces off-topic phrases.
- **Alternatives considered:** pseudo-relevance feedback (no LLM, weaker), static synonym
  expansion (brittle). LLM expansion chosen for quality.

# Failure Scenarios

- **No LLM configured / LLM error** → returns no phrases → pipeline behaves as plain dense (safe).
- **Over-broad phrases** → extra candidates are still distance-ranked, so noise tends to sort
  below genuine matches; reranking can further filter.

# Sequence Diagram

```
SearchService → LLM: expand_search_query(query) → phrases[]
loop phrase in [query, *phrases]
  SearchService → DB: dense search(phrase, limit=25)
  SearchService → SearchService: merge_candidates(best distance per incident)
```

# Component Diagram

```
IncidentSearchService._expand_query → LLMService.expand_search_query
                                     → search() per phrase → _merge_candidates
```

# Database Interaction

Reads only: one dense `search()` per phrase (`embeddings`+`incidents`).

# API Interaction

OpenAI (gpt-4o-mini) for phrase generation; PostgreSQL for each phrase search.

# Performance Considerations

Adds 1 LLM round-trip + P ANN probes (P = phrase count). Probes are cheap; the LLM call
dominates added latency.

# Operational Considerations

`expansion_phrase_count` and `candidate_count` are logged. Expansion is opt-in per call.

# Future Improvements

Make expansion default-on (its gain is consistent); cache phrase expansions for repeated
queries; bound/curate phrase count for latency.

# Interview Questions

- Why is expansion the "recovery" step and reranking is not?
- Why include the original query in the searched phrase set?
- Why does best-distance merge guarantee expansion never demotes a real candidate?
- What is the latency cost model of expansion, and what dominates it?
