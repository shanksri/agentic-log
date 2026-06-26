# 10 — Retrieval Pipeline

# Purpose

To orchestrate the read path: turn a query string into a ranked, confidence-labeled list of
incidents, with optional query expansion and LLM reranking layered on a dense base.

# Problem Statement

A single dense lookup is a strong default but has blind spots (rare jargon, terse queries) and
returns no quality signal. Retrieval must offer a tunable pipeline — cheap by default, richer
when needed — through one entry point used by both the search API and the investigation agent.

# High-Level Architecture

```
query ─► IncidentSearchService.retrieve(expand, rerank)
   1. embed query
   2. dense HNSW search (candidate_limit = 25 if expand/rerank else limit)
   3. [expand] LLM phrases → search each → merge candidates (best distance per incident)
   4. sort candidates by distance
   5. [rerank] LLM reorders candidate pool (fallback: distance order)
   6. classify_confidence(top1 similarity)
   → results + confidence
```

# Detailed Flow

Enters: query, `limit`, optional filters, `expand`, `rerank`. The base `search()` does the
dense HNSW lookup (doc 09). With expansion, the query plus LLM-generated phrases are each
searched and merged into a candidate map keyed by incident id, keeping the best (lowest)
distance per incident. Candidates are distance-sorted; if reranking is on, the LLM reorders the
pool (failing safe to distance order). Confidence is derived from the top-1 similarity. Leaves:
`IncidentSearchResult[]` (incident + distance/similarity) and a confidence label.

`search_debug()` is a backward-compatible alias for `retrieve(expand=True, rerank=True, limit=5)`.

# Design Decisions

- **One canonical entry point (`retrieve`).** Search routes and the agent share it; `search()`
  remains as the dense primitive and candidate generator.
- **Why expansion happens before reranking.** Expansion *grows* the candidate pool; reranking
  only *reorders* it. Reranking after expansion can promote a good candidate that expansion
  surfaced; the reverse order would have nothing extra to reorder.
- **Why reranking cannot recover missed candidates.** The reranker only sees the candidate pool
  produced by (expanded) dense search. If the correct incident never entered that pool, no
  amount of reranking can retrieve it — recovery is the job of candidate generation (expansion /
  future hybrid), not reranking.
- **Wider candidate pool when expand/rerank on (`25` vs `limit`).** Gives the LLM stages room to
  work; the final list is truncated to `limit`.
- **Confidence from top-1 similarity** (doc 14) — a single, interpretable signal computed at the
  end regardless of pipeline config.
- **Fail-safe LLM stages.** Any LLM error falls back to distance order; retrieval never hard-fails
  because of the optional stages.

# Tradeoffs

- **Advantage:** cheap deterministic default; opt-in quality; uniform interface; graceful
  degradation.
- **Disadvantage:** expand/rerank add LLM latency/cost and non-determinism; reranking is
  empirically near-neutral on score and occasionally reorders worse (doc 13).
- **Alternatives considered:** always-on reranking (cost, non-determinism) and dense-only
  (blind spots). The opt-in layering balances both.

# Failure Scenarios

- **LLM unavailable** → expansion returns no phrases / rerank falls back → behaves as dense.
- **Corpus drift / hub incidents** → a lexically-overlapping incident from another domain wins
  the dense step (e.g. Kafka "infinite loop … start a broker" hijacking "triggerer not
  starting"); expansion partially mitigates, but this is a candidate-generation limitation
  (doc 16).

# Sequence Diagram

```
Caller → SearchService: retrieve(query, expand, rerank)
SearchService → EmbeddingService: embed(query)
SearchService → DB: dense HNSW search (candidate_limit)
opt expand
  SearchService → LLM: expand_search_query(query)
  loop phrase: SearchService → DB: search(phrase); merge candidates
SearchService: sort candidates by distance
opt rerank
  SearchService → LLM: rerank(query, candidates) (fallback: distance order)
SearchService → Confidence: classify(top1)
SearchService → Caller: results + confidence
```

# Component Diagram

```
IncidentSearchService
 ├─ search()        dense HNSW primitive (doc 09)
 ├─ _expand_query() → LLMService.expand_search_query (doc 11)
 ├─ _merge_candidates() (doc 12)
 ├─ _rerank()       → LLMService.rerank_... (doc 13)
 └─ confidence_for()/classify_confidence (doc 14)
```

# Database Interaction

- **Reads:** `embeddings` + `incidents` (+`symptoms`) via `search()`; once per phrase when
  expanding.
- **Writes:** none (read path).

# API Interaction

OpenAI (gpt-4o-mini) for expansion/rerank; PostgreSQL/pgvector for dense search.

# Performance Considerations

Dense: one query embedding + one ANN probe. Expansion: 1 + P phrase searches (P probes) plus an
LLM call. Rerank: one LLM call over ≤25 candidates. Latency scales with the LLM stages, not the
corpus.

# Operational Considerations

`retrieval.search` / `retrieval.retrieve` structured logs capture config, candidate counts,
top-1 score, confidence, and duration. `call_site` distinguishes API vs agent traffic.

# Future Improvements

Hybrid candidate generation (dense+lexical) to fix the recovery blind spot; default-on
expansion; reranker guardrails (doc 13/17). All compatible with the current `retrieve` shape.

# Interview Questions

- Why does expansion run before reranking?
- Why can reranking never recover a missed candidate, and what stage owns recovery?
- Why widen the candidate pool to 25 when expand/rerank are enabled?
- How does the pipeline behave when the LLM is unavailable, and why is that acceptable?
