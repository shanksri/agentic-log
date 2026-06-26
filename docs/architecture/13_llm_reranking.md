# 13 — LLM Reranking

# Purpose

To reorder the candidate pool using an LLM's judgment of query–incident relevance, improving the
*ordering* of results beyond raw vector distance.

# Problem Statement

Vector distance is a coarse relevance proxy: two incidents at similar distances may differ greatly
in true relevance, and distance ignores fine-grained semantics (exact system, failure mode). A
reranker can apply richer judgment to the top candidates.

# High-Level Architecture

```
candidates[] (≤25, distance-sorted) ─► LLMService.rerank_incident_search_results(query, payloads, limit)
   → selected candidate ids ─► reorder ─► top-`limit`
   (fallback: distance order on any failure)
```

# Detailed Flow

When `rerank=True`, each candidate is rendered to a compact payload (title, owner/repo/source,
symptoms, severity, status, resolution_summary, similarity). The LLM is asked to select/order the
most relevant candidate ids; the service maps ids back to results, preserving the LLM order, then
backfills any shortfall from the distance-sorted remainder. Enters: candidate pool + query. Leaves:
a reordered top-`limit` list (or distance order if the LLM is absent/errors).

# Design Decisions

- **Why reranking operates only on the existing pool.** It is a *reordering* stage. It cannot fetch
  new incidents — recovery is candidate generation's job (docs 10, 12). This is the single most
  important property to internalize about reranking.
- **Compact candidate payloads.** Sending structured fields (not full bodies) keeps the prompt
  bounded and focuses the model on relevance-bearing signals.
- **Fail-safe to distance order.** Any LLM error or missing service yields the distance-sorted list,
  so reranking never degrades availability.
- **Backfill from distance order.** If the LLM returns fewer than `limit` ids, the remainder is
  filled by distance so the result is always full.

# Tradeoffs

- **Advantage:** can fix ordering where distance is ambiguous (e.g. preferring relational-DB incidents
  for "database requests slow").
- **Disadvantage:** empirically **near-neutral on top-1 score and occasionally harmful** — the model
  can over-anchor on topical affinity and discard a higher-similarity candidate (observed: "ISR shrink
  event", where rerank reverted an expansion gain from 0.50 back to 0.37). Adds LLM latency/cost and
  non-determinism.
- **Alternatives considered:** cross-encoder reranking (stronger, heavier — a future option), no
  reranking (loses ordering fixes). LLM rerank kept opt-in pending the evaluation framework.

# Failure Scenarios

- **LLM unavailable / error** → fallback to distance order (logged), retrieval continues.
- **LLM drops strong candidates** → ordering can regress vs. expansion-only; mitigated future-side by
  a guardrail that preserves materially-higher-similarity candidates (doc 17).
- **Malformed LLM id list** → only valid ids are honored; the rest backfilled by distance.

# Sequence Diagram

```
SearchService → SearchService: payloads = render(candidates)
SearchService → LLM: rerank(query, payloads, limit) → selected_ids
alt success
  SearchService: reorder by selected_ids; backfill from distance order
else failure
  SearchService: distance order[:limit]
```

# Component Diagram

```
IncidentSearchService._rerank
 ├─ _candidate_payload(index, result)
 ├─ LLMService.rerank_incident_search_results
 └─ map ids → results; backfill by distance
```

# Database Interaction

None directly — operates on already-retrieved candidates (their `incidents` rows are already loaded).

# API Interaction

OpenAI (gpt-4o-mini) for the rerank decision. No DB I/O.

# Performance Considerations

One LLM call over ≤25 compact payloads; latency is the LLM round-trip. Independent of corpus size.

# Operational Considerations

`reranking: true` is recorded in the retrieve log; failures emit `retrieval.retrieve.rerank_failed`
and fall back silently to distance order. Treat reranker gain (NDCG delta vs expansion-only) as a
measured quantity, not an assumption (doc 15).

# Future Improvements

A guardrail preventing the reranker from discarding much-higher-similarity candidates; cross-encoder
reranking evaluated through the framework; reranking only when the score gap signals ambiguity.

# Interview Questions

- Why can reranking never recover a missed candidate?
- Give a concrete case where reranking *hurt* ranking and explain the mechanism.
- Why does the reranker receive compact field payloads instead of full incident bodies?
- Why must reranking fail safe to distance order, and how is a short LLM response handled?
