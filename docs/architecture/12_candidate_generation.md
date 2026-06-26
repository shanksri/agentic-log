# 12 — Candidate Generation & Merge

# Purpose

To assemble a single, de-duplicated candidate pool from one or more dense searches (the
original query plus any expansion phrases), keeping each incident's best match.

# Problem Statement

Expansion issues multiple searches that overlap; the same incident may appear in several result
sets at different distances. The reranker and the final ranking need one clean pool with each
incident represented once, at its strongest score.

# High-Level Architecture

```
search(query)      ┐
search(phrase1)    ├─► _merge_candidates ─► {incident_id → best (lowest-distance) result}
search(phrase2)    ┘            │
                       sorted by distance ─► candidates[]
```

# Detailed Flow

Each dense search returns up to the candidate limit (25 when expand/rerank on). `_merge_candidates`
inserts each result into a map keyed by incident id, replacing an existing entry only if the new
distance is smaller. After all phrases, the map's values are sorted ascending by distance to form
the candidate list passed to reranking (or truncated to `limit` directly). Enters: per-phrase result
lists. Leaves: a distance-sorted, de-duplicated candidate list.

# Design Decisions

- **Keep best distance per incident.** Guarantees that adding phrases can only help an incident's
  standing — its represented distance is the minimum across all phrasings.
- **Key on incident id.** The natural de-dup unit for the read path (one incident, one candidate
  slot), independent of which phrase found it.
- **Distance-sort before rerank.** Gives the reranker a sensible prior and is the fallback order if
  reranking is skipped or fails.
- **Candidate limit of 25 under expand/rerank.** Wide enough for the LLM stages to find better
  orderings, bounded enough to keep the rerank prompt and latency in check.

# Tradeoffs

- **Advantage:** clean, minimal pool; monotonic improvement from expansion; deterministic given the
  searches.
- **Disadvantage:** a fixed pool size can exclude a relevant-but-distant incident that only a
  better candidate generator (hybrid/lexical) would surface.
- **Alternatives considered:** score-sum/voting across phrases (amplifies popular-but-generic hub
  incidents), reciprocal-rank fusion (viable future option). Best-distance chosen for simplicity and
  its monotonic guarantee.

# Failure Scenarios

- **Hub incident appears for many phrases** → it still occupies one slot at its best distance; it
  does not get vote-boosted (a deliberate benefit of best-distance over sum-of-scores).
- **Relevant incident outside every phrase's top-25** → never enters the pool; only candidate-generation
  improvements (doc 17) can fix this — reranking cannot (doc 13).

# Sequence Diagram

```
loop phrase
  SearchService → DB: search(phrase, limit=25)
  SearchService → SearchService: merge(candidate_map, results)  # keep min distance
SearchService → SearchService: candidates = sort(candidate_map.values, by distance)
```

# Component Diagram

```
IncidentSearchService._merge_candidates(candidate_map, results)
   → candidate_map[incident_id] = min-distance result
```

# Database Interaction

None of its own; consumes the outputs of `search()` (which reads `embeddings`+`incidents`).

# API Interaction

None (pure in-memory merge).

# Performance Considerations

O(total results) to build the map; O(n log n) to sort the unique candidates. Negligible vs. the
ANN probes and LLM calls.

# Operational Considerations

`candidate_count` (unique incidents in the pool) is logged on the retrieve path, useful for
diagnosing recall vs. precision behavior.

# Future Improvements

Reciprocal-rank fusion or hybrid (dense+lexical) candidate sources merged here; dynamic pool sizing
based on score gaps.

# Interview Questions

- Why keep the minimum distance per incident instead of summing scores across phrases?
- How does best-distance merge protect against generic "hub" incidents?
- Why is the candidate pool the true ceiling on recall, and which component can raise it?
- What determines whether a relevant incident can ever be reranked into the top results?
