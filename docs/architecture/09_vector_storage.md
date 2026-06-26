# 09 — Vector Storage (pgvector / HNSW)

# Purpose

To store incident embeddings and serve approximate nearest-neighbor (ANN) search
efficiently inside the same PostgreSQL database that holds the relational incident data.

# Problem Statement

Brute-force cosine over ~8,000 (growing) vectors per query would scale linearly and couple
poorly with relational filtering. We need fast similarity search co-located with incident
metadata so a single query can rank and filter together.

# High-Level Architecture

```
embeddings(embedding vector(384))  ──  HNSW index (vector_cosine_ops)
        │
SELECT incidents.*, embedding <=> :qvec AS distance
  JOIN embeddings ... WHERE model_name = :m
  ORDER BY distance LIMIT k
```

# Detailed Flow

The query vector is compared via pgvector's cosine distance operator against the HNSW index;
PostgreSQL returns the top-k incidents joined to their metadata, ordered by distance. Optional
relational filters (`source_type`, `owner`, `repo`, `source`, `state`, `tags`) are applied in
the same statement. Enters: a 384-dim vector + filters + k. Leaves: ranked `(incident,
distance)` rows.

# Design Decisions

- **Why pgvector (in-database) rather than a separate vector store.** Keeps vectors
  transactionally consistent with incidents, lets one SQL statement combine ANN ranking with
  relational filters, and avoids operating a second datastore for a v1 corpus of this size.
- **Why HNSW (not IVFFlat / brute force).** HNSW gives strong recall at low latency without a
  training step and degrades gracefully as the corpus grows; IVFFlat needs list tuning and
  re-training; brute force is linear.
- **Why cosine (`vector_cosine_ops`).** Sentence-embedding similarity is direction-based;
  cosine is the standard match for MiniLM outputs. Similarity surfaced as `1 − distance`.
- **One index over all sources.** Cross-source semantic search is a goal; partitioning by source
  would defeat it. Source scoping is done via WHERE filters, not separate indexes.

# Tradeoffs

- **Advantage:** transactional, single-query rank+filter, no extra infra, good ANN performance.
- **Disadvantage:** HNSW is approximate (rare recall misses); index build/insert cost grows with
  corpus; a dimension/model change requires rebuilding the index.
- **Alternatives considered:** dedicated vector DBs (more features/scale, more ops burden),
  IVFFlat (tuning overhead). pgvector+HNSW chosen for simplicity and consistency at current scale.

# Failure Scenarios

- **Corpus growth densifies neighborhoods** → genuine matches can drop in rank as competitors
  appear; this is corpus drift, addressed by the evaluation framework, not the index (doc 15).
- **Index rebuild after model upgrade** → handled offline; `model_name` filter isolates spaces
  during transition.

# Sequence Diagram

```
SearchService → DB: SELECT incident, embedding <=> qvec AS distance
                     JOIN embeddings WHERE model_name=m [filters]
                     ORDER BY distance LIMIT k
DB(HNSW) → SearchService: top-k (incident, distance)
```

# Component Diagram

```
PostgreSQL
 ├─ incidents (relational + JSONB metadata)
 └─ embeddings (vector(384)) ── HNSW(vector_cosine_ops)
```

# Database Interaction

- **Reads:** `embeddings` (vector search) joined to `incidents` (+`symptoms` eager-loaded).
- **Writes:** vectors inserted/updated by the embedding pipeline (doc 08).
- **Indexes:** HNSW on `embeddings.embedding`; supporting indexes on `incidents`
  (`source_type`, `owner`, `repo`, `source`, `state`, tags GIN, trigram full-text).

# API Interaction

PostgreSQL + pgvector extension only.

# Performance Considerations

ANN query is sub-linear (HNSW graph traversal); latency dominated by the single inference to
embed the query plus the index probe. Memory: HNSW graph + 384×4 bytes/vector. Insert cost rises
with graph size.

# Operational Considerations

Index defined in migration `0001`. Rebuilds should use `CREATE INDEX CONCURRENTLY` to avoid
locking. The `model_name` column is the migration-safety lever for embedding upgrades.

# Future Improvements

Hybrid retrieval combining HNSW with the existing trigram/BM25-style lexical index (doc 17);
HNSW parameter tuning (`ef_search`) evaluated via the framework; partitioning only if scale
demands it.

# Interview Questions

- Why store vectors in PostgreSQL instead of a dedicated vector database?
- Why HNSW over IVFFlat or brute force for this workload?
- How does corpus growth interact with ANN ranking, and whose responsibility is that?
- What has to happen to the index if the embedding model changes?
