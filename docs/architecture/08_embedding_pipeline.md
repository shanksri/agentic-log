# 08 — Embedding Pipeline

# Purpose

To turn each incident's `canonical_text` into a dense vector that supports semantic
similarity search, and to do so only when necessary.

# Problem Statement

Lexical search misses paraphrases ("triggerer not starting" vs "background scheduler refuses
to launch"). We need a vector representation that places semantically similar incidents near
each other, generated consistently for both ingested incidents and incoming queries.

# High-Level Architecture

```
canonical_text ─► EmbeddingService(all-MiniLM-L6-v2) ─► float32[384]
   ingest:  embed(incident.canonical_text) → embeddings.embedding
   query:   embed(query string)            → search vector
   guard:   re-embed only if text_hash changed
```

# Detailed Flow

**Ingest.** After a row is inserted/updated, `_upsert_embedding` computes `text_hash`,
checks whether a fresh embedding exists for the current model, and if not, calls
`embed_text(canonical_text)` and upserts the vector. **Query.** The same model embeds the
query string at retrieval time. Enters: a string. Leaves: a 384-dim vector.

# Design Decisions

- **Why the same model for ingest and query.** Similarity is only meaningful within one vector
  space; embedding queries and documents with the same model is mandatory for cosine comparison.
- **Why MiniLM-L6-v2 (384-dim).** Small, fast, CPU-friendly, good general semantic quality —
  appropriate for v1 where iteration speed and local operation matter more than top-end recall.
- **Why embed `canonical_text` (not raw body).** The normalized template concentrates signal
  (title, type, severity, symptoms, resolution) and removes noise, improving neighbor quality
  (doc 04).
- **Why re-embed is gated on `text_hash`.** Embedding is the dominant compute cost; skipping
  unchanged text makes incremental ingestion and re-runs cheap (doc 05).
- **Cosine similarity** is reported as `max(0, 1 − cosine_distance)` so scores are intuitive in
  `[0, 1]` and feed confidence calibration (doc 14).

# Tradeoffs

- **Advantage:** paraphrase-robust retrieval; cheap, local inference; deterministic per text.
- **Disadvantage:** 384-dim underfits very rich incidents (postmortems); a single general model
  is weaker on rare jargon and exact tokens (where lexical would help — doc 17); an embedding
  model upgrade invalidates the whole index and requires full re-embed.
- **Alternatives considered:** larger embedding models (higher recall, higher cost/latency),
  hosted embeddings (network dependency). MiniLM retained for v1.

# Failure Scenarios

- **Model/version change** → every stored vector is in a different space → must re-embed the
  whole corpus; `model_name` on each embedding row makes this detectable and stageable.
- **Embedding backend error mid-ingest** → `ingest()` raises before watermark advance; prior
  committed items stand; re-run is idempotent.

# Sequence Diagram

```
IngestionService → EmbeddingService: embed_text(canonical_text)
EmbeddingService → IngestionService: float32[384]
IngestionService → DB: upsert embeddings(incident_id, model_name, embedding, text_hash)
---
SearchService → EmbeddingService: embed_text(query)
EmbeddingService → SearchService: float32[384]
```

# Component Diagram

```
EmbeddingService(model_name = sentence-transformers/all-MiniLM-L6-v2)
 └─ embed_text(str) -> list[float]  (384)
```

# Database Interaction

- **Writes:** `embeddings(incident_id, model_name, embedding(vector(384)), text_hash)`,
  unique on `(incident_id, model_name)`.
- **Reads:** existing `embeddings` to decide freshness.

# API Interaction

Local model inference (no external API). PostgreSQL/pgvector stores the vectors.

# Performance Considerations

O(1) per text, but CPU-bound and the heaviest ingest step; batching would help throughput.
Vector size 384×4 bytes per incident. Query embedding adds one inference to read latency.

# Operational Considerations

`model_name` is stamped on every embedding for migration safety. Re-embedding is a controlled,
resumable batch (the backfill script). Freshness skip is the main cost control.

# Future Improvements

Batched embedding at ingest; an embedding-model upgrade path with dual-write/shadow index;
domain-tuned or larger models evaluated through the framework (doc 15) before adoption.

# Interview Questions

- Why must query and document embeddings use the same model?
- Why is re-embedding gated on `text_hash` and what does that save?
- What is the blast radius of upgrading the embedding model, and how is it made safe?
- Why embed `canonical_text` rather than the raw issue body?
