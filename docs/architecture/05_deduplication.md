# 05 — Deduplication & Hashing

# Purpose

Deduplication establishes incident *identity* and detects *content changes*, so
re-ingestion updates the right row and skips unchanged work.

# Problem Statement

The same incident is ingested repeatedly (backfills, incremental syncs, overlapping
watermark windows). Without stable identity we would create duplicates; without content
hashing we would re-embed unchanged incidents on every run (expensive) or, worse,
perpetually "update" rows whose stored form differs from the freshly-computed form.

# High-Level Architecture

```
NormalizedIncident ─► DeduplicationService
   incident_key  = SHA256({source_type, source_external_id})   → row identity
   payload_hash  = SHA256(canonical JSON of raw payload)        → raw-doc change
   text_hash     = SHA256(canonical_text)                       → embedding freshness
```

# Detailed Flow

- **`incident_key`** is the dedup key stored on `incidents` (unique). Two ingests of the
  same `(source_type, source_external_id)` map to the same row regardless of mutable content.
- **`payload_hash`** detects whether the stored raw document changed.
- **`text_hash`** is compared against the stored embedding's `text_hash`; if equal, the
  incident is **skipped** (no re-embed). If the incident exists but text changed, it is
  **updated** and re-embedded. If absent, it is **inserted**.

# Design Decisions

- **Why dedup (identity) happens before embedding.** Embedding is the most expensive step.
  Resolving identity and comparing `text_hash` first lets us skip embedding entirely for
  unchanged incidents — the common case on incremental runs and overlap windows.
- **Identity excludes mutable content.** `incident_key` hashes only `(source_type,
  source_external_id)`, so editing a title/body never forks a new incident.
- **Hash the persisted form.** `payload_hash` is computed from the *sanitized* payload (the
  exact bytes stored), and `text_hash` from the stored `canonical_text`. This guarantees the
  hash and the stored object always agree — otherwise sanitized vs. raw mismatches would cause
  endless false updates (see doc 07).
- **Canonical JSON** (`sort_keys`, tight separators) makes `payload_hash` order-independent.

# Tradeoffs

- **Advantage:** idempotent ingestion, cheap re-runs, no duplicate rows, embedding skipped
  when safe.
- **Disadvantage:** identity is only as good as `source_external_id` stability; a source that
  reissues IDs would mis-dedup.
- **Alternative considered:** content-based dedup (hash of body) — rejected; it would treat an
  edited incident as a new one and break update semantics.

# Failure Scenarios

- **Overlap window re-fetches** (watermark = run_start_time) → same `incident_key`,
  `text_hash` matches → skipped. No churn.
- **Two sources produce an identical raw payload** → distinct `incident_key`
  (`source_type` differs) but a shared `payload_hash` could collide with the
  `raw_documents.payload_hash` unique constraint; realistically negligible, flagged as an
  edge case.
- **Canonical text template change** → `text_hash` shifts for all incidents → controlled
  re-embed via the backfill script.

# Sequence Diagram

```
IngestionService → Dedup: incident_key(normalized)
IngestionService → DB: SELECT incident WHERE dedup_key = key
IngestionService → Dedup: text_hash(canonical_text)
alt existing AND embedding text_hash == text_hash
   IngestionService: return "skipped"
else existing
   IngestionService: update row + re-embed
else
   IngestionService: insert row + embed
```

# Component Diagram

```
DeduplicationService
 ├─ incident_key()  → incidents.deduplication_key (unique)
 ├─ payload_hash()  → raw_documents.payload_hash (unique)
 └─ text_hash()     → embeddings.text_hash
```

# Database Interaction

- **Reads:** `incidents` by `deduplication_key`; `embeddings` for `text_hash` comparison.
- **Writes (indirectly):** the keys are stored on `incidents.deduplication_key`,
  `raw_documents.payload_hash`, `embeddings.text_hash`.

# API Interaction

None — pure hashing over in-memory structures.

# Performance Considerations

SHA256 over JSON/text strings: microseconds per incident. The big win is *avoided*
embedding compute on skip. Lookups are indexed (unique constraints).

# Operational Considerations

Deterministic and stateless. The `payload_hash` unique constraint is the last-line guard
against duplicate raw documents.

# Future Improvements

Near-duplicate detection (semantic) across sources; configurable identity strategy per
source for systems with unstable external IDs.

# Interview Questions

- Why must dedup precede embedding rather than follow it?
- Why does `incident_key` exclude title/body?
- Why is `text_hash` computed from the stored canonical text rather than the raw body?
- What would happen if you hashed the pre-sanitization payload? (See doc 07.)
