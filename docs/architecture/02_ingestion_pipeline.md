# 02 — Ingestion Pipeline

# Purpose

To explain the orchestration layer that turns external source payloads into durable,
deduplicated, embedded incidents — and how new sources are dispatched uniformly.

# Problem Statement

We ingest from multiple, schema-incompatible sources and must do so idempotently,
incrementally, and resiliently, without per-source branches scattered through the
codebase. The pipeline must support both ad-hoc imports (corpus building) and
stored/scheduled ingestion (automation) over one execution path.

# High-Level Architecture

```
                 ┌─ Mode A: ad-hoc payload ─┐     ┌─ Mode B: stored source ─┐
 ingest_github_repo(...)  ingest_jira_project(...)   ingest_source(source_id)
        │  (auto get-or-create row)                     │ (load row + config)
        └──────────────────────┬────────────────────────┘
                               ▼
                _dispatch(source, config, force_backfill)
                   adapter = SourceRegistry.get(source.source_type)
                               ▼
                ingest_with_adapter(source, adapter, config)
                   ├─ WatermarkService.resolve → (mode, since)
                   ├─ adapter.collect(config, since) → payloads
                   ├─ adapter.normalize(payload)     → NormalizedIncident[]
                   ├─ ingest(): per item → upsert_raw → upsert_incident → upsert_embedding
                   ├─ commit incidents
                   └─ WatermarkService.advance → commit watermark
```

# Detailed Flow

**Enters:** a source identity (id or payload) + config.
1. **Acquire source row.** Mode A auto-creates an `incident_sources` row; Mode B loads one
   and reads its `config` JSONB.
2. **Resolve adapter** from `SourceRegistry` by `source.source_type`.
3. **Resolve watermark** → backfill (no `since`) or incremental (`since = last_ingested_at`).
4. **Collect** raw payloads (paginated, diagnostics-instrumented — doc 03).
5. **Normalize** each to `NormalizedIncident` (doc 04).
6. **Persist** per item: sanitize payload + hash (docs 05, 07), upsert `raw_documents`,
   upsert `incidents` (+`symptoms`), upsert `embeddings` (doc 08). Skip when content hash
   is unchanged.
7. **Commit incidents, then advance + commit watermark** (two-commit ordering — doc 06).

**Leaves:** counts (`fetched/inserted/updated/skipped`), `mode`, watermark bounds, and
collector diagnostics (`exit_reason`, `pages_traversed`, …).

# Design Decisions

- **Single `_dispatch` core.** Both modes and all sources funnel through one path that
  resolves the adapter via the registry. No hardcoded adapter instantiation, no
  `if source == "github"` branches.
- **Config in the database (`incident_sources.config`).** A new GitHub repo is onboarded
  by inserting a row — zero code. Secrets (GitHub token) inherit from the environment so
  the row need not store them.
- **Dedup/sanitize/hash live at the persistence boundary**, keeping collectors and
  normalizers pure and reusable.
- **Two-commit watermark ordering** guarantees a crash never advances past un-persisted data.

# Tradeoffs

- **Advantage:** onboarding cost approaches zero for issue-tracker sources; one tested path.
- **Disadvantage:** ad-hoc Mode A still creates an auto-managed source row (the
  `raw_documents.source_id` FK is `NOT NULL`); a truly rowless ephemeral import would need a
  schema change.
- **Alternative considered:** per-source ingestion services — rejected as duplicative and
  drift-prone.

# Failure Scenarios

- **Collector raises mid-run** → no commit, watermark unchanged → next run safely retries.
- **Embedding backend down** → `ingest()` raises before watermark advance; already-committed
  prior items stand; re-run is idempotent via content hash.
- **Partial/low-yield collection** → returns partial results with `exit_reason`; watermark
  still advances (intended for low-yield repos; use `force_backfill` to re-walk).

# Sequence Diagram

```
Caller → IngestionService: ingest_source(id)
IngestionService → DB: load incident_sources row (+config)
IngestionService → WatermarkService: resolve(source) → (mode, since)
IngestionService → Adapter: collect(config, since)
Adapter → IngestionService: payloads
loop per payload
  IngestionService → Adapter: normalize(payload)
  IngestionService → Sanitizer: sanitize_json(payload)
  IngestionService → DB: upsert raw_document, incident, embedding
IngestionService → DB: COMMIT incidents
IngestionService → WatermarkService: advance(run_start_time)
IngestionService → DB: COMMIT watermark
```

# Component Diagram

```
IngestionService
 ├─ SourceRegistry → SourceAdapter (collect/normalize)
 ├─ DeduplicationService
 ├─ EmbeddingService
 ├─ WatermarkService
 └─ json_sanitizer
```

# Database Interaction

- **Reads:** `incident_sources` (row + config + watermark); existing `raw_documents`,
  `incidents`, `embeddings` for upsert decisions.
- **Writes:** `incident_sources` (create + watermark), `raw_documents`, `incidents`,
  `symptoms`, `embeddings`.

# API Interaction

GitHub/Jira via the adapter's collector; OpenAI not involved on the write path;
PostgreSQL for all persistence.

# Performance Considerations

O(fetched) DB upserts + O(inserted/updated) embedding computations. The embedding step
dominates CPU; the collector network round-trips dominate wall-clock. Skipping unchanged
content (hash match) keeps re-ingestion cheap.

# Operational Considerations

One `ingestion_complete` structured log per run with mode, watermark bounds, counts, and
diagnostics. Idempotent and resumable. `force_backfill` forces a full re-walk.

# Future Improvements

A generic `POST /ingestion/sources/{id}/ingest` route (Mode B over HTTP); parallel
collection across sources; batch embedding.

# Interview Questions

- Why do ad-hoc and stored ingestion share `_dispatch` instead of separate services?
- Why does the watermark commit happen in a *second* transaction after the incident commit?
- A GitHub repo needs onboarding tonight — what is the minimum change required and why?
- What makes re-running an ingestion safe?
