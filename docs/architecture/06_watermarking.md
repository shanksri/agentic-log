# 06 — Watermarking (Incremental Ingestion)

# Purpose

Watermarks let ingestion run incrementally — fetching only what changed since the last
successful run — instead of re-walking thousands of historical incidents every time.

# Problem Statement

A full backfill of a large source on every run is untenable (network, time, rate limits).
We need a resumable "since" marker that is safe across crashes, partial failures, clock
skew, and pagination drift, and that never silently misses updates.

# High-Level Architecture

```
incident_sources.last_ingested_at  (NULL = never run)
        │
WatermarkService.resolve(source, force_backfill)
   NULL or force_backfill → ("backfill",  since=None)
   else                    → ("incremental", since=last_ingested_at)
        │
   collect(config, since)  →  ingest  →  COMMIT incidents
        │
WatermarkService.advance(source, run_start_time)  →  COMMIT watermark
```

# Detailed Flow

Enters: a source row. `resolve` returns mode + `since`. `run_start_time` is captured
**before** the first API call. Collection uses `since` (GitHub `since` param / Jira
`updated >=`). After the incident commit succeeds, `advance` sets
`last_ingested_at = run_start_time` and commits separately. Leaves: an advanced watermark
only on success.

# Design Decisions

- **Why the watermark is `run_start_time`, not `max(updated_at)` of the batch.** Using the
  max-seen timestamp risks missing items updated *during* the run and is vulnerable to
  pagination cursor drift (items shifting pages mid-run). `run_start_time` guarantees the next
  run's window `[run_start_time, ∞)` overlaps the current run's window, so anything changed
  during the run is caught next time. Cost: a small overlap re-fetch, which dedup skips.
- **Two-commit ordering (incidents first, watermark second).** If the process crashes between
  commits, the watermark is *not* advanced and the next run re-processes the overlap
  idempotently. The watermark can never run ahead of persisted data.
- **Advance even on empty/partial runs.** An empty incremental run still advances (nothing
  changed). A `low_yield`/`timeout_partial` collection also advances (intended for low-yield
  repos); operators use `force_backfill` to force a full re-walk.
- **Jira minute-resolution buffer.** JQL timestamps are minute-granular; truncation yields a
  0–59s safety margin that also absorbs index lag.

# Tradeoffs

- **Advantage:** O(changed) instead of O(corpus) per run; crash-safe; no missed updates by
  construction.
- **Disadvantage:** small repeated overlap re-fetch each run (cheap, dedup-skipped); a
  `timeout_partial` run advances the watermark, so un-scanned pages aren't auto-retried
  without `force_backfill`.
- **Alternatives considered:** `max_seen_updated_at` (silent-miss risk under drift/skew) and
  `max_seen − safety_window` (tuning-dependent, still drift-prone). `run_start_time` was chosen
  for correctness-by-construction.

# Failure Scenarios

- **Crash after incident commit, before watermark commit** → watermark stale → next run
  re-fetches overlap → all dedup-skip. Safe.
- **Item updated mid-run, missed by pagination** → caught by the next run because
  `since = run_start_time ≤ that update`. Safe.
- **Clock skew (seconds) between app and source** → absorbed by the overlap window (much
  larger than NTP skew).

# Sequence Diagram

```
IngestionService → WatermarkService: resolve(source) → (mode, since)
IngestionService: run_start_time = now()   # before any API call
IngestionService → Collector: collect(config, since)
IngestionService → DB: COMMIT incidents
IngestionService → WatermarkService: advance(source, run_start_time)
IngestionService → DB: COMMIT watermark
```

# Component Diagram

```
WatermarkService
 ├─ resolve(source, force_backfill) → (mode, since)
 └─ advance(source, new_watermark)  → sets last_ingested_at (caller commits)
```

# Database Interaction

- **Reads:** `incident_sources.last_ingested_at`.
- **Writes:** `incident_sources.last_ingested_at` (second commit, post-success).

# API Interaction

Indirect — `since` is translated by the collector into the source API's incremental filter
(GitHub `since`, Jira `updated >=`).

# Performance Considerations

Negligible service cost; the value is the *avoided* collection. Overlap re-fetch is bounded
by the run's wall-clock duration, not the inter-run interval.

# Operational Considerations

`mode`, `previous_watermark`, `new_watermark` are logged per run. `force_backfill=True`
resets to a full walk. NULL watermark = first run = backfill.

# Future Improvements

A separate "needs re-backfill" flag for `timeout_partial` runs; per-source configurable
overlap/buffer.

# Interview Questions

- Why `run_start_time` instead of the maximum `updated_at` observed?
- What happens if the watermark advances after a timeout-partial run?
- Why two commits instead of one, and what does the gap protect against?
- How does the design tolerate an item being updated while pagination is in flight?
