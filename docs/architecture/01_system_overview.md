# 01 — System Overview

> **Update note (docs 18–22):** the diagrams and prose below describe the system through
> Retrieval v1 / early Phase 16 — dense-only retrieval with a single-shot investigation agent.
> Both "Future Improvements" items this doc lists (hybrid retrieval, the evaluation platform) have
> since shipped. See [doc 18](18_adaptive_routing_and_hybrid_confidence.md) for adaptive
> routing/hybrid retrieval/confidence normalization, [doc 19](19_multi_agent_investigation.md) for
> the four-agent investigation framework that now exists alongside the single-shot agent described
> here, and [docs 20](20_reasoning_evaluation_and_judges.md)–[22](22_evaluation_api.md) for the
> full evaluation platform. None of docs 18/19 is wired into an API route yet — the read path
> described below remains what's actually reachable over HTTP today.

# Purpose

To give a single, accurate mental model of the entire platform: the components, how
data flows from an external incident to a ranked retrieval result, and the invariants
that hold across the system.

# Problem Statement

Engineering organizations accumulate incident knowledge across fragmented systems
(GitHub Issues, Jira, monitoring tools). When a new incident occurs, the relevant
prior art is hard to find: keyword search misses paraphrases, and the knowledge spans
multiple tools with incompatible schemas. This system unifies heterogeneous incident
sources into one semantically searchable corpus and serves "find similar past
incidents" with calibrated confidence.

# High-Level Architecture

```
                       ┌──────────── INGESTION (write path) ────────────┐
 GitHub API ─┐         │ Collector → Normalizer → Adapter                │
 Jira API  ──┼──►  SourceRegistry ──► IngestionService                   │
             │         │   ├─ sanitize_json (persistence boundary)       │
             │         │   ├─ DeduplicationService (identity/content)     │
             │         │   ├─ EmbeddingService (MiniLM-L6-v2, 384-d)      │
             │         │   └─ WatermarkService (incremental)              │
             │         └─────────────────────────────────────────────────┘
             ▼
        PostgreSQL  ──  incident_sources · raw_documents · incidents · symptoms · embeddings(pgvector HNSW)
             ▲
             │         ┌──────────── RETRIEVAL (read path) ─────────────┐
 User query ─┴──►  IncidentSearchService.retrieve()                     │
                       │   dense search → [expand] → merge → [rerank]    │
                       │   → confidence calibration → results            │
                       └────────────────────────────────────────────────┘
                                          │
                              Investigation Agent (hypotheses + evidence)
```

# Detailed Flow

**Write path (ingestion).** An external API payload enters a *collector*, which
paginates and returns raw issue dicts. A *normalizer* maps each raw dict to the
canonical `NormalizedIncident`. An *adapter* (resolved via `SourceRegistry`) wraps the
collector+normalizer behind a uniform interface. `IngestionService` then, at the
persistence boundary: sanitizes the raw payload, computes identity/content hashes,
upserts `raw_documents` and `incidents`, generates an embedding, and advances the
source watermark. What leaves: durable, deduplicated, embedded incidents.

**Read path (retrieval).** A query string enters `IncidentSearchService`. It is
embedded with the same model used at ingest, searched against the HNSW index (cosine),
optionally expanded (LLM generates related phrases, each searched, candidates merged),
optionally reranked (LLM reorders the candidate pool), and assigned a confidence level
from the top-1 similarity. What leaves: a ranked list of incidents with scores and a
confidence label.

# Design Decisions

- **Single canonical shape (`NormalizedIncident`).** Decouples N sources from the rest
  of the system; dedup/embedding/retrieval never branch on source type.
- **Registry-driven dispatch.** New sources are onboarded by registering an adapter (or,
  for GitHub repos, inserting a config row) — not by editing the pipeline.
- **Write-path heavy, read-path light.** Expensive normalization/embedding happens once
  at ingest; retrieval is a vector lookup plus optional LLM passes.
- **Identity ≠ UUID.** Stable identity is `(source_type, source_external_id)`; this
  survives re-ingestion and underpins dedup and evaluation.

# Tradeoffs

- **Advantage:** uniform downstream, cheap reads, source-agnostic evaluation.
- **Disadvantage:** the canonical shape is issue-tracker-flavored (title/body/resolution);
  non-issue sources (monitoring) will need schema accommodation (doc 17).
- **Alternative considered:** per-source retrieval indexes — rejected because cross-source
  semantic search ("find similar incidents regardless of tool") is the core value.

# Failure Scenarios

- **Source API outage / timeout** → collector returns partial results with diagnostics;
  watermark only advances on success (doc 06).
- **Malformed payload (control chars)** → sanitized at the persistence boundary; the
  incident is preserved, not dropped (doc 07).
- **Corpus growth changing rankings** → handled by the evaluation framework's
  fingerprinting, not by the runtime (doc 15).

# Sequence Diagram

```
Client → IngestionService: ingest(source)
IngestionService → Collector: collect(config, since)
Collector → SourceAPI: paginated GET
Collector → IngestionService: raw payloads + diagnostics
IngestionService → Normalizer: normalize(raw)
IngestionService → DB: sanitize → hash → upsert raw/incident
IngestionService → EmbeddingService: embed(canonical_text)
IngestionService → DB: upsert embedding; advance watermark
---
Client → SearchService: retrieve(query, expand, rerank)
SearchService → EmbeddingService: embed(query)
SearchService → DB(pgvector): HNSW cosine search
SearchService → LLM: expand / rerank (optional)
SearchService → Client: ranked results + confidence
```

# Component Diagram

```
SourceRegistry ──> [GitHubAdapter, JiraAdapter]
                       │            │
                  GitHubCollector  JiraCollector
                  GitHubNormalizer JiraNormalizer
                       └──────┬─────┘
                       IngestionService ──> Dedup, Embedding, Watermark, Sanitizer
                              └──> PostgreSQL(+pgvector) <── IncidentSearchService ──> LLMService
```

# Database Interaction

- **Written by ingestion:** `incident_sources`, `raw_documents`, `incidents`,
  `symptoms`, `embeddings`.
- **Read by retrieval:** `incidents` (joined) + `embeddings` (vector search).

# API Interaction

- **GitHub REST** (issues + comments), **Jira REST/JQL** (issues) — ingestion inbound.
- **OpenAI (gpt-4o-mini)** — query expansion + reranking on the read path; hypothesis
  generation in the agent.
- **PostgreSQL + pgvector** — primary store and vector index.

# Performance Considerations

Ingestion is O(incidents) and dominated by network + embedding compute. Retrieval is
sub-linear via HNSW for the dense step; expansion/rerank add LLM round-trips
(hundreds of ms to seconds). Embeddings are 384-dim float32.

# Operational Considerations

Structured logs at each boundary (`collection_complete`, `retrieval.search`,
`ingestion_complete`). Ingestion is idempotent and resumable. LLM steps degrade
gracefully (fall back to dense order on failure).

# Future Improvements

Hybrid (dense+lexical) retrieval, confidence recalibration for the grown corpus, and
the full evaluation platform (docs 15–17). No architectural redesign required.

**Status:** all three have shipped, unwired. Hybrid retrieval and adaptive routing exist as a
fully-built, unit-tested library (doc 18) but are not reachable through `app/api/routes/search.py`.
Confidence recalibration exists as a strategy-aware normalization layer (doc 18C) but is likewise
unwired. The evaluation platform shipped far beyond what doc 15 describes — reasoning evaluation,
LLM-as-judge, judge validation, gold-authoring tooling, an end-to-end pipeline, persistent
experiment tracking, and a 15-endpoint REST API (docs 20–22) — none of which this document's
diagrams reflect.

# Interview Questions

- Why is expensive work concentrated on the write path rather than the read path?
- Why is identity `(source_type, source_external_id)` rather than the row UUID?
- Where would a new non-issue-tracker source (e.g. PagerDuty) stress the canonical shape?
- Trace a query from string to ranked result, naming every external dependency touched.
