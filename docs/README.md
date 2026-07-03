# Incident Retrieval & Investigation System — Engineering Documentation

This is the internal engineering reference for the Incident Intelligence platform. Docs 01–17
describe the system through **Retrieval v1** (dense-only retrieval, a single-shot investigation
agent, and the retrieval-only evaluation platform); docs 18–22 cover everything built since —
adaptive routing/hybrid retrieval, a four-agent investigation framework, reasoning evaluation and
LLM-as-judge, evaluation-platform productionization, and a REST API; doc 23 covers production
hardening, API surface consolidation, authentication, and rate limiting — the layer that now sits
in front of everything docs 18–22 describe. It is written for engineers joining the team who need
to understand the whole system without reading the source first.

## What this system does

It ingests software incidents from multiple sources (GitHub Issues, Apache Jira),
normalizes them into a single canonical shape, embeds them into a vector space, and
serves **semantic incident retrieval** — "find past incidents similar to this
problem" — with optional LLM query expansion and reranking, plus a confidence
signal on every result. A downstream investigation agent consumes retrieval to
generate and ground root-cause hypotheses.

## Corpus at time of writing

~8,000 incidents: ~6,500 GitHub (16 repositories) + ~1,500 Jira
(KAFKA, SPARK, CASSANDRA).

## How to read these docs

Read top to bottom for a full mental model; jump to a numbered file to go deep on a
component.

| Doc | Component |
|---|---|
| [01](architecture/01_system_overview.md) | System overview & data lifecycle |
| [02](architecture/02_ingestion_pipeline.md) | Ingestion pipeline & registry dispatch |
| [03](architecture/03_collectors.md) | Source collectors (GitHub, Jira) |
| [04](architecture/04_normalization.md) | Normalization → `NormalizedIncident` |
| [05](architecture/05_deduplication.md) | Deduplication & hashing |
| [06](architecture/06_watermarking.md) | Incremental ingestion watermarks |
| [07](architecture/07_sanitization.md) | Payload sanitization |
| [08](architecture/08_embedding_pipeline.md) | Embedding generation |
| [09](architecture/09_vector_storage.md) | pgvector / HNSW storage |
| [10](architecture/10_retrieval_pipeline.md) | Retrieval orchestration |
| [11](architecture/11_query_expansion.md) | LLM query expansion |
| [12](architecture/12_candidate_generation.md) | Candidate merge |
| [13](architecture/13_llm_reranking.md) | LLM reranking |
| [14](architecture/14_confidence_calibration.md) | Confidence calibration |
| [15](architecture/15_evaluation_framework.md) | Evaluation framework |
| [16](architecture/16_current_limitations.md) | Current limitations |
| [17](architecture/17_future_roadmap.md) | Future roadmap (items 1–2 shipped — see docs 18–22) |
| [18](architecture/18_adaptive_routing_and_hybrid_confidence.md) | BM25, hybrid (RRF) retrieval, adaptive routing, strategy-aware confidence normalization |
| [19](architecture/19_multi_agent_investigation.md) | Multi-agent investigation framework (planner, hypothesis generation, critic, iterative orchestrator) |
| [20](architecture/20_reasoning_evaluation_and_judges.md) | Reasoning evaluation harness + LLM-as-judge framework |
| [21](architecture/21_evaluation_platform_productionization.md) | AI quality intelligence, judge validation, gold dataset authoring/labeling, end-to-end pipeline, experiment tracking |
| [22](architecture/22_evaluation_api.md) | Evaluation REST API (machine-facing + human-friendly interactive workflow) |
| [23](architecture/23_production_hardening_and_api_security.md) | Production hardening, API surface consolidation (27→21 endpoints), Bearer API-key authentication, endpoint-aware rate limiting |

## Core design principles (read these first)

1. **One canonical incident shape.** Every source is adapted into `NormalizedIncident`;
   everything downstream (dedup, embedding, retrieval, search) is source-agnostic.
2. **Stable identity, volatile rows.** An incident's identity is
   `(source_type, source_external_id)` — not its UUID. Re-ingestion may regenerate
   rows; identity is preserved via the deduplication key.
3. **The persistence boundary owns correctness concerns.** Sanitization and hashing
   happen at the database boundary, not in collectors or normalizers, which stay pure.
4. **Idempotent, resumable ingestion.** Watermarks + content hashing make re-runs safe
   and incremental.
5. **Retrieval quality is measured, not asserted.** Every change is judged against a
   versioned gold set with corpus fingerprinting (see doc 15).

## Status: what's built vs. what's wired

**Updated.** Doc 18's adaptive routing (Dense/BM25/Hybrid) is now the production retrieval engine
behind `app/api/routes/search.py`'s `/search/incidents` and `/search/debug`, and behind the
investigation orchestrator's default construction (`app/services/investigation_orchestrator.py`,
Phase 19D) — see doc 18's "Phase 18E — Production Adoption" section. It ships with
`routing_enabled=False` by default (`Settings.search_routing_enabled`, env var
`SEARCH_ROUTING_ENABLED`), so out-of-the-box behavior is unchanged from dense-only retrieval until
an operator opts in. Doc 19's three narrower single-purpose agents
(`HypothesisDrivenInvestigationAgent`/`PlannedInvestigationAgent`/`CriticReviewedInvestigationAgent`)
remain unwired — only the full Phase 19D orchestrator (reachable via `POST /agent/investigate`, the
single canonical investigation route since Phase 23A's API surface consolidation — see doc 19's
"Integration status") was adopted. Docs 20–21's evaluation platform is still only reachable through
the REST API in doc 22 or the CLI scripts in `scripts/`, not through automatic CI, and
`/evaluation/*`'s own orchestrator construction (`_build_orchestrator`) still deliberately pins a
plain dense `IncidentSearchService` for reproducible benchmarking, unaffected by the routed default.
Treat "documented" and "in production" as separate questions when reading docs 20–22.

**Phase 23/23A/23B/23C (production hardening, API consolidation, auth, rate limiting) — see doc
23, the newest doc in this series.** Input validation, graceful degradation, and a
security/load-testing pass were applied platform-wide (23); the public API surface was reduced
from 27 to 21 endpoints by removing duplicate/legacy routes (three investigation endpoints down to
one; four per-run filtered-view endpoints folded into `GET /evaluation/runs/{run_id}` — see doc
19's "Integration status" and doc 22's `RunDetailResponse` section) (23A); every business endpoint
now requires `Authorization: Bearer <API_KEY>` (23B); every business endpoint now enforces a
per-minute, per-caller rate limit sized to its cost, from 2/min (`/evaluation/full`) up to 100/min
(`/search`, `/incidents`) (23C). `/health` and `/health/ready` remain the only always-open,
always-unlimited routes.
