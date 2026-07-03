# Enterprise Incident Intelligence Platform

A production-oriented **multi-agent, retrieval-augmented** platform for software incident
investigation: it ingests real incidents from GitHub Issues and Jira, retrieves similar past
incidents via hybrid semantic + lexical search, runs a four-agent (planner → hypothesis generation
→ critic → orchestrator) investigation loop to propose grounded root causes, and — unusually for a
project this size — ships its own rigorous evaluation platform (retrieval metrics, reasoning
accuracy, LLM-as-judge, RAGAS-style grounding metrics, diagnostics/health dashboards) to measure
whether any of that actually works.

This is not a toy RAG demo. It's ~23 phases of incremental, tested development: real ingestion
pipelines, hybrid retrieval with adaptive routing, a genuine multi-agent orchestrator (not a single
prompt), a purpose-built evaluation harness, and a hardening pass (input validation, graceful
degradation, load/performance testing, security review) aimed at production readiness.

## What it does, end to end

```
GitHub Issues ──┐
Jira            ├─▶ Ingestion ─▶ Normalize ─▶ Deduplicate ─▶ Embed ─▶ pgvector
                │      (source-agnostic canonical incident shape)
                ▼
        Semantic + Hybrid (Dense/BM25/RRF) Retrieval, adaptively routed per query
                │
                ▼
   Multi-Agent Investigation:  Planner → Hypothesis Generation → Critic → Orchestrator
                │                (iterative: critic can send it back for more evidence)
                ▼
        Grounded root-cause report (confidence, evidence, rejected hypotheses)

                                    ▲
                                    │  measured by
                                    │
   Evaluation Platform: Retrieval metrics (Recall/MRR/NDCG) · Reasoning accuracy ·
   LLM-as-judge · BERTScore + RAGAS grounding (Faithfulness/Relevancy/Precision/Recall) ·
   evaluator-stability tracking · cost/skip diagnostics · health dashboards · trends
```

## Why this is more than "agentic RAG"

Most "agentic RAG" projects are one LLM call deciding when to retrieve. This platform has:

- **A real multi-agent loop**, not a single prompt — a `PlannerAgent` sets investigation strategy, a
  hypothesis-generation step proposes and evidence-checks root causes, a `CriticAgent` reviews the
  result and can force another iteration, and an orchestrator coordinates all of it with a bounded
  stopping condition. See [`docs/architecture/19`](docs/architecture/19_multi_agent_investigation.md).
- **Adaptive retrieval routing** — a rule-based policy picks Dense, BM25, or Hybrid (Reciprocal Rank
  Fusion) per query based on query shape (stack traces and exact error codes route to BM25; long
  multi-concept queries route to Hybrid), with strategy-aware confidence calibration. See
  [`docs/architecture/18`](docs/architecture/18_adaptive_routing_and_hybrid_confidence.md).
- **A real evaluation platform**, not just "looks good in a demo" — versioned gold datasets with
  corpus fingerprinting, Recall@K/MRR/NDCG, an LLM-as-judge framework with calibration/validation,
  BERTScore + RAGAS-style grounding metrics (Faithfulness, Answer Relevancy, Context Precision/
  Recall/Entity Recall) with configurable cost tiers and evaluator-stability (repeated-run variance)
  tracking, and a diagnostics layer that surfaces outliers, cost, and skip reasons without
  recomputing anything. See [`docs/architecture/15`](docs/architecture/15_evaluation_framework.md),
  [`20`](docs/architecture/20_reasoning_evaluation_and_judges.md),
  [`21`](docs/architecture/21_evaluation_platform_productionization.md).
- **A production-hardening pass** — input validation (UUID/length/format bounds on every endpoint),
  graceful degradation (typed errors instead of raw tracebacks when the DB/LLM/embedding backend is
  down), path-traversal and injection testing, load testing at 10/50/100 concurrent users, and
  performance profiling — see [Production hardening](#production-hardening-phase-23) below.

## Tech stack

FastAPI · SQLAlchemy 2 + Alembic · PostgreSQL + pgvector + `pg_trgm` · SentenceTransformers ·
OpenAI API · Docker / docker-compose · pytest (1,200+ tests) · Python 3.12

## Documentation map

This README is the entry point. The full engineering reference — written for someone joining the
project cold — lives in [`docs/README.md`](docs/README.md) and 22 numbered architecture docs
(`docs/architecture/01`–`22`), covering everything from ingestion and normalization through
retrieval, the multi-agent investigation framework, and the evaluation platform's REST API. Start
there for depth; this file is the map.

## Project structure

```
app/
  api/routes/       FastAPI routers: health, incidents, ingestion, search, agent, evaluation(+interactive)
  api/schemas.py     Pydantic request/response models (validated: length/format/UUID bounds)
  api/validation.py  Shared identifier/UUID validators (Phase 23)
  core/              Settings (pydantic-settings) and logging configuration
  db/                SQLAlchemy models + session management
  ingestion/         Source collectors (GitHub, Jira) and normalization
  services/          Embedding, LLM, retrieval (dense/BM25/hybrid/routed), the 4 investigation agents
  evaluation/        Gold datasets, harnesses, metrics, judges, experiment tracking, diagnostics — 37 modules
alembic/             Database migrations (creates the `vector` and `pg_trgm` extensions)
docs/                Full architecture documentation (22 numbered docs + index)
scripts/             Benchmark/evaluation CLI scripts, load_test.py, profile_performance.py
tests/               1,200+ tests: tests/unit, tests/api, tests/eval
Dockerfile, docker-compose.yml   Multi-stage build, non-root user, healthcheck (Phase 23)
```

## Quick start (Docker)

```bash
docker compose up --build
```

This starts PostgreSQL (with pgvector) and the API, running migrations automatically on
container start. API docs: `http://localhost:8000/docs`. Liveness: `GET /health`. Readiness
(checks DB connectivity): `GET /health/ready`.

Set `OPENAI_API_KEY` (required for the investigation agents and LLM-backed retrieval features) and
optionally `GITHUB_TOKEN` (avoids GitHub API rate limits) as environment variables before starting,
or in a `.env` file at the repo root — see `.env.example`.

## Manual setup (no Docker)

For a from-scratch local environment (Python venv, PostgreSQL via `docker run`, Alembic, running
the API and tests directly) see [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md) — a step-by-step
guide with verification commands for every stage.

## Configuration

All settings are read from environment variables (or a `.env` file) via `app/core/config.py`:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/incidents` | Required |
| `GITHUB_TOKEN` | — | Optional; avoids GitHub API rate limits |
| `OPENAI_API_KEY` | — | Required for `/agent/*` and any LLM-backed evaluation |
| `OPENAI_MODEL` | `gpt-4o-mini` | |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Must match the DB's vector dimension |
| `EMBEDDING_DIMENSIONS` | `384` | Matches the current migration's `VECTOR(384)` column |
| `LOG_LEVEL` | `INFO` | |
| `SEARCH_ROUTING_ENABLED` | `false` | Opt-in for adaptive Dense/BM25/Hybrid routing; `false` preserves dense-only behavior |

## API surface

27 endpoints across 6 routers (plus FastAPI's auto-generated `/docs`, `/redoc`, `/openapi.json`).

**Health** — `GET /health` (liveness, unconditional), `GET /health/ready` (readiness, checks DB —
returns 503 when unreachable)

**Incidents** — `GET /incidents` (paginated list), `GET /incidents/{incident_id}` (UUID-validated)

**Ingestion** — `POST /ingestion/github`, `POST /ingestion/jira` (both return 502 on upstream
failure rather than crashing)

**Search** — `POST /search/incidents`, `POST /search/debug` — adaptively routed Dense/BM25/Hybrid
retrieval with confidence scoring

**Agent** — `POST /agent/investigate` (single-shot), `POST /agent/investigate-advanced`
(structured report), `POST /agent/investigate-orchestrated` (**canonical** — the full
planner/hypothesis/critic/orchestrator loop)

**Evaluation** (16 endpoints) — `POST /evaluation/query|retrieval|reasoning|full`,
`GET /evaluation/runs`, `/runs/latest`, `/runs/{run_id}`, `/runs/{run_id}/failed-queries`,
`/failed-reasoning`, `/judge-disagreements`, `/diagnostics`, `/stats`, plus a human-friendly
interactive flow: `POST /evaluation/query/preview`, `/by-title`, `/{session_id}/evaluate`,
`GET /evaluation/query/{session_id}` — full detail in
[`docs/architecture/22`](docs/architecture/22_evaluation_api.md).

## Running tests

```bash
python -m pytest              # full suite — 1,200+ tests
python -m pytest tests/unit    # unit tests only (no HTTP layer)
python -m pytest tests/api     # FastAPI route tests (TestClient, no real DB/LLM)
python -m ruff check .         # lint
```

Every test runs against fakes/mocks for the database, LLM, and embedding backends — no live
Postgres or OpenAI credentials are needed to run the suite.

## Load testing and performance profiling

Added in the production-hardening pass (Phase 23):

```bash
python scripts/load_test.py         # 10/50/100 concurrent users; latency/throughput/error rate
python scripts/profile_performance.py  # retrieval, routing, generation, evaluation, orchestration
```

Both scripts run in-process against the real FastAPI app (via `httpx.ASGITransport`, no server
process needed) and write JSON reports to `.benchmarks/phase23_load_test/` and
`.benchmarks/phase23_performance/`. They measure real platform overhead (routing, validation,
serialization) — see each script's own docstring for what they do and don't cover without a live
database/LLM backend.

## Production hardening (Phase 23)

A dedicated validation-and-hardening pass, on top of the feature-complete platform:

- **Input validation** — every endpoint bounds string length, list size, and format (UUIDs
  validated before hitting the database; malformed input returns 422, not a silent 404 or a crash).
- **Graceful degradation** — a platform-wide exception handler converts unhandled failures (DB
  connection drop, LLM timeout, malformed LLM JSON response) into clean typed errors instead of
  leaking stack traces or internal details (connection strings, API key fragments) to the client.
  Typed errors: `LLMResponseError`, `EmbeddingServiceError`; DB failures → 503; upstream
  ingestion failures → 502.
- **Security testing** — SQL-injection-shaped and prompt-injection-shaped input verified to pass
  through as inert text (no server-side string concatenation); path-traversal attempts on
  filesystem-backed identifiers (`run_id`) rejected both at the API layer and via defense-in-depth
  containment checks in the repository layer itself.
- **Resilience** — corrupted evaluation-run files on disk are skipped (logged, not crashed);
  DB/LLM/embedding client timeouts configured (previously unbounded, could hang indefinitely).
- **Deployment readiness** — `Dockerfile` (multi-stage, non-root user, healthcheck) and
  `docker-compose.yml` added (the README referenced them before they existed); a `/health/ready`
  probe and app lifespan (startup DB check, clean shutdown) added.

1,200+ tests cover empty/malformed/oversized/unicode input, invalid UUIDs, missing resources,
simulated DB/LLM/embedding/upstream failures, and corrupted evaluation data.

## Corpus (at time of writing)

~8,000 incidents: ~6,500 from GitHub (16 repositories) + ~1,500 from Jira (KAFKA, SPARK, CASSANDRA).

## Current limitations

See [`docs/architecture/16`](docs/architecture/16_current_limitations.md) for the full list and
[`docs/architecture/17`](docs/architecture/17_future_roadmap.md) for the prioritized roadmap. Notable
ones: BM25/Hybrid retrieval requires a process restart to see newly-ingested incidents (the index is
built once and cached, not incrementally updated); the evaluation platform is reachable via its REST
API and CLI scripts but not wired into automatic CI; no rate limiting on any endpoint yet.
