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

Set `OPENAI_API_KEY` (required for the investigation agents and LLM-backed retrieval features),
`API_KEY` (required — see [Authentication](#authentication) below), and optionally `GITHUB_TOKEN`
(avoids GitHub API rate limits) as environment variables before starting, or in a `.env` file at
the repo root — see `.env.example`.

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
| `API_KEY` | — | **Required.** Shared secret for Bearer auth (Phase 23B) — see [Authentication](#authentication) |
| `RATE_LIMIT_ENABLED` | `true` | Global kill switch for rate limiting (Phase 23C) — see [Rate limiting](#rate-limiting) |
| `RATE_LIMIT_SEARCH_PER_MINUTE` | `100` | `/search/incidents`, `/search/debug` |
| `RATE_LIMIT_AGENT_PER_MINUTE` | `20` | `/agent/investigate` |
| `RATE_LIMIT_EVALUATION_QUERY_PER_MINUTE` | `20` | `POST /evaluation/query` |
| `RATE_LIMIT_EVALUATION_RETRIEVAL_PER_MINUTE` | `5` | `POST /evaluation/retrieval` |
| `RATE_LIMIT_EVALUATION_REASONING_PER_MINUTE` | `5` | `POST /evaluation/reasoning` |
| `RATE_LIMIT_EVALUATION_FULL_PER_MINUTE` | `2` | `POST /evaluation/full` (the most expensive single call in the API) |
| `RATE_LIMIT_INTERACTIVE_EVALUATION_PER_MINUTE` | `20` | `preview`/`by-title`/`{session_id}` routes |
| `RATE_LIMIT_INCIDENTS_PER_MINUTE` | `100` | Not in the original spec's suggested defaults — added so this router isn't left unlimited |
| `RATE_LIMIT_INGESTION_PER_MINUTE` | `10` | Not in the original spec's suggested defaults — ingestion triggers external HTTP calls, the platform's most abuse-prone surface |
| `RATE_LIMIT_EVALUATION_RUNS_PER_MINUTE` | `60` | Not in the original spec's suggested defaults — covers the read-only `GET /evaluation/runs*`, `/stats` views |

## Authentication

Every business endpoint (`/incidents`, `/ingestion`, `/search`, `/agent`, `/evaluation`) requires a
Bearer API key. `/health`, `/health/ready`, and the docs routes (`/docs`, `/redoc`,
`/openapi.json`) are intentionally left open — they're liveness/readiness probes and API
documentation, not business data.

This is deliberately **not** a user-management system — no accounts, passwords, sessions, JWTs,
OAuth, refresh tokens, roles, or a login endpoint. The platform is meant to run as an internal
service, behind an API gateway or inside a trusted network, so a single shared secret compared on
every request is the right amount of mechanism for that deployment model — see
`app/api/auth.py`'s module docstring for the fuller rationale.

**Making a request:**

```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/incidents
```

Every failure mode — missing header, malformed header, wrong key — returns the same `401` with the
same generic `{"detail": "Not authenticated."}` body and a `WWW-Authenticate: Bearer` header, so a
response never reveals which part of the request was wrong.

**Using Swagger:** open `http://localhost:8000/docs`, click the **Authorize** button (top right),
paste the raw key (no `Bearer` prefix — the UI adds it), and click **Authorize**. Every subsequent
"Try it out" request against a protected endpoint then carries the header automatically; public
endpoints show no lock icon and need no authorization.

**Configuration:** set `API_KEY` in `.env` (see `.env.example`) or as an environment variable — for
example, generate one with `openssl rand -hex 32`. There is no default: if `API_KEY` is unset, every
protected request is rejected (fail-closed), never silently allowed through.

The dependency (`app/api/auth.py`'s `require_api_key`) is centralized and applied once per
router — via each `APIRouter(dependencies=[Depends(require_api_key)])` — not repeated inside
individual endpoints, so a new route under a protected router is authenticated automatically with
no extra code.

## Rate limiting

Every business endpoint has a per-minute request limit, sized to the endpoint's actual cost —
cheap reads (`/incidents`, `/search`) allow far more traffic than expensive LLM-backed calls
(`/evaluation/full`, which can run retrieval + reasoning + judging + generation in one request,
allows only 2/minute). `/health` and `/health/ready` are always unlimited. Limits are per **caller
identity** (the presented Bearer token, or client IP as a fallback) and per **endpoint group** —
exhausting one endpoint's quota never affects another's, and different callers never share a quota.

**On exceeding a limit:**

```json
HTTP/1.1 429 Too Many Requests
Retry-After: 37
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1735689660

{"detail": "Rate limit exceeded for 'agent': 20 requests per 60 seconds. Retry after 37 seconds."}
```

Every response from a rate-limited endpoint — successful or not — carries `X-RateLimit-Limit`,
`X-RateLimit-Remaining`, and `X-RateLimit-Reset` so a well-behaved client can back off before
hitting the limit, not just after.

**Implementation:** a fixed 60-second window, counted in-process (no Redis — the platform runs
single-process today; see `app/api/rate_limit.py`'s module docstring for why this is a deliberate,
documented tradeoff). The counting logic sits behind a `RateLimitBackend` abstraction so a
distributed backend could replace the in-memory one later without touching any route. Like auth,
it's wired once per router/route via `dependencies=[Depends(<group>_rate_limit)]` — never
duplicated inside an endpoint body.

**Configuration:** every limit is a `Settings` field (see the Configuration table above), overridable
via environment variable, e.g. `RATE_LIMIT_SEARCH_PER_MINUTE=200`. Set `RATE_LIMIT_ENABLED=false` to
disable rate limiting entirely (health remains unaffected either way).

**In Swagger:** every protected route documents a `429` response (see the **Responses** section of
each endpoint at `/docs`) alongside its usual success responses.

## API surface

21 endpoints across 6 routers (plus FastAPI's auto-generated `/docs`, `/redoc`, `/openapi.json`).
Reduced from 27 in Phase 23A by removing duplicate/legacy routes — one canonical endpoint per
business capability; see [Phase 23A: API surface consolidation](#phase-23a-api-surface-consolidation)
below.

**Health** — `GET /health` (liveness, unconditional), `GET /health/ready` (readiness, checks DB —
returns 503 when unreachable)

**Incidents** — `GET /incidents` (paginated list), `GET /incidents/{incident_id}` (UUID-validated)

**Ingestion** — `POST /ingestion/github`, `POST /ingestion/jira` (both return 502 on upstream
failure rather than crashing)

**Search** — `POST /search/incidents`, `POST /search/debug` — two genuinely distinct capabilities,
not duplicates: `/incidents` is plain filtered retrieval (arbitrary limit, full incident objects,
no LLM); `/debug` is the LLM-expanded + LLM-reranked variant (fixed at 5 results, lightweight
response shape, echoes the resolved filters) — currently the *only* way to invoke query expansion
and reranking over the API, despite the "debug" name.

**Agent** — `POST /agent/investigate` — the single canonical investigation endpoint, backed by the
full planner/hypothesis/critic/orchestrator loop. (Phase 23A retired `/investigate-advanced` and
the separate `/investigate-orchestrated` path — three routes for one business capability at three
generations of sophistication became one.)

**Evaluation** (12 endpoints) — `POST /evaluation/query|retrieval|reasoning|full`,
`GET /evaluation/runs`, `/runs/latest`, `/runs/{run_id}` (now includes failed-query/
failed-reasoning/judge-disagreement views, plus an opt-in `?include_diagnostics=true` diagnostics
dashboard — Phase 23A folded four separate GET routes into this one), `/stats`, plus a
human-friendly interactive flow: `POST /evaluation/query/preview`, `/by-title`,
`/{session_id}/evaluate`, `GET /evaluation/query/{session_id}` — full detail in
[`docs/architecture/22`](docs/architecture/22_evaluation_api.md).

## Phase 23A: API surface consolidation

A dedicated refactoring pass, on top of the feature-complete, production-hardened platform: same
capabilities, fewer routes to secure/document/maintain. Reviewed every endpoint group; consolidated
only genuine duplication.

- **Agent (3 → 1):** `/investigate`, `/investigate-advanced`, `/investigate-orchestrated` were three
  successive implementations of "investigate this problem and report a root cause," not three
  capabilities — the orchestrated one was already documented as canonical. Removed the other two
  routes; `POST /agent/investigate` now serves the orchestrator directly. The underlying single-shot
  agent classes (`InvestigationAgent`, `AdvancedInvestigationAgent`) are untouched and still
  independently unit-tested — only their HTTP routes were retired.
- **Evaluation run views (4 → 0 extra routes):** `/runs/{id}/failed-queries`, `/failed-reasoning`,
  `/judge-disagreements`, `/diagnostics` each did nothing but load the same run and return one
  filtered view of it. Folded into `GET /runs/{run_id}` — the three failure/disagreement lists are
  free (precomputed at save time, always included); diagnostics is real computation, so it stays
  opt-in via `?include_diagnostics=true`.
- **Search — reviewed, kept both.** `/search/incidents` and `/search/debug` looked redundant by
  name but aren't: different retrieval behavior (plain vs. LLM-expanded+reranked), different
  response shapes. Consolidating would have meant either losing a capability or growing
  `/search/incidents`'s contract — out of scope for a routes-only refactor.
- **Evaluation benchmarks and the interactive query workflow — reviewed, kept as-is.**
  `/query`, `/retrieval`, `/reasoning`, `/full` have different response shapes and scopes (not
  provably interchangeable without touching evaluation logic, which this phase didn't). The
  interactive `preview` → `evaluate` flow and the one-shot `by-title` endpoint serve two different
  callers — a human reviewing results before committing vs. a caller that already knows the answer
  — not one workflow expressed three ways.

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
