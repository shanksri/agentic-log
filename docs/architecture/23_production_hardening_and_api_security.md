# 23 — Production Hardening, API Consolidation, Authentication & Rate Limiting (Phases 23, 23A, 23B, 23C)

This document covers four back-to-back phases that took the platform from "feature complete" to
"safe to expose": **Phase 23** (validation & hardening across every subsystem), **23A** (API
surface consolidation — one canonical endpoint per business capability), **23B** (Bearer API-key
authentication), and **23C** (endpoint-aware rate limiting). None of the four changed retrieval,
routing, investigation, or evaluation logic — all four are explicitly scoped to the API/service
boundary. Docs 18–22 remain the authority on *what* the platform computes; this doc is the
authority on *how a request reaches it safely*.

---

## Phase 23 — Production Validation & Hardening

### Goal

Review every subsystem (ingestion, retrieval, routing, investigation, evaluation, experiment
tracking, diagnostics, API, persistence, configuration) for production readiness and fix what
could be fixed without redesigning anything, without adding features, and without changing any
algorithm.

### What changed

- **Input validation** (`app/api/validation.py`, new): shared `validate_uuid`/`validate_safe_identifier`
  helpers. Every route now bounds string length, list size, and format — a malformed `incident_id`
  returns `422` instead of a silent `404`; oversized payloads (multi-KB strings, 1000+ item lists)
  are rejected before touching business logic.
- **Graceful degradation**: a platform-wide exception handler (`app/main.py`) converts any
  unhandled exception into a clean `500` (`{"detail": "An internal error occurred."}`) and any
  `sqlalchemy.exc.OperationalError` into a `503`, logging the real exception server-side only —
  no stack trace, connection string, or API key fragment ever reaches the client. `LLMService` and
  `EmbeddingService` gained typed exceptions (`LLMResponseError`, `EmbeddingServiceError`) and a
  30-second client timeout (previously unbounded). `app/api/routes/agent.py` and `ingestion.py`
  wrap previously-unguarded service calls (`503` for missing config, `502` for upstream GitHub/Jira
  failure).
- **Security testing**: SQL-injection- and prompt-injection-shaped input verified to pass through
  as inert text (no server-side string concatenation); path-traversal on filesystem-backed
  identifiers (`run_id`) rejected both at the API layer and via a defense-in-depth containment
  check added to `run_repository.py`/`experiment_tracking.py` (a `run_id` that would resolve
  outside the storage directory is treated as not-found, never read).
- **Load testing** (`scripts/load_test.py`, new): drives the real ASGI app in-process via
  `httpx.ASGITransport` at 10/50/100 concurrent workers, reporting latency/throughput/error-rate
  per endpoint. Explicitly scoped: measures FastAPI/Pydantic/handler overhead against in-process
  fakes, not real database or OpenAI round-trip latency (no live credentials in this environment).
- **Performance profiling** (`scripts/profile_performance.py`, new): profiles the pure-computation
  paths that don't need live infrastructure — BM25 indexing/scoring, routing decisions, BERTScore
  matching, Recall/MRR/NDCG computation, orchestration control flow with a fake instant LLM. All
  sub-millisecond to low-single-digit-ms, confirming none of it is a bottleneck; real production
  latency is dominated by DB/LLM I/O this environment can't measure.
- **Deployment readiness**: `Dockerfile` and `docker-compose.yml` added (the README referenced
  `docker compose up --build` before either file existed). `GET /health/ready` added (DB
  connectivity check, `503` when unreachable) alongside the unconditional `GET /health`
  (liveness). A FastAPI `lifespan` handler added — best-effort DB probe at startup (never blocks
  boot), clean connection-pool disposal at shutdown.

### Testing

1,065 → 1,238 tests over the course of this phase. New files: `tests/unit/test_llm_service.py`,
`tests/unit/test_run_repository_hardening.py`, `tests/api/test_incidents_api.py`,
`tests/api/test_health_api.py`, `tests/api/test_production_hardening.py` (27 cross-cutting tests:
oversized/malformed/unicode input, DB/LLM/upstream failures, no-exception-leak assertions).

### Remaining risks (as of Phase 23)

No authentication, no rate limiting, filesystem-only experiment tracking (incompatible with
horizontal scaling), no observability stack (metrics/tracing/correlation IDs), unpinned
dependencies, single uvicorn worker. **The first two were resolved by Phases 23B and 23C below.**
The rest remain open — see the Phase 23 production-readiness review for the full breakdown (not
captured as a standalone doc; delivered as a structured report at the time).

---

## Phase 23A — API Surface Consolidation

### Goal

Reduce API surface area — one canonical endpoint per business capability — without losing any
capability, without redesigning architecture, without touching retrieval/investigation/evaluation
logic.

### What changed

**Agent: 3 routes → 1.** `/investigate` (single-shot `InvestigationAgent`), `/investigate-advanced`
(single-shot `AdvancedInvestigationAgent`), and `/investigate-orchestrated`
(`MultiAgentInvestigationOrchestrator`, already documented as canonical — see doc 19) were three
generations of the same capability. The first two were retired; `POST /agent/investigate` now
serves the orchestrator directly. `InvestigationAgent`/`AdvancedInvestigationAgent` are untouched
in `app/services/` and still independently unit-tested — only their HTTP routes were removed. See
doc 19's "Integration status" for the full before/after.

**Evaluation run views: 4 routes → 0 extra routes.** `/runs/{id}/failed-queries`,
`/failed-reasoning`, `/judge-disagreements`, `/diagnostics` each loaded the same run and returned
one filtered view of data already available. Folded into `GET /evaluation/runs/{run_id}`: the
three failure/disagreement lists are always present (`failed_queries`, `failed_reasoning`,
`judge_disagreements` — precomputed at save time, free to include); `diagnostics` (Phase 22C's
`build_health_report`) is real computation, so it's opt-in via `?include_diagnostics=true`. See
doc 22's `RunDetailResponse` section.

**Reviewed, kept as-is:** `/search/incidents` vs. `/search/debug` (genuinely different retrieval
behavior — `/debug` is the only way to invoke LLM query expansion + reranking over the API, despite
the name); `/evaluation/query|retrieval|reasoning|full` (different response shapes and scopes,
collapsing them would have required touching `EvaluationPipeline`); the interactive evaluation
workflow (`preview`/`evaluate`/`by-title` serve two different callers — a human reviewing results
before committing vs. a caller that already knows the answer, not one workflow expressed three
ways).

### Result

27 → 21 API endpoints. Full before/after table, endpoint-by-endpoint justification, and the
"kept, not removed" reasoning were delivered as the Phase 23A report at the time this phase shipped.

---

## Phase 23B — Bearer API-Key Authentication

### Goal

Secure the API with lightweight authentication appropriate for an internal enterprise platform —
explicitly **not** a user-management system (no accounts, passwords, sessions, JWTs, OAuth,
refresh tokens, roles, database-backed auth, or a login endpoint).

### Architecture

```
Authorization: Bearer <API_KEY>
        │
        ▼
require_api_key()  ← app/api/auth.py, wraps fastapi.security.HTTPBearer(auto_error=False)
        │
        ├─ credentials is None (missing / malformed / wrong scheme / empty token)  → 401
        ├─ settings.api_key is unset                                              → 401 (fail-closed)
        ├─ secrets.compare_digest(token, settings.api_key) is False               → 401
        └─ match                                                                   → request proceeds
```

`auto_error=False` is required, not cosmetic: `HTTPBearer`'s default raises `403` for a
missing/malformed header, not `401`. Every failure mode — missing header, malformed header, wrong
key — returns the identical `401` + `{"detail": "Not authenticated."}` + `WWW-Authenticate: Bearer`,
so a response never reveals which part of the request was wrong (no oracle for guessing towards a
valid key).

Wired once per router — `APIRouter(dependencies=[Depends(require_api_key)])` — on `incidents`,
`ingestion`, `search`, `agent`, `evaluation`, `evaluation_interactive`. `health.py` has no such
dependency, so `/health` and `/health/ready` stay open. `/docs`, `/redoc`, `/openapi.json` are
plain Starlette routes outside FastAPI's dependant tree entirely, never reachable by router-level
dependencies regardless.

`HTTPBearer` is a `fastapi.security.SecurityBase` instance, so FastAPI auto-registers it in the
OpenAPI schema's `components.securitySchemes` even though it's nested inside `require_api_key`
rather than used directly as a route parameter — this is what makes Swagger's single **Authorize**
button work: paste the key once, every subsequent "Try it out" call on a locked (🔒) endpoint
carries the header automatically.

### Configuration

`Settings.api_key: str | None = None` (`API_KEY` env var). No default — an unset key means every
protected request is rejected (fail-closed), never silently allowed through.

### Testing

`tests/api/test_authentication.py` (21 tests) — missing/malformed/invalid key all return `401`
with the identical message; an unconfigured key fails closed; a valid key reaches real business
logic on a representative endpoint from every one of the six protected routers (proving the
dependency actually executes on each, not just one); all five public routes reachable with zero
headers; OpenAPI security-scheme registration and per-route `security` requirements verified
directly against the generated schema. Every other existing API test file's `_client()` helper
gained one line — `app.dependency_overrides[require_api_key] = lambda: None` — so tests exercising
business logic don't also need to fabricate a valid key; this is the same dependency-override
pattern already used for `get_db` throughout the test suite.

### Remaining risks

A single shared secret has no per-caller identity or revocation granularity — if it leaks, every
caller rotates together. No rate limiting was in place at the time this phase shipped (closed by
23C, below). No expiry/rotation mechanism. `/health/ready` stays unauthenticated and reveals DB
reachability (low sensitivity, intentional).

---

## Phase 23C — Endpoint-Aware Rate Limiting

### Goal

Protect the API from accidental or malicious abuse while preserving usability, with limits sized
to each endpoint's actual cost — without adding authentication, without redesigning routing,
without touching business logic.

### Architecture

```
Request → require_api_key (Phase 23B, unchanged)
        → <group>_rate_limit dependency
              │
              ▼
        settings.rate_limit_enabled?  ──false──▶ pass through, no headers
              │ true
              ▼
        limit = settings.rate_limit_<group>_per_minute
              │
              ▼
        identity = Bearer token if present, else client IP
        RateLimitBackend.check(f"{group}:{identity}", limit, window=60s)
              │
        ┌─────┴─────┐
     allowed      exceeded
        │             │
   X-RateLimit-*   429 + Retry-After + X-RateLimit-* + clear message
   headers set,
   request proceeds
```

`RateLimitBackend` (`app/api/rate_limit.py`, ABC) has one implementation —
`InMemoryRateLimitBackend`, a fixed-window counter behind a `threading.Lock` (FastAPI's sync route
handlers run in a bounded threadpool, so concurrent requests genuinely race on the same counter).
The abstraction exists so a distributed backend (Redis `INCR`+`EXPIRE`, etc.) could replace it
later behind the same `check()`/`reset()` interface with zero route changes — required by the
phase spec, satisfied literally rather than only in spirit.

**Identity** is resolved from the raw `Authorization` header directly (`f"key:{token}"`), not by
re-validating against `Settings.api_key` — every rate-limited router already requires
authentication (23B), so re-deriving "the API key" here would just duplicate that check. Falls
back to `f"ip:{request.client.host}"` if no Bearer token is present. Under the *current* wiring
every rate-limited request is already authenticated, so the IP-fallback branch is not reachable
over HTTP — it exists (and is directly unit-tested) so the dependency stays generically reusable on
a future endpoint that doesn't require auth.

Wired exactly like 23B: one named dependency per endpoint-cost group, router-level for routers
with a single cost tier, route-level for `evaluation.py` (whose four POST routes each need a
different limit within the same router).

### Limits

| Group | Default | Spec origin |
|---|---|---|
| Health | unlimited | Spec |
| Search | 100/min | Spec |
| Agent Investigation | 20/min | Spec |
| Evaluation Query | 20/min | Spec |
| Evaluation Retrieval | 5/min | Spec |
| Evaluation Reasoning | 5/min | Spec |
| Evaluation Full | 2/min | Spec |
| Interactive Evaluation | 20/min | Spec |
| Incidents | 100/min | **Added** — not in the spec's suggested list; left unlimited would contradict the phase's own objective |
| Ingestion | 10/min | **Added** — triggers external HTTP calls, the platform's most abuse-prone surface (see Phase 23's SSRF/cost-exhaustion findings) |
| Evaluation runs/stats (GET) | 60/min | **Added** — covers the read-only views the spec didn't name |

All ten are `Settings` fields (`RATE_LIMIT_*_PER_MINUTE` env vars), plus a global
`RATE_LIMIT_ENABLED` kill switch.

### Behavior

Every response from a rate-limited endpoint — success or `429` — carries `X-RateLimit-Limit`,
`X-RateLimit-Remaining`, `X-RateLimit-Reset`. A `429` additionally carries `Retry-After` and a
message naming the group, the limit, and the window (e.g. `"Rate limit exceeded for 'agent': 20
requests per 60 seconds. Retry after 37 seconds."`) — deliberately more informative than 23B's
generic auth-failure message, since transparency about limits is good UX, not a security leak.

### Testing

`tests/api/test_rate_limiting.py` (14 tests) — below/at/above limit; independent quotas across
endpoint groups (including two routes on the *same* router); independent quotas for two different
Bearer tokens; the IP-fallback identity path (unit-tested directly); `/health` staying unlimited
even when every other group is set to 1; window reset (and, importantly, *not* resetting
prematurely — this caught a real test bug where a wall-clock-derived start time landed near a
window boundary); the global kill switch; direct unit tests of the swappable in-memory backend.

`tests/api/conftest.py` (new) — an autouse fixture resetting the shared in-memory backend before
every test in `tests/api/`. Necessary because the backend is a process-local singleton shared by
every `TestClient` request across the whole test session: without a reset, a low-limit endpoint
(`/evaluation/full` at 2/min) would accumulate hits across dozens of unrelated test functions
within one test run's 60-second window and fail them with spurious `429`s — this is exactly what
happened the first time the existing suite ran against the new limiter, before this fixture existed.

### Remaining risks

A single shared API key (23B) limits how much "per-key" rate limiting actually discriminates
between callers today — solving it means multiple named keys, out of scope for both phases.
Fixed-window counting allows short bursts up to ~2× the configured limit at a window boundary
(accepted tradeoff; a sliding-window or token-bucket algorithm would close this at real complexity
cost). In-memory state resets on every process restart and — if ever scaled beyond one process —
is enforced independently per replica (`limit × replica_count`), explicitly accepted for the
current single-process deployment; the backend abstraction exists specifically so this becomes a
config swap, not a rewrite, if it matters later. No visibility/alerting on rate-limit rejections in
production.

---

## Cumulative effect on the API surface

| | Before Phase 23 | After Phase 23C |
|---|---|---|
| Endpoint count | 27 | 21 |
| Input validation | Partial, inconsistent | Every route bounds length/size/format |
| Unhandled-exception behavior | Raw tracebacks/leaked details possible | Platform-wide handler, no leaks, typed errors |
| Authentication | None | Bearer API key, centralized, fail-closed |
| Rate limiting | None | Per-group, per-caller, centralized, configurable |
| Deployment artifacts | README referenced Docker files that didn't exist | `Dockerfile` + `docker-compose.yml`, `/health/ready`, lifespan hooks |
| Test count | 1,065 | 1,272 |

Docs 18–22 describe what the platform computes and how well; this doc describes the layer that
now sits in front of all of it. A request to any business endpoint passes through, in order:
route-level input validation → `require_api_key` (23B) → the endpoint's rate-limit dependency
(23C) → business logic → the platform-wide exception handler (23) if anything still goes wrong.
