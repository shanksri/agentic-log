# 22 — Evaluation API (Phases 21G–21H)

This document covers the two phases that expose the evaluation platform (docs 15, 20, 21) over
HTTP: **Phase 21G** (`app/api/routes/evaluation.py`, the machine-facing REST surface) and
**Phase 21H** (`app/api/routes/evaluation_interactive.py`, a human-facing session workflow built
on top of it without modifying it). Both routers mount at prefix `/evaluation` and share the
`evaluation` Swagger tag, so they appear as one grouped API surface even though they are two
independently-shipped phases. Wiring is confirmed in `app/main.py`:

```python
from app.api.routes import agent, evaluation, evaluation_interactive, health, incidents, ingestion, search
...
app.include_router(evaluation.router)
app.include_router(evaluation_interactive.router)
```

No architecture doc in this series documents any REST endpoint prior to this one — `/search`,
`/incidents`, `/ingestion`, `/agent` remain undocumented at the API level. This doc's scope is
strictly 21G/21H; the pre-existing gap is noted in doc 17.

---

## Phase 21G — Evaluation REST API

### Goal

Expose the evaluation framework built across Phases 16–21F — retrieval evaluation, reasoning
evaluation, judges, the full pipeline, and persisted experiment history — as a set of FastAPI
endpoints, so evaluation runs can be triggered and inspected without a local Python shell or the
CLI scripts (`scripts/run_full_evaluation.py`, `scripts/inspect_evaluation_run.py`).

### Motivation

Every phase through 21F produced a Python library, not a service. Running an evaluation meant
importing `app.evaluation.*` directly. The module docstring is explicit about the boundary this
phase must not cross: *"This module introduces NO new evaluation logic — every endpoint
delegates to already-existing public APIs and returns their results in Pydantic-typed response
envelopes."* The problem being solved is purely one of access — turning library calls into HTTP
calls — not evaluation correctness, which docs 15/20/21 already own.

### Architecture

Router: `APIRouter(prefix="/evaluation", tags=["evaluation"])`. Persistence dependency:

```python
_DEFAULT_RUNS_DIR = Path(".evaluation_runs")

def _get_repo() -> ExperimentRepository:
    return ExperimentRepository(base_dir=_DEFAULT_RUNS_DIR)

ExperimentRepo = Depends(_get_repo)
```

**Request models** (all Pydantic `BaseModel`, defined inline in the route module):

| Model | Fields |
|---|---|
| `QueryEvalRequest` | `query: str` (min_length=1) · `expected_incident_ids: list[str] = []` · `k: int = 10` (1–100) |
| `RetrievalBenchmarkRequest` | `dataset_path: str` · `persist: bool = True` · `experiment_name: str = "default"` · `k: int = 10` |
| `ReasoningBenchmarkRequest` | `dataset_path: str` · `judge: str = "none"` (pattern `^(rule|none)$`) · `experiment_name: str = "default"` · `persist: bool = True` |
| `FullPipelineRequest` | `retrieval_dataset: str \| None` · `reasoning_dataset: str \| None` · `judge: str = "none"` · `experiment_name: str = "default"` · `persist: bool = True` · `k: int = 10` |

**Response models:**

| Model | Fields |
|---|---|
| `RetrievedIncidentItem` | `incident_id, title, similarity_score, rank, is_expected` |
| `QueryEvalResponse` | `query, k, retrieved: list[RetrievedIncidentItem], recall_at_k, reciprocal_rank, ndcg_at_k, rank_of_first_expected, failures: list[dict]` |
| `RetrievalBenchmarkResponse` | `run_id \| None, experiment_name, evaluation_report: dict, regression_report: dict \| None, warnings: list[str], errors: list[str]` |
| `ReasoningBenchmarkResponse` | `run_id \| None, experiment_name, reasoning_report: dict, judge_aggregate: dict \| None, regression_report: None (always), warnings, errors` |
| `FullPipelineResponse` | `run_id \| None, experiment_name, retrieval_report, reasoning_report, judge_report, quality_report, validation_report, retrieval_regression, reasoning_regression: dict \| None each, execution_summary: dict, warnings, errors` |
| `RunSummary` | `run_id, timestamp, experiment_name, duration: float, git_commit: str \| None` |
| `RunDetailResponse` | `metadata: dict, summary: dict, quality_report: dict \| None, recommendations: list[dict], retrieval_report, reasoning_report, judge_report, validation_report: dict \| None each` |
| `FailedQueriesResponse` / `FailedReasoningResponse` / `JudgeDisagreementsResponse` | `run_id, total: int, <field>: list[dict]` |
| `StatsResponse` | `total_runs: int, best_mrr, best_ndcg, best_reasoning_accuracy: float \| None each, latest_run: str \| None, trend: list[str]` |

**Design constraints stated in the module docstring:**
- MUST NOT compute metrics, re-run evaluation, or duplicate serialization.
- MUST NOT import `IncidentSearchService` internals, `LLMService`, or agent implementation
  classes directly in the response-model layer.
- `GET /evaluation/runs/latest` is registered **before** `GET /evaluation/runs/{run_id}` so
  FastAPI's literal-path matching wins over the variable segment.

### Lifecycle — endpoint by endpoint

**`POST /evaluation/query`** — single-query diagnostic, no persistence.
1. `_build_search_service(db)` constructs `IncidentSearchService(db, EmbeddingService())`; raises
   `503` on failure.
2. `search_service.search(query, limit=k, call_site="evaluation_api")` runs live retrieval; any
   exception → `500`.
3. Each string in `expected_incident_ids` is parsed as a UUID; a bad UUID → `422`.
4. A synthetic single-query gold record is built in memory — `GoldQuery` (category
   `"lexical-overlap"` if any expected incidents were supplied, else `"no-match-expected"`;
   difficulty hardcoded `"medium"`) wrapped in a `ResolvedGoldQuery` with each expected UUID
   pre-resolved via `ResolvedIdentity(source_type="api", ...)` — and scored with
   `app.evaluation.metrics.score_query` (Phase 16C), the same primitive doc 15's harness uses.
5. `rank_of_first_expected` is computed by a linear scan of the ranked results (1-indexed, `None`
   if no expected incident appears).
6. If `recall_at_k < 1.0`, a **minimal** `EvaluationReport` (one query, hand-built
   `AggregateMetrics`/`CoverageBreakdown`/`CorpusStatistics`/`GoldDatasetResolutionSummary`) is
   assembled purely so `app.evaluation.failure_analysis.analyze_retrieval_failures` (Phase 21A)
   can be called on it; any exception here is swallowed (`except Exception: pass`) — failure
   analysis is explicitly best-effort for this endpoint.

**`POST /evaluation/retrieval`** — full Gold Dataset benchmark.
1. `_load_gold_dataset(path)` — `400` if the file doesn't exist or fails to parse/validate
   (`GoldDatasetParseError` / `GoldDatasetValidationError` from Phase 16B's loader).
2. `app.evaluation.harness.evaluate(dataset, search_service, k=k)` (Phase 16D) runs the benchmark;
   exception → `500`.
3. Regression is **not** actually computed: the handler checks `repo.latest()` and, if a prior run
   exists, only appends a warning string — *"Regression comparison against file-persisted runs
   not supported via API; run the full pipeline CLI for regression tracking."* `regression_report`
   is always `None` from this endpoint.
4. If `persist=True`, the report is wrapped in a synthetic minimal `EvaluationPipelineResult` (via
   `_make_minimal_pipeline_result`) and saved through `ExperimentRepository.save()`; a save failure
   is caught and appended to `errors`, not raised — the endpoint still returns `200`.

**`POST /evaluation/reasoning`** — reasoning benchmark, optional judge pass.
1. `_load_reasoning_dataset(path)` manually parses the JSON (`InvestigationScenario(**s)` per
   scenario) rather than delegating to a `load_reasoning_dataset` helper — `400` on missing file,
   bad JSON, or a `KeyError`/`TypeError` while constructing the dataclasses.
2. `_build_orchestrator(db)` constructs `MultiAgentInvestigationOrchestrator(search, llm)` (Phase
   19D) — `503` if search or LLM construction fails.
3. `_build_judge(judge)` — `"rule"` → `RuleJudge()` (Phase 20B); `"none"` → `None`; anything else →
   `400`.
4. `app.evaluation.reasoning_harness.evaluate_reasoning_dataset(dataset, orchestrator)` (Phase 20A)
   runs the benchmark; exception → `500`.
5. If a judge was requested, each `result.problem`/`result.session` in the report is scored one at
   a time via `judge.evaluate_session(...)`; per-scenario judge failures are appended to `errors`
   and do **not** abort the run. Successful evaluations are aggregated with
   `app.evaluation.judge_benchmark.aggregate_judge_evaluations` (Phase 20B).
6. Persistence follows the same pattern as `/retrieval` (best-effort, errors collected not raised).
   `regression_report` is hardcoded `None` — no reasoning-regression wiring exists at the API layer
   at all (contrast with `app.evaluation.reasoning_regression`, Phase 20A, which exists but is
   unreachable via this endpoint).

**`POST /evaluation/full`** — the Phase 21E pipeline end-to-end.
1. Datasets are loaded only if paths are supplied (`retrieval_dataset` / `reasoning_dataset` are
   both optional) — either or both stages can be skipped.
2. Service construction is best-effort and swallows `HTTPException`: if `_build_search_service`
   raises, `search_service` stays `None`; the orchestrator is only attempted at all if a reasoning
   dataset was supplied, and likewise degrades to `None` on failure. This is a deliberate contrast
   with `/retrieval` and `/reasoning`, which hard-fail (`503`) if their one required service is
   unavailable — `/full` instead hands `None` services into the pipeline and lets
   `EvaluationPipeline.run()` (Phase 21E) decide how to degrade.
3. `EvaluationPipelineConfig` is built with `run_retrieval`/`run_reasoning`/`run_judge` flags
   derived from what was actually loaded, `run_failure_analysis=True`, `run_validation=True`
   always on, and **`persist_results=False`** — persistence is deliberately handled by this route's
   own call to `ExperimentRepository.save()` afterward, not by the pipeline's own repositories
   (which are freshly constructed in-memory instances: `InMemoryBenchmarkRepository`,
   `InMemoryReasoningBenchmarkRepository`, `InMemoryJudgedReasoningBenchmarkRepository` — created
   per-request and discarded).
4. `pipeline.run(PipelineInputs(...))` — any exception → `500`.
5. Persisting the result (`repo.save(result, ...)`) is wrapped in a bare `except Exception: pass`
   — a persistence failure here is completely silent (not even appended to `errors`), unlike
   `/retrieval` and `/reasoning`, which at least record persistence failures in `errors`. This is a
   real inconsistency between endpoints (see Risks).
6. The response always includes `execution_summary` (from `result.execution_summary`), and
   `warnings`/`errors` are read from that summary object, not accumulated locally by the route
   handler (unlike the two single-stage endpoints above).

**`GET /evaluation/runs`** — `repo.list_runs()` then reversed (newest first); each ID is loaded and
reduced to a `RunSummary`; runs that fail to load (`repo.load(rid) is None`) are silently skipped.

**`GET /evaluation/runs/latest`** — `repo.latest()`; `404` if no runs exist; otherwise delegates to
`_run_to_detail()`.

**`GET /evaluation/runs/{run_id}`** — `repo.load(run_id)`; `404` if not found; same
`_run_to_detail()` helper as `latest`. `_run_to_detail` pulls `recommendations` out of
`run.quality_report["recommendations"]` if present, else `[]`.

**`GET /evaluation/runs/{run_id}/failed-queries|failed-reasoning|judge-disagreements`** — three
structurally identical endpoints; each loads the run (`404` if missing) and returns
`len(collection)` plus the raw `list(collection)` from the persisted run object
(`run.failed_queries`, `run.failed_reasoning`, `run.judge_disagreements` respectively — all
produced upstream by Phase 21A/21B/21F machinery, not recomputed here).

**`GET /evaluation/stats`** — `repo.stats()` (Phase 21F), mapped field-for-field into
`StatsResponse`.

### Design decisions

- **Zero new evaluation logic, by rule.** Every numeric result traces to a Phase ≤21F function.
  The API layer's only responsibilities are input validation, service construction, and
  dict/Pydantic shaping.
- **Best-effort persistence and best-effort judge/failure-analysis passes.** Across all four POST
  endpoints, secondary side effects (persistence, judge scoring, failure clustering) are wrapped in
  broad `except Exception` blocks and degrade to `None`/`[]`/an appended warning rather than
  failing the whole request — the primary evaluation result is judged more valuable than a
  wrap-around feature.
- **Route ordering as a correctness mechanism.** The `latest`-before-`{run_id}` ordering is called
  out explicitly in the module docstring as something a maintainer must not reorder.
- **In-memory repositories for the pipeline, on-disk repository for the API's own persistence.**
  `/full` deliberately keeps `EvaluationPipeline`'s own bookkeeping repositories in memory and
  layers `ExperimentRepository` (file-backed, Phase 21F) on top — avoiding a second on-disk store
  with a different shape.

### Interfaces

Depends on (imports from): `app.evaluation.experiment_tracking` (`ExperimentRepository`,
`_to_jsonable`), `.gold_loader`, `.gold_dataset`, `.harness`, `.metrics`, `.failure_analysis`,
`.reasoning_dataset`, `.reasoning_harness`, `.reasoning_benchmark`, `.judge_benchmark`,
`.rule_judge`, `.evaluation_pipeline`, `.benchmark` (`InMemoryBenchmarkRepository`),
`app.services.search`, `app.services.embedding_service`, `app.services.investigation_orchestrator`,
`app.services.llm_service`. Public surface: the router object `evaluation.router`, plus
module-level helpers re-imported by Phase 21H (`_build_search_service`, `_to_dict`,
`QueryEvalResponse`, `RetrievedIncidentItem`) — Phase 21H explicitly reuses rather than
reimplements these.

**Update (Phase 18E):** `app.services.investigation_orchestrator` now imports
`app.services.routed_search`/`.search_factory` at module level (for its own default
`search_service` construction, doc 18/19), so importing it here transitively imports doc 18's
routing stack too — the "not imported by, nor importing, routing/routed_search" claim this section
previously made no longer holds at the *import* level. Functionally, though, `_build_orchestrator`
(above) still explicitly constructs and passes its own plain dense `IncidentSearchService` as
`search_service`, which short-circuits the orchestrator's routed default — so `/evaluation/reasoning`
and `/evaluation/full` still evaluate against dense-only retrieval in practice, deliberately, for
reproducible benchmarking. Only `/search/incidents`, `/search/debug`, and
`/agent/investigate-orchestrated` actually execute routing/BM25/Hybrid.

### Testing

`tests/api/test_evaluation_api.py` covers, per endpoint: `POST /query` (503 with no DB/service,
422 on bad UUID, success path, empty-expected no-crash); `POST /retrieval` (400 missing dataset,
success, persist creates a `run_id`, `persist=False` leaves `run_id=None`); `POST /reasoning` (400
missing dataset, success, 503 when orchestrator unavailable); `POST /full` (empty result when no
datasets given, persist saves a run, retrieval report present when a retrieval dataset is given,
`execution_summary` always present, 400 on an invalid judge string); `GET /runs` (empty list,
newest-first ordering, required fields present); `GET /runs/latest` (404 with no runs, returns most
recent); `GET /runs/{run_id}` (404, metadata, summary); the three failure/disagreement sub-routes
(404, empty collections on a fresh run); `GET /stats` (empty repo, aggregates after several saves,
`latest_run` populated); and two route-registration checks — the `evaluation` tag appears in the
OpenAPI schema, and all 11 expected routes are registered.

### Risks

- **`/retrieval` and `/reasoning` have no regression comparison at all.** A prior "dead
  `regression_report` field" was removed (it was always `None` on both endpoints); regression
  tracking (Phase 16E/20A) remains reachable only via the CLI (`scripts/run_full_evaluation.py`),
  not via HTTP. `/full`'s `retrieval_regression`/`reasoning_regression` fields are a separate,
  still-live mechanism (see below).
- **Inconsistent error handling for persistence across endpoints.** `/retrieval` and `/reasoning`
  append a persistence failure to `errors`; `/full` swallows it with a bare `pass` and reports
  nothing to the caller. A client polling `/full` with `persist=True` has no way to know
  persistence silently failed.
- **No authentication, no rate limiting.** Any caller can trigger a full pipeline run (LLM calls
  included) or read the entire experiment history.
- **Synchronous, blocking handlers.** A `/full` call with both datasets and an LLM judge runs
  entirely inside one FastAPI request; there is no background-job model, so slow evaluations block
  the request thread.

### Future work

No "future phase" language appears in this module's docstring; the only forward-looking note is
that `/retrieval`'s response still points a caller at the CLI as the current recommended path for
regression tracking, implying API-level regression support may be built in a future phase.

---

## Phase 21H — Human-Friendly Interactive Evaluation API

### Goal

Let a human annotate retrieval quality without ever handling a UUID: run a query once, see
ranked, titled results, pick the correct one(s) by eye, and get Recall/MRR/NDCG back — while
guaranteeing that scoring reuses the exact retrieval results the human looked at (no re-run, no
drift between what was shown and what was scored).

### Motivation

Phase 21G's `POST /evaluation/query` requires the caller to already know the expected incidents'
UUIDs — usable by a script, not by a person reviewing results in Swagger UI. The module docstring
frames this precisely: a three-step workflow (**preview → human selects → evaluate**) plus a
`by-title` shortcut for when the reviewer already knows the incident's title. The explicit
non-goal, stated three times in the docstring, is touching Phase 21G at all: *"MUST NOT modify or
duplicate any endpoint from Phase 21G,"* *"MUST NOT rerun retrieval inside the /evaluate step,"*
*"MUST NOT compute metrics, re-derive failures, or duplicate serialisation beyond what already
exists in Phase 21G's helpers."*

### Architecture

Router: same prefix/tag as 21G (`APIRouter(prefix="/evaluation", tags=["evaluation"])`) — a second
router instance, not the same object, but grouped identically in OpenAPI/Swagger.

**Session data model:**

```python
SESSION_TTL_SECONDS: int = 1800  # 30 minutes

@dataclass
class _SearchHit:               # plain-data snapshot, no SQLAlchemy objects retained
    incident_id: str
    title: str
    similarity_score: float
    rank: int
    repo: str | None
    source: str | None
    source_type: str

@dataclass
class PreviewSession:
    session_id: str
    query: str
    k: int
    hits: list[_SearchHit]
    created_at: str              # ISO 8601, UTC
    expires_at: str              # ISO 8601, UTC
    status: str = "pending"      # "pending" | "evaluated"

_DEFAULT_STORE: dict[str, PreviewSession] = {}   # module-level, process-local
SessionStore = Depends(lambda: _DEFAULT_STORE)   # overridable per-test
```

**Request/response models:** `PreviewRequest {query, k=10}` → `PreviewResponse {session_id, query,
k, expires_at, retrieved: list[PreviewHit]}`; `PreviewHit {incident_id, title, similarity_score,
rank, repo, source, source_type}`; `EvaluateSessionRequest {selected_incident_ids: list[str]}` (can
be empty — an explicit "none of these are correct" signal) → reuses 21G's `QueryEvalResponse`;
`SessionStatusResponse {session_id, query, k, status, created_at, expires_at, retrieved}`;
`ByTitleRequest {query, expected_titles: list[str] (min_length=1), k=10}` → also 21G's
`QueryEvalResponse`.

Imports `QueryEvalResponse`, `RetrievedIncidentItem`, `_build_search_service`, and `_to_dict`
directly from `app.api.routes.evaluation` — the concrete mechanism by which this phase avoids
duplicating 21G's response shapes and service-construction logic.

### Lifecycle

**`POST /evaluation/query/preview`** (registered before the session-ID routes, for the same
literal-vs-variable path reason as 21G's `runs/latest`):
1. `_build_search_service(db)` → `503` on failure.
2. `search_service.search(query, limit=k, call_site="evaluation_interactive")` → `500` on failure.
3. Each raw result becomes a `_SearchHit` (rank assigned by enumeration order, 1-indexed;
   `repo`/`source`/`source_type` pulled via `getattr(..., None)`/`getattr(..., "")` so a missing
   attribute degrades gracefully rather than raising).
4. `_prune_expired(store)` runs, then a new `PreviewSession` is created with a random
   `uuid.uuid4()` ID, `created_at = now`, `expires_at = now + 1800s`, `status = "pending"`, and
   stored in `store[session_id]`.
5. Response includes the session ID, the human-readable hit list, and `expires_at` so a client can
   know when the session will disappear.

**`GET /evaluation/query/{session_id}`**: `_require_session` prunes expired sessions then looks up
`session_id`; `404` (`"Session {id!r} not found or has expired"`) if absent (including if it just
expired). Returns the full session state, including current `status`.

**`POST /evaluation/query/{session_id}/evaluate`**:
1. `_require_session` — `404` under the same conditions as above.
2. Each string in `selected_incident_ids` is parsed as a UUID — `422` on failure. An empty list is
   valid and means "none of the retrieved results are correct."
3. `_score_hits_against(session.hits, expected_uuids, session.k, session.query)` — **the session's
   cached `hits` are scored directly; `search_service.search()` is never called again.** This
   helper is structurally identical to 21G's single-query scoring path (same synthetic
   `GoldQuery`/`ResolvedGoldQuery` construction, same `score_query` call, same best-effort
   `analyze_retrieval_failures` wrapped in `except Exception: pass`), duplicated in this module
   specifically because it must operate on `_SearchHit` (plain dataclass) rather than 21G's live
   ORM-backed search results.
4. `session.status = "evaluated"` is set **in place** (the dataclass is mutable) — there is no
   guard against calling `/evaluate` again afterward; the endpoint will happily re-score against
   a different `selected_incident_ids` and remains at `status="evaluated"`. Prior evaluation
   results are not retained anywhere — only the last call's response was ever returned to a caller.

**`POST /evaluation/query/by-title`**:
1. Each `expected_titles` entry is lowercased; `db.query(Incident.id, Incident.title).filter(
   func.lower(Incident.title).in_(title_lower)).all()` resolves titles to rows in one query
   (`func.lower` chosen explicitly so the comparison works identically on SQLite, used in tests,
   and Postgres, used in production).
2. Rows are folded into `by_lower: dict[str, uuid.UUID]`, keeping only the **first** row
   encountered per lowercased title (`if key not in by_lower`). The result list preserves the
   caller's `expected_titles` order, and any title with no DB match is silently dropped.
   **Note:** the handler docstring claims *"If a title matches multiple incidents the most
   recently created one is used,"* but the query has no `ORDER BY` — the row actually kept is
   whatever order the database happens to return, not necessarily the most recent. This is a
   real docstring/behavior mismatch (see Risks).
3. Retrieval runs fresh (`call_site="evaluation_by_title"`) — this endpoint does not use or create
   a `PreviewSession` at all, it is a single-call shortcut.
4. `_score_hits_against(hits, resolved_uuids, k, query)` produces the same `QueryEvalResponse` as
   the session-based path.

### Design decisions

- **In-memory, TTL'd session store instead of persistence.** Explicitly "process-local, no
  persistence required per the brief" — the workflow is meant for one interactive review sitting,
  not durable multi-day annotation campaigns.
- **Reuse over reimplementation.** `QueryEvalResponse`, `RetrievedIncidentItem`, `_build_search_service`,
  and `_to_dict` are imported from Phase 21G rather than redefined; only the scoring helper
  (`_score_hits_against`) is duplicated, and only because its input type (`_SearchHit`, a frozen
  snapshot) differs from what 21G's inline scoring code operates on.
  Cross-reference: doc 21F/21G's `ExperimentRepository`-based persistence is deliberately **not**
  used here — nothing from a preview/evaluate session is ever written to `.evaluation_runs/`.
- **Empty `selected_incident_ids` is a first-class signal**, not an error — modeling "reviewer
  confirms no correct answer" as distinct from "reviewer hasn't decided yet."
- **Same router prefix/tag as 21G, by design**, purely so Swagger UI groups both phases' endpoints
  together for a developer browsing the API.

### Interfaces

Depends on: `app.api.routes.evaluation` (reuses four names, see above), `app.api.dependencies`
(`DbSession`), `app.db.models.Incident`, `app.evaluation.failure_analysis`, `.gold_dataset`,
`.gold_loader`, `.harness`, `.metrics` (all the same Phase 16/21A primitives 21G's single-query
path uses), `sqlalchemy.func`. Does not depend on and is not depended on by any Phase 18/19 module.

### Testing

`tests/api/test_evaluation_interactive.py` covers, per endpoint/concern: `POST /query/preview`
(session ID + human-readable fields returned, session actually stored, `repo`/`source`/`source_type`
present, `k` respected, `expires_at` present, 500 on retrieval failure, 503 on service
unavailability, ranks assigned sequentially across multiple results); `GET /query/{session_id}`
(data returned including status, 404 for unknown ID, status flips to `"evaluated"` after the
evaluate call); `POST /query/{session_id}/evaluate` (recall/rank computed correctly, confirms
retrieval is called exactly once in preview and not again on evaluate, 404 for unknown session,
422 for a bad UUID, empty-selection → `rank_of_first_expected=None`, multi-expected recall,
partial-recall case, MRR = 1/2 when the first expected hit is at rank 2, `status` flips to
`"evaluated"`); session-expiry behavior (expired session → 404, expired session pruned on access,
a session with a future expiry is not pruned, `SESSION_TTL_SECONDS == 1800` asserted directly);
`POST /query/by-title` (titles resolved and scored, case-insensitivity, unmatched titles ignored
silently, multiple titles resolved together, empty `expected_titles` → 422); one explicit
backward-compatibility check that Phase 21G's `POST /evaluation/query` still works unmodified; and
two route-registration checks (all four interactive routes appear in the OpenAPI schema and are
tagged `evaluation`).

### Risks

- **No authentication or per-user isolation on sessions.** Any caller who obtains a `session_id`
  (e.g. it leaks into a log or is guessed) can evaluate or inspect it — session IDs are random
  UUIDs but carry no ownership check.
- **Process-local store means all sessions are lost on restart or across multiple app instances** —
  this API cannot be deployed behind more than one worker process/replica without sessions
  randomly 404ing depending on which instance handled the `/preview` call.
- **Lazy-only pruning.** Expired sessions are removed only when *some* request touches the store
  (any preview or lookup call); an idle store accumulates expired entries in memory indefinitely
  between requests.
- **No concurrency control on session mutation.** `session.status = "evaluated"` is a plain
  attribute write on a shared dict entry; two concurrent `/evaluate` calls on the same session
  race with no lock.
- **`/evaluate` is callable repeatedly with different selections and always ends in `"evaluated"`**
  with no history — only the most recent scoring result was ever returned to a client; there's no
  way to recover an earlier evaluation of the same session from the API itself.
- **`by-title`'s multi-match tie-break does not match its own docstring** (see Lifecycle) — the
  "most recently created" claim is not implemented; behavior is dependent on unspecified row
  order from the database driver.

### Future work

No "future phase" language appears in this module's docstring. The natural next steps implied by
the Risks above — persistent/shared session storage, session ownership, and an explicit tie-break
order for `by-title` — are not mentioned as planned work anywhere in the source.
</content>
