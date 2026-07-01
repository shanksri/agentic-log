"""Evaluation REST API (Phase 21G).

Exposes the existing evaluation framework (Phases 16–21F) through FastAPI
endpoints.  This module introduces NO new evaluation logic — every endpoint
delegates to already-existing public APIs and returns their results in
Pydantic-typed response envelopes.

# Endpoint map

  POST /evaluation/query              — single-query retrieval evaluation
  POST /evaluation/retrieval          — full retrieval benchmark against a dataset
  POST /evaluation/reasoning          — reasoning benchmark against a dataset
  POST /evaluation/full               — complete Phase 21E pipeline

  GET  /evaluation/runs               — list persisted experiment runs
  GET  /evaluation/runs/latest        — shortcut to the most recent run
  GET  /evaluation/runs/{run_id}      — load one run by ID
  GET  /evaluation/runs/{run_id}/failed-queries       — retrieval failures only
  GET  /evaluation/runs/{run_id}/failed-reasoning     — reasoning failures only
  GET  /evaluation/runs/{run_id}/judge-disagreements  — low-scoring judge cases

  GET  /evaluation/stats              — aggregate statistics across run history

# Design constraints

- MUST NOT compute metrics, re-run evaluation, or duplicate serialisation.
- MUST NOT import ``IncidentSearchService`` internals, LLMService, or any
  agent implementation class directly in the response-model layer.
- All evaluation work happens inside existing public APIs; this module only
  wires them together and shapes the HTTP responses.
- ``GET /evaluation/runs/latest`` is registered BEFORE
  ``GET /evaluation/runs/{run_id}`` so FastAPI matches the literal path
  segment before treating it as a variable.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import DbSession
from app.evaluation.experiment_tracking import ExperimentRepository, _to_jsonable

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# ── Default repository path (override-able via dependency) ────────────────────

_DEFAULT_RUNS_DIR = Path(".evaluation_runs")


def _get_repo() -> ExperimentRepository:
    return ExperimentRepository(base_dir=_DEFAULT_RUNS_DIR)


ExperimentRepo = Depends(_get_repo)


# ── Request models ────────────────────────────────────────────────────────────


class QueryEvalRequest(BaseModel):
    """Evaluate a single retrieval query against known expected incidents."""

    query: str = Field(min_length=1)
    expected_incident_ids: list[str] = Field(
        default=[],
        description="Stable UUIDs of the incidents expected to be found.",
    )
    k: int = Field(default=10, ge=1, le=100)


class RetrievalBenchmarkRequest(BaseModel):
    dataset_path: str = Field(min_length=1)
    persist: bool = True
    experiment_name: str = "default"
    k: int = Field(default=10, ge=1, le=100)


class ReasoningBenchmarkRequest(BaseModel):
    dataset_path: str = Field(min_length=1)
    judge: str = Field(default="none", pattern="^(rule|none)$")
    experiment_name: str = "default"
    persist: bool = True


class FullPipelineRequest(BaseModel):
    retrieval_dataset: str | None = None
    reasoning_dataset: str | None = None
    judge: str = Field(default="none", pattern="^(rule|none)$")
    experiment_name: str = "default"
    persist: bool = True
    k: int = Field(default=10, ge=1, le=100)


# ── Response models ───────────────────────────────────────────────────────────


class RetrievedIncidentItem(BaseModel):
    incident_id: str
    title: str
    similarity_score: float
    rank: int
    is_expected: bool


class QueryEvalResponse(BaseModel):
    query: str
    k: int
    retrieved: list[RetrievedIncidentItem]
    recall_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    rank_of_first_expected: int | None
    failures: list[dict[str, Any]]


class RetrievalBenchmarkResponse(BaseModel):
    run_id: str | None
    experiment_name: str
    evaluation_report: dict[str, Any]
    warnings: list[str]
    errors: list[str]


class ReasoningBenchmarkResponse(BaseModel):
    run_id: str | None
    experiment_name: str
    reasoning_report: dict[str, Any]
    judge_aggregate: dict[str, Any] | None
    warnings: list[str]
    errors: list[str]


class FullPipelineResponse(BaseModel):
    run_id: str | None
    experiment_name: str
    retrieval_report: dict[str, Any] | None
    reasoning_report: dict[str, Any] | None
    judge_report: dict[str, Any] | None
    quality_report: dict[str, Any] | None
    validation_report: dict[str, Any] | None
    retrieval_regression: dict[str, Any] | None
    reasoning_regression: dict[str, Any] | None
    execution_summary: dict[str, Any]
    warnings: list[str]
    errors: list[str]


class RunSummary(BaseModel):
    run_id: str
    timestamp: str
    experiment_name: str
    duration: float
    git_commit: str | None = None


class RunDetailResponse(BaseModel):
    metadata: dict[str, Any]
    summary: dict[str, Any]
    quality_report: dict[str, Any] | None
    recommendations: list[dict[str, Any]]
    retrieval_report: dict[str, Any] | None
    reasoning_report: dict[str, Any] | None
    judge_report: dict[str, Any] | None
    validation_report: dict[str, Any] | None


class FailedQueriesResponse(BaseModel):
    run_id: str
    total: int
    failed_queries: list[dict[str, Any]]


class FailedReasoningResponse(BaseModel):
    run_id: str
    total: int
    failed_reasoning: list[dict[str, Any]]


class JudgeDisagreementsResponse(BaseModel):
    run_id: str
    total: int
    disagreements: list[dict[str, Any]]


class StatsResponse(BaseModel):
    total_runs: int
    best_mrr: float | None
    best_ndcg: float | None
    best_reasoning_accuracy: float | None
    latest_run: str | None
    trend: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert any evaluation dataclass to a JSON-safe dict."""
    return _to_jsonable(obj)


def _load_gold_dataset(path: str):
    """Load a Gold Dataset from disk; raise 400 if missing or invalid."""
    from app.evaluation.gold_loader import (
        GoldDatasetParseError,
        GoldDatasetValidationError,
        load_gold_dataset,
    )
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {path!r}")
    try:
        return load_gold_dataset(p)
    except (GoldDatasetParseError, GoldDatasetValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {exc}") from exc


def _load_reasoning_dataset(path: str):
    """Load a ReasoningGoldDataset from disk; raise 400 if missing or invalid."""
    from app.evaluation.reasoning_dataset import InvestigationScenario, ReasoningGoldDataset

    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {path!r}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        scenarios = tuple(InvestigationScenario(**s) for s in raw.get("scenarios", []))
        return ReasoningGoldDataset(
            version=raw["version"],
            description=raw["description"],
            created_at=raw["created_at"],
            scenarios=scenarios,
            author=raw.get("author"),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid reasoning dataset: {exc}") from exc


def _build_search_service(db):
    """Build IncidentSearchService; raise 503 if unavailable."""
    try:
        from app.services.embedding_service import EmbeddingService
        from app.services.search import IncidentSearchService

        return IncidentSearchService(db=db, embedding_service=EmbeddingService())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Search service unavailable: {exc}",
        ) from exc


def _build_orchestrator(db):
    """Build the multi-agent orchestrator; raise 503 if unavailable."""
    try:
        from app.services.investigation_orchestrator import (
            MultiAgentInvestigationOrchestrator,
        )
        from app.services.llm_service import LLMService
        from app.services.search import IncidentSearchService
        from app.services.embedding_service import EmbeddingService

        search = IncidentSearchService(db=db, embedding_service=EmbeddingService())
        llm = LLMService()
        return MultiAgentInvestigationOrchestrator(db, search_service=search, llm_service=llm)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Orchestrator unavailable: {exc}",
        ) from exc


def _build_judge(judge_arg: str):
    if judge_arg == "none":
        return None
    if judge_arg == "rule":
        from app.evaluation.rule_judge import RuleJudge
        return RuleJudge()
    raise HTTPException(status_code=400, detail=f"Unknown judge: {judge_arg!r}")


# ── Shared scoring helper ────────────────────────────────────────────────────
#
# Both this endpoint and Phase 21H's session-based /evaluate and /by-title
# endpoints need to score an already-retrieved, rank-ordered result set
# against a set of expected incident UUIDs. Both independently built the same
# synthetic GoldQuery/ResolvedGoldQuery glue to reuse Phase 16C's score_query
# — this is that logic, written once and shared via
# app.api.routes.evaluation_interactive's import of this function.


def _score_query_against_expected(
    *,
    query_id: str,
    query: str,
    k: int,
    retrieved: list[tuple[uuid.UUID, str, float]],
    expected_uuids: list[uuid.UUID],
) -> QueryEvalResponse:
    """Score ``retrieved`` (rank-ordered ``(incident_id, title,
    similarity_score)`` triples) against ``expected_uuids``. ``query_id`` is
    the synthetic ``GoldQuery.id`` used for this one-off scoring call — it
    surfaces in any resulting failure records' ``subject_id``, so distinct
    call sites should pass distinct ids (e.g. ``"api-single-query"`` vs.
    ``"interactive-session"``).
    """
    from app.evaluation.failure_analysis import analyze_retrieval_failures
    from app.evaluation.gold_dataset import (
        RELEVANCE_MAX,
        CorpusFingerprintPlaceholder,
        ExpectedIncident,
        GoldQuery,
    )
    from app.evaluation.gold_loader import (
        GoldDatasetResolutionSummary,
        ResolvedExpectedIncident,
        ResolvedGoldQuery,
        ResolvedIdentity,
    )
    from app.evaluation.harness import (
        AggregateMetrics,
        CoverageBreakdown,
        EvaluationConfig,
        EvaluationDatasetInfo,
        EvaluationReport,
        CorpusStatistics,
        QueryEvaluationOutcome,
    )
    from app.evaluation.metrics import score_query

    # Build a synthetic ResolvedGoldQuery so we can reuse score_query directly
    expected_incidents = tuple(
        ExpectedIncident(
            source_type="api",
            source_external_id=str(uid),
            relevance=RELEVANCE_MAX,
        )
        for uid in expected_uuids
    )
    gold_q = GoldQuery(
        id=query_id,
        query=query,
        category="lexical-overlap" if expected_incidents else "no-match-expected",
        difficulty="medium",
        expected_incidents=expected_incidents,
    )
    resolved_incidents = tuple(
        ResolvedExpectedIncident(
            expected=ei,
            resolved=ResolvedIdentity(
                source_type="api",
                source_external_id=str(uid),
                incident_id=uid,
            ),
        )
        for ei, uid in zip(expected_incidents, expected_uuids)
    )
    resolved_q = ResolvedGoldQuery(query=gold_q, resolved_incidents=resolved_incidents)

    retrieved_ids = [incident_id for incident_id, _title, _score in retrieved]
    metric = score_query(retrieved_ids, resolved_q, k=k)

    # Build retrieved items list
    expected_set = set(expected_uuids)
    retrieved_items = [
        RetrievedIncidentItem(
            incident_id=str(incident_id),
            title=title,
            similarity_score=similarity_score,
            rank=i + 1,
            is_expected=incident_id in expected_set,
        )
        for i, (incident_id, title, similarity_score) in enumerate(retrieved)
    ]

    # Rank of first expected incident (1-indexed, or None)
    rank_of_first: int | None = None
    for i, (incident_id, _title, _score) in enumerate(retrieved):
        if incident_id in expected_set:
            rank_of_first = i + 1
            break

    # Run failure analysis only when we have enough to build a minimal report
    failures: list[dict[str, Any]] = []
    if metric is not None and metric.recall_at_k is not None and metric.recall_at_k < 1.0:
        try:
            outcome = QueryEvaluationOutcome(
                query_id=query_id,
                category=gold_q.category,
                difficulty=gold_q.difficulty,
                num_relevant=len(expected_uuids),
                num_unresolved_expected=0,
                skipped=False,
                skip_reason=None,
                metric=metric,
            )
            corpus_fp = CorpusFingerprintPlaceholder()
            agg = AggregateMetrics(
                num_queries=1,
                mean_recall_at_k=metric.recall_at_k,
                mean_reciprocal_rank=metric.reciprocal_rank,
                mean_ndcg_at_k=metric.ndcg_at_k,
                resolution_coverage=1.0,
                queries_with_unresolved_incidents=0,
            )
            mini_report = EvaluationReport(
                dataset=EvaluationDatasetInfo(
                    version="api",
                    description="Single-query API evaluation",
                    created_at="",
                    author=None,
                    corpus_fingerprint=corpus_fp,
                ),
                config=EvaluationConfig(k=k, expand=False, rerank=False),
                corpus_statistics=CorpusStatistics(
                    corpus_fingerprint=corpus_fp,
                    distinct_retrieved_incident_count=len(retrieved_ids),
                ),
                num_evaluated=1,
                num_skipped=0,
                aggregate_metrics=agg,
                per_query=(outcome,),
                coverage=CoverageBreakdown(
                    total_queries=1,
                    no_match_expected_queries=0,
                    fully_resolved_queries=0,
                    partially_resolved_queries=1,
                    fully_unresolved_queries=0,
                ),
                resolution_summary=GoldDatasetResolutionSummary(
                    total_expected_incidents=len(expected_uuids),
                    resolved_count=len(expected_uuids),
                    unresolved_identities=(),
                ),
                category_breakdown={},
                difficulty_breakdown={},
                started_at="",
                finished_at="",
                duration_seconds=0.0,
            )
            failure_records = analyze_retrieval_failures(mini_report)
            failures = [_to_dict(f) for f in failure_records]
        except Exception:  # noqa: BLE001
            pass  # failure analysis is best-effort for single-query mode

    return QueryEvalResponse(
        query=query,
        k=k,
        retrieved=retrieved_items,
        recall_at_k=metric.recall_at_k if metric else None,
        reciprocal_rank=metric.reciprocal_rank if metric else None,
        ndcg_at_k=metric.ndcg_at_k if metric else None,
        rank_of_first_expected=rank_of_first,
        failures=failures,
    )


# ── POST /evaluation/query ────────────────────────────────────────────────────


@router.post("/query", response_model=QueryEvalResponse)
def evaluate_query(request: QueryEvalRequest, db: DbSession) -> QueryEvalResponse:
    """Evaluate a single retrieval query against expected incident IDs.

    Runs live retrieval and scores against the supplied expected incidents.
    No persistence — purely a diagnostic endpoint.
    """
    search_service = _build_search_service(db)

    try:
        results = search_service.search(
            request.query, limit=request.k, call_site="evaluation_api"
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    expected_uuids: list[uuid.UUID] = []
    for eid in request.expected_incident_ids:
        try:
            expected_uuids.append(uuid.UUID(eid))
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"Invalid UUID in expected_incident_ids: {eid!r}"
            )

    retrieved = [(r.incident.id, r.incident.title, r.similarity_score) for r in results]
    return _score_query_against_expected(
        query_id="api-single-query",
        query=request.query,
        k=request.k,
        retrieved=retrieved,
        expected_uuids=expected_uuids,
    )


# ── POST /evaluation/retrieval ────────────────────────────────────────────────


@router.post("/retrieval", response_model=RetrievalBenchmarkResponse)
def run_retrieval_benchmark(
    request: RetrievalBenchmarkRequest,
    db: DbSession,
    repo: ExperimentRepository = ExperimentRepo,
) -> RetrievalBenchmarkResponse:
    """Run a full retrieval benchmark against a Gold Dataset JSON file.

    Regression comparison against a prior run is not available through this
    endpoint (persisted runs are stored as plain dicts, not typed
    ``EvaluationReport``s) — use the full pipeline CLI
    (``scripts/run_full_evaluation.py``) for regression tracking.
    """
    from app.evaluation.harness import evaluate

    dataset = _load_gold_dataset(request.dataset_path)
    search_service = _build_search_service(db)

    warnings: list[str] = []
    errors: list[str] = []

    try:
        report = evaluate(
            dataset,
            search_service,
            k=request.k,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Retrieval evaluation failed: {exc}") from exc

    run_id: str | None = None
    if request.persist:
        try:
            run_id = repo.save(
                _make_minimal_pipeline_result(retrieval_report=report),
                experiment_name=request.experiment_name,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Persistence failed: {exc!r}")

    return RetrievalBenchmarkResponse(
        run_id=run_id,
        experiment_name=request.experiment_name,
        evaluation_report=_to_dict(report),
        warnings=warnings,
        errors=errors,
    )


# ── POST /evaluation/reasoning ────────────────────────────────────────────────


@router.post("/reasoning", response_model=ReasoningBenchmarkResponse)
def run_reasoning_benchmark(
    request: ReasoningBenchmarkRequest,
    db: DbSession,
    repo: ExperimentRepository = ExperimentRepo,
) -> ReasoningBenchmarkResponse:
    """Run a reasoning benchmark against a ReasoningGoldDataset JSON file."""
    from app.evaluation.reasoning_harness import evaluate_reasoning_dataset

    dataset = _load_reasoning_dataset(request.dataset_path)
    orchestrator = _build_orchestrator(db)
    judge = _build_judge(request.judge)

    warnings: list[str] = []
    errors: list[str] = []

    try:
        report = evaluate_reasoning_dataset(dataset, orchestrator)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Reasoning evaluation failed: {exc}"
        ) from exc

    judge_aggregate: dict[str, Any] | None = None
    if judge is not None:
        try:
            from app.evaluation.judge_benchmark import (
                aggregate_judge_evaluations,
                create_judged_benchmark_run,
            )
            from app.evaluation.reasoning_benchmark import create_reasoning_benchmark_run

            judge_evals = []
            for result in report.results:
                try:
                    je = judge.evaluate_session(result.problem, result.session)
                    judge_evals.append(je)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Judge failed for scenario {result.scenario_id!r}: {exc!r}")

            if judge_evals:
                agg = aggregate_judge_evaluations(judge_evals)
                judge_aggregate = _to_dict(agg)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Judge aggregation failed: {exc!r}")

    run_id: str | None = None
    if request.persist:
        try:
            run_id = repo.save(
                _make_minimal_pipeline_result(reasoning_report=report),
                experiment_name=request.experiment_name,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Persistence failed: {exc!r}")

    return ReasoningBenchmarkResponse(
        run_id=run_id,
        experiment_name=request.experiment_name,
        reasoning_report=_to_dict(report),
        judge_aggregate=judge_aggregate,
        warnings=warnings,
        errors=errors,
    )


# ── POST /evaluation/full ─────────────────────────────────────────────────────


@router.post("/full", response_model=FullPipelineResponse)
def run_full_pipeline(
    request: FullPipelineRequest,
    db: DbSession,
    repo: ExperimentRepository = ExperimentRepo,
) -> FullPipelineResponse:
    """Execute the complete Phase 21E evaluation pipeline."""
    from app.evaluation.benchmark import InMemoryBenchmarkRepository
    from app.evaluation.evaluation_pipeline import (
        EvaluationPipeline,
        EvaluationPipelineConfig,
        PipelineInputs,
        PipelineRepositories,
    )
    from app.evaluation.judge_benchmark import InMemoryJudgedReasoningBenchmarkRepository
    from app.evaluation.reasoning_benchmark import InMemoryReasoningBenchmarkRepository

    gold_dataset = None
    if request.retrieval_dataset:
        gold_dataset = _load_gold_dataset(request.retrieval_dataset)

    reasoning_dataset = None
    if request.reasoning_dataset:
        reasoning_dataset = _load_reasoning_dataset(request.reasoning_dataset)

    # Services: best-effort — pipeline handles None gracefully
    search_service = None
    orchestrator = None
    try:
        search_service = _build_search_service(db)
    except HTTPException:
        pass

    if reasoning_dataset is not None:
        try:
            orchestrator = _build_orchestrator(db)
        except HTTPException:
            pass

    judge = _build_judge(request.judge)

    config = EvaluationPipelineConfig(
        experiment_name=request.experiment_name,
        run_retrieval=gold_dataset is not None,
        run_reasoning=reasoning_dataset is not None,
        run_judge=judge is not None,
        run_failure_analysis=True,
        run_validation=True,
        persist_results=False,  # we persist via ExperimentRepository instead
        retrieval_k=request.k,
    )

    pipeline = EvaluationPipeline(
        config=config,
        repositories=PipelineRepositories(
            retrieval_repo=InMemoryBenchmarkRepository(),
            reasoning_repo=InMemoryReasoningBenchmarkRepository(),
            judged_repo=InMemoryJudgedReasoningBenchmarkRepository(),
        ),
    )

    try:
        result = pipeline.run(
            PipelineInputs(
                gold_dataset=gold_dataset,
                search_service=search_service,
                reasoning_dataset=reasoning_dataset,
                orchestrator=orchestrator,
                judge=judge,
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    run_id: str | None = None
    if request.persist:
        try:
            run_id = repo.save(result, experiment_name=request.experiment_name)
        except Exception as exc:  # noqa: BLE001
            pass  # non-fatal

    s = result.execution_summary
    return FullPipelineResponse(
        run_id=run_id,
        experiment_name=request.experiment_name,
        retrieval_report=_to_dict(result.retrieval_report) if result.retrieval_report else None,
        reasoning_report=_to_dict(result.reasoning_report) if result.reasoning_report else None,
        judge_report=_to_dict(result.judge_report) if result.judge_report else None,
        quality_report=_to_dict(result.quality_report) if result.quality_report else None,
        validation_report=_to_dict(result.judge_validation_report) if result.judge_validation_report else None,
        retrieval_regression=_to_dict(result.retrieval_regression) if result.retrieval_regression else None,
        reasoning_regression=_to_dict(result.reasoning_regression) if result.reasoning_regression else None,
        execution_summary=_to_dict(s),
        warnings=list(s.warnings),
        errors=list(s.errors),
    )


# ── GET /evaluation/runs ──────────────────────────────────────────────────────


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    repo: ExperimentRepository = ExperimentRepo,
) -> list[RunSummary]:
    """List all persisted experiment runs, newest first."""
    run_ids = repo.list_runs()
    summaries: list[RunSummary] = []
    for rid in reversed(run_ids):  # newest first
        run = repo.load(rid)
        if run is None:
            continue
        m = run.metadata
        summaries.append(
            RunSummary(
                run_id=m.run_id,
                timestamp=m.timestamp,
                experiment_name=m.experiment_name,
                duration=m.duration,
                git_commit=m.git_commit,
            )
        )
    return summaries


# ── GET /evaluation/runs/latest  (must be before /{run_id}) ──────────────────


@router.get("/runs/latest", response_model=RunDetailResponse)
def get_latest_run(
    repo: ExperimentRepository = ExperimentRepo,
) -> RunDetailResponse:
    """Load the most recently persisted experiment run."""
    run = repo.latest()
    if run is None:
        raise HTTPException(status_code=404, detail="No runs found.")
    return _run_to_detail(run)


# ── GET /evaluation/runs/{run_id} ─────────────────────────────────────────────


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run(
    run_id: str,
    repo: ExperimentRepository = ExperimentRepo,
) -> RunDetailResponse:
    """Load a specific experiment run by ID."""
    run = repo.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return _run_to_detail(run)


# ── GET /evaluation/runs/{run_id}/failed-queries ──────────────────────────────


@router.get("/runs/{run_id}/failed-queries", response_model=FailedQueriesResponse)
def get_failed_queries(
    run_id: str,
    repo: ExperimentRepository = ExperimentRepo,
) -> FailedQueriesResponse:
    """Return retrieval failures (recall < 1.0 or skipped) for one run."""
    run = repo.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return FailedQueriesResponse(
        run_id=run_id,
        total=len(run.failed_queries),
        failed_queries=list(run.failed_queries),
    )


# ── GET /evaluation/runs/{run_id}/failed-reasoning ───────────────────────────


@router.get("/runs/{run_id}/failed-reasoning", response_model=FailedReasoningResponse)
def get_failed_reasoning(
    run_id: str,
    repo: ExperimentRepository = ExperimentRepo,
) -> FailedReasoningResponse:
    """Return reasoning failures for one run."""
    run = repo.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return FailedReasoningResponse(
        run_id=run_id,
        total=len(run.failed_reasoning),
        failed_reasoning=list(run.failed_reasoning),
    )


# ── GET /evaluation/runs/{run_id}/judge-disagreements ────────────────────────


@router.get("/runs/{run_id}/judge-disagreements", response_model=JudgeDisagreementsResponse)
def get_judge_disagreements(
    run_id: str,
    repo: ExperimentRepository = ExperimentRepo,
) -> JudgeDisagreementsResponse:
    """Return judge disagreement cases (score < 5.0) for one run."""
    run = repo.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return JudgeDisagreementsResponse(
        run_id=run_id,
        total=len(run.judge_disagreements),
        disagreements=list(run.judge_disagreements),
    )


# ── GET /evaluation/stats ─────────────────────────────────────────────────────


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    repo: ExperimentRepository = ExperimentRepo,
) -> StatsResponse:
    """Return aggregate statistics across the full experiment run history."""
    stats = repo.stats()
    return StatsResponse(
        total_runs=stats.total_runs,
        best_mrr=stats.best_mrr,
        best_ndcg=stats.best_ndcg,
        best_reasoning_accuracy=stats.best_reasoning_accuracy,
        latest_run=stats.latest_run,
        trend=list(stats.trend),
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _run_to_detail(run: Any) -> RunDetailResponse:
    qual = run.quality_report
    recs: list[dict[str, Any]] = []
    if qual is not None:
        recs = qual.get("recommendations") or []
    return RunDetailResponse(
        metadata=_to_jsonable(run.metadata),
        summary=run.summary,
        quality_report=qual,
        recommendations=recs,
        retrieval_report=run.retrieval_report,
        reasoning_report=run.reasoning_report,
        judge_report=run.judge_report,
        validation_report=run.validation_report,
    )


def _make_minimal_pipeline_result(
    *,
    retrieval_report=None,
    reasoning_report=None,
) -> Any:
    """Build a minimal EvaluationPipelineResult for persistence purposes."""
    from app.evaluation.evaluation_pipeline import (
        EvaluationPipelineResult,
        ExecutionSummary,
    )
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    summary = ExecutionSummary(
        start_time=now,
        end_time=now,
        duration_seconds=0.0,
        retrieval_queries=(
            retrieval_report.num_evaluated + retrieval_report.num_skipped
            if retrieval_report else 0
        ),
        reasoning_scenarios=(
            reasoning_report.metrics.num_scenarios if reasoning_report else 0
        ),
        judge_evaluations=0,
        warnings=(),
        errors=(),
    )
    return EvaluationPipelineResult(
        retrieval_report=retrieval_report,
        retrieval_regression=None,
        retrieval_benchmark=None,
        reasoning_report=reasoning_report,
        reasoning_regression=None,
        reasoning_benchmark=None,
        judge_report=None,
        judge_validation_report=None,
        quality_report=None,
        execution_summary=summary,
    )
