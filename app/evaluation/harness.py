"""Evaluation Harness (Phase 16D).

Orchestrates the end-to-end retrieval evaluation pipeline for a Gold Dataset
v2 against a live ``IncidentSearchService``. This module implements NONE of
the following — it only coordinates the modules that already implement
them:

- **Retrieval** — delegated entirely to ``IncidentSearchService.retrieve()``
  (app/services/search.py). The harness never embeds a query, never builds
  a SQL statement, never calls the LLM directly.
- **Identity resolution** — delegated entirely to
  ``app.evaluation.gold_loader.resolve_gold_dataset`` /
  ``summarize_resolution`` (Phase 16B), which itself delegates to
  ``IdentityResolver`` (Phase 16A). The harness never imports
  ``IdentityResolver`` and never queries the database directly; it only
  reuses ``search_service.db`` as the session those calls need.
- **Metrics** — delegated entirely to ``app.evaluation.metrics.score_query``
  (Phase 16C). The harness never computes Recall/MRR/DCG/NDCG itself; it
  only aggregates (means, counts, bucket groupings) over already-computed
  ``QueryMetricResult`` values.

# Evaluation lifecycle

```
evaluate(dataset, search_service, k=..., expand=..., rerank=...)
  1. resolve_gold_dataset(search_service.db, dataset)   [Phase 16B, once for
     the whole dataset — see "Why resolution happens once" below]
  2. summarize_resolution(resolved_queries)              [Phase 16B]
  3. for each ResolvedGoldQuery:
       a. search_service.retrieve(query.query, limit=k, expand=, rerank=)
          [Phase: existing IncidentSearchService]
          - on exception: record a skipped QueryEvaluationOutcome, continue
       b. retrieved_ids = [r.incident.id for r in results]
       c. score_query(retrieved_ids, resolved_query, k=k)  [Phase 16C]
       d. record an evaluated QueryEvaluationOutcome
  4. aggregate across all outcomes (overall, per-category, per-difficulty)
  5. compute the query-level coverage breakdown (Phase-16B-derived, search-
     independent)
  6. assemble and return an immutable EvaluationReport
```

# Why resolution happens once, not per query

The end-to-end flow is described per-query ("for each GoldQuery, resolve
... then retrieve ..."), but resolution is implemented as a single batched
call across the whole dataset (Phase 16B's ``resolve_gold_dataset`` already
batches by design — one query for all identities). Resolving per query
individually would defeat that batching with no benefit: resolution does
not depend on anything produced by retrieval, so there is no ordering
requirement that forces it to happen query-by-query. The harness resolves
once up front and then iterates the resolved results per query for the
retrieval/scoring steps, which do genuinely need to happen one query at a
time (each is an independent search call).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from app.evaluation.gold_dataset import CorpusFingerprintPlaceholder, GoldDataset
from app.evaluation.gold_loader import (
    GoldDatasetResolutionSummary,
    ResolvedGoldQuery,
    resolve_gold_dataset,
    summarize_resolution,
)
from app.evaluation.metrics import QueryMetricResult, score_query
from app.services.search import IncidentSearchResult, IncidentSearchService

NO_MATCH_EXPECTED_CATEGORY = "no-match-expected"


# ── Report data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationConfig:
    """The retrieval configuration this evaluation run used. Mirrors
    ``IncidentSearchService.retrieve``'s own knobs exactly, so a report is
    self-describing about what pipeline produced it.
    """

    k: int
    expand: bool
    rerank: bool


@dataclass(frozen=True)
class EvaluationDatasetInfo:
    """A primitive-only snapshot of the dataset's own metadata (Phase 16B's
    ``GoldDataset`` fields, minus ``queries`` — the queries themselves are
    reported individually via ``EvaluationReport.per_query``, not duplicated
    here).
    """

    version: str
    description: str
    created_at: str
    author: str | None
    corpus_fingerprint: CorpusFingerprintPlaceholder


@dataclass(frozen=True)
class CorpusStatistics:
    """Facts about the corpus this run observed.

    ``corpus_fingerprint`` is passed through from the dataset unchanged —
    this phase does not implement corpus fingerprinting (that remains a
    placeholder per Phase 16B; real fingerprinting is a later phase per
    docs/architecture/15_evaluation_framework.md). Only
    ``distinct_retrieved_incident_count`` is genuinely computed here, and it
    is computed from data the harness already has (the union of every
    query's retrieved ids) — no additional corpus-wide query is issued.
    """

    corpus_fingerprint: CorpusFingerprintPlaceholder
    distinct_retrieved_incident_count: int


@dataclass(frozen=True)
class QueryEvaluationOutcome:
    """The per-query result: either a scored outcome, or a skip with a
    reason. ``num_relevant``/``num_unresolved_expected`` are captured
    directly here (not only inside ``metric``) because they come from
    identity resolution, which happens before — and independently of —
    retrieval. This way a query that was skipped due to a *search* failure
    still reports accurate resolution-coverage information; coverage
    statistics are never silently lost just because retrieval failed.
    """

    query_id: str
    category: str
    difficulty: str
    num_relevant: int
    num_unresolved_expected: int
    skipped: bool
    skip_reason: str | None
    metric: QueryMetricResult | None


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate statistics over a set of ``QueryEvaluationOutcome``s — used
    both for the dataset-wide totals and for each category/difficulty
    bucket (the same shape, scoped to different query subsets).

    ``num_queries`` is the bucket's total query count, including skipped
    queries. The mean fields are computed only over queries that produced a
    *defined* value for that metric (excluding skipped queries, and
    excluding mathematically-undefined ``None`` results from Phase 16C —
    e.g. a no-match-expected query contributes nothing to
    ``mean_recall_at_k``, the same way it is excluded from a single query's
    own Recall/NDCG). ``resolution_coverage`` and
    ``queries_with_unresolved_incidents`` are computed over ALL queries in
    the bucket (including skipped ones), since resolution coverage is
    independent of whether retrieval later succeeded.
    """

    num_queries: int
    mean_recall_at_k: float | None
    mean_reciprocal_rank: float | None
    mean_ndcg_at_k: float | None
    resolution_coverage: float | None
    queries_with_unresolved_incidents: int


@dataclass(frozen=True)
class CoverageBreakdown:
    """Query-level resolution coverage, independent of search outcome.

    Distinct from ``AggregateMetrics.resolution_coverage`` (a single
    fraction over *expected incidents*) and from
    ``EvaluationReport.resolution_summary`` (Phase 16B's incident-level
    summary): this breakdown classifies each *query* into exactly one of
    four buckets, useful for spotting whether unresolved coverage is
    concentrated in a few queries or spread thinly across many.
    """

    total_queries: int
    no_match_expected_queries: int
    fully_resolved_queries: int
    partially_resolved_queries: int
    fully_unresolved_queries: int


@dataclass(frozen=True)
class EvaluationReport:
    """The complete, immutable result of one evaluation run. Every nested
    object is itself frozen and primitive-only (no ``Incident``, no
    ``IncidentSearchResult``, no SQLAlchemy session) so the whole report is
    safe to hold, compare, or serialize well after ``search_service``'s
    session has been closed. No visualization, no regression comparison —
    those are later phases (16E+).
    """

    dataset: EvaluationDatasetInfo
    config: EvaluationConfig
    corpus_statistics: CorpusStatistics
    num_evaluated: int
    num_skipped: int
    aggregate_metrics: AggregateMetrics
    per_query: tuple[QueryEvaluationOutcome, ...]
    coverage: CoverageBreakdown
    resolution_summary: GoldDatasetResolutionSummary
    category_breakdown: dict[str, AggregateMetrics]
    difficulty_breakdown: dict[str, AggregateMetrics]
    started_at: str
    finished_at: str
    duration_seconds: float


# ── Orchestration ─────────────────────────────────────────────────────────────


def evaluate(
    dataset: GoldDataset,
    search_service: IncidentSearchService,
    *,
    k: int = 10,
    expand: bool = False,
    rerank: bool = False,
) -> EvaluationReport:
    """Run the complete evaluation pipeline for ``dataset`` against
    ``search_service`` and return an ``EvaluationReport``.

    Identity resolution reuses ``search_service.db`` as its session,
    rather than taking a separate database argument. This is deliberate:
    resolution and retrieval must read the same corpus snapshot for the
    evaluation to be coherent (the same reasoning Phase 16A/16B already
    apply within a single resolution call); accepting a second, independent
    session would risk silently evaluating identity resolution against one
    corpus state and retrieval against another.

    Error handling (see module docstring's "Why resolution happens once"
    for the resolution-call decision):
    - A search failure for one query (``search_service.retrieve`` raises)
      is caught and recorded as a skipped ``QueryEvaluationOutcome`` with a
      ``skip_reason``; it does not abort the run or affect any other query.
    - A failure in the one-time, whole-dataset identity-resolution call is
      NOT caught here and propagates to the caller. This is a deliberate
      choice, not an oversight: resolution failure is a single infrastructure
      event affecting every query identically (e.g. the database is
      unreachable), not a per-query data condition. Fabricating per-query
      skip reasons for a dataset-wide infrastructure failure would
      misrepresent it as a gold-data problem rather than an infrastructure
      one, and a caller that wants to retry or page on this failure needs
      the original exception, not a report that looks like "100% of gold
      data is bad."
    - An empty dataset (``dataset.queries == ()``) is not an error: every
      step below operates correctly on an empty input, producing a report
      with zeroed counts and ``None`` aggregate metrics — no special-casing
      is needed because Phase 16A/16B/16C already define correct behavior
      for empty inputs at each layer.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")

    started_at = datetime.now(UTC)
    started_perf = time.monotonic()

    resolved_queries = resolve_gold_dataset(search_service.db, dataset)
    resolution_summary = summarize_resolution(resolved_queries)

    outcomes: list[QueryEvaluationOutcome] = []
    retrieved_universe: set[uuid.UUID] = set()

    for resolved_query in resolved_queries:
        outcome, retrieved_ids = _evaluate_one(
            resolved_query, search_service, k=k, expand=expand, rerank=rerank
        )
        outcomes.append(outcome)
        retrieved_universe.update(retrieved_ids)

    num_skipped = sum(1 for outcome in outcomes if outcome.skipped)
    num_evaluated = len(outcomes) - num_skipped

    finished_at = datetime.now(UTC)
    return EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version=dataset.version,
            description=dataset.description,
            created_at=dataset.created_at,
            author=dataset.author,
            corpus_fingerprint=dataset.corpus_fingerprint,
        ),
        config=EvaluationConfig(k=k, expand=expand, rerank=rerank),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=dataset.corpus_fingerprint,
            distinct_retrieved_incident_count=len(retrieved_universe),
        ),
        num_evaluated=num_evaluated,
        num_skipped=num_skipped,
        aggregate_metrics=_aggregate(outcomes),
        per_query=tuple(outcomes),
        coverage=_coverage_breakdown(resolved_queries),
        resolution_summary=resolution_summary,
        category_breakdown=_bucket_aggregate(outcomes, key=lambda o: o.category),
        difficulty_breakdown=_bucket_aggregate(outcomes, key=lambda o: o.difficulty),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=time.monotonic() - started_perf,
    )


def _evaluate_one(
    resolved_query: ResolvedGoldQuery,
    search_service: IncidentSearchService,
    *,
    k: int,
    expand: bool,
    rerank: bool,
) -> tuple[QueryEvaluationOutcome, list[uuid.UUID]]:
    """Evaluate a single resolved gold query. Returns the outcome plus the
    raw retrieved id list (only used by the caller to build the dataset-wide
    distinct-retrieved-id count; not stored on the outcome itself, since
    ``QueryMetricResult`` already reports ``num_retrieved``).
    """
    query = resolved_query.query
    num_relevant = sum(1 for entry in resolved_query.resolved_incidents if entry.is_resolved)
    num_unresolved_expected = resolved_query.unresolved_count

    try:
        results: list[IncidentSearchResult] = search_service.retrieve(
            query.query,
            limit=k,
            expand=expand,
            rerank=rerank,
            call_site="evaluation_harness",
        )
    except Exception as exc:  # noqa: BLE001 - intentionally broad; see module docstring
        return (
            QueryEvaluationOutcome(
                query_id=query.id,
                category=query.category,
                difficulty=query.difficulty,
                num_relevant=num_relevant,
                num_unresolved_expected=num_unresolved_expected,
                skipped=True,
                skip_reason=f"search_failed: {exc!r}",
                metric=None,
            ),
            [],
        )

    retrieved_ids = [result.incident.id for result in results]
    metric = score_query(retrieved_ids, resolved_query, k=k)
    outcome = QueryEvaluationOutcome(
        query_id=query.id,
        category=query.category,
        difficulty=query.difficulty,
        num_relevant=num_relevant,
        num_unresolved_expected=num_unresolved_expected,
        skipped=False,
        skip_reason=None,
        metric=metric,
    )
    return outcome, retrieved_ids


# ── Aggregation helpers ────────────────────────────────────────────────────────


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(outcomes: Sequence[QueryEvaluationOutcome]) -> AggregateMetrics:
    metrics = [outcome.metric for outcome in outcomes if outcome.metric is not None]
    recalls = [m.recall_at_k for m in metrics if m.recall_at_k is not None]
    reciprocal_ranks = [m.reciprocal_rank for m in metrics if m.reciprocal_rank is not None]
    ndcgs = [m.ndcg_at_k for m in metrics if m.ndcg_at_k is not None]

    total_relevant = sum(outcome.num_relevant for outcome in outcomes)
    total_expected = total_relevant + sum(
        outcome.num_unresolved_expected for outcome in outcomes
    )
    resolution_coverage = (total_relevant / total_expected) if total_expected else None
    queries_with_unresolved = sum(
        1 for outcome in outcomes if outcome.num_unresolved_expected > 0
    )

    return AggregateMetrics(
        num_queries=len(outcomes),
        mean_recall_at_k=_mean(recalls),
        mean_reciprocal_rank=_mean(reciprocal_ranks),
        mean_ndcg_at_k=_mean(ndcgs),
        resolution_coverage=resolution_coverage,
        queries_with_unresolved_incidents=queries_with_unresolved,
    )


def _bucket_aggregate(
    outcomes: Sequence[QueryEvaluationOutcome], *, key: Callable[[QueryEvaluationOutcome], str]
) -> dict[str, AggregateMetrics]:
    buckets: dict[str, list[QueryEvaluationOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(key(outcome), []).append(outcome)
    return {bucket_key: _aggregate(items) for bucket_key, items in buckets.items()}


def _coverage_breakdown(resolved_queries: Sequence[ResolvedGoldQuery]) -> CoverageBreakdown:
    no_match = 0
    fully_resolved = 0
    partially_resolved = 0
    fully_unresolved = 0

    for resolved_query in resolved_queries:
        if resolved_query.query.category == NO_MATCH_EXPECTED_CATEGORY:
            no_match += 1
            continue
        total = len(resolved_query.resolved_incidents)
        unresolved = resolved_query.unresolved_count
        if unresolved == 0:
            fully_resolved += 1
        elif unresolved == total:
            fully_unresolved += 1
        else:
            partially_resolved += 1

    return CoverageBreakdown(
        total_queries=len(resolved_queries),
        no_match_expected_queries=no_match,
        fully_resolved_queries=fully_resolved,
        partially_resolved_queries=partially_resolved,
        fully_unresolved_queries=fully_unresolved,
    )
