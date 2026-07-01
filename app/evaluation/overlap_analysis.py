"""Retrieval Characteristics: Dense/BM25 Candidate Overlap (Phase 17C).

Answers a question the Evaluation Matrix's Recall/MRR/NDCG numbers cannot:
not "which strategy scores higher" but "how much of what BM25 retrieves is
something dense already found, versus genuinely new." This is deliberately
NOT a relevance metric (it does not consult the Gold Dataset's expected
incidents at all) and is not computed by, or added to, Phase 16C's
``score_query``/``AggregateMetrics`` — it is a separate, candidate-set-only
analysis, exactly as Phase 17C's "Retrieval Characteristics" section
specifies as distinct from its "Metrics" section.

# Why this needs its own pass, not a harness extension

The Phase 16D harness (``app.evaluation.harness.evaluate``) discards each
query's raw retrieved-id list after scoring it — ``QueryMetricResult`` keeps
only counts (``num_retrieved``, etc.), never the ids themselves (see
``harness.py``'s ``EvaluationReport`` — no field holds per-query candidate
sets). Overlap analysis needs the actual id sets from both retrievers for
the same query, which the harness's output simply does not retain. Re-
deriving it from two independent ``evaluate()`` runs would also require the
harness to start retaining that data, modifying Phase 16D — out of scope
for this phase. Calling dense ``.search()`` and BM25 ``.retrieve()``
directly, once per query, is the only way to get this without touching
existing code.

# Why ``.search()``, not ``.retrieve()``

Same reasoning as ``HybridRetriever`` (Phase 17B): the dense side of this
comparison must be the bare candidate set, with no query expansion or LLM
reranking layered on, so the overlap measured is "dense candidates vs. BM25
candidates" and not "dense-plus-expansion candidates vs. BM25 candidates."

# Overlap lifecycle

```
compute_overlap(dataset, dense, bm25, limit=k)
  for each GoldQuery in dataset.queries:
    dense_ids = { str(r.incident.id) for r in dense.search(query.query, limit=k) }
    bm25_ids  = { r.document_id      for r in bm25.retrieve(query.query, limit=k) }
    overlap   = dense_ids & bm25_ids
    jaccard   = len(overlap) / len(dense_ids | bm25_ids)   (0.0 if both empty)
  -> OverlapReport(per_query=..., mean_jaccard=..., ...)
```

A low mean Jaccard with a high ``mean_bm25_only_count`` means BM25 is
surfacing candidates dense retrieval never sees at all — the concrete
signal needed to answer "is BM25 useful by itself, or does it just
duplicate dense" (Phase 17C's stated question). A high mean Jaccard means
the two retrievers are largely redundant for this gold set, which would
argue against the complexity of adding BM25/Hybrid for limited marginal
benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.evaluation.gold_dataset import GoldDataset
from app.services.bm25_search import BM25Retriever
from app.services.search import IncidentSearchService

_OVERLAP_CALL_SITE = "phase17c_overlap_analysis"


@dataclass(frozen=True)
class QueryOverlapResult:
    """Candidate-set comparison for one gold query. ``dense_ids``/
    ``bm25_ids`` are stringified incident ids (matching the convention
    ``app.services.hybrid_search`` already uses for cross-retriever id
    comparison), kept on the result for callers that want to inspect which
    specific incidents diverged, not just the counts.
    """

    query_id: str
    dense_ids: frozenset[str]
    bm25_ids: frozenset[str]
    overlap_count: int
    dense_only_count: int
    bm25_only_count: int
    jaccard: float


@dataclass(frozen=True)
class OverlapReport:
    """Aggregate candidate-overlap statistics across an entire dataset."""

    num_queries: int
    mean_jaccard: float | None
    mean_overlap_count: float | None
    mean_dense_only_count: float | None
    mean_bm25_only_count: float | None
    per_query: tuple[QueryOverlapResult, ...]


def _score_one(
    query_id: str, dense_ids: frozenset[str], bm25_ids: frozenset[str]
) -> QueryOverlapResult:
    overlap = dense_ids & bm25_ids
    union = dense_ids | bm25_ids
    jaccard = len(overlap) / len(union) if union else 0.0
    return QueryOverlapResult(
        query_id=query_id,
        dense_ids=dense_ids,
        bm25_ids=bm25_ids,
        overlap_count=len(overlap),
        dense_only_count=len(dense_ids - bm25_ids),
        bm25_only_count=len(bm25_ids - dense_ids),
        jaccard=jaccard,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_overlap(
    dataset: GoldDataset,
    dense: IncidentSearchService,
    bm25: BM25Retriever,
    *,
    limit: int = 10,
) -> OverlapReport:
    """Run every gold query's text against both retrievers' bare candidate
    generation (dense ``.search()``, BM25 ``.retrieve()`` — never
    expansion/reranking) and report dense/BM25/hybrid-relevant candidate
    overlap. Does not consult ``expected_incidents`` at all — this is a
    retrieval-characteristics analysis, not a relevance metric.
    """
    per_query: list[QueryOverlapResult] = []
    for gold_query in dataset.queries:
        dense_results = dense.search(
            gold_query.query, limit=limit, call_site=_OVERLAP_CALL_SITE
        )
        dense_ids = frozenset(str(result.incident.id) for result in dense_results)
        bm25_results = bm25.retrieve(gold_query.query, limit=limit)
        bm25_ids = frozenset(result.document_id for result in bm25_results)
        per_query.append(_score_one(gold_query.id, dense_ids, bm25_ids))

    return OverlapReport(
        num_queries=len(per_query),
        mean_jaccard=_mean([r.jaccard for r in per_query]),
        mean_overlap_count=_mean([float(r.overlap_count) for r in per_query]),
        mean_dense_only_count=_mean([float(r.dense_only_count) for r in per_query]),
        mean_bm25_only_count=_mean([float(r.bm25_only_count) for r in per_query]),
        per_query=tuple(per_query),
    )
