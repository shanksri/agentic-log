"""Hybrid Retrieval — Dense + BM25 via Reciprocal Rank Fusion (Phase 17B).

Orchestrates the two already-independent, already-validated retrieval
engines — ``IncidentSearchService`` (dense, Phases 1-3A) and
``BM25Retriever`` (lexical, Phase 17A) — without modifying either. This
module is the ONLY thing in the codebase that imports both; neither
``app.services.search`` nor ``app.services.bm25_search`` imports the other
or this module. That asymmetry is deliberate: the two retrievers remain
independently swappable and independently benchmarkable; only the
orchestration layer needs to know both exist.

This phase introduces orchestration only:

- No reranking. ``IncidentSearchService.retrieve()`` (the expand/rerank
  pipeline) is never called here — only the dense primitive,
  ``IncidentSearchService.search()``, which performs no expansion and no
  LLM reranking. Calling ``.retrieve()`` instead would silently violate the
  "do not invoke reranking / query expansion" constraint, so this is a
  load-bearing detail, not an arbitrary method choice.
- No investigation integration, no API route changes, no evaluation
  changes. Nothing outside this module calls it yet.
- No redesign of either retrieval implementation. ``IncidentSearchService``
  and ``BM25Retriever`` are used exactly as Phases 1-3A/17A built them.

# Updated architecture (where this fits)

```
                              Query
                                │
                ┌───────────────┴───────────────┐
                ▼                                ▼
   IncidentSearchService.search()         BM25Retriever.retrieve()
   (dense, Phases 1-3A — unmodified)       (lexical, Phase 17A — unmodified)
                │                                │
                ▼                                ▼
      list[IncidentSearchResult]         list[BM25SearchResult]
                │                                │
                └───────────────┬────────────────┘
                                 ▼
                  HybridRetriever (THIS PHASE)
                    1. Reciprocal Rank Fusion
                    2. Candidate deduplication (built into the same step)
                                 ▼
                     list[HybridSearchResult]
                                 │
                 (terminates here — no reranking, no investigation
                  integration, no API wiring; future phases decide
                  how to consume this)
```

# Hybrid retrieval lifecycle

```
HybridRetriever(dense, bm25, config=HybridConfig(...))
  .retrieve(query, limit=...)
    1. dense_results  = dense.search(query, limit=config.dense_limit,
                                      call_site="hybrid_retriever")
       (caught: a dense failure degrades to dense_results=[], not an abort)
    2. bm25_results   = bm25.retrieve(query, limit=config.bm25_limit)
       (caught: a BM25 failure degrades to bm25_results=[], not an abort)
    3. fuse(dense_results, bm25_results, rrf_k=config.rrf_k)
         -> rank each side 1-based, RRF-score every distinct document,
            deduplicating by document id as part of the same pass
            (see "Candidate deduplication" below)
    4. sort by (-rrf_score, document_id), truncate to `limit`
    -> list[HybridSearchResult]
```

# Reciprocal Rank Fusion

For a document ``d``, let ``rank_dense(d)`` and ``rank_bm25(d)`` be its
1-based rank in each retriever's result list (``None`` if that retriever
did not return ``d`` at all). With fusion constant ``k`` (``config.rrf_k``):

```
RRF(d) = [ 1 / (k + rank_dense(d)) if d in dense_results else 0 ]
       + [ 1 / (k + rank_bm25(d))  if d in bm25_results  else 0 ]
```

A document retrieved by both retrievers sums both contributions; a document
retrieved by only one gets only that one term. There is no "missing rank"
penalty term subtracted for absence — absence simply contributes 0, never a
negative adjustment.

## Why RRF instead of score normalization

Dense similarity scores (cosine, roughly `[0, 1]` per doc 08) and BM25
scores (an unbounded, corpus- and query-dependent quantity — see Phase 17A's
idf formula) live on fundamentally different, non-comparable scales. Fusing
them by normalizing each to `[0, 1]` (e.g. min-max per query) and summing
would require deciding how to normalize an *unbounded* BM25 score
distribution that can shift drastically between queries (a query matching
one rare term behaves very differently from one matching many common
terms) — there is no principled, query-independent way to do this without
introducing tunable, hard-to-justify normalization parameters.

RRF sidesteps the entire problem by discarding the scores and using only
*rank* — a document's position in each retriever's ordering, not its raw
score. Rank is already a comparable, bounded quantity (`1, 2, 3, ...`)
regardless of what scoring function produced it. This is precisely why RRF
is the standard choice for fusing heterogeneous retrievers in the
information-retrieval literature (Cormack, Clarke & Buettcher, 2009) and
why this project — which has already hit real fusion-adjacent failure
modes from score-scale mismatches (doc 12's "hub incident" problem,
doc 13's reranker discarding higher-similarity candidates) — uses it rather
than inventing a bespoke normalization scheme.

## Why rank-based fusion is robust across heterogeneous retrievers

Because RRF never looks at the underlying score, it is agnostic to *how*
either retriever produces its ranking. Dense retrieval could be replaced by
a different embedding model (doc 08's documented future upgrade path) and
BM25 could be replaced by a different tokenizer or even a different lexical
scoring function (doc 17A's documented tokenizer-injection point) — as long
as each still returns "a ranked list," RRF's fusion logic does not need to
change. This robustness is what makes RRF appropriate for *this specific*
architecture, where the explicit goal (this phase's own design philosophy)
is keeping both retrievers independently swappable.

## Configurable fusion constant (k)

``HybridConfig.rrf_k`` (default ``60.0``, the literature/Elasticsearch-
convention default) controls how much a document's exact rank matters
versus merely being present. A larger ``k`` flattens the curve — rank 1 and
rank 10 contribute more similar amounts — favoring documents that appear in
*both* lists somewhere over documents that rank #1 in only one. A smaller
``k`` sharpens the curve, weighting a #1 rank much more heavily than a #5
rank in the same list. ``rrf_k`` must be `> 0` (enforced at construction);
nothing else about its value is enforced, since the "right" value is an
empirical question for a future benchmarking phase (explicitly out of scope
here), not something this phase should hardcode an opinion about beyond a
reasonable default.

# Candidate deduplication

A document appearing in both ``dense_results`` and ``bm25_results`` must
appear exactly once in the fused output, with its score being the *sum* of
both RRF contributions — never two separate entries. This falls directly
out of the fusion algorithm's data structure rather than being a separate
post-processing step: dense and BM25 ranks are each collected into a
``{document_id: rank}`` mapping first, and the fused candidate set is the
*union of those mappings' keys* (one entry per distinct ``document_id``, by
construction — a Python ``dict``/``set`` cannot hold two entries for the
same key). There is no separate "merge step" that could be skipped or get
the dedup logic wrong; deduplication is a structural property of the
chosen representation, not a behavior that had to be separately implemented
and could drift out of sync with the scoring logic.

If either retriever's own result list happens to contain a duplicate
``document_id`` (a defect in that retriever, not expected from either
``IncidentSearchService`` or ``BM25Retriever`` as built), only the first
(best-ranked) occurrence is kept per side — the same "keep the best rank"
philosophy already used by the dense pipeline's own candidate merge (doc
12) and by ``IdentityResolver``'s defensive relevance-grade handling
(Phase 16C).

# Configuration design

``HybridConfig`` exposes exactly the four parameters this phase is
responsible for — ``dense_limit``, ``bm25_limit``, ``rrf_k``,
``final_limit`` — and nothing else. It does NOT re-expose dense retrieval's
own filters (``owner``/``repo``/``source``/``state``/``tags``) or BM25's
tokenizer/``k1``/``b`` — those remain configuration of their own retrievers
(``IncidentSearchService``/``BM25Config``), constructed and owned by the
caller before being handed to ``HybridRetriever``. This is "keep
configuration isolated from both retrievers" taken literally: this
module's config is a strict subset of "fusion-specific" knobs, not a
superset re-exposing everything either retriever already configures.
Validation happens once, at construction (``__post_init__``), so an invalid
config fails immediately rather than producing a confusing downstream error
during fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.bm25_search import BM25Retriever, BM25SearchResult
from app.services.search import IncidentSearchResult, IncidentSearchService

_HYBRID_CALL_SITE = "hybrid_retriever"


@dataclass(frozen=True)
class HybridConfig:
    """Fusion-specific configuration only — see module docstring's
    "Configuration design" for what is deliberately excluded.
    """

    dense_limit: int = 25
    bm25_limit: int = 25
    rrf_k: float = 60.0
    final_limit: int = 10

    def __post_init__(self) -> None:
        if self.dense_limit < 1:
            raise ValueError(f"dense_limit must be >= 1, got {self.dense_limit!r}")
        if self.bm25_limit < 1:
            raise ValueError(f"bm25_limit must be >= 1, got {self.bm25_limit!r}")
        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, got {self.rrf_k!r}")
        if self.final_limit < 1:
            raise ValueError(f"final_limit must be >= 1, got {self.final_limit!r}")


@dataclass(frozen=True)
class HybridSearchResult:
    """One fused candidate. ``dense_rank``/``bm25_rank`` are 1-based ranks
    in each retriever's own result list, or ``None`` if that retriever did
    not return this document at all. ``dense_result``/``bm25_result`` carry
    the original per-retriever result (also ``None`` if absent from that
    side) so a future consumer can access incident detail (via
    ``dense_result.incident``) or the BM25 score without a second lookup —
    this phase does not decide how that data is used, only that it is not
    thrown away.
    """

    document_id: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_result: IncidentSearchResult | None
    bm25_result: BM25SearchResult | None


class HybridRetriever:
    """Orchestrates dense (``IncidentSearchService``) and lexical
    (``BM25Retriever``) retrieval via Reciprocal Rank Fusion. Calls only
    ``IncidentSearchService.search()`` (never ``.retrieve()``) — see module
    docstring for why this distinction matters.
    """

    def __init__(
        self,
        dense: IncidentSearchService,
        bm25: BM25Retriever,
        *,
        config: HybridConfig | None = None,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._config = config or HybridConfig()

    @property
    def config(self) -> HybridConfig:
        return self._config

    def retrieve(self, query: str, *, limit: int | None = None) -> list[HybridSearchResult]:
        """Fuse dense and BM25 results for ``query`` and return the top
        ``limit`` (default: ``config.final_limit``) fused candidates.

        A failure in either underlying retriever is caught and treated as
        "that retriever returned no results" — the other retriever's
        results still get fused and returned. This mirrors the project's
        existing fail-safe conventions (e.g. reranking falling back to
        distance order on LLM failure, doc 13) rather than letting one
        retriever's outage take down hybrid retrieval entirely.
        """
        effective_limit = limit if limit is not None else self._config.final_limit
        if effective_limit < 1:
            raise ValueError(f"limit must be >= 1, got {effective_limit!r}")

        dense_results = self._safe_dense_search(query)
        bm25_results = self._safe_bm25_retrieve(query)

        return _fuse(
            dense_results, bm25_results, rrf_k=self._config.rrf_k, limit=effective_limit
        )

    def _safe_dense_search(self, query: str) -> list[IncidentSearchResult]:
        try:
            return self._dense.search(
                query, limit=self._config.dense_limit, call_site=_HYBRID_CALL_SITE
            )
        except Exception:
            return []

    def _safe_bm25_retrieve(self, query: str) -> list[BM25SearchResult]:
        try:
            return self._bm25.retrieve(query, limit=self._config.bm25_limit)
        except Exception:
            return []


def _fuse(
    dense_results: list[IncidentSearchResult],
    bm25_results: list[BM25SearchResult],
    *,
    rrf_k: float,
    limit: int,
) -> list[HybridSearchResult]:
    dense_rank_by_id: dict[str, int] = {}
    dense_result_by_id: dict[str, IncidentSearchResult] = {}
    for rank, result in enumerate(dense_results, start=1):
        document_id = str(result.incident.id)
        if document_id not in dense_rank_by_id:  # keep the best (first) rank on a duplicate
            dense_rank_by_id[document_id] = rank
            dense_result_by_id[document_id] = result

    bm25_rank_by_id: dict[str, int] = {}
    bm25_result_by_id: dict[str, BM25SearchResult] = {}
    for rank, result in enumerate(bm25_results, start=1):
        if result.document_id not in bm25_rank_by_id:
            bm25_rank_by_id[result.document_id] = rank
            bm25_result_by_id[result.document_id] = result

    # The union of both rank maps' keys IS the deduplicated candidate set —
    # see module docstring's "Candidate deduplication".
    all_document_ids = set(dense_rank_by_id) | set(bm25_rank_by_id)

    fused: list[HybridSearchResult] = []
    for document_id in all_document_ids:
        dense_rank = dense_rank_by_id.get(document_id)
        bm25_rank = bm25_rank_by_id.get(document_id)
        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (rrf_k + dense_rank)
        if bm25_rank is not None:
            score += 1.0 / (rrf_k + bm25_rank)
        fused.append(
            HybridSearchResult(
                document_id=document_id,
                rrf_score=score,
                dense_rank=dense_rank,
                bm25_rank=bm25_rank,
                dense_result=dense_result_by_id.get(document_id),
                bm25_result=bm25_result_by_id.get(document_id),
            )
        )

    ranked = sorted(fused, key=lambda result: (-result.rrf_score, result.document_id))
    return ranked[:limit]
