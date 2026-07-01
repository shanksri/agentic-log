"""Retrieval Strategy Adapters for Evaluation (Phase 17C).

The Phase 16D harness's ``evaluate()`` takes any object exposing ``.db``
and ``.retrieve(query, *, limit, expand, rerank, call_site)`` — this is
already how ``tests/unit/test_harness.py``'s ``FakeSearchService`` plugs
into it without any harness change. This module exploits that exact same
seam for real evaluation: it adapts BM25 (Phase 17A) and Hybrid (Phase 17B)
retrieval to the identical interface, so the *unmodified* harness can run
against any of the three strategies. ``IncidentSearchService`` itself needs
no adapter — it already satisfies the contract directly.

This phase implements NONE of the following — it only wires existing,
already-validated retrieval implementations behind a shared interface:

- **No new retrieval algorithm.** Every adapter below delegates entirely to
  ``IncidentSearchService`` (dense), ``BM25Retriever`` (Phase 17A), or
  ``HybridRetriever`` (Phase 17B). None of those implementations is
  modified.
- **No production wiring.** Nothing here is imported by
  ``app.api`` routes or the investigation agent. These adapters exist only
  to let the evaluation harness run BM25/Hybrid; production retrieval is
  untouched (see ``app.services.search`` — not modified by this phase).

# Why "configuration, not code paths"

Phase 17C's brief requires evaluating three strategies "through
configuration rather than modifying code" with "no separate evaluation
paths for each strategy." Because ``evaluate()`` already only needs an
object satisfying the ``.db`` + ``.retrieve(...)`` shape, the "evaluation
path" is single and unconditional for every strategy:

```
evaluate(dataset, <any adapter satisfying the shape>, k=..., expand=False, rerank=False)
```

What varies is only *which adapter instance* gets passed in — chosen by
the caller before the call, not branched on inside any evaluation code.
``build_strategy()`` below is the one place that maps a strategy name to
the corresponding adapter; it is a constructor selector, not a second
code path through ``evaluate()``.

# Why ``expand``/``rerank`` are pinned to ``False`` for BM25 and Hybrid

``IncidentSearchService.retrieve()``'s query expansion and LLM reranking
are dense-retrieval-specific refinements (Phase: pre-16) that neither
``BM25Retriever`` nor ``HybridRetriever`` (Phase 17B, by design — see its
module docstring) implement. The adapters below raise ``ValueError`` if
called with ``expand=True`` or ``rerank=True`` rather than silently
ignoring them — a caller requesting a feature a strategy doesn't have
should get a loud error, not a quietly-downgraded result that looks like a
fair comparison but isn't. The Evaluation Matrix in this phase always
passes ``expand=False, rerank=False`` for every strategy, including dense
— so the only variable across the three runs is the retrieval strategy
itself, never expansion/reranking. This isolates exactly what doc 17C
wants measured: dense vs. BM25 vs. RRF fusion, not dense-plus-expansion vs.
bare BM25.

# Adapter lifecycle

```
load_bm25_retriever(db)                 -> BM25Retriever
    1. SELECT id, canonical_text FROM incidents     (one query, full corpus)
    2. BM25Document(document_id=str(id), text=canonical_text) per row
    3. BM25Retriever.from_documents(documents)
       (same corpus text dense embeddings are built from — see
       app.services.incident_ingestion: embed_text(incident.canonical_text) —
       so BM25 and dense are compared on identical source text, not on two
       different views of each incident.)

BM25RetrievalAdapter(db, bm25_retriever)
HybridRetrievalAdapter(db, hybrid_retriever)
    .retrieve(query, *, limit, expand=False, rerank=False, call_site=None)
        -> list[IncidentSearchResult]   (synthetic: incident is a
                                          SimpleNamespace exposing only
                                          .id, since the harness reads only
                                          result.incident.id; never a real
                                          ORM-loaded Incident)
```

# Why ``distance`` on adapter results is not a real distance

``IncidentSearchResult.distance`` is the *only* other field the harness
type carries. BM25 and RRF scores are not cosine distances (BM25's score is
unbounded and higher-is-better; RRF's score is also higher-is-better, on
yet another scale — see Phase 17A/17B's own "why RRF, not normalization"
sections). The harness never reads ``.distance`` (only ``.incident.id``),
so the adapters store ``-score`` there purely so that *if* something later
sorts by distance ascending, "better" still means "smaller" — a
traceability nicety, not a value any consumer should treat as a true
cosine distance.

# Why ``build_strategy`` does not also handle ``"dense"``

Dense retrieval needs no adapter at all — ``IncidentSearchService``
already exposes ``.db`` and a compatible ``.retrieve(...)`` signature
(the harness has run against it unmodified since Phase 16D). Wrapping it
in a no-op adapter here would be an indirection with no behavior to add;
``build_strategy("dense", ...)`` returns the ``IncidentSearchService``
instance itself, unchanged.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Incident
from app.services.bm25_search import BM25Document, BM25Retriever
from app.services.hybrid_search import HybridRetriever
from app.services.search import IncidentSearchResult, IncidentSearchService

StrategyName = Literal["dense", "bm25", "hybrid"]

_EVALUATION_CALL_SITE = "phase17c_evaluation"


def load_bm25_retriever(db: Session) -> BM25Retriever:
    """Build a fresh ``BM25Retriever`` indexed over every incident's
    ``canonical_text`` currently in ``db`` — the same text dense embeddings
    are built from (see module docstring). One query, full corpus; intended
    to be called once per evaluation run, not per query.
    """
    rows = db.execute(select(Incident.id, Incident.canonical_text)).all()
    documents = [
        BM25Document(document_id=str(incident_id), text=canonical_text)
        for incident_id, canonical_text in rows
    ]
    return BM25Retriever.from_documents(documents)


def _require_no_expand_or_rerank(*, expand: bool, rerank: bool, strategy: str) -> None:
    if expand or rerank:
        raise ValueError(
            f"{strategy} retrieval does not support expand/rerank "
            f"(got expand={expand!r}, rerank={rerank!r}); the evaluation matrix "
            "must hold these fixed at False for every strategy"
        )


def _to_incident_search_result(document_id: str, score: float) -> IncidentSearchResult:
    return IncidentSearchResult(
        incident=SimpleNamespace(id=uuid.UUID(document_id)), distance=-score
    )


class BM25RetrievalAdapter:
    """Adapts ``BM25Retriever`` to the harness's
    ``.db`` + ``.retrieve(query, *, limit, expand, rerank, call_site)``
    contract.
    """

    def __init__(self, db: Session, bm25: BM25Retriever) -> None:
        self.db = db
        self._bm25 = bm25

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        expand: bool = False,
        rerank: bool = False,
        call_site: str | None = None,
    ) -> list[IncidentSearchResult]:
        _require_no_expand_or_rerank(expand=expand, rerank=rerank, strategy="BM25")
        results = self._bm25.retrieve(query, limit=limit)
        return [_to_incident_search_result(r.document_id, r.score) for r in results]


class HybridRetrievalAdapter:
    """Adapts ``HybridRetriever`` (Phase 17B) to the harness's
    ``.db`` + ``.retrieve(query, *, limit, expand, rerank, call_site)``
    contract.
    """

    def __init__(self, db: Session, hybrid: HybridRetriever) -> None:
        self.db = db
        self._hybrid = hybrid

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        expand: bool = False,
        rerank: bool = False,
        call_site: str | None = None,
    ) -> list[IncidentSearchResult]:
        _require_no_expand_or_rerank(expand=expand, rerank=rerank, strategy="Hybrid")
        results = self._hybrid.retrieve(query, limit=limit)
        return [_to_incident_search_result(r.document_id, r.rrf_score) for r in results]


def build_strategy(
    name: StrategyName,
    *,
    search_service: IncidentSearchService,
    bm25: BM25Retriever | None = None,
    hybrid_factory: Callable[[], HybridRetriever] | None = None,
) -> IncidentSearchService | BM25RetrievalAdapter | HybridRetrievalAdapter:
    """Return the object to pass as ``evaluate()``'s ``search_service``
    argument for strategy ``name``. The single selection point referenced
    in the module docstring's "configuration, not code paths" — every
    branch here returns an object satisfying the same ``.db``/``.retrieve``
    shape, never a different evaluation code path.

    ``bm25`` is required for ``"bm25"``/``"hybrid"`` (build it once via
    ``load_bm25_retriever`` and reuse across strategies/runs — rebuilding
    per call would re-tokenize the whole corpus for no reason). ``hybrid``
    additionally requires ``hybrid_factory``, a zero-argument callable
    returning a configured ``HybridRetriever`` (typically
    ``lambda: HybridRetriever(search_service, bm25, config=...)``) — taking
    a factory rather than a pre-built instance keeps this function agnostic
    to whatever ``HybridConfig`` the caller wants, without re-exposing every
    one of its fields as a parameter here.
    """
    if name == "dense":
        return search_service
    if name == "bm25":
        if bm25 is None:
            raise ValueError("bm25 retriever is required for strategy 'bm25'")
        return BM25RetrievalAdapter(search_service.db, bm25)
    if name == "hybrid":
        if hybrid_factory is None:
            raise ValueError("hybrid_factory is required for strategy 'hybrid'")
        return HybridRetrievalAdapter(search_service.db, hybrid_factory())
    raise ValueError(f"unknown retrieval strategy {name!r}")
