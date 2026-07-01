"""Production ``RoutedSearchService`` construction.

Single shared construction point for every production caller that needs a
fully-wired ``RoutedSearchService`` (Dense + BM25 + Hybrid + routing) — used
by ``app/api/routes/search.py`` and by
``MultiAgentInvestigationOrchestrator``'s default construction (Phase 19D) —
so neither duplicates the "build dense, index BM25, wrap Hybrid, wire a
routing engine" assembly.

# Why the BM25 index is cached, not rebuilt per call

Building a ``BM25Retriever`` requires reading and tokenizing every
incident's ``canonical_text`` in the corpus (see
``app.services.bm25_search``'s own docstring: indexing is a one-shot,
whole-corpus operation, "intended to be called once ... not per query").
Rebuilding it on every request would make every search O(corpus size)
regardless of routing decision. This module builds it once, lazily, on
first use, and reuses the same immutable index for the lifetime of the
process — the same "full rebuild, not incremental" tradeoff
``BM25Index`` itself already documents; this is simply the first place in
the codebase that actually serves requests with it, so it needs a caching
boundary that Phase 18's evaluation-only callers (which build a fresh index
once per benchmark run, not per query) never needed.

Consequence: newly-ingested incidents are not lexically searchable via
BM25/Hybrid until the process restarts (dense search is unaffected — it
reads the database directly on every call). This is a known, documented
limitation, not an oversight; see docs/architecture/18's "Future work".
"""

from __future__ import annotations

import logging
import threading

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Incident
from app.services.bm25_search import BM25Document, BM25Retriever
from app.services.hybrid_search import HybridRetriever
from app.services.llm_service import LLMService
from app.services.routed_search import RoutedSearchConfig, RoutedSearchService
from app.services.routing import DefaultRuleBasedRoutingPolicy, RoutingEngine
from app.services.search import IncidentSearchService

logger = logging.getLogger(__name__)

_bm25_cache: BM25Retriever | None = None
_bm25_lock = threading.Lock()


def _build_bm25_retriever(db: Session) -> BM25Retriever:
    rows = db.execute(select(Incident.id, Incident.canonical_text)).all()
    documents = [
        BM25Document(document_id=str(incident_id), text=canonical_text or "")
        for incident_id, canonical_text in rows
    ]
    return BM25Retriever.from_documents(documents)


def get_bm25_retriever(db: Session) -> BM25Retriever:
    """Return the process-local BM25 index, building it once (thread-safe,
    double-checked locking) and reusing it on every subsequent call. ``db``
    is only used if a build is actually needed.
    """
    global _bm25_cache
    if _bm25_cache is None:
        with _bm25_lock:
            if _bm25_cache is None:
                logger.info("search_factory.bm25_index_build_started")
                index = _build_bm25_retriever(db)
                logger.info(
                    "search_factory.bm25_index_build_complete",
                    extra={"corpus_size": index.index.size},
                )
                _bm25_cache = index
    return _bm25_cache


def reset_bm25_cache() -> None:
    """Drop the cached index so the next call to ``get_bm25_retriever``
    rebuilds it. Exposed for tests; production code has no automatic
    invalidation trigger yet (see module docstring).
    """
    global _bm25_cache
    with _bm25_lock:
        _bm25_cache = None


def build_routed_search_service(
    db: Session, *, llm_service: LLMService | None = None
) -> RoutedSearchService:
    """The single production construction point for ``RoutedSearchService``.
    ``routing_enabled`` is read from ``Settings.search_routing_enabled``
    (default ``False`` — dense-only behavior, unchanged from before this
    module existed, unless explicitly opted in).
    """
    dense = IncidentSearchService(db, llm_service=llm_service)
    bm25 = get_bm25_retriever(db)
    hybrid = HybridRetriever(dense, bm25)
    routing_engine = RoutingEngine(DefaultRuleBasedRoutingPolicy())
    config = RoutedSearchConfig(routing_enabled=settings.search_routing_enabled)
    return RoutedSearchService(
        dense,
        bm25=bm25,
        hybrid=hybrid,
        routing_engine=routing_engine,
        config=config,
    )
