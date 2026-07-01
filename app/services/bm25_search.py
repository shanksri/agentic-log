"""BM25 lexical retrieval infrastructure (Phase 17A).

A completely independent retrieval strategy, sibling to dense retrieval
(``app.services.search.IncidentSearchService``), not an extension of it.
This module has ZERO imports from ``app.services.search``, ``app.db``, or
pgvector — it knows nothing about ``Incident``, ``Embedding``, SQLAlchemy,
or cosine distance. It operates purely on ``(document_id, text)`` pairs
supplied by the caller.

This phase implements ONLY the lexical retrieval layer itself. It does
**not** implement Hybrid Search, Reciprocal Rank Fusion, dense/BM25 fusion,
cross-encoder reranking, or any change to query expansion, confidence
calibration, or evaluation — those are later phases (17B+).

# Where BM25 fits in the retrieval pipeline

Today: nowhere yet. This module is freestanding infrastructure — no route,
no agent, and no evaluation harness wires into it in this phase. It exists
so that Phase 17B can later orchestrate

    DenseRetriever.retrieve()   (today: IncidentSearchService.search/retrieve)
    BM25Retriever.retrieve()    (this module)

side by side, fusing their results, WITHOUT either implementation importing
or knowing about the other. Phase 17A's job is to make that future
orchestration possible by building a retriever that already looks and
behaves like a sibling, not by wiring it into anything yet.

# BM25 lifecycle

```
documents: Iterable[BM25Document]            (caller-supplied (id, text) pairs;
                                                e.g. a future Phase 17B loader
                                                reading Incident.canonical_text —
                                                NOT this module's concern)
        │
        ▼
BM25Index()              # empty, tokenizer configured
   .build(documents)      # one-shot: tokenize each doc, populate postings,
                           #   doc lengths, doc order. Raises if called twice
                           #   or if a document_id repeats.
        │
        ▼
BM25Retriever(index, config=BM25Config(k1=..., b=...))
   .retrieve(query, limit=k)  -> list[BM25SearchResult]
```

# Index lifecycle

``BM25Index`` holds ONLY corpus statistics, never scoring parameters:

- ``_postings: dict[term, dict[document_id, term_frequency]]`` — the
  inverted index. Scoring a query only ever touches the postings for the
  query's own (usually small) term set, never every document in the corpus.
- ``_doc_lengths: dict[document_id, token_count]``
- ``_doc_order: list[document_id]`` — insertion order, used only for
  ``size``; not used for scoring or tie-breaking (scoring tie-breaks on
  ``document_id`` value itself, see "Design decisions").

**Rebuild strategy: construct a new index, don't mutate one in place.**
``build()`` may be called exactly once per ``BM25Index`` instance; calling
it again raises. To rebuild (e.g. after a batch of incidents changed),
construct a new ``BM25Index`` and ``build()`` it, then swap the reference
held by whatever owns it (e.g. atomically replace
``retriever = BM25Retriever(new_index, config=retriever.config)``). This is
the same "copy-on-build, swap the reference" pattern used for the dense
embedding index's `model_name`-gated re-embed cycle (doc 08) — readers never
observe a half-rebuilt structure, because there is no in-place mutation to
observe mid-rebuild.

**Incremental updates are intentionally not implemented in this phase.**
Adding/updating/removing individual documents without a full rebuild would
require postings pruning, ``avgdl`` maintenance, and deletion bookkeeping —
real complexity that has no concrete trigger or consumer yet (no caller
exists that updates the corpus incrementally and needs an incremental BM25
index to match). Building it now would be speculative. The full-rebuild
strategy above is sufficient until a future phase identifies an actual
rebuild-cadence requirement (e.g. mirroring ingestion's `text_hash`-gated
re-embed, doc 08) that needs it.

# Retrieval lifecycle

```
query: str
   │
   ▼
tokenizer(query) -> set of unique query terms      (duplicates within the
                                                      query do not double-count;
                                                      see "Design decisions")
   │
   ▼
for each query term present in the index's postings:
   idf(term)                                          (corpus-wide, query-independent)
   for each (document_id, term_frequency) in that term's postings:
       accumulate idf * saturation(term_frequency, document_length, avgdl, k1, b)
   │
   ▼
sort by (-score, document_id)                        (deterministic tie-break)
   │
   ▼
top `limit` -> list[BM25SearchResult(document_id, score)]
```

# Design decisions

- **No coupling to dense retrieval or pgvector.** The index and retriever
  accept plain ``(document_id: str, text: str)`` pairs. Wiring this to the
  actual incident corpus (e.g. reading `Incident.canonical_text`) is
  deliberately left to whatever integrates this module later (Phase 17B's
  orchestrator, or a dedicated indexing script) — this phase's surface area
  stays minimal and is trivially unit-testable with zero database fixture.
- **Why BM25 is implemented independently from dense retrieval.** Dense and
  lexical retrieval fail on different, largely non-overlapping query
  classes (doc 16: dense misses rare jargon/exact error codes; lexical
  trivially catches those but misses paraphrases dense handles well). Each
  is independently valuable, independently measurable, and independently
  swappable — coupling them now (e.g. having `BM25Retriever` import
  `IncidentSearchResult` or vice versa) would make it impossible to
  benchmark either in isolation (a stated Phase 17A goal:
  "independently benchmarkable") and would presume a fusion strategy before
  one has been designed (Phase 17B's job, not this phase's).
- **Why Hybrid Search is intentionally postponed.** Fusing two ranking
  signals (RRF or otherwise) is a design decision with real failure modes
  documented elsewhere in this project's own history — e.g. the
  candidate-merge "hub incident" problem (doc 12) and the reranker's
  documented tendency to discard higher-similarity candidates (doc 13).
  Building fusion logic before BOTH retrievers exist and are independently
  validated would repeat the exact mistake the Phase 16 evaluation platform
  was built to prevent: shipping an algorithm change with no way to
  attribute its effect to one component or the other. BM25 must exist and
  be independently correct first; fusion is Phase 17B's job, evaluated
  through the now-existing Phase 16 harness, not guessed at here.
- **Classic (non-query-term-frequency-weighted) BM25.** Each *unique* query
  term contributes ``idf(term) * saturation(...)`` once, regardless of how
  many times that term appears in the query string itself. This is the
  textbook Okapi BM25 form (Robertson et al.); query-term-frequency
  weighting is a documented extension some implementations add, omitted
  here for the simplest correct baseline.
- **Smoothed idf (``ln(1 + (N - df + 0.5) / (df + 0.5))``).** The classic
  Robertson idf (without the ``+1`` inside the log) can go negative for
  terms occurring in more than half the corpus, which would make a
  "matching" document score *worse* than one that doesn't contain the term
  at all — a well-known pathology. The ``+1`` (the same smoothing used by
  Lucene's BM25 implementation) guarantees ``idf >= 0`` always, at the cost
  of being a slightly different formula than the original 1994 paper — a
  standard, deliberate trade-off.
- **``k1``/``b`` live on the retriever, not the index.** They are pure
  scoring-time constants — corpus statistics (postings, lengths, avgdl) are
  independent of them. This lets two ``BM25Retriever``s with different
  ``k1``/``b`` share one built ``BM25Index`` (no re-tokenization cost),
  which is exactly the shape a future Phase 17B parameter sweep would want.
- **Deterministic tie-break by ``document_id``.** Two documents scoring
  identically must return in a stable order across runs and across Python
  versions/hash seeds — sorting by ``(-score, document_id)`` rather than
  relying on dict/set iteration order achieves this without depending on
  insertion order (which Python dicts preserve but which is an
  implementation detail this module should not lean on for correctness).
- **Tokenizer is a swappable, injected function, not a hardcoded
  pipeline.** ``default_tokenizer`` is the simplest tokenizer that produces
  stable, predictable matches for exact-keyword and lexical-overlap queries
  (the same "lexical-overlap" category already used by the Phase 0 gold-set
  plan and Gold Dataset v2's category schema). Stemming, stopword removal,
  or n-gram tokenization are natural Phase 17B+ refinements, swappable via
  the ``tokenizer`` constructor parameter without touching this module's
  internals.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

TokenizerFn = Callable[[str], list[str]]

_WORD_PATTERN = re.compile(r"\w+")


def default_tokenizer(text: str) -> list[str]:
    """Lowercase, extract ``\\w+`` word tokens. No stemming, no stopword
    removal — see module docstring's "Design decisions" for why this is the
    deliberate v1 default rather than an oversight.
    """
    return _WORD_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Document:
    """One unit of input to ``BM25Index.build()``: an opaque caller-defined
    ``document_id`` (e.g. ``str(Incident.id)`` — this module never knows
    that) paired with the text to index (e.g. ``Incident.canonical_text``,
    also not this module's concern).
    """

    document_id: str
    text: str


@dataclass(frozen=True)
class BM25Config:
    """Scoring-time BM25 parameters. Deliberately separate from
    ``BM25Index`` — see module docstring's "Design decisions".
    """

    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class BM25SearchResult:
    document_id: str
    score: float


class BM25Index:
    """Immutable-after-build corpus statistics for BM25 scoring.

    Holds no scoring parameters (``k1``/``b``) — only postings, document
    lengths, and document count, from which average document length and
    per-term document frequency are derived. See module docstring's "Index
    lifecycle" for the rebuild strategy.
    """

    def __init__(self, *, tokenizer: TokenizerFn = default_tokenizer) -> None:
        self._tokenizer = tokenizer
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_order: list[str] = []
        self._built = False

    @property
    def tokenizer(self) -> TokenizerFn:
        return self._tokenizer

    @property
    def size(self) -> int:
        return len(self._doc_order)

    @property
    def average_document_length(self) -> float:
        if not self._doc_order:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_order)

    def document_frequency(self, term: str) -> int:
        return len(self._postings.get(term, {}))

    def document_length(self, document_id: str) -> int:
        return self._doc_lengths[document_id]

    def postings(self, term: str) -> dict[str, int]:
        """``{document_id: term_frequency}`` for every document containing
        ``term``, or ``{}`` if no document does. Returns a fresh dict (a
        defensive copy) — callers (i.e. ``BM25Retriever``) must not be able
        to corrupt index state through the returned mapping.
        """
        return dict(self._postings.get(term, {}))

    def build(self, documents: Iterable[BM25Document]) -> None:
        """Tokenize and index every document. May be called exactly once.

        Raises ``RuntimeError`` if called a second time on the same
        instance (construct a new ``BM25Index`` to rebuild — see module
        docstring). Raises ``ValueError`` on a duplicate ``document_id``
        within ``documents`` (never silently overwrites or merges, the same
        convention as ``BenchmarkRepository.save`` in Phase 16F).
        """
        if self._built:
            raise RuntimeError(
                "BM25Index.build() may only be called once; construct a new "
                "BM25Index to rebuild"
            )
        for document in documents:
            if document.document_id in self._doc_lengths:
                raise ValueError(f"duplicate document_id {document.document_id!r}")
            tokens = self._tokenizer(document.text)
            self._doc_lengths[document.document_id] = len(tokens)
            self._doc_order.append(document.document_id)
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, {})[document.document_id] = count
        self._built = True


class BM25Retriever:
    """The lexical sibling of ``IncidentSearchService``'s dense retrieval —
    a self-contained ``retrieve(query, limit=...)`` entry point that never
    imports, calls, or knows about dense retrieval.
    """

    def __init__(self, index: BM25Index, *, config: BM25Config | None = None) -> None:
        self._index = index
        self._config = config or BM25Config()

    @property
    def index(self) -> BM25Index:
        return self._index

    @property
    def config(self) -> BM25Config:
        return self._config

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[BM25Document],
        *,
        config: BM25Config | None = None,
        tokenizer: TokenizerFn = default_tokenizer,
    ) -> BM25Retriever:
        """Convenience constructor: build a fresh ``BM25Index`` from
        ``documents`` and wrap it in one call, for the common case where
        nothing else needs to share the index.
        """
        index = BM25Index(tokenizer=tokenizer)
        index.build(documents)
        return cls(index, config=config)

    def retrieve(self, query: str, *, limit: int = 10) -> list[BM25SearchResult]:
        """Rank indexed documents against ``query`` by BM25 score,
        descending, deterministically tie-broken by ``document_id``
        ascending. Returns ``[]`` for an empty index or a query with no
        vocabulary overlap with the corpus — neither is an error. Raises
        ``ValueError`` if ``limit < 1``.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit!r}")
        if self._index.size == 0:
            return []

        query_terms = set(self._index.tokenizer(query))
        average_document_length = self._index.average_document_length
        k1, b = self._config.k1, self._config.b
        scores: dict[str, float] = {}

        for term in query_terms:
            postings = self._index.postings(term)
            if not postings:
                continue
            idf = self._idf(term)
            for document_id, term_frequency in postings.items():
                document_length = self._index.document_length(document_id)
                # average_document_length can only be 0.0 when every
                # document has length 0, in which case no document ever
                # appears in any term's postings (a zero-token document
                # contributes no postings entries) — so this division is
                # never reached with average_document_length == 0.
                length_norm = 1 - b + b * (document_length / average_document_length)
                denominator = term_frequency + k1 * length_norm
                contribution = idf * (term_frequency * (k1 + 1)) / denominator
                scores[document_id] = scores.get(document_id, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return [
            BM25SearchResult(document_id=document_id, score=score)
            for document_id, score in ranked[:limit]
        ]

    def _idf(self, term: str) -> float:
        n = self._index.size
        df = self._index.document_frequency(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))
