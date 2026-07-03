# 18 — Adaptive Routing, Hybrid Retrieval & Confidence Normalization (Phases 17A–18E)

This document covers seven phases that, together, replace dense-only retrieval with a
strategy-aware system now serving production traffic: **17A** builds an independent BM25 lexical
retriever, **17B** fuses it with dense retrieval via Reciprocal Rank Fusion, **18A** builds a
deterministic query router that picks a strategy per query, **18B** wires that router into a
production-facing service, **18C** normalizes each strategy's native score onto a common `[0, 1]`
scale before classification, **18D** benchmarks all of it against the live corpus, and **18E**
wires the whole stack into `/search/incidents`, `/search/debug`, and the investigation
orchestrator's default construction — the production adoption pass. Phases 17A/17B predate 18A–18D
and are technically outside the audited 18A–21H range, but no architecture doc has ever covered
them either, and 18A–18D cannot be explained without them — they are included here as the natural
home. **Adaptive routing is now the production retrieval engine** for `/search/*` and the
investigation orchestrator (Phase 18E) — not an evaluation-only subsystem — though it stays
behaviorally dense-only by default (`routing_enabled=False`) until explicitly opted in.

---

## Phase 17A — BM25 Lexical Retrieval

### Goal

Build a completely independent BM25 lexical retrieval engine — a sibling to dense retrieval, not
an extension of it — operating on plain `(document_id, text)` pairs with zero coupling to the
database, pgvector, or `Incident` models.

### Motivation

Dense and lexical retrieval fail on different, largely non-overlapping query classes: dense
misses rare jargon and exact error codes; lexical trivially catches those but misses paraphrases
dense handles well (doc 16, limitation 1). The module docstring frames each as "independently
valuable, independently measurable, and independently swappable" — BM25 is built standalone
specifically so either retriever can be benchmarked in isolation without coupling them, avoiding
"the exact mistake the Phase 16 evaluation platform was built to prevent: shipping an algorithm
change with no way to attribute its effect to one component or the other."

### Architecture

```python
BM25Document(document_id: str, text: str)          # frozen, input unit
BM25Config(k1: float = 1.5, b: float = 0.75)        # frozen, tuning parameters — separated from the index
BM25SearchResult(document_id: str, score: float)    # frozen, output unit

BM25Index:
    .build(documents)                # one-shot; raises RuntimeError on a second call
    _postings: dict[term, dict[doc_id, term_frequency]]     # inverted index
    _doc_lengths: dict[doc_id, token_count]
    .size, .average_document_length, .document_frequency(term), .document_length(doc_id)

BM25Retriever(index, config=None):
    .retrieve(query, limit=10) -> list[BM25SearchResult]
    .from_documents(documents, config=None, tokenizer=default_tokenizer)   # factory
    _idf(term)                       # log(1 + (n - df + 0.5) / (df + 0.5))

default_tokenizer(text) -> list[str]      # lowercase + \w+ word extraction, no stemming
TokenizerFn = Callable[[str], list[str]]  # swappable tokenizer type
```

### Lifecycle

**Index construction**: for each document, tokenize via the injected tokenizer, record
`_doc_lengths[document_id] = len(tokens)`, and for each unique `(term, count)` in
`Counter(tokens)`, store `_postings[term][document_id] = count`. A second `.build()` call or a
duplicate `document_id` both raise.

**Retrieval**: tokenize the query into a **set** of unique terms (repeats in the query string are
deduplicated — classic, not query-term-frequency-weighted BM25). For each query term present in
the postings: compute `idf = ln(1 + (n - df + 0.5) / (df + 0.5))`; for each `(document_id, tf)` in
that term's postings, compute `length_norm = 1 - b + b * (doc_length / avgdl)` and accumulate
`scores[document_id] += idf * (tf * (k1 + 1)) / (tf + k1 * length_norm)`. Results are sorted by
`(-score, document_id)` (descending score, ascending id for deterministic ties) and truncated to
`limit`. An empty corpus or a query with no vocabulary overlap returns `[]`.

### Design decisions

- **Zero coupling to dense retrieval or the database** — input is opaque `(id, text)` pairs;
  callers own reading `Incident.canonical_text` and converting to UUIDs. This keeps the module
  trivially unit-testable with no DB fixture.
- **Hybrid fusion is deliberately deferred to 17B** — fusing two ranking signals has documented
  failure modes elsewhere in this project (the candidate-merge "hub incident" problem, doc 12; the
  reranker's tendency to discard higher-similarity candidates, doc 13); building fusion before both
  retrievers exist and are independently validated would repeat Phase 16's mistake.
- **Smoothed IDF (`ln(1 + ...)`), not classic Robertson IDF** — the unsmoothed form can go negative
  for terms in more than half the corpus, making a matching document score *worse* than a
  non-matching one; the `+1` (the Lucene convention) guarantees `idf >= 0` at the cost of departing
  from the 1994 paper — a standard, deliberate trade-off.
- **`k1`/`b` live on the retriever, not the index** — scoring parameters are pure constants,
  independent of corpus statistics, so two retrievers with different `k1`/`b` can share one built
  index without re-tokenizing.
- **Deterministic tie-break by `document_id`** ensures stable ordering across runs without relying
  on dict iteration order.
- **Tokenizer is an injected, swappable function** — `default_tokenizer` (lowercase + `\w+`, no
  stemming/stopwords) is the simplest baseline; stemming/stopword-removal/n-grams are deferred.
- **No incremental index updates** — add/update/remove would require postings pruning and `avgdl`
  maintenance with no concrete trigger yet; full-rebuild-and-swap is sufficient until a future phase
  identifies an actual need.

### Interfaces

Imports only the standard library (`math`, `re`, `Counter`, `dataclass`). Public surface:
`BM25Document`, `BM25Config`, `BM25SearchResult`, `BM25Index`, `BM25Retriever`,
`default_tokenizer`, `TokenizerFn`. **Not imported by any route** — freestanding infrastructure
consumed by Phase 17B and later.

### Testing

`tests/unit/test_bm25_search.py` (38 tests) covers: tokenizer behavior (lowercasing, punctuation
stripping, empty/punctuation-only input); index construction (size, lengths, average length,
document frequency, postings returning a defensive copy, rejecting duplicate ids, rejecting a
second `.build()`, handling empty-text documents without breaking `avgdl`); a hand-computed BM25
score for a single term; exact-keyword ranking above non-matching documents; multiple matches
ordered correctly; confirmation that repeated query terms are *not* weighted by query-term
frequency; smoothed IDF never going negative for a term in every document; empty corpus and
unknown-query edge cases; deterministic tie-breaking and ordering stability across repeated calls;
`limit` handling including rejecting non-positive limits; different `k1`/`b` on the same index
producing different scores; the `from_documents` factory; and result-object immutability.

### Risks

No explicit "Risks" section in the docstring. Implicit ones: IDF smoothing departs from the
original Robertson formula (documented trade-off); a caller supplying duplicate `document_id`s gets
a hard `ValueError` rather than silent overwriting (deliberate, to prevent silent data loss).

### Future work

From the docstring: "stemming, stopword removal, or n-gram tokenization are natural Phase 17B+
refinements, swappable via the tokenizer injection point." Hybrid fusion (17B) is the explicitly
named next consumer.

---

## Phase 17B — Hybrid Retrieval (Dense + BM25 via Reciprocal Rank Fusion)

### Goal

Orchestrate the two already-independent, already-validated retrieval engines —
`IncidentSearchService` (dense, pre-16) and `BM25Retriever` (lexical, 17A) — via Reciprocal Rank
Fusion, without modifying either. This module is the *only* thing in the codebase that imports
both; neither retriever imports the other or this module.

### Motivation

Dense similarity scores (`[0, 1]`-ish, doc 08) and BM25 scores (unbounded, corpus- and
query-dependent) live on "fundamentally different, non-comparable scales." Normalizing each to
`[0, 1]` and summing would require a principled way to normalize an unbounded distribution that
doesn't exist. RRF sidesteps this by discarding scores entirely and fusing on **rank** —
comparable and bounded (`1, 2, 3, ...`) regardless of the scoring function that produced it. This
is the standard choice in IR literature (Cormack, Clarke &amp; Buettcher, 2009), and — per the
docstring — deliberately not attempted until both retrievers were independently validated, for the
same reason 17A deferred fusion: this project has already hit the "hub incident" (doc 12) and
reranker-discard (doc 13) failure modes and didn't want to reintroduce either by fusing early.

### Architecture

```python
HybridConfig(dense_limit=25, bm25_limit=25, rrf_k=60.0, final_limit=10)   # fusion-only knobs;
    # does NOT re-expose either retriever's own config (filters, tokenizer, k1/b)
    # __post_init__ validates all four are >= 1 (rrf_k must be > 0)

HybridSearchResult(document_id, rrf_score, dense_rank: int|None, bm25_rank: int|None,
                    dense_result: IncidentSearchResult|None, bm25_result: BM25SearchResult|None)

HybridRetriever(dense, bm25, config=None):
    .retrieve(query, limit=None) -> list[HybridSearchResult]
    _safe_dense_search(query)   # try/except around dense.search(..., call_site="hybrid_retriever")
    _safe_bm25_retrieve(query)  # try/except around bm25.retrieve(...)

_fuse(dense_results, bm25_results, rrf_k, limit)   # private, pure fusion function
```

### Lifecycle

1. Dense search: `dense.search(query, limit=config.dense_limit, call_site="hybrid_retriever")`,
   wrapped in try/except — any exception degrades to `[]`, never aborts.
2. BM25 retrieval: `bm25.retrieve(query, limit=config.bm25_limit)`, same try/except degradation.
3. **Fusion**: build `{document_id: rank}` maps for each side (1-based, first occurrence wins on a
   duplicate); the candidate set is the union of both maps' keys (one entry per distinct id by
   construction — this *is* the deduplication step, not a separate one). For each candidate:
   `score = [1/(rrf_k + dense_rank) if present else 0] + [1/(rrf_k + bm25_rank) if present else 0]`.
   Sort by `(-rrf_score, document_id)`; return the top `limit` (parameter overrides
   `config.final_limit` if supplied; effective limit must be `>= 1`).

RRF formula: `RRF(d) = 1/(k + rank_dense(d))` (0 if absent) `+ 1/(k + rank_bm25(d))` (0 if absent)
— absence contributes zero, never a penalty; a document found by both sums both terms.

### Design decisions

- **Rank-based fusion, not score normalization** — RRF is agnostic to how each retriever produces
  its ranking, so either side can be swapped (different embedding model, different tokenizer) with
  zero change to fusion logic.
- **`HybridConfig` exposes only fusion-specific knobs** — dense's own filters and BM25's own
  tokenizer/`k1`/`b` remain the concern of retrievers constructed before being handed to
  `HybridRetriever`; validation happens once, at construction, via `__post_init__`.
- **`rrf_k = 60.0`** (the literature/Elasticsearch default) controls how much exact rank matters vs.
  mere presence in both lists — larger `k` flattens the curve (favoring documents present in both),
  smaller `k` sharpens it (favoring a #1 rank strongly). Tuning is explicitly out of scope; the
  default is not fitted to gold data.
- **`dense.search()`, never `dense.retrieve()`** — a load-bearing method choice: `.retrieve()`
  would silently perform expansion/reranking, violating the constraint that this module supplies
  only the raw candidate-generation primitive.
- **Graceful degradation on either retriever's failure** — mirrors the project's existing
  reranker-failure-falls-back-to-distance-order convention; one retriever's outage doesn't take
  down hybrid entirely.

### Interfaces

Imports `BM25Retriever`/`BM25SearchResult` (17A) and `IncidentSearchResult`/`IncidentSearchService`
(pre-16). Public surface: `HybridConfig`, `HybridSearchResult`, `HybridRetriever`. **Not imported by
any route** — Phase 18B later wraps this module's results in the production pipeline, but no
`app/api/routes/*.py` file imports it directly.

### Testing

`tests/unit/test_hybrid_search.py` (27 tests) covers: fusing disjoint results from each retriever;
a document present in both appearing exactly once; a hand-verified RRF score for an overlapping
document at `k=60`; `None` dense/BM25 rank and result fields when a document is absent from that
side; descending sort by RRF score with deterministic tie-breaking by `document_id`; `final_limit`
from config vs. an overriding `limit` parameter, and rejection of a non-positive override; the
exact `dense_limit`/`call_site` used for the dense call; graceful degradation when either side
raises (falls back to the other side's results alone); both-empty returning `[]`; config validation
rejecting non-positive `dense_limit`/`bm25_limit`/`rrf_k`/`final_limit`; sensible config defaults
(25/25/60.0/10); and result immutability.

### Risks

No explicit "Risks" section; implicit one: try/except degradation around each retriever means a
silent partial failure (one side down) is possible by design — chosen deliberately so one
retriever's outage doesn't take down hybrid retrieval entirely, but it means a caller cannot
distinguish "both sides healthy, genuinely disjoint results" from "one side silently failed" without
inspecting logs.

### Future work

The docstring frames the next question as "how to consume this" — answered by Phase 18A/18B
(routing integration).

---

## Phase 18A — Adaptive Retrieval Routing Framework

### Goal

Introduce a routing *architecture* — the extension point Phase 17D's benchmarking argued for
(different query types benefit from different strategies) — without committing to any particular
routing algorithm. Ship one simple, deterministic, fully explainable policy
(`DefaultRuleBasedRoutingPolicy`) behind an abstract `RoutingPolicy` interface. Explicitly not in
scope: ML classifiers, LLM-based routers, or tuning the thresholds.

### Motivation

Short keyword queries or exact error signatures favor BM25's exact matching; long, multi-concept
queries benefit from hybrid's fused candidates (Phase 17D's benchmark findings, doc 17). The
module docstring is explicit that "the rule thresholds below are illustrative v1 defaults, not
tuned values" — the goal of this phase is the pluggable architecture, not a well-tuned policy.

### Architecture

```python
RoutingStrategy(str, Enum): DENSE="dense", BM25="bm25", HYBRID="hybrid"
    # values deliberately match Phase 17C's StrategyName/build_strategy() vocabulary

RoutingSignals(frozen): query, token_count, has_exact_error_signature, has_stack_trace,
                        has_quoted_identifier, lexical_density: float

RoutingDecision(frozen): strategy: RoutingStrategy, reason: str, signals: RoutingSignals

RoutingPolicy (ABC): decide(query, signals) -> RoutingDecision

DefaultRuleBasedRoutingPolicy(RoutingPolicy):
    SHORT_QUERY_TOKEN_THRESHOLD = 3
    LONG_QUERY_TOKEN_THRESHOLD = 12

RoutingEngine(policy):
    .route(query) -> RoutingDecision       # computes signals itself, then delegates to policy
    .policy                                 # property

extract_routing_signals(query) -> RoutingSignals   # pure, regex/tokenization only, no I/O
```

Signal detection: `_STACK_TRACE_PATTERNS` (Python traceback marker, Java-style frame, Python
`File "...", line N`); `_ERROR_SIGNATURE_PATTERN` (camelcase word ending Error/Exception/Fault/
Timeout, or a ticket-like id such as `KAFKA-17`); `_QUOTED_IDENTIFIER_PATTERN` (backtick or
no-space-inside quoted string). `lexical_density = unique_tokens / total_tokens` (0 if empty).

### Lifecycle

`DefaultRuleBasedRoutingPolicy.decide()` checks, in fixed priority order, **first match wins**:

1. `has_stack_trace` → BM25 ("exact frame text; embeddings blur stack traces")
2. `has_exact_error_signature` → BM25 ("precise tokens... exactly what BM25's idf-weighted exact
   match rewards")
3. `has_quoted_identifier` → BM25 ("explicit signal the user wants an exact string match")
4. `token_count <= 3` → BM25 ("very short queries read as keyword lookups")
5. `token_count >= 12` → HYBRID ("long, multi-clause queries read as multi-concept")
6. otherwise → DENSE ("a natural-language, single-concept query")

`RoutingEngine.route(query)` always computes `RoutingSignals` itself (never delegates extraction to
the policy) and passes both raw `query` and `signals` to `decide()` — so every policy sees signals
computed the same way, and a future LLM-based policy can still read the raw query text.

### Design decisions

- **Five specific signals chosen** because stack traces/error signatures/quoted identifiers are all
  "this must match exactly" — the opposite of what dense embeddings excel at and exactly what BM25
  rewards; token count is the simplest complexity proxy; `lexical_density` is computed even though
  the default policy never reads it, deliberately, so a future policy can use it without a
  signal-extraction change.
- **Thresholds (3, 12) are explicitly not tuned** — reasonable, explainable defaults, not fit to any
  gold dataset; a future phase can replace or parameterize `DefaultRuleBasedRoutingPolicy` without
  touching `RoutingEngine`.
- **Priority order, not scoring/voting** — "first matching rule wins" is a one-sentence description
  of the whole policy's behavior; scoring/voting is explicitly deferred as a step toward ML-style
  routing.
- **Narrowest-first, broadest-last ordering** mirrors 19B's planner: auth/network/stack-trace
  signals are rarely ambiguous; token-count rules are broad enough to catch almost anything, so
  they're checked last.
- **`RoutingDecision` always carries its originating `signals`** — a decision that can't be traced
  to its inputs isn't explainable.
- **`RoutingEngine` never touches a retriever** — its output is a label
  (`RoutingStrategy.DENSE`/`BM25`/`HYBRID`), never a retriever instance; dispatching is the caller's
  job (demonstrated only in this phase's own integration tests). `RoutingStrategy`'s values
  deliberately match Phase 17C's `StrategyName` vocabulary so a caller *can* dispatch directly via
  `build_strategy(decision.strategy.value, ...)`.

### Interfaces

Standard library only (`re`, `abc`, `dataclasses`, `enum`). Public surface: `RoutingStrategy`,
`RoutingSignals`, `RoutingDecision`, `RoutingPolicy`, `DefaultRuleBasedRoutingPolicy`,
`RoutingEngine`, `extract_routing_signals`. **Not imported by any route** — Phase 18B is the
integration.

### Testing

`tests/unit/test_routing.py` (40 tests) covers: stack-trace detection (Python traceback, Java-style
frame, Python file/line frame, negative case); error-signature detection (camelcase, Exception
suffix, ticket-like id, negative case); quoted-identifier detection (backtick, double-quoted,
negative case for a plain quoted sentence); token count, empty-query zero-density, lexical-density
hand-computation, original query preservation, and immutability; each individual rule in isolation;
three explicit priority-order tests (stack trace beats short-query rule, error signature beats
long-query rule, quoted identifier beats short-query rule); determinism of both `decide()` and
`route()`; policy injection and swappability (a stub policy changes the decision without changing
the engine); confirmation `RoutingStrategy` values match Phase 17C's `StrategyName` vocabulary; and
two tests demonstrating dispatch via `build_strategy()` in the caller (not inside this module).

### Risks

Thresholds are documented as illustrative, not tuned — no gold-set validation backs the specific
values 3 and 12.

### Future work

From the docstring: "a future phase that wants to tune them can do so by replacing
`DefaultRuleBasedRoutingPolicy` (or parameterizing it) without touching `RoutingEngine`."

---

## Phase 18B — Adaptive Routing Integration

### Goal

Activate 18A's `RoutingEngine` as the mechanism choosing between Dense (pre-16), BM25 (17A), and
Hybrid (17B) for every incoming query — without modifying any of the three, without modifying
`IncidentSearchService`, and without touching 18A's policy/rules. Produce the integration point,
ready to be adopted; adoption itself is explicitly out of scope for this phase.

### Motivation

Phase 17D's benchmarking showed different query types benefit from different strategies. 18A built
the pluggable routing architecture; 18B wires it into a production-shaped service while keeping
today's callers completely unaffected — routing is opt-in, off by default.

### Architecture

```python
RoutedSearchConfig(routing_enabled: bool = False)     # single opt-in switch

RoutingObservation(frozen): query, call_site, routing_enabled, policy_strategy,
    effective_strategy, reason, override_reason: str|None, signals: RoutingSignals

RoutedSearchService(dense, bm25=None, hybrid=None, routing_engine=None, config=None):
    .retrieve(query, *, limit=10, source_type=None, tags=..., owner=..., repo=..., source=...,
              state=..., expand=False, rerank=False, call_site=None) -> list[IncidentSearchResult]
    .last_observation                          # most recent RoutingObservation
    .db                                         # from dense service
    @staticmethod confidence_for(results)       # delegates to IncidentSearchService.confidence_for

_ProductionCandidatePipeline(llm_service=None):   # private, strategy-agnostic expand/merge/rerank
    .run(query, generate, limit, expand, rerank, call_site, strategy_label)
    _expand_query, _merge, _rerank, _payload      # identical algorithm dense already uses

_EXPAND_CANDIDATE_LIMIT = 25   # pool size when expanding or reranking, same as dense's own default
```

### Lifecycle

```
retrieve(query, *, limit, expand, rerank, call_site, filters...)
  1. decision = routing_engine.route(query)          # ALWAYS computed, even if routing disabled
  2. effective_strategy =
       DENSE, override="routing disabled"              if not config.routing_enabled
       DENSE, override="query has filters..."           elif any filter supplied
       decision.strategy, override=None                 else
  3. record RoutingObservation; log it (structured logger, key
       "retrieval.routed_search.routing_decision")
  4. dispatch:
       DENSE  -> dense.retrieve(...unchanged...)                       [pre-16, entire pipeline reused]
       BM25   -> _pipeline.run(..., generate=_bm25_generate, ...)
       HYBRID -> _pipeline.run(..., generate=_hybrid_generate, ...)
  5. return list[IncidentSearchResult]   # identical shape regardless of strategy
```

The routing decision affects step 4 **only** — which primitive produces the initial candidate
pool. Expansion, merge, reranking, and confidence classification are the same algorithm on every
branch: the dense branch delegates entirely to `IncidentSearchService.retrieve()` (unmodified);
the BM25/Hybrid branches share one `_ProductionCandidatePipeline`, parameterized by a
`generate(phrase, limit)` callable, implementing the identical 25-candidate pool sizing, "keep
lowest distance on repeat" merge rule, reranker payload shape, and reranker-failure fallback to
distance order that dense's own pipeline uses. Unlike Phase 17C/17D's evaluation-only adapters
(which only needed candidate ids), BM25/Hybrid candidates here are converted to real, DB-fetched
`Incident` objects (via `_fetch_incident_result`/`_hybrid_to_incident_result`) before merging, so
downstream code sees the exact same populated `IncidentSearchResult` shape regardless of strategy.

### Design decisions

- **`routing_enabled=False` by default** — with routing off, `.retrieve()` always takes the dense
  branch with the caller's exact arguments unchanged; today's production behavior is untouched
  until a caller explicitly opts in.
- **Filters force dense even when routing is enabled** — BM25/Hybrid were built without filter
  support (`source_type`/`tags`/`owner`/`repo`/`source`/`state` — 17A/17B's own scope decisions).
  Silently dropping a caller's filter would be a correctness bug, not a routing optimization
  question, so this check happens *before* consulting the policy's decision at all.
- **Signals are computed and recorded even when routing is disabled** — cheap, pure regex/
  tokenization with no I/O, so "shadow observability" is free: what would routing have chosen on
  real traffic can be evaluated before ever flipping the switch on.
- **One shared candidate pipeline for BM25 and Hybrid**, not two near-duplicates — parameterized
  solely by the `generate()` callable.
- **`RoutingObservation` exposed as a plain property**, not just a log line — a caller or test can
  inspect the most recent routing decision directly with no log-scraping.

### Interfaces

Imports `Incident` (db), `BM25Retriever` (17A), `HybridRetriever`/`HybridSearchResult` (17B),
`LLMService`, `DefaultRuleBasedRoutingPolicy`/`RoutingEngine`/`RoutingSignals`/`RoutingStrategy`
(18A), `IncidentSearchResult`/`IncidentSearchService` (pre-16). Public surface:
`RoutedSearchConfig`, `RoutingObservation`, `RoutedSearchService`. **Not imported by any route** —
this phase produces the integration point, not the adoption.

### Testing

`tests/unit/test_routed_search.py` (35+ parameterized tests) covers: routing-disabled delegating to
dense unchanged (including a regression test matching calling dense directly) while still recording
an observation; dense/BM25/Hybrid dispatch when routing is enabled, including missing-incident
handling (skipped, not crashed) and an unconfigured retriever raising; six parameterized tests
confirming every individual filter (owner/repo/source_type/tags/source/state) forces dense despite
the policy's decision, with `override_reason` set; expansion compatibility (candidate merge keeping
lowest distance; Hybrid expansion falling back to the original query alone with no LLM); reranking
compatibility (identical payload shape, LLM-failure fallback to distance order); confidence
compatibility across all three strategies; confirmation that no downstream consumer can tell which
strategy produced a result (identical `IncidentSearchResult` shape); and two end-to-end tests using
the real `DefaultRuleBasedRoutingPolicy` (not a stub) confirming a short query routes to BM25 and a
medium signal-free query routes to dense.

### Risks

- BM25/Hybrid's lack of filter support is a structural limitation, not a bug to fix inside this
  module — documented explicitly so a future phase doesn't expect filters to work through routed
  BM25/Hybrid.
- **Distance/score sign confusion for a future integration**: this service stores `distance = -score`
  for BM25/Hybrid candidates as an internal sorting convenience. A future integration wiring 18C's
  confidence normalization into this service must pass the strategy's *original* score
  (`BM25SearchResult.score`, `HybridSearchResult.rrf_score`), re-negating `IncidentSearchResult.distance`
  back to the raw score — the docstring calls this out explicitly as "an easy mistake for a future
  integration to make silently."

### Future work

Per the docstring: "this phase produces the integration point, ready to be adopted, not the
adoption itself." Wiring `RoutedSearchService` into `app/api/routes/search.py` (or the investigation
agents) is left to a later, unspecified phase.

---

## Phase 18C — Strategy-Aware Confidence Normalization

### Goal

Normalize each retrieval strategy's native score to a common `[0.0, 1.0]` range before it reaches
the existing, unmodified `app.services.confidence.classify_confidence` (thresholds 0.40/0.55,
calibrated for dense only, doc 14). This phase is architecture only — explicitly **not** statistical
calibration (Platt scaling, isotonic regression, temperature scaling), not ML, and not a change to
`classify_confidence` or the routing modules.

### Motivation

Dense scores are cosine distances (`[0, 2]`, usually near `[0, 1]`); BM25 scores are unbounded,
non-negative, idf-weighted sums with no fixed ceiling; Hybrid/RRF scores are sums of `1/(k+rank)`
terms — tiny, typically well under 0.04 for `k=60`. Feeding any of these directly into
`classify_confidence` (thresholds calibrated against dense similarity only) is meaningless for
BM25/Hybrid: a raw BM25 score of 3.5 has no relationship to the `[0, 1]` range those thresholds
assume, and would always classify HIGH regardless of actual retrieval quality. This phase inserts a
per-strategy normalization layer between "the strategy's native score" and the shared classifier.

### Architecture

```python
BM25_MIDPOINT = 4.0        # saturating midpoint, order-of-magnitude from Phase 17C/17D benchmarks
HYBRID_MIDPOINT = 0.016    # saturating midpoint, from RRF's bounded max at rrf_k=60

NormalizedConfidence(frozen): value: float, level: str, strategy: RoutingStrategy,
                               raw_score: float | None   # raw_score/strategy: traceability only

ConfidenceNormalizer (ABC): strategy: RoutingStrategy (class attr); normalize(raw_score) -> NormalizedConfidence

DenseConfidenceNormalizer:   value = clamp(1.0 - raw_score, 0, 1)                    # distance in
BM25ConfidenceNormalizer:    value = clamp(raw_score / (raw_score + BM25_MIDPOINT), 0, 1)
HybridConfidenceNormalizer:  value = clamp(raw_score / (raw_score + HYBRID_MIDPOINT), 0, 1)

_NORMALIZERS: dict[RoutingStrategy, ConfidenceNormalizer]     # registry
get_confidence_normalizer(strategy) -> ConfidenceNormalizer   # lookup, raises if unregistered
register_confidence_normalizer(strategy, normalizer) -> None  # swap/extend the registry
normalize_confidence(strategy, raw_score) -> NormalizedConfidence   # public entry point
```

### Lifecycle

`normalize_confidence(strategy, raw_score)` looks up the strategy's normalizer in `_NORMALIZERS` and
calls `.normalize(raw_score)`. **Dense**: `value = clamp(1 - distance, 0, 1)` — the exact transform
`IncidentSearchResult.similarity_score` already performs, so dense's normalized value and
classification are numerically identical to today's production behavior (backward compatible).
**BM25/Hybrid**: both use the same saturating form `score / (score + midpoint)` — 0 at score=0, 0.5
at the midpoint, monotonically approaching (never reaching) 1.0 — re-scaled per strategy by a single
midpoint constant chosen from the order of magnitude observed in Phase 17C/17D benchmarks, not
fitted to gold data. `value` and `level` (via the unmodified `classify_confidence`) are then packed
into an immutable `NormalizedConfidence`, with `strategy`/`raw_score` retained only for traceability.

### Design decisions

- **Classification is not reimplemented here** — every strategy's normalized value is judged by the
  literal same `classify_confidence` thresholds; there is no way for BM25's and Dense's
  classification logic to silently drift apart, because there is only one classifier function.
- **`raw_score` is the strategy's own native score, not `IncidentSearchResult.distance`** — Phase
  18B stores `distance = -score` for BM25/Hybrid as an internal sorting convenience, which is an
  implementation detail of 18B's pipeline, not a semantically meaningful distance. This module's
  own docstring flags this as "an easy mistake for a future integration to make silently" (repeated
  from 18B's own Risks section).
- **Registry, not an if/elif chain** — `get_confidence_normalizer`/`register_confidence_normalizer`
  mirror Phase 17C's `build_strategy()` name-to-object lookup and Phase 18A's single-swap-point
  `RoutingPolicy` interface; a future phase can register a Platt-scaled `DenseConfidenceNormalizer`
  once labeled data exists, without touching call sites.
- **Statistical calibration is deliberately deferred** — Platt scaling/isotonic regression/
  temperature scaling require a labeled dataset across the operating range large enough to avoid
  overfitting; the project's current labeled data is Phase 17C/17D's 36 hand-authored gold
  queries, "nowhere near enough to fit three independent per-strategy calibration curves with
  confidence they reflect reality rather than noise." This phase builds only the part that doesn't
  depend on dataset size: a stable interface a properly-resourced future phase can implement
  against.
- **Consumers are expected to read only `.value`/`.level`** — `strategy`/`raw_score` exist purely
  for debugging/future-calibration inspection; downstream behavior must not branch on which
  strategy produced a confidence.

### Interfaces

Imports `classify_confidence` (`app.services.confidence`, pre-16/Phase 6A) and `RoutingStrategy`
(18A). Public surface: `NormalizedConfidence`, `ConfidenceNormalizer`,
`DenseConfidenceNormalizer`/`BM25ConfidenceNormalizer`/`HybridConfidenceNormalizer`,
`get_confidence_normalizer`, `register_confidence_normalizer`, `normalize_confidence`. **Not
imported by any route or service** — Phase 18D's benchmark script is its only consumer so far.

### Testing

`tests/unit/test_confidence_normalization.py` (41 tests) covers: Dense normalization (distance 0 →
full confidence HIGH; distance 1 → zero confidence LOW; exact match to the `similarity_score`
formula; `None` raw_score → 0; out-of-range clamping); BM25 normalization (zero → zero; at the 4.0
midpoint → 0.5; large scores approaching but never reaching 1.0; negative scores treated as zero;
monotonicity); Hybrid normalization (zero → zero; at the 0.016 midpoint → 0.5; a typical rank-1 RRF
score landing in a sane range); output clamping to `[0, 1]` for all three normalizers under extreme
inputs (`None`, negative, huge); shared-classifier threshold behavior including exact-boundary
cases; strategy-independence (a downstream consumer reading only `.value`/`.level` sees identical
behavior for equivalent confidence regardless of originating strategy); Dense backward-compatibility
against the historical `similarity_score` formula; and registry mechanics (`get_`/`register_`
swapping a normalizer without changing call sites, `NormalizedConfidence` immutability).

### Risks

The module's own "Risks discovered" section: confusing `distance` with the strategy's native
`raw_score` is called out explicitly as "an easy mistake for a future integration to make silently"
— any future code path wiring this module into `RoutedSearchService` must re-negate the stored
distance back to BM25's/Hybrid's original score before normalizing, not use the distance directly.

### Future work

Per the docstring: "a future, properly-resourced calibration phase can implement [statistically fit
normalizers] against [this] stable interface... without anything downstream (the classifier, the
investigation agent) needing to change when a heuristic normalizer is swapped for a fitted one" —
via `register_confidence_normalizer`, once labeled data at sufficient scale exists.

---

## Phase 18D — Benchmark Evaluating 18A/18B/18C

### Goal

Evaluate Adaptive Routing (18A/18B) and Confidence Normalization (18C) against the live corpus and
Phase 17C's 36-query gold dataset, across three configurations run through the same harness at the
same `k=10, expand=True, rerank=True`: **A** (Dense, routing disabled), **B** (Hybrid, routing
disabled — "always hybrid"), **C** (Adaptive Routing enabled, `DefaultRuleBasedRoutingPolicy`).

### Motivation

Phase 17C/17D's benchmarking established that different query types benefit from different
strategies. This script measures whether 18A/18B's routing framework actually selects appropriate
strategies per query, and whether 18C's confidence normalization produces usable, quality-correlated
confidence levels across all three strategies — informing whether the approach is worth
productionalizing.

### Architecture

Evaluation-only wrappers: `_CostTrackingLLMService` (monkeypatches the OpenAI client to record
token usage and estimate cost) and `_RetryingLLMService` (exponential backoff up to 6 retries on
transient rate-limit errors, mirroring Phase 17D's benchmark runner). Constants:
`GOLD_PATH = tests/eval/gold/phase17c_benchmark_v1.json`, `REPO_DIR = .benchmarks/phase18d`, `K=10`,
illustrative cost rates `PROMPT_COST_PER_1K=0.00015`/`COMPLETION_COST_PER_1K=0.0006` (approximate
GPT-4o-mini rates, explicitly "not guaranteed current").

### Lifecycle

1. Load the 36-query gold dataset; build a BM25 index over the live corpus.
2. Construct the three services: A = `RoutedSearchService(..., RoutedSearchConfig(routing_enabled=False))`;
   B = Phase 17D's `HybridProductionAdapter` ("always hybrid"); C =
   `RoutedSearchService(..., bm25=..., hybrid=..., RoutedSearchConfig(routing_enabled=True))`.
3. Run all three through Phase 16's harness (`evaluate(dataset, service, k=10, expand=True,
   rerank=True)`); compare LLM call counts/tokens/estimated cost across configs.
4. Routing analysis (pure, free — no LLM/DB): route every gold query via `routing_engine.route()`
   directly, recording strategy/reason/token_count; aggregate strategy distribution, rule
   utilization, per-strategy average token counts, and routing latency in microseconds.
5. Retrieval-only latency (expand/rerank off, zero LLM cost) across all three services.
6. Confidence-normalization latency: 1000 calls to `normalize_confidence(DENSE, 0.3)`.
7. Full-pipeline latency for Config C only (expand+rerank on).
8. Confidence analysis (Config C only): for each query, compute normalized confidence and correlate
   its LOW/MEDIUM/HIGH level against that query's recall@K and MRR.
9. Regression analysis (Phase 16F): A-vs-C and B-vs-C via `compare_runs()`.
10. Save all artifacts to `.benchmarks/phase18d/`: `dense.json`, `hybrid.json`, `routed.json` (full
    benchmark runs), `routing_records.json`, `confidence_records.json`.

Example routing record: `{"query_id": "v2-lex-02", "strategy": "hybrid", "reason": "long,
multi-clause queries read as multi-concept...", "token_count": 12}`. Example confidence record:
`{"query_id": "v2-lex-01", "strategy": "dense", "value": 0.7198, "level": "HIGH", "recall_at_k":
1.0, "reciprocal_rank": 1.0}` — HIGH confidence correlating with perfect recall/MRR for that query.

### Design decisions

- **Real OpenAI calls, deliberately** — expand/rerank are left on so there is real LLM cost data to
  compare across configurations, not just retrieval-quality metrics.
- **Routing/confidence-normalization analyses are run separately from the harness** because both
  are pure and free (no DB, no LLM) — measuring them via 36 direct calls is cheaper and more precise
  than inferring them from harness runs.
- **Same gold dataset, same k, same expand/rerank settings across all three configs** so metric
  differences are attributable to strategy choice alone, not confounded by evaluation-setting
  differences.

### Interfaces

Imports from `app.evaluation.benchmark` (`FileBenchmarkRepository`, `compare_runs`,
`create_benchmark_run`), `.gold_loader`, `.harness`, `.production_pipeline`
(`HybridProductionAdapter`), `.retrieval_strategies` (`load_bm25_retriever`),
`app.services.confidence_normalization`, `.embedding_service`, `.hybrid_search`, `.llm_service`,
`.routed_search`, `.routing`, `.search`, `app.db.session`. This is a standalone script
(`python scripts/run_phase18d_benchmark.py`), not a library module — no downstream consumer.

### Testing

No dedicated unit test file exists (this is a benchmark script, not library code); the script
itself, run against the live corpus and gold dataset, is the validation mechanism, producing
artifacts for manual inspection and comparison.

### Risks

Real OpenAI API calls carry real, non-trivial cost; the cost-rate constants are explicitly labeled
"illustrative, approximate... NOT guaranteed current"; persistent API outages (beyond the 6-retry
backoff) will fail the benchmark run outright.

### Future work

This benchmark's original open question — whether adaptive routing should be productionalized —
has been resolved: it now is (see "Integration status" below). What remains open: whether the
routing thresholds (3/12 tokens) need tuning against production traffic, and — if confidence
normalization doesn't show the expected correlation with retrieval quality — whether a future
calibration phase should implement statistical fitting per doc 18C's registry extension point.

---

## Phase 18E — Production Adoption

### Goal

Wire `RoutedSearchService` into the actual production request path — `/search/incidents`,
`/search/debug`, and `MultiAgentInvestigationOrchestrator`'s default construction — so adaptive
routing is the retrieval engine those callers use, not a parallel, evaluation-only path sitting
next to it.

### Motivation

Every phase through 18D shipped a fully-built, fully-tested `RoutedSearchService` that nothing in
production ever called — see this document's original "Integration status," which stated flatly
that no file under `app/api/routes/` imported `routing`, `routed_search`, `hybrid_search`, or
`bm25_search`. This phase closes that gap without introducing a new algorithm: it is entirely
construction and wiring, reusing Phase 18B's `RoutedSearchService.retrieve()` and the shared
`app.services.candidate_pipeline.CandidatePipeline` (doc-adjacent simplification pass) exactly as
they were.

### Architecture

```python
# app/services/search_factory.py — the single production construction point
DEFAULT_CANDIDATE_LIMIT  # (unrelated; see app.services.candidate_pipeline)

_bm25_cache: BM25Retriever | None   # process-local, lazily built, thread-safe (double-checked lock)

get_bm25_retriever(db) -> BM25Retriever        # builds once per process, reuses thereafter
build_routed_search_service(db, *, llm_service=None) -> RoutedSearchService
    # dense = IncidentSearchService(db, llm_service=llm_service)
    # bm25  = get_bm25_retriever(db)              (cached)
    # hybrid = HybridRetriever(dense, bm25)
    # config = RoutedSearchConfig(routing_enabled=settings.search_routing_enabled)
```

```python
# app/core/config.py
Settings.search_routing_enabled: bool = False   # env var SEARCH_ROUTING_ENABLED; opt-in switch
```

Two methods were added to `RoutedSearchService` itself (`app/services/routed_search.py`) to make it
a drop-in replacement for every call shape `IncidentSearchService` previously supported:

```python
RoutedSearchService.search(query, *, limit=10, source_type=None, ..., call_site=None)
    -> list[IncidentSearchResult]
    # delegates to self.retrieve(..., expand=False, rerank=False) — already documented (Phase 18B)
    # as behaviorally identical to a plain search for whichever strategy is selected.

RoutedSearchService.search_debug(query, *, owner=None, repo=None, source=None, state=None,
                                  call_site=None) -> list[IncidentSearchResult]
    # delegates to self.retrieve(query, limit=5, expand=True, rerank=True, ...) — the same
    # "canonical pipeline, 5 results" alias IncidentSearchService.search_debug() already was.
```

### Lifecycle

**`/search/incidents`** and **`/search/debug`** (`app/api/routes/search.py`): construct
`build_routed_search_service(db)` (optionally with an `LLMService()` for `/debug`'s
expand+rerank), then call `.search(...)` / `.search_debug(...)` exactly where they previously
called the same-named methods on a plain `IncidentSearchService`. No new endpoints; no change to
either route's request/response schema.

**`MultiAgentInvestigationOrchestrator`** (Phase 19D, `app/services/investigation_orchestrator.py`):
its `__init__` now defaults `search_service` to `build_routed_search_service(db,
llm_service=self.llm_service)` instead of a plain `IncidentSearchService(db)`, when no explicit
`search_service` is passed. Since `HypothesisEvaluator` (Phase 19A) is constructed from
`self.search_service` and calls `.search()` on it for each hypothesis's evidence lookup, **both**
the orchestrator's initial `.retrieve()` call and every per-hypothesis evidence `.search()` call
now go through routing when enabled — investigations, not just top-level search, benefit
adaptively. A caller that passes its own `search_service` explicitly (e.g.
`app/api/routes/evaluation.py`'s `_build_orchestrator`, which pins a plain dense
`IncidentSearchService` for reproducible benchmarking) is unaffected — this only changes the
*default*.

**BM25 index lifecycle**: `get_bm25_retriever` builds the index once, lazily, on the first call
that needs it in a given process, and reuses the same immutable `BM25Retriever` for every
subsequent call — building it per-request would make every search O(corpus size) regardless of
which strategy the router picks (see `bm25_search`'s own docstring: indexing is a one-shot,
whole-corpus operation). Consequence: newly-ingested incidents are not lexically searchable via
BM25/Hybrid until the process restarts; dense search is unaffected (it reads the database on every
call). `reset_bm25_cache()` exists for tests; there is no automatic production invalidation trigger
yet (see Future work).

### Design decisions

- **`routing_enabled` defaults to `False`, sourced from `Settings.search_routing_enabled`
  (env var `SEARCH_ROUTING_ENABLED`)** — preserves dense-only behavior exactly, out of the box,
  until an operator explicitly opts in. This satisfies the same "purely additive" guarantee Phase
  18B's own config already established, now actually exercised in production.
- **One shared construction point (`build_routed_search_service`), not duplicated per caller** —
  `/search/incidents`, `/search/debug`, and the orchestrator's default all call the same factory
  function, so "how do we build Dense+BM25+Hybrid+routing" exists in exactly one place.
- **`.search()`/`.search_debug()` added to `RoutedSearchService` rather than inlining the
  expand=False/rerank=False call at each call site** — keeps `RoutedSearchService` a complete,
  symmetric replacement for `IncidentSearchService`'s public surface, and keeps
  `HypothesisEvaluator` (which calls `.search()`, never `.retrieve()`, by Phase 19A's own design)
  working unmodified against either backend.
- **BM25 caching is process-local and lazy, not eager-at-startup** — avoids paying the full-corpus
  indexing cost for processes that never receive a request needing it (e.g. a worker that only
  handles ingestion), and avoids blocking application startup on a database read.
- **Filters still force dense, unconditionally** — Phase 18B's own rule (BM25/Hybrid don't support
  `source_type`/`tags`/`owner`/`repo`/`source`/`state`) is unchanged and unconditionally still
  applies through `/search/incidents`' filter parameters.

### Interfaces

New module: `app/services/search_factory.py`, importing `app.core.config`, `app.db.models`,
`app.services.bm25_search`, `app.services.hybrid_search`, `app.services.llm_service`,
`app.services.routed_search`, `app.services.routing`, `app.services.search`. Consumed by
`app/api/routes/search.py` and `app/services/investigation_orchestrator.py`. `RoutedSearchService`
gained `.search()`/`.search_debug()` (no new imports needed — both delegate to the existing
`.retrieve()`).

### Testing

`tests/api/test_search_api.py` (new): routing-disabled parity for both `/search/incidents` and
`/search/debug` (dense called with the expected `expand`/`rerank` flags, BM25/Hybrid never
touched, response body matches the dense backend's canned results exactly); filters forcing dense
even with `routing_enabled=True`; routing-enabled tests proving a short query is actually routed to
and executed by BM25 (dense never called) and a long query to Hybrid, in both cases asserting
`RoutedSearchService.last_observation` reflects the real decision. `tests/unit/test_investigation_orchestrator.py`
(extended): the orchestrator's default `search_service` is a real `RoutedSearchService` instance,
not a plain `IncidentSearchService`; an explicitly-passed `search_service` bypasses the routed
default entirely (proving the evaluation platform's pinned-dense callers are unaffected); and an
end-to-end test showing both the initial `retrieve()` and a hypothesis's evidence `search()` route
to BM25 for a short problem statement, with `last_observation` populated accordingly.

### Risks

- BM25 index staleness after ingestion (see Lifecycle) is a known, accepted limitation, not yet
  mitigated by any invalidation hook.
- The first request in a process that needs BM25/Hybrid pays the full corpus-indexing cost
  synchronously (inside that request's handling), not amortized at startup — a cold-start latency
  spike on whichever request happens to trigger it first.
- `search_routing_enabled` is a single global switch — no per-request or per-tenant override exists;
  enabling it affects every `/search` and orchestrator call in the process at once.

### Future work

A cache-invalidation/rebuild hook for the BM25 index (on ingestion completion, or on a schedule);
pre-warming the index at application startup instead of on first request; per-request or gradual
(percentage-based) rollout of `routing_enabled` instead of a single global switch.

---

## Integration status

Adaptive routing is now the production retrieval engine for `/search/incidents`, `/search/debug`,
and the investigation orchestrator's default construction (Phase 18E, above) — not a parallel,
evaluation-only path. `routing_enabled` defaults to `False` (`Settings.search_routing_enabled`),
so out-of-the-box behavior is unchanged from dense-only retrieval until explicitly opted in via the
`SEARCH_ROUTING_ENABLED` environment variable. `scripts/run_phase18d_benchmark.py` and the
evaluation-only adapters in `app/evaluation/{overlap_analysis,production_pipeline,retrieval_strategies}.py`
remain separate, additional consumers of the same underlying classes, used for benchmarking rather
than serving requests.

**Since Phase 23B/23C:** both `/search/incidents` and `/search/debug` require
`Authorization: Bearer <API_KEY>` and share one rate-limit bucket, `RATE_LIMIT_SEARCH_PER_MINUTE`
(default 100/min) — the highest limit in the API, reflecting that dense/BM25/hybrid retrieval is
comparatively cheap next to the LLM-heavy `/agent` and `/evaluation/full` endpoints. See doc 23.
</content>
