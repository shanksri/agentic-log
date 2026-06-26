# Retrieval V2.5 Implementation Plan

Follow-up to the retrieval design review (low similarity scores 0.25–0.30,
weakly-related top results for queries like "database timeout during peak
traffic"). This plan sequences the four priorities into shippable phases,
each with impact, complexity, risk, and how to measure success.

## Sequencing note (read first)

The priorities as given are: (1) canonical text cleanup, (2) promote
reranking/expansion, (3) hybrid retrieval, (4) eval methodology. For
*execution order*, two adjustments are worth considering:

- **Build a minimal eval harness before changing anything.** Without a
  before/after baseline, none of phases 1–3 can be validated — you'll be
  guessing whether scores improved. Treat a small (~20–30 query) gold set as
  "V2.5.0", done first, then expand it into the full methodology (phase 4)
  once real changes start landing.
- **Hybrid retrieval (3) probably belongs before reranking (2).** Reranking
  quality is bounded by the candidate set it sees — if the dense+lexical
  candidate pool is still missing the relevant incident entirely, no amount
  of reranking fixes that. Improving candidate recall first makes the
  reranking investment pay off more.

Recommended order: **eval baseline → canonical text cleanup → hybrid
retrieval → reranking/expansion promotion → full eval methodology rollout.**
The sections below are written per-priority as requested, but flag where
this reordering matters.

---

## Phase 0 (prerequisite): Minimal eval baseline

Before touching `_canonical_text`, `search()`, or adding hybrid retrieval,
snapshot current behavior so every later change can be measured against it.

- Curate 20–30 (query, expected incident ids) pairs. Sources: `is_gold_labeled`
  resolved incidents (use their title/symptoms to generate a natural-language
  query via LLM, then verify the original incident is the expected hit),
  plus a handful of hand-written "hard" queries (the DB timeout case, and
  similar paraphrase-style queries with no literal overlap).
- Tag each query by type: *lexical-overlap* (shares words with target doc),
  *paraphrase/semantic* (no shared vocabulary), *multi-concept*, and
  *no-match-expected* (negative controls — nothing in the corpus should be
  highly relevant).
- Run current `search()` against this set, record Recall@5, Recall@10, MRR,
  NDCG@10, and the raw similarity score distribution, broken out by query
  type. Store as `eval_baseline_v2.4.json` (or similar) tagged with the
  current `embedding_model_name`.

This is cheap (no production code change) and de-risks everything downstream.

---

## 1. Canonical text cleanup

**Change:** Restructure `_canonical_text()` in `github_normalizer.py`:
- Drop the raw dict repr of `Environment` (`f"Environment: {environment}"`)
  — replace with a short natural-language phrase (e.g., repo/owner only,
  phrased as prose rather than a Python dict literal).
- Front-load the most discriminating fields — title, symptoms, incident
  type, affected components, severity — ahead of the description, since
  MiniLM's 256-token window means whatever comes last is most likely to be
  truncated or down-weighted.
- Truncate description more aggressively (e.g., ~1,500 chars instead of
  4,000) — the extra length isn't being seen by the model anyway, it's just
  diluting the part that is.
- Reduce repeated boilerplate labels that are identical across every
  document ("Title:", "Status:", "Tags:") — these add shared signal that
  compresses variance between documents without adding any.

**Backfill:** Any change to `_canonical_text` changes `text_hash`, which
`_is_embedding_fresh()` will detect — but only on the *next ingestion run*
for incidents already stored. Existing incidents need a one-off backfill
job that recomputes `canonical_text` from stored normalized fields (or
re-normalizes from `RawDocument.payload`) and re-embeds.

**Expected impact:** Moderate. Improves signal-to-noise within the existing
model's window; should raise discrimination between topically distinct
incidents and modestly raise true-positive similarity scores. Won't fix the
fundamental model-capability ceiling (see Phase 3.5 in the original review —
model swap is out of scope for V2.5 but this sets up for it later, since
cleaner input text benefits any embedding model).

**Complexity:** Low. Pure text-construction change in the normalizer, plus
a backfill script (re-normalize + re-embed all incidents for the active
`model_name`). No schema changes.

**Risks:**
- Full re-embedding pass is CPU-bound (sentence-transformers on CPU) —
  estimate runtime against current corpus size before running in production;
  may need batching/throttling.
- Changing structure could *remove* signal some current borderline-working
  queries depend on — this is exactly why Phase 0's baseline matters; run
  the eval set before and after this change specifically.
- If the backfill job is interrupted partway, incidents end up on a mix of
  old/new `text_hash` for the same `model_name` — `_upsert_embedding`
  overwrites in place, so a resumable/idempotent backfill script is
  important (can re-run safely, just re-checks `text_hash`).

**Success measurement:**
- Eval harness (Phase 0): Recall@5/10, MRR, NDCG@10 — focus on the
  *lexical-overlap* and *paraphrase* query buckets, where signal dilution
  most likely hurts.
- Score distribution: check whether true-positive similarity scores shift
  upward and whether the gap between top-1 and top-5 widens (more
  discrimination, less "everything is 0.25–0.30").
- Manual spot-check on the "database timeout during peak traffic" query and
  2–3 similar variants.

---

## 2. Promote reranking/query expansion into production `search()`

**Change:** `IncidentSearchService.search_debug()` already implements query
expansion (`_expand_query`, via `LLMService.expand_search_query`) and LLM
reranking (`_rerank`, via `rerank_incident_search_results`). Promote this
logic into the default path used by `investigation_agent.py` and
`advanced_investigation_agent.py`, which currently call plain `.search()`
with no `llm_service`.

Recommended approach: make expansion/reranking an *optional layer* — e.g., a
`search_with_rerank()` method (or a flag on `search()`) that wraps the base
candidate retrieval, so callers can opt in/out and the behavior is
feature-flaggable for rollback. Both investigation agents should pass an
`LLMService` instance and call the reranked path for their primary searches.

**Expected impact:** Potentially high for the originally reported symptom —
an LLM looking at structured candidate payloads (title, symptoms, severity,
resolution) can apply semantic judgment that a 384-dim cosine score can't,
independent of base embedding quality. This is likely the fastest way to
improve *perceived* result quality for end users, even before the embedding
model itself improves.

If sequenced after Phase 3 (hybrid retrieval), reranking operates over a
better candidate pool and the combined effect should compound.

**Complexity:** Medium. Mostly plumbing — wiring `LLMService` into the
investigation agents — but with real operational concerns:
- Latency: expansion (N extra LLM calls for expanded phrases, each
  triggering its own vector search) + reranking (1 more LLM call) adds
  meaningfully to per-search latency. Investigation agents that call
  `search()` multiple times per workflow step multiply this.
- Cost: every search now costs additional LLM tokens. For agentic
  workflows with multiple search calls per investigation, this adds up.
- Fallback behavior: `_rerank` already falls back to distance-sorted
  candidates if `llm_service is None` or returns nothing usable — confirm
  this fallback also covers LLM errors/timeouts, not just absence.

**Risks:**
- **Cost/latency blowup** in agentic workflows — consider capping the number
  of expansion phrases, running expansion searches in parallel, and/or
  caching expansion results per query string.
- **Nondeterminism**: LLM reranking may vary run-to-run, making eval noisy.
  Use temperature 0 for the rerank/expansion calls where supported, and run
  eval multiple times to get a variance estimate, not a single number.
- **Topic drift from expansion**: expanded query phrases can pull in
  tangentially related candidates, increasing recall at the cost of
  precision if the reranker doesn't filter them well. The existing
  `candidate_map` merge (keeps best distance per incident) helps but doesn't
  prevent the reranker from being presented with more noise.
- **Silent degradation**: if the LLM service is down, agents should still
  get usable (if lower-quality) results via the distance-sorted fallback —
  verify this path explicitly, since it's the difference between "degraded"
  and "broken."

**Success measurement:**
- Eval harness: Recall@5/10 and NDCG@10 with reranking ON vs OFF, across all
  query-type buckets — expect the biggest gains on *paraphrase* and
  *multi-concept* queries where lexical/dense retrieval alone struggles.
- Latency: p50/p95 added per search call, and end-to-end investigation
  workflow time before/after.
- Cost: LLM tokens/cost per search and per investigation run.
- Regression check: *no-match-expected* queries shouldn't start returning
  confident-looking but wrong results due to reranker overconfidence —
  track whether reranking ever promotes a low-similarity candidate to rank 1
  for these.

---

## 3. Hybrid retrieval using the existing trigram index

**Change:** Add a lexical retrieval path alongside the existing pgvector
cosine search, using the GIN trigram index already defined on
`title`/`description`/`canonical_text` (`ix_incidents_full_text`,
`gin_trgm_ops`). Run both retrievals for a query, then fuse results —
Reciprocal Rank Fusion (RRF) is a reasonable default since it doesn't
require score normalization across two different scales (cosine distance vs.
trigram similarity).

Two lexical options to evaluate:
- **pg_trgm similarity** (`%` operator / `similarity()`) on the existing
  index — good for fuzzy/substring matches, no schema change needed.
- **Postgres full-text search** (`tsvector` + `ts_rank`) — closer to a
  BM25-style term-frequency relevance signal, better for queries like
  "database timeout" where term *importance* (not just presence) matters,
  but requires adding a `tsvector` column + new GIN index (schema migration).

Given the existing index is trigram-based, start with pg_trgm for V2.5 (zero
migration) and treat tsvector/ts_rank as a fast-follow if trigram fusion
proves insufficient.

**Expected impact:** High for the specific symptom in the original report —
queries with strong literal term overlap ("database", "timeout", "peak")
will now surface documents that share that vocabulary even when the dense
embedding similarity is weak. This directly targets the "top results are
weakly related" failure mode by giving literal-match documents a path into
the candidate set that doesn't depend on embedding quality at all.

**Complexity:** Medium.
- Query layer: run two queries (or one query with a CTE/UNION) and merge in
  application code (`IncidentSearchService.search`), applying RRF before
  metadata filters or after — need to decide whether filters apply to both
  legs identically (they should, for consistency).
- Fusion weight/method needs to be configurable, since the right balance is
  empirical and should be tunable per query-type without a redeploy ideally.
- If moving to tsvector later: schema migration (generated column + GIN
  index) on a potentially large `incidents` table — plan for migration
  timing (online index creation via `CREATE INDEX CONCURRENTLY` to avoid
  locking).

**Risks:**
- Trigram similarity can resurface noisy partial matches — an incident that
  mentions "timeout" once in a long, otherwise-unrelated description could
  get pulled in. Fusion weighting and/or a minimum-similarity floor for the
  lexical leg will need tuning.
- Fusion parameters tuned against a small eval set risk overfitting — use
  the *query-type* breakdown (Phase 0) to check that lexical fusion helps
  *lexical-overlap* queries without degrading *paraphrase* queries (where
  the lexical leg should contribute little to nothing).
- Combining with Phase 1's canonical text cleanup: shorter, less
  boilerplate-heavy `canonical_text` also changes what the trigram index
  matches against — sequence this after Phase 1, or re-run the Phase 1
  eval baseline after this change too, so the two effects aren't conflated.

**Success measurement:**
- Eval harness: Recall@5/10 and NDCG@10 specifically on the *lexical-overlap*
  bucket — this is the bucket this phase is designed for, and should show
  the clearest improvement.
- Regression check on *paraphrase* bucket — confirm hybrid fusion doesn't
  pull in lexically-similar-but-semantically-wrong results that dense-only
  search was correctly excluding.
- Direct re-test of the original "database timeout during peak traffic"
  query — confirm PendingRollbackError/OOM/retry-state-machine results no
  longer dominate the top-5, and that any genuinely relevant DB-timeout
  incident in the corpus now surfaces (or, if none exists, that the
  *no-match-expected* behavior is reasonable).

---

## 4. Retrieval evaluation methodology (full rollout)

Phase 0 establishes a minimal baseline; this phase generalizes it into a
standing regression suite usable for all future retrieval changes (including
a future embedding model swap, which was flagged as out of scope for V2.5
but benefits from this infrastructure existing).

**Change:**
- Expand the gold query set beyond the initial ~20–30 to cover more of the
  corpus's incident types/components, maintaining the query-type tagging
  (lexical-overlap, paraphrase/semantic, multi-concept, no-match-expected).
- Version eval results by `(embedding_model_name, search_config)` so results
  from different phases/configs are directly comparable — store as
  timestamped JSON snapshots, not just printed numbers.
- Define the standing metric set: Recall@5, Recall@10, MRR, NDCG@10, plus
  similarity score distributions (to track whether the "everything is
  0.25–0.30" ceiling is moving).
- Decide on a process: eval harness runs (a) before/after each phase change
  during V2.5 rollout, and (b) as a regression check before any future
  retrieval-affecting change ships.

**Expected impact:** This is infrastructure, not a retrieval-quality change
on its own — its value is making the impact of phases 1–3 *measurable* and
preventing future regressions. Given phases 1–3 can't be properly validated
without it, its practical value is high even though it doesn't move metrics
itself.

**Complexity:** Medium. Curating a representative gold set (manual +
LLM-assisted query generation from `is_gold_labeled` incidents) is the main
effort; the harness itself is straightforward (run queries, compute
Recall/MRR/NDCG against known relevant ids).

**Risks:**
- Gold set representativeness — if skewed toward queries that closely match
  `canonical_text` vocabulary (likely if LLM-generated from incident titles),
  metrics may overstate real-world performance on more colloquial user
  queries. Mix in hand-written queries deliberately phrased differently from
  the source incidents.
- LLM-assisted query generation and any LLM-as-judge relevance scoring
  introduce the same nondeterminism concerns as Phase 2 — fix
  temperature/seed where possible, and treat single eval runs as noisy
  point estimates rather than ground truth.
- Maintenance burden — as the corpus grows, gold set ids may become stale
  (incidents deduplicated, updated, or deleted). Build a periodic check that
  gold set ids still resolve to incidents in the database.

**Success measurement:**
- The harness itself: does it produce stable (low run-to-run variance),
  reproducible numbers for an unchanged configuration? Run it 2–3 times on
  the same config before trusting it for phase comparisons.
- Coverage: query-type bucket sizes should be roughly balanced enough that
  no single bucket dominates the aggregate metrics.
- End state for V2.5: a documented before/after comparison across all four
  phases, run through this harness, showing the cumulative effect on
  Recall@5/10, MRR, NDCG@10, and score distribution — this is the artifact
  that closes out V2.5.

---

## Summary table

| Phase | Change | Impact | Complexity | Key risk |
|---|---|---|---|---|
| 0 | Minimal eval baseline | Enables measurement of everything else | Low–Medium | Small/unrepresentative gold set |
| 1 | Canonical text cleanup + backfill | Moderate | Low | Full re-embed cost; signal removal for borderline queries |
| 3 | Hybrid retrieval (trigram fusion) | High for lexical-overlap queries | Medium | Noisy fusion weighting; needs re-baseline after Phase 1 |
| 2 | Promote reranking/expansion to `search()` | High, compounds with Phase 3 | Medium | Latency/cost in agent workflows; nondeterminism |
| 4 | Full eval methodology rollout | Infrastructure — makes 1–3 measurable | Medium | Gold set representativeness/maintenance |

(Table ordered by recommended execution sequence, not the original
priority numbering.)
