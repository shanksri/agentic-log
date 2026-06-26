# Phase 2 Retrieval Results: Promoting Reranking/Expansion to Production

## What changed

- Added `IncidentSearchService.retrieve()` ([app/services/search.py](app/services/search.py)) — the
  canonical retrieval pipeline. It wraps the existing `search()` candidate
  generation with optional `expand` (LLM query expansion) and `rerank` (LLM
  reranking) flags. `_rerank` failures are caught and fall back to
  distance-sorted candidates, so the pipeline degrades gracefully if the LLM
  is unavailable.
- `search_debug()` is now a thin wrapper: `retrieve(..., limit=5, expand=True,
  rerank=True)`. `/search/debug` behavior is unchanged.
- `/search/incidents` and all existing `search()` callers are unchanged —
  `search()` itself was not modified beyond the structured-logging fields
  added in Phase 0.
- **`InvestigationAgent.investigate()`** now calls
  `retrieve(problem, limit=5, expand=True, rerank=True)`.
- **`AdvancedInvestigationAgent.investigate()`** (initial retrieval) now calls
  `retrieve(problem, limit=10, expand=True, rerank=True)`.
- **`AdvancedInvestigationAgent._collect_evidence()`** (per-hypothesis
  retrieval) is unchanged — still plain `search()`, dense-only, as specified
  (latency/cost control; up to 5 hypotheses × 1 call each).
- No changes to embeddings, `canonical_text`, or hybrid retrieval.

Test fixtures updated for the new `call_site`/`**kwargs` signatures; all 12
unit tests pass.

---

## Metric deltas: baseline (v2.4, dense-only) vs rerank-enabled (v2.5)

Same 24-query gold set ([tests/eval/gold_queries.json](tests/eval/gold_queries.json)), same corpus (384 incidents), same embedding model. Full results: [baseline_v2.4.json](tests/eval/results/baseline_v2.4.json) / [rerank_v2.5.json](tests/eval/results/rerank_v2.5.json).

### Overall

| Metric | Baseline (dense-only) | Rerank+Expand | Delta |
|---|---|---|---|
| Recall@5 | 0.975 | 1.000 | +0.025 |
| Recall@10 | 1.000 | 1.000 | 0 |
| MRR | 0.9125 | 0.975 | **+0.0625** |
| NDCG@10 | 0.9194 | 0.9815 | **+0.0622** |
| top1_score_mean | 0.548 | 0.559 | +0.011 |
| top5_mean_score_mean | 0.407 | 0.409 | +0.002 |

### By query type

| Bucket | Metric | Baseline | Rerank+Expand | Delta |
|---|---|---|---|---|
| lexical-overlap (n=10) | Recall@5 / MRR / NDCG@10 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 |
| paraphrase (n=6) | Recall@5 / MRR / NDCG@10 | 1.0 / 0.792 / 0.844 | 1.0 / **1.0** / **1.0** | 0 / **+0.208** / **+0.156** |
| multi-concept (n=4) | Recall@5 / MRR / NDCG@10 | 0.875 / 0.875 / 0.831 | **1.0** / 0.875 / **0.908** | **+0.125** / 0 / **+0.077** |
| no-match-expected (n=4) | top1_score_mean | 0.252 | 0.272 | +0.020 |

### Interpretation

- **Lexical-overlap** was already perfect — no room to move, and reranking
  didn't disturb it.
- **Paraphrase queries gained the most**: MRR went from 0.792 → 1.0. In the
  baseline, the correct incident was often retrieved but not ranked #1 (e.g.
  rank 2 or 4); reranking consistently promotes it to #1. This is the
  expected outcome — an LLM judging structured candidate payloads (title,
  symptoms, resolution) can recognize a paraphrase match that raw cosine
  distance ranks lower.
- **Multi-concept** gained on Recall@5 (one previously-missing expected
  incident, `b093992f` for `multi-01`, now appears in the top 5) and NDCG@10.
  MRR unchanged (0.875) because the *first* hit was already correct in both
  configs for the queries that matter to MRR.
- **No-match-expected**: top1_score_mean rose slightly (0.252 → 0.272). This
  is a regression-risk signal to watch, not yet a problem — none of the 4
  negative-control queries returned a confident (>0.5) top-1 result in either
  config, so reranking isn't yet promoting false positives to high apparent
  confidence. Worth re-checking after Phase 3 (hybrid retrieval), since
  lexical fusion is more likely to inflate these.

---

## Latency impact

Measured directly (warm model, second call to exclude one-time
SentenceTransformer load):

| Path | Latency (warm) |
|---|---|
| `search()` (dense-only, current `/search/incidents` and evidence collection) | **~85 ms** |
| `retrieve(expand=True, rerank=True)` (new default for both agents' primary retrieval) | **~3.7 s** |

The ~3.7s figure includes: 1 base dense search + N expansion-phrase dense
searches (LLM generates 3–5 phrases, each triggers its own `search()` call
candidate_limit=25) + 1 LLM reranking call. This is **~40x** the dense-only
latency, dominated by sequential LLM round-trips (query expansion +
reranking), not by the additional vector searches themselves (each ~85ms).

**Per-agent impact:**
- `InvestigationAgent.investigate()`: 1 retrieval call → adds ~3.6s to total
  investigation latency (on top of the existing `generate_investigation` LLM
  call, which was already present).
- `AdvancedInvestigationAgent.investigate()`: 1 retrieval call (initial) at
  ~3.6s added; `_collect_evidence()` unchanged at ~85ms × up to 5 hypotheses
  (~425ms total) — unchanged from before.

**Cost impact:** each `retrieve(expand=True, rerank=True)` call now makes 2
additional LLM calls (`expand_search_query`, `rerank_incident_search_results`)
beyond whatever the agent was already calling. For `AdvancedInvestigationAgent`,
this is on top of `generate_hypotheses` + `evaluate_investigation_evidence` —
total LLM calls per advanced investigation goes from 2 to 4 (evidence
collection's per-hypothesis searches remain LLM-free).

The expansion-phrase searches currently run **sequentially** — parallelizing
them (noted as a risk in the V2.5 plan) would cut a meaningful chunk of the
3.7s, since each phrase search is only ~85ms but there are 3-5 of them plus
2 LLM round-trips. Not done in this phase per scope (measurement only).

---

## Qualitative examples

**`para-01`** — *"the airflow scheduler keeps restarting because of a null
database id field on the task instance"* (expected: `e3bfe559` — "Scheduler
crashloops with `ValidationError: UUID input should be a string`...")

- Baseline top-5: `[fd6ae144, e913aaed, a04f8f29, e3bfe559, fcbc6109]` —
  correct incident at **rank 4** (MRR 0.25).
- Rerank top-5: `[e3bfe559, fd6ae144, fcbc6109, 176e05af, e913aaed]` —
  correct incident promoted to **rank 1** (MRR 1.0).
- This is the clearest example of the mechanism working as designed: dense
  similarity alone ranked three other Airflow incidents above the actual
  match; the reranker, given titles/symptoms/resolutions, correctly
  identified the scheduler-crashloop incident as most relevant to a query
  about restarts caused by a null DB field.

**`para-05`** — *"the generated API docs show the same operation identifier
twice for an endpoint that supports several HTTP verbs"* (expected: `5dba5df8`
— "Duplicated OperationID when adding route with multiple methods")

- Baseline top-5: `[e1f61e5b, 5dba5df8, e2f41732, fe9c0ec3, b8ddf602]` —
  correct incident at rank 2 (MRR 0.5).
- Rerank top-5: `[5dba5df8, b74479a8, 481069ab, de1208c6, fe9c0ec3]` —
  correct incident promoted to rank 1 (MRR 1.0).

**`multi-01`** — *"memory exhaustion and crash during large file compilation
in watch mode"* (expected: `a9a17361` + `b093992f`, both memory/OOM-related
TypeScript compiler issues)

- Baseline top-5 included `a9a17361` at rank 1 but `b093992f` was outside
  the top 5 (Recall@5 = 0.875 for this bucket overall, driven by this query).
- Rerank top-5: `[a9a17361, b093992f, a562aa3c, 7bc74b8a, 0aabf550]` — both
  expected incidents now in top 2.

**`neg-03`** (no-match-expected) — *"spreadsheet export formatting broken
when opened in Excel 2010"*

- top1_score rose from 0.252 (baseline) to 0.351 (rerank) — still well below
  the ~0.5-0.8 scores seen on genuine matches, but the largest increase among
  the negative controls. Flagging for attention in Phase 3/4 eval — not a
  failure today, but the bucket to watch if hybrid retrieval is added.

---

## Summary

Promoting expansion + reranking into the production retrieval path (as used
by both investigation agents) produced a clear quality win on exactly the
query types it was expected to help — **paraphrase** (MRR +0.21, NDCG@10
+0.16) and **multi-concept** (Recall@5 +0.125, NDCG@10 +0.08) — with no
regression on lexical-overlap and only a small (+0.02) increase in
no-match-expected top1 scores that doesn't yet look like a precision problem.

The cost is a ~40x latency increase (85ms → ~3.7s) per retrieval call for the
agents' primary searches, plus 2 additional LLM calls each. This is consistent
with the V2.5 plan's predicted risk and is the deliberate tradeoff requested —
`_collect_evidence()` was kept dense-only specifically to bound this, and that
constraint was preserved.

**Suggested next steps** (not done here, per scope):
- Parallelize the expansion-phrase searches within `retrieve()` to reduce the
  3.7s figure.
- Re-run this harness after Phase 3 (hybrid retrieval) — paraphrase/
  multi-concept gains here should compound with improved candidate recall,
  but re-check the no-match-expected bucket since lexical fusion is more
  likely to inflate those top1 scores than reranking was.
