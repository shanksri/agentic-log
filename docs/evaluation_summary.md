# Evaluation Summary — All Runs to Date

This is a single reference point for every evaluation result produced for this project so far:
retrieval quality (dense vs. reranking), confidence calibration, root-cause hypothesis/investigation
quality, and generation + grounding quality. It exists because the results are scattered across
`tests/eval/results/*.json` and `.evaluation_runs/history/*/` with no single index tying them
together with what config produced them.

**Scope note:** this file reports numbers exactly as they appear in the source JSON files listed
under each section (paths given so every number here is traceable and reproducible). Where a
metric was never actually run (e.g. a persisted hybrid/BM25 retrieval benchmark), that gap is
called out explicitly in [What has *not* been benchmarked](#what-has-not-been-benchmarked) rather
than estimated.

---

## 1. Retrieval evaluation — dense vs. reranking

Two independent generations of this benchmark exist, both against the same 24-query v1 gold set
(`tests/eval/gold_queries.json`: 10 lexical-overlap, 6 paraphrase, 4 multi-concept, 4
no-match-expected). Both use `sentence-transformers/all-MiniLM-L6-v2` embeddings. In both
generations, the "rerank" configuration turns on **query expansion AND LLM reranking together**
(`expansion: true, reranking: true`) versus a **dense-only baseline**
(`expansion: false, reranking: false, hybrid: false`) — there is no persisted run that isolates
reranking's effect from expansion's effect.

### Generation 1 — `baseline_v2.4` → `rerank_v2.5` (2026-06-13, ~07:1x UTC)

| Metric | Dense-only (`baseline_v2.4`) | Expansion+Rerank (`rerank_v2.5`) | Δ |
|---|---|---|---|
| Recall@5 | 0.9750 | 1.0000 | +0.0250 |
| Recall@10 | 1.0000 | 1.0000 | 0 |
| MRR | 0.9125 | 0.9750 | +0.0625 |
| NDCG@10 | 0.9194 | 0.9815 | +0.0622 |
| Top-1 score (mean) | 0.5481 | 0.5591 | +0.0110 |
| Top-5 mean score (mean) | 0.4073 | 0.4090 | +0.0018 |

By query type:

| Query type | Metric | Dense | Rerank |
|---|---|---|---|
| lexical-overlap (n=10) | Recall@5 / @10 / MRR / NDCG@10 | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 / 1.0 |
| paraphrase (n=6) | Recall@5 / @10 / MRR / NDCG@10 | 1.0 / 1.0 / **0.7917** / **0.8436** | 1.0 / 1.0 / **1.0** / **1.0** |
| multi-concept (n=4) | Recall@5 / @10 / MRR / NDCG@10 | **0.875** / 1.0 / 0.875 / 0.8314 | 1.0 / 1.0 / 0.875 / 0.9077 |
| no-match-expected (n=4) | (recall/MRR/NDCG undefined — 0 expected incidents) | — | — |

Source: `tests/eval/results/baseline_v2.4.json`, `tests/eval/results/rerank_v2.5.json`.

### Generation 2 — `canonical_v3a_dense` → `canonical_v3a_rerank` (2026-06-13, ~20:5x UTC, "canonical" corpus snapshot)

Same 24-query gold set, run again later the same day against a refreshed/canonical corpus state —
the dense-only baseline alone improved noticeably over Generation 1 (MRR 0.9125 → 0.975), so these
two generations are **not** directly comparable as a rerank-effect measurement across time; compare
dense-vs-rerank *within* each generation instead.

| Metric | Dense-only (`canonical_v3a_dense`) | Expansion+Rerank (`canonical_v3a_rerank`) | Δ |
|---|---|---|---|
| Recall@5 | 1.0000 | 1.0000 | 0 |
| Recall@10 | 1.0000 | 1.0000 | 0 |
| MRR | 0.9750 | 0.9750 | 0 |
| NDCG@10 | 0.9754 | 0.9775 | +0.0021 |
| Top-1 score (mean) | 0.5843 | 0.5990 | +0.0147 |
| Top-5 mean score (mean) | 0.4155 | 0.4261 | +0.0106 |

By query type:

| Query type | Metric | Dense | Rerank |
|---|---|---|---|
| lexical-overlap (n=10) | Recall@5 / @10 / MRR / NDCG@10 | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 / 1.0 |
| paraphrase (n=6) | Recall@5 / @10 / MRR / NDCG@10 | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 / 1.0 |
| multi-concept (n=4) | Recall@5 / @10 / MRR / NDCG@10 | 1.0 / 1.0 / 0.875 / 0.8770 | 1.0 / 1.0 / 0.875 / 0.8877 |
| no-match-expected (n=4) | (recall/MRR/NDCG undefined) | — | — |

MRR is **identical** (0.975) dense vs. rerank in this generation — the one query pulling MRR below
1.0 in both configs is `multi-02` ("scheduler database error causing repeated crash loops..."),
where the correct incident lands at rank 2 in *both* dense and rerank ordering (`mrr: 0.5` for that
query in both files) — reranking did not fix it. NDCG@10 gain from reranking is small (+0.0021
overall). This matches the documented finding in
[docs/architecture/13_llm_reranking.md](architecture/13_llm_reranking.md): reranking is
**"empirically near-neutral on top-1 score and occasionally harmful"** — the doc cites an observed
case ("ISR shrink event") where reranking reverted an expansion gain from 0.50 back to 0.37.

Source: `tests/eval/results/canonical_v3a_dense.json`, `tests/eval/results/canonical_v3a_rerank.json`.

### Reading both generations together

- Recall@10 is 1.0 in every configuration across both generations — the dense retriever alone
  never fails to place the correct incident somewhere in the top 10 on this 24-query set.
- Reranking's main measured value is **ordering polish** (NDCG@10, top-1 score), not recall — it
  cannot recover a candidate the dense/expansion stage didn't already retrieve (see doc 13, "Why
  can reranking never recover a missed candidate?").
- No-match-expected queries (4 per generation) correctly have no recall/MRR/NDCG (undefined when
  there are 0 expected incidents) — see [Section 2](#2-confidence-calibration) for how those are
  scored instead (top-1 similarity score + confidence banding).

---

## 1.5. Hybrid (BM25 + dense, RRF) vs. dense-only

Two generations exist, both against the 36-query `phase17c_benchmark_v1.json` gold set:

**Generation 1 — Phase 18D (2026-06-29, `.benchmarks/phase18d/`), expand=True/rerank=True.** Ran
Dense, "Always Hybrid," and Adaptive Routing through live, separately-constructed `LLMService`
instances per config. Dense: recall@10 0.9643, MRR 0.9196, NDCG@10 0.9218. Always-Hybrid: recall@10
0.9286, MRR 0.9286, NDCG@10 0.9092 (worse recall than dense). Routed: recall@10 0.9643, MRR 0.9643,
NDCG@10 0.9449 (best of the three). **Caveat:** since expand/rerank fire live per config with no
repetition, a single-repetition per-query difference between two nominally-identical strategy
choices isn't distinguishable from LLM sampling noise — this generation alone couldn't establish
whether Hybrid's aggregate regression, or Routing's aggregate improvement over both, was a real
strategy effect.

**Generation 2 — deterministic recheck (2026-09-01, `.benchmarks/hybrid_deterministic_recheck.json`),
expand=False/rerank=False**, run after restoring the corpus (see below) specifically to remove LLM
variance from the comparison. Confirmed a real, reproducible regression: pure RRF fusion dropped
query `v2-para-10`'s correct incident entirely out of the top 10 (dense alone ranked it 8th) —
ten other same-repo issues sharing generic vocabulary with the paraphrased query scored on both
retrievers and outranked it via double RRF credit. Full root-cause and fix: doc 18's "Update — dense
floor" note under Phase 17B. Post-fix (dense floor added to `_fuse()` in `hybrid_search.py`), Hybrid
strictly dominates Dense on this gold set: identical recall@10 (0.9643) but better MRR (0.9092 vs.
0.8646) and NDCG@10 (0.9018 vs. 0.8702) — fusion's genuine ranking-quality wins (3 other queries,
MRR gains of 0.5–0.75, all cases where the correct incident was already visible to both retrievers)
survive untouched, and the one catastrophic loss is closed.

**Corpus caveat on Generation 2's absolute numbers:** this recheck ran after the 2026-08-25 laptop
reset wiped the database to a 5-incident smoke test; the corpus was re-ingested (16 GitHub repos +
KAFKA/SPARK/CASSANDRA Jira, ~7,400 incidents vs. the original ~8,000) and the 12 gold-referenced
issues outside any repo's 500-most-recently-updated window were fetched individually by issue
number to restore 34/34 gold resolution. The corpus is not byte-identical to Generation 1's, so
Generation 1 vs. Generation 2 absolute numbers aren't a clean before/after comparison — but the
dense-floor fix's effect (Hybrid: 0.9286→0.9643 recall, same corpus, same code otherwise) is a
clean within-generation before/after.

Source: `.benchmarks/phase18d/{dense,hybrid,routed}.json`, `.benchmarks/phase18d/routing_records.json`,
`.benchmarks/hybrid_deterministic_recheck.json`, `scripts/run_hybrid_deterministic_check.py`.

---

## 2. Confidence calibration

One run: `tests/eval/results/confidence_v4.json` (2026-06-16), built on top of the
`canonical_v3a_dense` run's per-query top-1/top-2/top-5 scores (24 queries, same gold set as
Section 1).

**Thresholds:** `low_confidence_threshold = 0.4`, `high_confidence_threshold = 0.55`.

**Score separation:** MATCH queries (n=20) scored top-1 similarity in **[0.4223, 0.8372]**;
NO_MATCH queries (n=4) scored **[0.2324, 0.3443]** — a clean gap between the two populations on
this gold set (no overlap).

**Confidence-level breakdown:** MATCH → 16 HIGH, 4 MEDIUM (0 LOW); NO_MATCH → 4 LOW (as expected).

**Threshold sweep (match/no-match classification via top-1 score):**

| Threshold | TP | FP | TN | FN | Precision | Recall | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 20 | 1 | 3 | 0 | 0.9524 | 1.0000 | 0.25 | 0.00 |
| 0.35 | 20 | 0 | 4 | 0 | 1.0000 | 1.0000 | 0.00 | 0.00 |
| 0.40 | 20 | 0 | 4 | 0 | 1.0000 | 1.0000 | 0.00 | 0.00 |
| 0.42 | 20 | 0 | 4 | 0 | 1.0000 | 1.0000 | 0.00 | 0.00 |
| 0.45 | 19 | 0 | 4 | 1 | 1.0000 | 0.9500 | 0.00 | 0.05 |
| 0.55 | 16 | 0 | 4 | 4 | 1.0000 | 0.8000 | 0.00 | 0.20 |
| 0.60 | 13 | 0 | 4 | 7 | 1.0000 | 0.6500 | 0.00 | 0.35 |

The 0.35–0.42 band is the sweet spot on this gold set: perfect precision and recall (no false
positives, no false negatives). The chosen production thresholds (0.40 low / 0.55 high) sit inside
that clean zone, trading some recall at the high end (0.55 → 4 false negatives) for a stricter HIGH
confidence bar.

---

## 3. Root-cause hypothesis / investigation evaluation

This is a separate, smaller benchmark (8 cases, 7 with a real expected root cause + 1 negative
case) measuring the downstream agentic layer — not raw retrieval — using `gpt-4o-mini` for
hypothesis generation. Three generations exist, in order:

| Run | Generated at | Retrieval Recall@5 | Root-cause Recall@1 | Root-cause Recall@3 | Root-cause MRR | Validation keyword Recall@5 | Pass/fail breakdown |
|---|---|---|---|---|---|---|---|
| `hypothesis_v5` | 2026-06-16 03:32 | 1.0000 | 0.8571 (6/7) | 0.8571 (6/7) | 0.8929 | 0.8571 | pass: 5, hypothesis_failure: 1, validation_keyword_failure: 1, negative_case: 1 |
| `hypothesis_v6a` | 2026-06-16 03:49 | 1.0000 | **1.0000 (7/7)** | **1.0000 (7/7)** | **1.0000** | 0.8571 | pass: 6, validation_keyword_failure: 1, negative_case: 1 |
| `hypothesis_v7` | 2026-06-18 09:19 | 1.0000 | 0.7143 (5/7) | 0.8571 (6/7) | 0.8214 | — (replaced by keyword_strategy_a/c below) | pass: 4, hypothesis_failure: 1, validation_keyword_failure: 2, negative_case: 1 |

v7 also reports two alternative keyword-matching strategies instead of a single validation-keyword
metric: `keyword_strategy_a` and `keyword_strategy_c`, both scoring case-level Recall@5 = 0.7143 and
per-hypothesis recall = 0.6286 — identical to each other in this run.

**v6a was the high-water mark** (perfect root-cause recall/MRR); **v7 regressed** back toward v5's
level. `root_cause_match_threshold = 0.55` in all three runs.

**Confidence correlation** (does the agent's stated confidence track actual correctness?), point-biserial correlation between confidence and correctness across all 35 hypotheses generated per run:

| Run | Correct / Incorrect | Raw confidence correlation | Composite confidence correlation |
|---|---|---|---|
| `hypothesis_v5` | 24 / 11 | 0.3035 | — (not computed this run) |
| `hypothesis_v6a` | 23 / 12 | 0.1711 | **0.5866** |
| `hypothesis_v7` | 18 / 17 | 0.2509 | 0.4165 |

Composite confidence (introduced in v6a) correlates with correctness noticeably better than raw
confidence in both runs that report it — raw confidence alone is a weak signal
(correlations of 0.17–0.30).

**v7 latency** (only run with timing data): mean per-case evidence-keyword derivation 0.242s, total
1.939s across 8 cases (range 0.19s–0.36s per case).

Source: `tests/eval/results/hypothesis_v5.json`, `hypothesis_v6a.json`, `hypothesis_v7.json`,
`hypothesis_structure_v7.json` (per-case root-cause statement classification/rationale for v7, not
separately summarized above — see the file for the raw per-case rows).

---

## 4. Generation & grounding evaluation (Phase 22)

Measures the *downstream answer* an LLM produces from retrieved context, both for semantic
similarity to a human reference answer (BERTScore) and for RAGAS-style groundedness (Faithfulness,
Answer Relevancy, Context Precision, Context Recall, Context Entity Recall). Dataset:
`tests/eval/gold/phase22d_generation_v1.json` — 10 queries, a curated subset of the 36-query
`tests/eval/gold/phase17c_benchmark_v1.json` retrieval benchmark (that larger 36-query set has
**not** itself been run through the Section 1 dense/rerank retrieval benchmark — see
[Section 5](#what-has-not-been-benchmarked)). Both persisted runs below used `generation_mode=fast`
(BERTScore + Faithfulness only; Answer Relevancy/Context Precision/Context Recall/Context Entity
Recall are all skipped by design in fast mode) and `generation_repetitions=1`.

| Run | Persisted at | Real services? | BERTScore F1 (mean / median / min / max) | Faithfulness (mean / median / min / max, n scored) | Duration | Failures |
|---|---|---|---|---|---|---|
| `20260810_172732_phase22d_real_generation_eval` | 2026-08-10 | yes (live retrieval, LLM, embeddings) | 0.5554 / 0.5508 / 0.3636 / 0.6977 | 0.3371 / 0.1875 / 0.0 / 1.0 (n=8/10) | 227.4s | 2 faithfulness calls returned malformed verdict-array lengths (isolated, non-fatal) |
| `20260814_194720_phase22d_real_generation_eval_v2` | 2026-08-14/15 (this session, "Part 6") | yes (live retrieval, LLM, embeddings) | 0.5631 / 0.5590 / 0.3993 / 0.7206 | 0.5899 / 0.7333 / 0.0 / 0.8462 (n=8/10) | 186.5s | 2 different faithfulness calls returned malformed verdict-array lengths (isolated, non-fatal) |

Both runs answered all 10/10 queries, skipped 0, failed 0 at the query level (the 2 faithfulness
parse failures per run are metric-level, isolated by the harness, and don't fail the query). The
second run's full per-query breakdown (generated vs. reference answers, retrieved
incidents/contexts per query, per-metric skip reasons, and the negative-control query's specific
behavior) is published as an artifact:
<https://claude.ai/code/artifact/26146007-c2e8-4c05-ad76-43b4f3bc0358>.

**Negative control finding (both runs, query `v2-neg-01`):** the system does **not** correctly
decline out-of-corpus queries. Retrieval still returns its top-k nearest incidents by cosine
distance (there is no relevance floor), and the answer-generation LLM treats them as evidence
rather than recognizing none of them describe the queried system — it fabricates plausible-sounding
root causes instead of stating "no matching incident." Faithfulness correctly flags this at
**0.0** in both runs (the grounding LLM catches that the claims aren't supported by context), and
BERTScore vs. the reference "no match" answer is the lowest or near-lowest of all 10 queries in
both runs. This is a real, reproduced (not one-off) behavior gap between what the system does and
what the reference answer expects.

Source: `.evaluation_runs/history/20260810_172732_phase22d_real_generation_eval/generation_report.json`,
`.evaluation_runs/history/20260814_194720_phase22d_real_generation_eval_v2/generation_report.json`
(also mirrored to `.evaluation_runs/latest/`).

---

## 5. What has *not* been benchmarked

Called out explicitly rather than estimated:

- **Adaptive routing (Adaptive Routing config specifically) with expand/rerank isolated from LLM
  noise.** Section 1.5 covers a deterministic (`expand=False`) Dense-vs-Hybrid recheck, but the
  three-way comparison including Adaptive Routing has only ever been run with `expand=True`
  (Phase 18D, Generation 1) — it is not yet known whether Routing's apparent aggregate edge over
  both Dense and Hybrid alone in that run reflects real routing-policy value or was itself
  partly attributable to LLM query-expansion variance across the three separately-timed configs.
  Per docs/README.md, `/evaluation/*`'s orchestrator construction "still deliberately pins a plain
  dense `IncidentSearchService` for reproducible benchmarking" — hybrid/routing quality is
  documented and now partially measured (Section 1.5), but not through the standard `/evaluation/*`
  pipeline.
- **Reranking's effect in isolation.** Every persisted rerank run also has query expansion on
  (`expansion: true, reranking: true` together) versus a dense-only baseline — there is no
  `expansion: true, reranking: false` (or vice versa) run to attribute the gain/loss to one
  mechanism specifically.
- **Retrieval-only (MRR/Recall@K/NDCG) numbers for the 36-query `phase17c_benchmark_v1.json` /
  10-query `phase22d_generation_v1.json` gold sets.** Section 1's dense/rerank numbers are all
  against the older, smaller 24-query `gold_queries.json` set. The newer, larger gold sets have
  only been used for generation/grounding evaluation (Section 4), not run through the retrieval
  harness (`app/evaluation/harness.py`) directly.
- **`generation_mode=standard` or `full`** (Answer Relevancy, Context Precision, Context Recall,
  Context Entity Recall) — every persisted generation run used `fast` mode; those four metrics have
  never been computed on a real run (cost control per doc 22B).
- **Evaluator-stability repetitions** (`generation_repetitions > 1`) — never run; both persisted
  generation runs used `repetitions=1`, so no measured variance/confidence band exists for any
  metric.

---

## 6. Source file index

| File | What it is |
|---|---|
| `tests/eval/results/baseline_v2.4.json` | Retrieval: dense-only, 24-query gold set, gen. 1 |
| `tests/eval/results/rerank_v2.5.json` | Retrieval: expansion+rerank, 24-query gold set, gen. 1 |
| `tests/eval/results/canonical_v3a_dense.json` | Retrieval: dense-only, 24-query gold set, gen. 2 (canonical corpus) |
| `tests/eval/results/canonical_v3a_rerank.json` | Retrieval: expansion+rerank, 24-query gold set, gen. 2 (canonical corpus) |
| `tests/eval/results/confidence_v4.json` | Confidence calibration / threshold sweep, built on `canonical_v3a_dense` |
| `tests/eval/results/hypothesis_v5.json` | Root-cause hypothesis evaluation, gen. 1 (8 cases) |
| `tests/eval/results/hypothesis_v6a.json` | Root-cause hypothesis evaluation, gen. 2 (8 cases) — best result |
| `tests/eval/results/hypothesis_v7.json` | Root-cause hypothesis evaluation, gen. 3 (8 cases) — regression vs. v6a |
| `tests/eval/results/hypothesis_structure_v7.json` | Per-case root-cause statement structure classification for v7 |
| `tests/eval/results/token_lengths_v3a.json` | Corpus text-length profiling (old vs. new `canonical_text`, 384 incidents) — not a quality metric |
| `.evaluation_runs/history/20260810_172732_phase22d_real_generation_eval/` | Generation+grounding eval, real services, run 1 |
| `.evaluation_runs/history/20260814_194720_phase22d_real_generation_eval_v2/` | Generation+grounding eval, real services, run 2 (incl. `retrieved_contexts.json`) |
| `tests/eval/gold_queries.json` | v1 gold format, 24 queries — used by Section 1 & 2 |
| `tests/eval/gold/phase17c_benchmark_v1.json` | v2 gold format, 36 queries — source of the Section 4 subset |
| `tests/eval/gold/phase22d_generation_v1.json` | v2 gold format + reference answers, 10-query subset — used by Section 4 |
| `.benchmarks/phase18d/{dense,hybrid,routed}.json` | Dense/Hybrid/Routing, 36-query gold set, expand+rerank ON (Section 1.5, Gen. 1) |
| `.benchmarks/hybrid_deterministic_recheck.json` | Dense/Hybrid/Routing per-query recheck, expand/rerank OFF (Section 1.5, Gen. 2, post dense-floor fix) |
