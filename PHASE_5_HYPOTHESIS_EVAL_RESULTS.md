# Phase 5 Results: Hypothesis Generation Evaluation

## What was built

- [tests/eval/hypothesis_gold.json](tests/eval/hypothesis_gold.json) — 8 cases
  derived from existing `gold_queries.json` entries (7 positive: para-01,
  para-02, para-04, para-03, para-05, para-06, multi-04; 1 negative: neg-01).
  Each positive case carries `expected_incident_ids` plus a primary
  `expected_root_causes` description and 1-2
  `acceptable_alternative_root_causes` phrasings, written from each
  incident's actual title/resolution_summary.
- [tests/eval/run_hypothesis_eval.py](tests/eval/run_hypothesis_eval.py) —
  harness that, per case: runs dense `search()` (retrieval correctness),
  builds a minimal incident context, calls
  `AdvancedInvestigationAgent.llm_service.generate_hypotheses()` unmodified,
  scores each hypothesis's `root_cause` against the acceptable root causes via
  MiniLM cosine similarity (threshold 0.55), runs each hypothesis's
  `validation_keywords` back through `search()`, and classifies each case into
  a single failure stage. Results: [tests/eval/results/hypothesis_v5.json](tests/eval/results/hypothesis_v5.json).
- No prompts, retrieval, or confidence-calibration code were modified.

---

## Headline metrics (n=7 positive cases, n=8 total)

| Metric | Value |
|---|---|
| Retrieval Recall@5 | 1.0 (7/7) |
| Root-cause Recall@1 | 0.857 (6/7) |
| Root-cause Recall@3 | 0.857 (6/7) |
| Root-cause MRR | 0.893 |
| Validation-keyword Recall@5 | 0.857 (6/7) |

## Confidence-vs-correctness

| | Value |
|---|---|
| n hypotheses scored (excl. negative case) | 35 |
| n correct (root-cause match) | 24 |
| n incorrect | 11 |
| mean `confidence_score`, correct | 0.685 |
| mean `confidence_score`, incorrect | 0.600 |
| point-biserial correlation | **0.30** |

A correlation of 0.30 is weak-positive: correct hypotheses do tend to carry
slightly higher self-reported confidence, but the overlap is large enough
that `confidence_score` is not a reliable correctness signal on its own (see
hyp-02 below, where the correct hypothesis has *lower* confidence than three
incorrect ones).

## Failure breakdown (n=8 cases)

| Stage | Count | Cases |
|---|---|---|
| pass | 5 | hyp-01, hyp-04, hyp-05, hyp-06, hyp-07 |
| retrieval_failure | 0 | — |
| hypothesis_failure | 1 | hyp-02 |
| validation_keyword_failure | 1 | hyp-03 |
| negative_case (excluded from recall stats) | 1 | hyp-08 |

**Retrieval is not the bottleneck here** — every positive case's expected
incident was retrieved (consistent with Phase 3A/4's near-saturated dense
retrieval). Both failures occur strictly downstream, in hypothesis generation
and validation-keyword construction.

---

## Failure case detail

### hyp-02 — `hypothesis_failure`: *"the background trigger service fails to launch after deployment"* (expected: `375a627d`, triggerer not starting)

Generated hypotheses, ranked:

| Rank | Root cause | Confidence | Match |
|---|---|---|---|
| 1 | "In-process Execution API lifespan startup failures due to synchronous changes" | 0.80 | ✗ |
| 2 | "Missing connection ID for Azure Blob storage" | 0.70 | ✗ |
| 3 | "Remote logging not enabled in configuration" | 0.60 | ✗ |
| 4 | "Potential issues with the triggerer's cleanup process" | 0.50 | ✓ (sim 0.588) |
| 5 | "Insufficient error handling in the triggerer's startup sequence" | 0.40 | ✓ (sim 0.704) |

The **correct** hypothesis class (triggerer startup/cleanup) is present, but
ranked 4th-5th with the *lowest* confidence scores (0.40-0.50), while three
**unrelated** hypotheses (Azure Blob storage, remote logging config, "lifespan
API" changes — likely pattern-matched from other retrieved Airflow incidents
in the context, not this one) were ranked above it with *higher* confidence
(0.60-0.80). The model had the right answer available but buried it beneath
more confidently-stated wrong ones — the exact inversion that weakens the
confidence-correctness correlation overall.

### hyp-03 — `validation_keyword_failure`: *"the type checker crashes with a memory error while watching files for changes"* (expected: `a9a17361`, --watch segfault/OOM)

Rank-1 hypothesis ("Memory leak in type checker when using --watch flag",
sim=0.694, confidence=0.80) **correctly matches**. But its
`validation_keywords` → query `"out of memory --watch memory leak"` retrieves
**neither** `a9a17361` (this incident) **nor** any incident containing
`a9a17361` in its top 5 — `validation_keyword_eval.recall_at_5 = 0.0`.

This corpus contains *two* near-duplicate memory/OOM TypeScript incidents
(`a9a17361` = `--watch` segfault, `b093992f` = heap OOM on large bundles —
the same pair flagged as `multi-01`'s two expected incidents in the retrieval
gold set). The keyword query `"out of memory --watch memory leak"` is generic
enough that dense search resolves it toward the *other* memory incident
(`b093992f`) instead of the `--watch`-specific one. The hypothesis's root
cause was correct, but its validation query was too generic to confirm
*which* of the two similar incidents it actually corresponds to —
`_collect_evidence()` would surface evidence for the wrong incident while the
reasoning text sounds right.

### hyp-08 — `negative_case`: *"credit card payment processing failure in the checkout flow"* (expected: none)

Per Phase 4, this query's dense retrieval top1_score should fall in the LOW
band (no genuine match exists in this corpus — confirmed: `retrieved_top5_ids`
are unrelated incidents). Despite this, `generate_hypotheses()` produced 5
hypotheses with confidence **0.60-0.80** — squarely in the "plausible" range,
indistinguishable from the confidence scores on genuinely-matched cases
above:

| Rank | Root cause | Confidence |
|---|---|---|
| 1 | "Inconsistent handling of task failures in SDKs leading to terminal states without retries." | 0.80 |
| 2 | "Incomplete response handling in the structured output parsing mechanism." | 0.75 |
| 3 | "Lack of error handling for invalid or unexpected input during payment processing." | 0.70 |
| 4 | "Issues with the schema versioning and migration processes in the Dag Processor." | 0.65 |
| 5 | "Client-side decision-making in SDKs that does not account for server-side states." | 0.60 |

These are generic incident-shaped statements extrapolated from the (irrelevant)
retrieved context — none relate to "credit card payment processing" at all
(they reference Airflow SDK/Dag Processor concepts from the retrieved
incidents). **`generate_hypotheses()`'s `confidence_score` field carries no
information about retrieval confidence** — it appears to reflect the model's
confidence in its own reasoning given whatever context it was handed, not
whether that context was relevant. This is the clearest evidence yet that
Phase 4's `retrieval_confidence` (LOW for this query) is *not* propagated into
hypothesis-level confidence, even though `_build_incident_context()` includes
the LOW-confidence header text in the prompt context.

---

## Observations

1. **Hypothesis generation, not retrieval, is now the binding constraint** on
   investigation quality for non-trivial cases — both observed failures
   (hyp-02, hyp-03) occur with perfect retrieval. This confirms the Phase 4
   prediction that hypothesis generation is the next bottleneck.
2. **`confidence_score` is a weak signal (r=0.30)** and in at least one case
   (hyp-02) is *inversely* related to correctness for the specific hypothesis
   that matters — the correct hypothesis was the least confident of five. Do
   not treat per-hypothesis `confidence_score` as a reliable filter or
   ranking signal without further work.
3. **Validation-keyword genericity causes silent evidence misattribution**
   (hyp-03): a correct root cause can still produce
   `_collect_evidence()` results for the *wrong* near-duplicate incident when
   keywords aren't specific enough to disambiguate similar incidents in the
   corpus. This is a distinct failure mode from "wrong root cause" and would
   not be visible without the keyword-recall@5 check added in this phase.
4. **Negative cases get confidently-stated, irrelevant hypotheses** (hyp-08):
   even though `_build_incident_context()` already includes a LOW-confidence
   /"no strong historical match" header (Phase 4), `generate_hypotheses()`
   still returns 5 hypotheses at 0.60-0.80 confidence drawn from irrelevant
   retrieved incidents. The LOW-confidence framing in the context does not
   suppress confident-sounding hypothesis output.
5. **Small sample (n=7 positive, n=8 total)** — these findings establish a
   *baseline and failure taxonomy*, not statistically robust rates. The value
   here is in the failure-stage classification (retrieval vs. hypothesis vs.
   validation-keyword), which a larger gold set could populate with more
   confidence.

## Recommendations for future refinement (not implemented here)

- Investigate whether `evaluate_investigation_evidence`'s
  `ranked_hypotheses`/`confidence_assessment` re-orders or down-weights
  low-confidence hypotheses better than raw `generate_hypotheses` output does
  — i.e., does the *final report* recover from hyp-02-style ranking errors?
  (Out of scope for this phase, which evaluates `generate_hypotheses` in
  isolation, but a natural Phase 6 question.)
- Consider whether `generate_hypotheses`'s prompt should be told the
  retrieval confidence level explicitly (as a structured field, not just
  prose in the context) and instructed to lower all `confidence_score` values
  proportionally when retrieval confidence is LOW — addressing observation 4.
- For validation-keyword specificity (observation 3), consider whether
  `validation_keywords` generation should be aware of *which* retrieved
  incident the hypothesis was drawn from, and required to include
  disambiguating terms when multiple similar incidents exist in the initial
  context.
- Expand `hypothesis_gold.json` beyond 8 cases, particularly with more
  negative cases and cases involving near-duplicate incidents (like the
  `a9a17361`/`b093992f` pair), to turn the failure-stage counts into stable
  rates.
