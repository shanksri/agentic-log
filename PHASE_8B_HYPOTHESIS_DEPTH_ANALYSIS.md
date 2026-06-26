# Phase 8B: Hypothesis Depth Analysis

**Question**: Do ranks 4 and 5 add enough signal to justify their cost in generic
hypotheses and evidence-selection noise?

Source data: `tests/eval/results/hypothesis_v7.json` and
`tests/eval/results/hypothesis_structure_v7.json`. Computed analytically — no
new DB queries or LLM calls.

Evaluation scope: 7 positive cases, 35 hypotheses total (5 per case), 1 negative
case (hyp-08) excluded throughout.

---

## Metric definitions

**Root Cause Recall@k** — fraction of positive cases (out of 7) where at least
one hypothesis in ranks 1–k is `is_match=True` (cosine ≥ 0.55 vs gold incident
root cause).

**Keyword Recall (strategy A, case-level)** — fraction of cases where the first
correct hypothesis in top-k (or rank 1 if none) has `validation_keyword_recall_ok_a=True`.

**Keyword Recall (strategy C, case-level)** — same but using
`validation_keyword_recall_ok_c` (evidence-oriented keywords).

**Generic hypothesis rate** — fraction of the k×7 = 7k hypotheses in top-k
classified as `neither` by the LLM classifier.

**Evidence-selection accuracy** — fraction of the 7k hypotheses in top-k where
C_hyp's cosine step selected the gold incident (`chose_gold=True`).

---

## Primary results table

| Depth | RC Recall | Kw-A (case) | Kw-C (case) | Generic rate | Evidence acc |
|---|---|---|---|---|---|
| **@1** | **5/7 = 71.4%** | 6/7 = 85.7% | 5/7 = 71.4% | **0/7 = 0.0%** | **6/7 = 85.7%** |
| **@2** | **6/7 = 85.7%** | 5/7 = 71.4% | 4/7 = 57.1% | 1/14 = 7.1% | 11/14 = 78.6% |
| **@3** | **6/7 = 85.7%** | 5/7 = 71.4% | 4/7 = 57.1% | **3/21 = 14.3%** | 15/21 = 71.4% |
| @5 (full) | 7/7 = 100.0% | 5/7 = 71.4% | 5/7 = 71.4% | 7/35 = 20.0% | 24/35 = 68.6% |

---

## Per-rank incremental analysis

Each row shows what that rank *adds* to the cumulative pool.

| Rank | New RC cases | New "neither" | Gold-selection (this rank only) | Net |
|---|---|---|---|---|
| **1** | 5/7 = 71.4% | 0/7 | 6/7 = 85.7% | **High value** |
| **2** | 1/7 = 14.3% (hyp-04) | 1/7 (hyp-05 r2) | 5/7 = 71.4% | **Moderate value** |
| **3** | 0/7 | 2/7 (hyp-06 r3, hyp-07 r3) | 4/7 = 57.1% | **Low value** |
| **4** | 1/7 = 14.3% (hyp-02) | 1/7 (hyp-03 r4) | 5/7 = 71.4% | **Moderate value** |
| **5** | 0/7 | 2/7 (hyp-04 r5, hyp-07 r5) | 4/7 = 57.1% | **Low value** |

**Pattern**: ranks 3 and 5 contribute zero new correct cases and add the most
generic hypotheses (2 each). Ranks 2 and 4 contribute one new case each and
introduce one generic hypothesis each. The odd ranks after 1 are the least
efficient.

The alternation exists because the model exhausts its primary evidence reading
at ranks 1–2 and then generates diversity-driven hypotheses at rank 3 before
latching onto a new pool incident at rank 4 and repeating the pattern at rank 5.

---

## Root Cause Recall by depth

```
@1  ████████████████████░░░░░░░░  71.4%  (5/7)
@2  ████████████████████████░░░░  85.7%  (6/7)
@3  ████████████████████████░░░░  85.7%  (6/7)  ← no gain from rank 3
@5  ████████████████████████████ 100.0%  (7/7)
```

Which case fails at @3 but succeeds at @5?

**hyp-02** ("background trigger service fails to launch after deployment"):
- Rank 1: "Synchronous startup of Execution API…" → is_match=False
- Rank 2: "Missing connection ID for Azure Blob…" → is_match=False (wrong incident)
- Rank 3: "Remote logging configuration not enabled…" → is_match=False
- **Rank 4**: "Inadequate error handling during the triggerer's startup process." → is_match=True ← first correct
- Rank 5: "Issues with triggerer's cleanup process…" → is_match=True

The correct hypothesis for hyp-02 is only reachable at rank 4 because the gold
incident ("Triggerer not starting") is an extremely brief title with almost no
discriminating vocabulary. The LLM generates 3 wrong hypotheses (grounded in 3
other incidents in the pool) before arriving at the gold case at rank 4. This
is a retrieval context problem, not a generation-order problem: the pool has 3
semantically adjacent but incorrect incidents that fill ranks 1–3.

**Cost of cutting to top-3**: lose hyp-02 as a correct case. RC Recall stays
at 6/7 = 85.7% instead of reaching 7/7 = 100%.

---

## Generic hypothesis rate by depth

```
@1   0/7   ░░░░░░░░░░  0.0%
@2   1/14  ████░░░░░░  7.1%
@3   3/21  ██████░░░░ 14.3%
@5  7/35  █████████░ 20.0%
```

"Neither" hypotheses by rank position:
- Rank 1: 0
- Rank 2: 1 — hyp-05 r2 "Lack of clear documentation on…defining routes"
- Rank 3: 2 — hyp-06 r3 "Inconsistent handling of audience claims across components", hyp-07 r3 "Inconsistent behavior of type narrowing across different enum types"
- Rank 4: 1 — hyp-03 r4 "Incompatibility or bugs introduced in recent TypeScript versions"
- Rank 5: 3 — hyp-04 r5 "Incompatibility with specific TypeScript versions", hyp-07 r4/r5

Cutting to top-3 eliminates 4 of 7 "neither" hypotheses (57%) while retaining
3. The remaining 3 are at ranks 2 and 3 — earlier in the generation order —
and are driven by problem statement framing (hyp-05 r2) or rank-3 abstraction
(hyp-06 r3, hyp-07 r3) rather than the vocabulary-absorption from deep pool
incidents that characterises ranks 4–5 failures.

---

## Evidence-selection accuracy by depth

```
@1  6/7  = 85.7%  ██████████████████████████████
@2 11/14 = 78.6%  ████████████████████████░░░░░░
@3 15/21 = 71.4%  █████████████████████░░░░░░░░░
@5 24/35 = 68.6%  ████████████████████░░░░░░░░░░
```

Rank-by-rank gold-selection failures:

| Rank | Mismatch cases | Cause |
|---|---|---|
| 1 | hyp-05 r1 (gap=+0.032) | Near-tie; SSE APIRouter incident marginally scores higher |
| 2 | hyp-02 r2 (gap=+0.705) | Azure blob hypothesis strongly matches azure blob incident |
| 3 | hyp-02 r3 (gap=+0.601), hyp-03 r3 (gap=+0.108), hyp-04 r3 (gap=+0.259) | 3 mismatches |
| 4 | hyp-03 r4 (gap=+0.374), hyp-07 r4 (gap=+0.209) | 2 mismatches |
| 5 | hyp-04 r5 (gap=+0.253), hyp-07 r5 (gap=+0.580) | 2 mismatches |

Gold-selection failure rate per rank: rank 1 = 14.3%, rank 2 = 28.6%, rank 3 = 42.9%, rank 4 = 28.6%, rank 5 = 42.9%. Ranks 3 and 5 have the highest failure rates (same 42.9%), consistent with those being the ranks where "neither" hypotheses concentrate.

---

## Keyword recall by depth

### Strategy A (LLM-generated keywords)

| Depth | Case-level recall | Change vs previous |
|---|---|---|
| @1 | 6/7 = 85.7% | — |
| @2 | 5/7 = 71.4% | −14.3% |
| @3 | 5/7 = 71.4% | 0% |
| @5 | 5/7 = 71.4% | 0% |

Counter-intuitive: strategy A keyword recall is **highest at @1**. The mechanism:

At top-1, hyp-04 has no correct hypothesis → fallback to rank 1 (hyp-04 r1
"Insufficient memory allocation for large bundles", `kw_recall_a=True`). At
top-2+, hyp-04's first correct hypothesis is r2 ("Inefficient memory management
in the TypeScript compiler", `kw_recall_a=False`). Using the correct hypothesis
at rank 2 actually hurts keyword recall for that case because the correct
hypothesis happens to have worse keywords than the incorrect rank-1 hypothesis.

This is an artifact of the gold keyword evaluation: `kw_recall_a=False` for
hyp-04 r2 means "compiler hang maximum call stack size exceeded TypeScript" did
not retrieve `b093992f` ("JavaScript heap out of memory for 10s of MB of
source") in the keyword search top-5 — a known failure first identified in
Phase 7 and explained by the gold incident's title not matching those
mechanism-level terms.

### Strategy C (evidence-oriented keywords)

| Depth | Case-level recall | Change vs previous |
|---|---|---|
| @1 | 5/7 = 71.4% | — |
| @2 | 4/7 = 57.1% | −14.3% |
| @3 | 4/7 = 57.1% | 0% |
| @5 | 5/7 = 71.4% | +14.3% |

Strategy C's top-5 parity with top-1 is explained by a single case: hyp-02.
At top-5, hyp-02's first correct hypothesis is rank 4 with `kw_c=True` (the
evidence keywords from the gold-incident title "Triggerer not starting" are
found by the cosine step because the rank-4 hypothesis "Inadequate error
handling during the triggerer's startup process" cosines correctly to the gold
title). At top-3, hyp-02 falls back to rank 1, where `kw_c=False` (rank-1 maps
to the gold incident but the C_hyp extraction fails because "Triggerer not
starting" is too short — 3 words — to extract 5 discriminating terms).

---

## Marginal value vs marginal cost by rank

| Rank | RC gain | Generic penalty | Evidence noise | Keyword change (A) | Net verdict |
|---|---|---|---|---|---|
| 1 | +71.4% | 0% | 14.3% miss rate | baseline 85.7% | **Keep — essential** |
| 2 | +14.3% | +7.1% | 28.6% miss rate | −14.3% | **Keep — adds 1 case, low cost** |
| 3 | 0% | +14.3% | 42.9% miss rate | 0% | **Marginal — no RC gain, 2 new neither** |
| 4 | +14.3% | +7.1% | 28.6% miss rate | 0% | **Conditional — adds 1 case, but only for "shallow pool" scenarios** |
| 5 | 0% | +14.3% | 42.9% miss rate | 0% | **Cut — no RC gain, 2 new neither** |

---

## Scenario analysis: cut to top-3

**What is gained**: eliminate 4 "neither" hypotheses (3 from hyp-07 r4/r5 and
hyp-04 r5, 1 from hyp-03 r4), reduce generic rate from 20% to 14.3%, improve
per-hypothesis evidence-selection accuracy from 68.6% to 71.4%, reduce LLM
generation cost by 40%.

**What is lost**: hyp-02 drops from RC recall = 1.0 (using rank-4 correct
hypothesis) to RC recall = 0.0 within the top-3 window. Case-level RC recall
drops from 7/7 = 100% to 6/7 = 85.7%. Strategy C keyword recall drops from
5/7 to 4/7 (because hyp-02 r4 provides the only correct evidence keyword for
that case).

**Conclusion**: the loss is real but concentrated in one specific failure mode:
a case where the retrieval pool contains 3 incorrect-but-adjacent incidents that
fill ranks 1–3, and the correct hypothesis only emerges at rank 4. This is a
retrieval problem, not a generation problem, and is not fixed by extending to
rank 4 in general — it is fixed by improving the retrieval pool quality for
cases like hyp-02.

---

## Scenario analysis: cut to top-2

**What is gained over top-3**: 2 fewer "neither" hypotheses (hyp-06 r3, hyp-07 r3),
evidence accuracy improves from 71.4% to 78.6%.

**What is lost over top-3**: the incremental RC gain from rank 2 (hyp-04's first
correct at r2) is retained, so RC recall is identical at 85.7%. Keyword A drops
from 71.4% to 71.4% (no change). Keyword C drops further from 57.1% to 57.1%
(no change — same bottleneck cases). The only change is 2 fewer neither
hypotheses and better evidence accuracy.

Wait — top-2 and top-3 have **identical RC recall and keyword recall** but
top-2 has fewer "neither" hypotheses. This means **rank 3 is a pure cost centre
under the current dataset**: it adds no RC cases, introduces 2 "neither"
hypotheses, and degrades evidence-selection accuracy from 78.6% to 71.4%.

---

## Recommendation

The evidence supports a **top-2 default with adaptive extension**:

1. **Always generate top-2**: rank 2 adds one RC case (hyp-04 +14.3%) with
   only 1 "neither" introduced (7.1% generic rate) and 78.6% evidence accuracy.

2. **Extend to rank 4 only for low-confidence retrievals**: hyp-02 (the case
   that needs rank 4) has retrieval confidence = MEDIUM (top1 = 0.527). A
   threshold rule such as "generate ranks 3–4 when initial retrieval confidence
   is MEDIUM and no correct hypothesis found in top-2" would recover the hyp-02
   RC gain without penalising the 6 other cases. This is a retrieval-confidence-
   gated depth strategy.

3. **Never extend to rank 5**: rank 5 provides zero RC gain across the full
   dataset and introduces 2 "neither" hypotheses with 42.9% evidence-selection
   failures.

4. **Rank 3 is borderline**: under the current dataset rank 3 adds nothing and
   costs 2 "neither" hypotheses. It would become valuable only if a future case
   has its first correct hypothesis at rank 3. Given the 0/7 = 0% gain here,
   omitting it from the default path is defensible and would drop the generic
   rate from 14.3% to 7.1%.

---

## Summary statistics (all metrics at all depths)

```
Metric                        @1      @2      @3      @5
─────────────────────────────────────────────────────────
Root Cause Recall            71.4%   85.7%   85.7%  100.0%
Kw-A (case-level)            85.7%   71.4%   71.4%   71.4%
Kw-C (case-level)            71.4%   57.1%   57.1%   71.4%
Generic rate (neither %)      0.0%    7.1%   14.3%   20.0%
Evidence-selection accuracy  85.7%   78.6%   71.4%   68.6%
Total hypotheses                 7      14      21      35
"Neither" hypotheses             0       1       3       7
Gold-selection failures          1       3       6      11
```

Key takeaways:

- **Rank 1** is the highest-quality rank on every metric: 0% generic rate, 85.7%
  evidence accuracy, 85.7% keyword A recall.

- **Rank 2** provides the best marginal return: +14.3% RC recall for +7.1% generic
  rate and −14.3% keyword A accuracy (the keyword regression is an evaluation
  artefact, not a real degradation).

- **Ranks 3 and 5** are symmetric: both add 0% RC recall, 2 "neither" hypotheses,
  and 42.9% evidence-selection miss rates. They are the lowest-value ranks.

- **Rank 4** recovers 14.3% RC recall (hyp-02) but is only needed for one specific
  class of failure (shallow retrieval pool with ≥3 incorrect-but-adjacent
  incidents). A confidence-gated depth rule can capture this benefit selectively.

- **The optimal depth** under this dataset is top-2 by default, with rank 3–4
  extension gated on MEDIUM retrieval confidence and absence of a correct
  hypothesis in top-2.
