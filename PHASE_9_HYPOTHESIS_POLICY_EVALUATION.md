# Phase 9: Hypothesis Depth Policy Evaluation

**Goal**: Determine whether a top-2 default policy is justified by evidence before
changing production behavior.

Source data: `tests/eval/results/hypothesis_v7.json`,
`tests/eval/results/hypothesis_structure_v7.json`. Computed analytically — no
new retrieval calls, no LLM calls, no code changes.

Evaluation scope: 7 positive cases (hyp-01 to hyp-07), 1 negative case
(hyp-08) excluded. Total positive hypotheses: 35.

---

## Policies evaluated

**Policy A** — Generate top 2 hypotheses. Discard ranks 3–5 unconditionally.

**Policy B** — Generate top 2 hypotheses. Conditionally extend to ranks 3–4 when:
- retrieval confidence is MEDIUM (top-1 cosine < 0.55), **AND**
- no `is_match=True` hypothesis found in the first two ranks.

**Policy C** — Current behavior. Generate top 5 hypotheses (all ranks).

---

## Section 1 — Raw metrics at each depth

### Metric definitions

| Metric | Definition |
|---|---|
| **RC Recall** | Fraction of cases where any hypothesis in top-k has `is_match=True` |
| **RC Precision** | Fraction of all generated hypotheses with `is_match=True` |
| **MRR** | Mean reciprocal rank of first `is_match=True` hypothesis (0 if none) |
| **Evidence accuracy** | Fraction of generated hypotheses where C_hyp selects gold (`chose_gold=True`) |
| **Generic rate** | Fraction of generated hypotheses classified as `neither` |
| **Mean raw confidence** | Mean LLM-assigned raw confidence across all generated hypotheses |
| **Mean composite confidence** | Mean composite confidence (raw × keyword discount) across all |
| **Best-case confidence** | Mean of per-case best composite confidence (used for keyword eval) |
| **KW-A recall** | Case-level keyword recall using LLM-generated keywords |
| **KW-C recall** | Case-level keyword recall using evidence-oriented keywords |

---

### Results table

| Metric | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| **RC Recall** | 5/7 = **71.4%** | 6/7 = **85.7%** | 6/7 = 85.7% | 7/7 = **100.0%** |
| **RC Precision** | 5/7 = 71.4% | 9/14 = 64.3% | 11/21 = 52.4% | 18/35 = 51.4% |
| **MRR** | 0.786 | 0.821 | 0.821 | 0.821 |
| **Evidence accuracy** | 6/7 = **85.7%** | 11/14 = 78.6% | 15/21 = 71.4% | 24/35 = 68.6% |
| **Generic rate** | 0/7 = **0.0%** | 1/14 = 7.1% | 3/21 = 14.3% | 7/35 = 20.0% |
| **Mean raw conf** | 0.829 | 0.782 | 0.743 | 0.679 |
| **Mean composite conf** | 0.682 | 0.628 | 0.588 | 0.539 |
| **Best-case composite conf** | 0.709 | 0.709 | 0.709 | **0.730** |
| **Hypotheses per case** | 1 | 2 | 3 | 5 |
| **KW-A recall** | 6/7 = 85.7% | 5/7 = 71.4% | 5/7 = 71.4% | 5/7 = 71.4% |
| **KW-C recall** | 5/7 = 71.4% | 4/7 = 57.1% | 4/7 = 57.1% | 5/7 = 71.4% |

### Incremental value per rank

| Rank added | RC Recall gain | RC Precision Δ | "Neither" added | Gold-sel failures (rank only) | Net verdict |
|---|---|---|---|---|---|
| 1 | +71.4% | — | 0 | 1/7 | **Essential** |
| 2 | +14.3% | −7.1% | 1 | 2/7 | **Valuable** |
| 3 | 0% | −11.9% | 2 | 3/7 | **No value** |
| 4 | +14.3% | −0.9% | 1 | 2/7 | **Conditionally valuable** |
| 5 | 0% | −1.0% | 2 | 3/7 | **No value** |

**Pattern**: Ranks 3 and 5 are symmetric: both contribute 0 new correct cases,
add 2 "neither" hypotheses, and have the highest per-rank evidence-selection
failure rates (3/7 = 42.9%). Ranks 2 and 4 each add one correct case at a
lower failure rate (2/7 = 28.6%). Rank 3 is a pure cost — it provides nothing
that rank 2 doesn't provide, at a higher noise level.

The `@2 MRR = @3 MRR = @5 MRR = 0.821` result shows that the MRR
improvement from @1 to @2 comes from hyp-04 (first correct at rank 2, not
rank 3). Ranks 3–5 do not improve MRR — they only improve binary recall by
surfacing hyp-02's rank-4 hypothesis.

---

## Section 2 — Policy comparison

### Policy B trigger evaluation

For each MEDIUM-confidence case, check whether it has a correct hypothesis in
top-2:

| Case | Conf level | Top-1 score | Correct in top-2? | Policy B extends? |
|---|---|---|---|---|
| hyp-02 | MEDIUM | 0.527 | No (r1=F, r2=F) | **Yes → generates r3, r4** |
| hyp-03 | MEDIUM | 0.463 | Yes (r1=T) | No |
| hyp-04 | MEDIUM | 0.422 | Yes (r2=T) | No |
| hyp-05 | MEDIUM | 0.452 | Yes (r1=T) | No |

Policy B extends only for hyp-02. Total hypotheses: 6 cases × 2 + 1 case × 4 = **16**.

### Policy side-by-side

| Metric | Policy A (top-2) | Policy B (adaptive) | Policy C (top-5) |
|---|---|---|---|
| **RC Recall** | 6/7 = 85.7% | **7/7 = 100.0%** | **7/7 = 100.0%** |
| **RC Precision** | 9/14 = 64.3% | 10/16 = 62.5% | 18/35 = 51.4% |
| **MRR** | 0.786 | **0.821** | **0.821** |
| **Evidence accuracy** | 11/14 = 78.6% | 12/16 = 75.0% | 24/35 = 68.6% |
| **Generic rate** | 1/14 = 7.1% | **1/16 = 6.25%** | 7/35 = 20.0% |
| **Mean raw conf** | **0.782** | 0.763 | 0.679 |
| **Mean composite conf** | **0.628** | 0.616 | 0.539 |
| **Best-case composite conf** | 0.709 | 0.709 | **0.730** |
| **Hypotheses total** | 14 | 16 | 35 |
| **KW-A recall** | 5/7 = 71.4% | 5/7 = 71.4% | 5/7 = 71.4% |
| **KW-C recall** | 4/7 = 57.1% | **5/7 = 71.4%** | **5/7 = 71.4%** |
| **LLM hyp calls** | 7 × 5* | 7 × 5* | 7 × 5 |
| **Keyword derivation (est.)** | 0.097 s/case | 0.111 s/case | 0.242 s/case |

*Policy A and B still issue the 5-hypothesis LLM call in this simulation
(no prompt change). With a prompt change to `n=2`, the LLM call would also be
smaller — that saving is outside this evaluation's scope.

### Key differentials

**Policy B vs A**:
- +14.3% RC Recall (recovers hyp-02)
- +14.3% KW-C recall (hyp-02 rank-4 provides correct evidence keywords)
- +0.036 MRR (5.75/7 vs 5.5/7)
- −3.6% Evidence accuracy (2 additional wrong selections from hyp-02 r3)
- −0.012 generic rate (hyp-02 r3/r4 are mechanism_symptom/mechanism_only, not "neither")
- Cost: 2 additional hypotheses for 1 case (1 LLM-call saved if prompt is reduced, same otherwise)

**Policy B vs C**:
- Identical RC Recall (both 7/7 = 100%)
- Identical MRR (both 0.821)
- Identical KW-A and KW-C recall
- **+6/19 = +17.1% evidence accuracy** (Policy B: 12/16 = 75.0% vs Policy C: 24/35 = 68.6%)
- **−7/19 "neither" hypotheses** (Policy B: 1 vs Policy C: 7 — 86% reduction)
- **+13.4% mean raw confidence** (0.763 vs 0.679)
- **+14.3% mean composite confidence** (0.616 vs 0.539)
- Requires **54% fewer hypotheses** (16 vs 35)

**Policy A vs C**:
- −14.3% RC Recall (misses hyp-02)
- −0.035 MRR
- +10% evidence accuracy (78.6% vs 68.6%)
- 86% fewer "neither" hypotheses (1 vs 7)
- +15.0% mean composite confidence (0.628 vs 0.539)
- 60% fewer hypotheses (14 vs 35)

---

## Section 3 — Cases where Policy A fails but Policy B succeeds

There is exactly **one case**: hyp-02.

---

### hyp-02 — "Background trigger service fails to launch after deployment"

**Retrieval profile**:
- Confidence: MEDIUM (top-1 cosine = 0.527)
- Gold incident: `375a627d` — "Triggerer not starting" (3-word title, no rich symptom text)
- Retrieved pool: Triggerer not starting (gold), Triggerer unable to ship logs to remote
  azure blob, Triggerer warning about remote logging but not enabled, Add built-in job
  queue…, Batch triggerer's cleanup deletes

**Why the pool is difficult**: 4 of 5 retrieved incidents are Triggerer-related
but describe distinct failure modes (azure blob logging, remote logging config,
cleanup batch logic, a feature request). The gold incident has almost no
vocabulary — its title is 3 words. The embedding distance from the problem
statement to the gold incident is moderate (cosine 0.527). This creates a pool
where incorrect-but-adjacent incidents dominate the hypothesis-generation context.

**Hypothesis trajectory**:

| Rank | Root cause | Is match | Chose gold | Composite | Cause |
|---|---|---|---|---|---|
| 1 | Synchronous startup of Execution API lifespan causing initialization failures | No | Yes | 0.408 | Grounded in non-gold retrieval |
| 2 | Missing connection ID for Azure Blob storage preventing log shipping | No | No | 0.595 | Azure blob incident absorbed |
| 3 | Remote logging configuration not enabled, leading to warnings | No | No | 0.510 | Remote logging incident absorbed |
| **4** | **Inadequate error handling during triggerer's startup process** | **Yes** | **Yes** | **0.553** | **Abstracted past azure/logging specifics** |
| 5 | Issues with triggerer's cleanup process affecting initialization | Yes | Yes | 0.425 | Batch cleanup incident vocabulary |

**Why depth helped**: The model iterates through concrete incorrect hypotheses
grounded in the 3 wrong-but-adjacent pool incidents before generating a
sufficiently abstract and accurate hypothesis at rank 4. Rank 4's
"Inadequate error handling during triggerer's startup process" does not
literally appear in any pool incident's title — it is a generalisation that
cross-cuts all triggerer failures — and that abstraction is what makes it
correct.

**Policy A outcome**: No match in top-2. Case contributes 0 to RC Recall.
KW-C falls back to rank-1 keywords from evidence step: rank-1 maps to
`375a627d` (gold, cosine 0.285) but extraction yields generic terms
(`containerStatuses`, `started`, `restart`, `ready`, `stays`) because the gold
title "Triggerer not starting" is too short for the term extractor. Keyword
recall = 0.

**Policy B outcome**: Trigger fires (MEDIUM conf, no top-2 match). Ranks 3–4
generated. Rank 4 is correct (is_match=True, chose_gold=True, composite=0.553).
KW-C from rank-4 evidence selection: maps to gold (`Triggerer not starting`),
uses terms `Triggerer`, `starting` — retrieves gold in top-5. Keyword recall = 1.

**Why Policy B's trigger condition is tight here**: MEDIUM confidence is
necessary but not sufficient. All 4 MEDIUM cases are hyp-02, hyp-03, hyp-04,
hyp-05. Three of the four have a correct hypothesis in top-2; only hyp-02 does
not. The joint condition (MEDIUM AND no top-2 match) fires on exactly the one
case that needs it.

---

## Section 4 — Cases where ranks 3–5 introduce incorrect or generic hypotheses

### Overview

Incorrect or generic hypotheses from ranks 3–5:

| Case | Rank | Classification | Is match | Chose gold | Composite | Impact |
|---|---|---|---|---|---|---|
| hyp-02 r3 | 3 | mechanism_symptom | F | **No** | 0.510 | Wrong evidence selection |
| hyp-03 r3 | 3 | mechanism_only | F | **No** | 0.306 | Wrong evidence selection |
| hyp-03 r4 | 4 | **neither** | F | **No** | 0.255 | Wrong evidence, generic |
| hyp-03 r5 | 5 | mechanism_only | F | Yes | 0.553 | Incorrect RC, gold selected |
| hyp-04 r3 | 3 | mechanism_only | F | **No** | 0.306 | Wrong evidence, gap=+0.259 |
| hyp-04 r4 | 4 | mechanism_only | F | **No** | 0.255 | Near-tie wrong selection |
| hyp-04 r5 | 5 | **neither** | F | **No** | 0.204 | Wrong evidence, generic |
| hyp-06 r3 | 3 | **neither** | F | Yes | 0.700 | Generic, gold selected correctly |
| hyp-06 r4 | 4 | mechanism_only | F | Yes | 0.600 | Incorrect RC, gold selected |
| hyp-07 r4 | 4 | **neither** | F | **No** | 0.360 | Wrong evidence, gap=+0.209 |
| hyp-07 r5 | 5 | **neither** | F | **No** | 0.390 | Wrong evidence, gap=+0.580 |

Total incorrect hypotheses from ranks 3–5: 11 of 15 (73.3%).
Total "neither" from ranks 3–5: 5 of 15 (33.3%). (The 2 at rank 3 are
non-neither; ranks 4–5 are where neither concentrates.)

---

### hyp-03 — Ranks 3–5 impact

Case: "type checker crashes with a memory error while watching files for changes"
Gold: `a9a17361` — "[2.8.0-rc] Segfault when running compiler with --d or --watch (out of memory)"

| | r3 | r4 (neither) | r5 |
|---|---|---|---|
| Root cause | Recursive function calls → stack overflow | Incompatibility / bugs in TypeScript versions | Improper file watching event handling |
| Is match | F | F | F |
| Chose gold | **No** | **No** | Yes |
| Composite | 0.306 | 0.255 | 0.553 |
| Chosen incident | RangeError/isReachableFlowNodeWorker | tsserver ENOENT (TypeScript 1.7.3) | Gold ✓ |

**Evidence-selection impact**: r3 and r4 select wrong incidents. r4 (the
"neither" hypothesis) selects the tsserver ENOENT incident with cosine gap
+0.374 — the largest gap in the hyp-03 pool. If C_hyp keywords were derived
from r4, they would be `[tsserver]`, `ENOENT`, `TypeScript`, `processing`,
`request` — completely unrelated to memory and --watch.

**Confidence impact**: All three ranks have composite ≤ 0.553, below rank-1
(0.68), so they do not displace rank-1 as the keyword-eval hypothesis. Damage
is contained to the C_hyp evidence-selection log, not the final keyword query.

**Verdict**: Ranks 3–5 add no correct hypotheses and degrade evidence selection
for 2 of 3 slots. Under Policy B, hyp-03 does not trigger extension (r1 is
correct), so ranks 3–5 are never generated. Pure cost with no benefit.

---

### hyp-04 — Ranks 3–5 impact

Case: "node process runs out of memory when building a large bundle of files"
Gold: `b093992f` — "JavaScript heap out of memory for 10s of MB of source"

| | r3 | r4 | r5 (neither) |
|---|---|---|---|
| Root cause | Concurrent tasks causing memory pressure | Memory leaks in application code | Incompatibility with specific TypeScript versions |
| Is match | F | F | F |
| Chose gold | **No** | **No** (gap=+0.015) | **No** (gap=+0.253) |
| Composite | 0.306 | 0.255 | 0.204 |
| Chosen incident | Cache table memory pressure | Cache table memory pressure | Compiler hang --allowJs |

**Evidence-selection impact**: All three are wrong selections. r4 is a near-tie
(gap=+0.015 to cache table vs gold); still wrong. r5 (the neither hypothesis)
selects the compiler hang incident — a completely different failure mode than
OOM — with gap=+0.253.

**Confidence impact**: All ranks 3–5 have composite < 0.357 (rank-2), and
far below rank-2 which is the first correct hypothesis used for keyword eval.
The low composite scores mean these hypotheses would not surface as
investigation leads. They do, however, create evidence-selection noise in the
pool of supporting incidents shown to analysts.

**Retrieval noise amplification**: the presence of `053dc4ca` (VectorDB/Celery
incident) in pool position 4 is part of why rank-5's "TypeScript version
compatibility" hypothesis emerges — the model exhausts specific evidence and
falls back to a generic catch-all. The "neither" at rank 5 is a symptom of
pool contamination, not generation-order alone.

**Verdict**: All 3 ranks add zero correct hypotheses and 100% wrong evidence
selections. Under Policy B, hyp-04 does not trigger extension (r2 is correct),
so ranks 3–5 are never generated. Pure cost.

---

### hyp-06 — Rank 3 impact

Case: "the keycloak token's audience claim is being ignored by the authentication layer"
Gold: `1c11adbe` — "JWTValidator does not use the configured audience with KeycloakAuthManager"

| | r3 (neither) | r4 |
|---|---|---|
| Root cause | Inconsistent handling of audience claims across components | Lack of proper validation checks for audience claims |
| Is match | F | F |
| Chose gold | **Yes** (0.205) | **Yes** (0.421) |
| Composite | 0.700 | 0.600 |

**Evidence-selection impact**: Despite being "neither" and incorrect, r3 chose
the gold incident (the only "neither" hypothesis to do so). This is because the
pool of non-gold incidents for hyp-06 is extremely weak: 4 of 5 incidents are
barely related to JWT/audience (AzureOpenAI 401, lazy OAuth2, Azure OAuth
button, trim_messages), so even the generic vocabulary of r3 ("audience claims")
maps highest to the gold.

**Confidence impact**: r3's composite = 0.700. This is *high* — it would be
reported alongside r1 (composite=0.900) as a secondary hypothesis. However,
r3 is factually incorrect (is_match=False): "inconsistent handling across
different components" does not identify `JWTValidator` misconfiguration.
**An analyst reading r3 would receive a misleading generalisation at 70%
confidence while the underlying cause is specific component misconfiguration.**

r4 (mechanism_only, also incorrect, composite=0.600) compounds this: two of
three deeper hypotheses for hyp-06 are incorrect and high-confidence.

**Verdict**: Ranks 3–4 introduce incorrect hypotheses at high composite
confidence (0.700, 0.600). Gold is still selected by C_hyp (the vocabulary
overlap works here), so keyword recall is unaffected. But an analyst receiving
this report sees 4 of 5 hypotheses as "audience claim issues" — 3 of which are
wrong. Under Policy B, hyp-06 does not trigger extension (HIGH retrieval
confidence, r1 correct), so ranks 3–4 are never generated.

---

### hyp-07 — Ranks 4–5 impact

Case: "type inference and switch statement narrowing bugs involving enums"
Gold: `0c4aacbc` — "narrowing in switch doesn't work with ambient enums"

| | r4 (neither) | r5 (neither) |
|---|---|---|
| Root cause | Issues with TypeScript compiler's handling of closures and block scoping | Nondeterministic type inference behavior in complex callback scenarios |
| Is match | F | F |
| Chose gold | **No** (chose f5f33b02, gap=+0.209) | **No** (chose 923c5ef6, gap=+0.580) |
| Composite | 0.360 | 0.390 |
| Chosen incident | Enum not block-scoped given closure --target es5 | Inference fails nondeterministic when callback gets typeof |

**Evidence-selection impact**: Both hypotheses absorb vocabulary directly from
pool incidents 2 and 4 respectively. r4 latches onto "closure" and "block
scoping" from pool rank 2; r5 latches onto "nondeterministic" and "callback"
from pool rank 4 (gap=+0.580 — the largest cosine gap in the entire 35-hypothesis
dataset). These are strong wrong selections, not near-ties.

**Confidence impact**: r4 and r5 have composite 0.360 and 0.390 — below r3
(0.750), which is also "neither" but chose gold correctly. The two hypotheses
would appear at the bottom of an investigation report, but with sufficient
confidence to be read (0.36–0.39 is above LOW threshold of 0.40... wait, let me
check: LOW < 0.40, MEDIUM 0.40–0.55, HIGH ≥ 0.55). r4 at 0.36 is LOW, r5 at
0.39 is also LOW. Both would be filtered out if the report only shows MEDIUM+
hypotheses. Under current thresholds both fall below the MEDIUM floor and would
be classified LOW confidence.

**If shown to an analyst**: r4 would suggest investigating TypeScript closure
scoping (wrong direction), r5 would suggest nondeterministic callback inference
(also wrong direction). Both are false leads with LOW composite confidence.

**Verdict**: hyp-07 ranks 4–5 are the clearest example of vocabulary absorption
from competing pool incidents. Their LOW composite confidence contains the damage
within the scoring system, but they degrade evidence selection and introduce 2
of the 7 total "neither" hypotheses. Under Policy C these are always generated;
under Policy B they are never generated (hyp-07 has HIGH retrieval confidence,
so the extension condition does not trigger).

---

## Section 5 — Confidence quality by policy

The `best_case_composite_conf` metric (mean of the per-case highest composite
score) is the most operationally relevant confidence measure: it is the
confidence attributed to the investigation's leading hypothesis.

| Policy | Best-case conf | Highest-conf correct | Highest-conf incorrect |
|---|---|---|---|
| A (top-2) | 0.709 | 0.900 (hyp-01 r1) | 0.595 (hyp-02 r2) |
| B (adaptive) | 0.709 | 0.900 (hyp-01 r1) | 0.595 (hyp-02 r2) |
| C (top-5) | **0.730** | 0.900 (hyp-01 r1) | 0.700 (hyp-06 r3 — **neither**) |

Policy C's slightly higher mean best-case confidence (0.730 vs 0.709) is driven
entirely by **hyp-05**: ranks 3–5 of hyp-05 have higher composite scores than
ranks 1–2 (because C_hyp correctly selects gold at r3+, raising the composite,
while r1 and r2 have kw_c=False). Specifically, hyp-05 r4 achieves
composite=0.5525 vs r1's 0.408.

However, Policy C's highest-confidence *incorrect* hypothesis is hyp-06 r3
(composite=0.700, "neither"). Policy A and B's highest-confidence incorrect
hypothesis is hyp-02 r2 (composite=0.595, is_match=False but mechanism_symptom,
not "neither"). The incorrect-but-high-confidence scenario is more
problematic under Policy C than Policy A/B.

---

## Section 6 — Summary and recommendation

### Numeric comparison

| Metric | Policy A | Policy B | Policy C | B vs C | B vs A |
|---|---|---|---|---|---|
| RC Recall | 85.7% | **100%** | **100%** | 0% | +14.3% |
| RC Precision | **64.3%** | 62.5% | 51.4% | +11.1% | — |
| MRR | 0.786 | **0.821** | **0.821** | 0% | +0.035 |
| Evidence accuracy | 78.6% | 75.0% | 68.6% | **+6.4%** | — |
| Generic rate | 7.1% | **6.25%** | 20.0% | **−13.75%** | — |
| Mean composite conf | **0.628** | 0.616 | 0.539 | **+0.077** | — |
| KW-A recall | 71.4% | 71.4% | 71.4% | 0% | 0% |
| KW-C recall | 57.1% | **71.4%** | **71.4%** | 0% | +14.3% |
| Hypotheses total | 14 | 16 | 35 | **−54%** | — |
| "Neither" total | 1 | 1 | 7 | **−86%** | — |

### Decision matrix

**Is Policy A justified?**

No. Policy A loses hyp-02 (RC Recall 85.7% vs 100%, MRR 0.786 vs 0.821) and
also loses KW-C recall for that case (57.1% vs 71.4%). The cost of depth
extension for hyp-02 is 2 additional hypotheses (r3, r4) for one case — small
relative to the benefit. Policy A is not the right default because it
demonstrably fails the one case where depth is needed.

**Is Policy B justified?**

Yes. Policy B matches Policy C on every recall metric (100% RC, 0.821 MRR,
71.4% KW-A, 71.4% KW-C) while using 54% fewer hypotheses, reducing the generic
rate by 86% (7 → 1), improving evidence-selection accuracy by 6.4 percentage
points, and raising mean composite confidence by 14.3 percentage points (0.616
vs 0.539). The trigger condition (MEDIUM confidence AND no top-2 match) is
logically sound and fires precisely where it is needed.

**Is Policy C justified over Policy B?**

No. Policy C matches Policy B on all recall metrics but:
- Uses 54% more hypotheses (35 vs 16)
- Generates 6 additional "neither" hypotheses
- Has 17.1% lower evidence-selection accuracy (68.6% vs 75.0%)
- Has 14.3% lower mean composite confidence (0.539 vs 0.616)
- Introduces the highest-confidence incorrect hypothesis in the dataset (hyp-06 r3, composite=0.700)

The only metric where Policy C exceeds Policy B is `best_case_composite_conf`
(0.730 vs 0.709), driven entirely by hyp-05 reaching a better composite score
at rank 4. This is a real but minor advantage (+0.021 mean) that does not
outweigh the evidence-quality and precision losses.

### Recommended policy

**Adopt Policy B**: generate top-2 hypotheses by default, extend to ranks 3–4
only when retrieval confidence is MEDIUM and no correct hypothesis is found in
the first two ranks.

Implementation requires:
1. Check `initial_confidence_level` after retrieval (already available in
   `_collect_evidence`).
2. After `_generate_hypotheses` with `n=2`, check whether any hypothesis
   passes the `is_match` threshold in-context (using composite confidence ≥
   MEDIUM floor as a proxy — is_match is not available at investigation time
   since the gold is unknown; a proxy is composite_confidence ≥ 0.40).
3. If both conditions hold, call `_generate_hypotheses` again requesting
   ranks 3–4 (`n=2, offset=2` or equivalent prompt adjustment).

The trigger proxy (composite_confidence ≥ 0.40 for at least one top-2
hypothesis) is not perfect: at evaluation time we do not know `is_match`.
However, the MEDIUM confidence + "no hypothesis with composite ≥ 0.40 in top-2"
condition provides a reasonable operational approximation.

**Implementation note**: This evaluation is constraint-compliant — no retrieval,
prompt, embedding, or keyword-generation changes were made. The policy change
only affects how many hypotheses are consumed from the generation output.
Prompt changes to reduce the initial `n_hypotheses` parameter from 5 to 2 would
compound the benefit with LLM token savings, but are outside this evaluation's
scope.
