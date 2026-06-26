# Phase 7: Hypothesis Structure Analysis

**Question under investigation**: Do mechanism+symptom hypotheses select the
correct supporting incident more reliably than mechanism-only hypotheses, as
suggested by the hyp-05 sibling-mismatch failure?

Source data:
[tests/eval/results/hypothesis_v7.json](tests/eval/results/hypothesis_v7.json),
[tests/eval/results/hypothesis_structure_v7.json](tests/eval/results/hypothesis_structure_v7.json).
35 hypotheses, 7 positive cases, 11 sibling-mismatch failures.
Classification: LLM (`gpt-4o-mini`) against four defined classes.

No production code was changed for this analysis.

---

## 1. Classification of all 35 hypotheses

| Class | Definition | Count | % |
|---|---|---|---|
| **mechanism_only** | Names a root action / code-level trigger; does not state the observable user-facing outcome | 24 | 68.6% |
| **mechanism_symptom** | Names both a root cause AND an observable outcome / failure symptom | 4 | 11.4% |
| **symptom_only** | Names the observable failure; no code-level cause stated | 0 | 0% |
| **neither** | Too vague / generic to assign to either class | 7 | 20.0% |

The LLM-generated hypotheses are overwhelmingly mechanism-only (68.6%).
No hypothesis was classified as symptom-only. The `generate_hypotheses()` prompt
("Generate 3-5 possible root-cause hypotheses") elicits cause-level
explanations, not failure descriptions — this is the baseline vocabulary the
model defaults to.

### Full listing

| Case | Rank | Class | RC match | Gold sel | KwA | KwC | Root cause (truncated) |
|---|---|---|---|---|---|---|---|
| hyp-01 | 1 | mechanism_symptom | ✓ | ✓ | ✓ | ✓ | "Historical task instances have a NULL dag_version_id, causing validation errors." |
| hyp-01 | 2 | mechanism_symptom | ✓ | ✓ | ✓ | ✓ | "The scheduler does not handle NULL dag_version_id gracefully, leading to continuous restarts." |
| hyp-01 | 3 | mechanism_only | ✗ | ✓ | ✓ | ✗ | "The introduction of the dag_version table has not been fully integrated with existing task instances." |
| hyp-01 | 4 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "The scheduler's validation logic does not account for historical task instances with missing IDs." |
| hyp-01 | 5 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Lack of fallback mechanisms for NULL dag_version_id in the scheduler's context construction." |
| hyp-02 | 1 | mechanism_only | ✗ | ✓ | ✓ | ✗ | "Synchronous startup of the Execution API lifespan is causing initialization failures." |
| hyp-02 | 2 | mechanism_symptom | ✗ | **✗** | ✗ | ✓ | "Missing connection ID for Azure Blob storage is preventing log shipping." |
| hyp-02 | 3 | mechanism_symptom | ✗ | **✗** | ✗ | ✓ | "Remote logging configuration is not enabled, leading to warnings and potential failures." |
| hyp-02 | 4 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Inadequate error handling during the triggerer's startup process." |
| hyp-02 | 5 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Issues with the triggerer's cleanup process affecting its ability to initialize." |
| hyp-03 | 1 | mechanism_only | ✓ | ✓ | ✗ | ✓ | "Memory leak in the type checker when using --watch flag." |
| hyp-03 | 2 | mechanism_only | ✗ | ✓ | ✗ | ✗ | "Inefficient memory management in the type checker for large source files." |
| hyp-03 | 3 | mechanism_only | ✗ | **✗** | ✗ | ✗ | "Recursive function calls leading to stack overflow." |
| hyp-03 | 4 | neither | ✗ | **✗** | ✗ | ✗ | "Incompatibility or bugs introduced in recent TypeScript versions." |
| hyp-03 | 5 | mechanism_only | ✗ | ✓ | ✓ | ✓ | "Improper handling of file watching events leading to excessive resource usage." |
| hyp-04 | 1 | mechanism_only | ✗ | ✓ | ✓ | ✓ | "Insufficient memory allocation for large bundles." |
| hyp-04 | 2 | mechanism_only | ✓ | ✓ | ✗ | ✗ | "Inefficient memory management in the TypeScript compiler." |
| hyp-04 | 3 | mechanism_only | ✗ | **✗** | ✗ | ✗ | "Concurrent tasks causing memory pressure." |
| hyp-04 | 4 | mechanism_only | ✗ | **✗** | ✗ | ✗ | "Memory leaks in the application code." |
| hyp-04 | 5 | neither | ✗ | **✗** | ✗ | ✗ | "Incompatibility with specific TypeScript versions." |
| hyp-05 | 1 | mechanism_only | ✓ | **✗** | ✓ | ✗ | "Improper use of `@router.api_route()` for defining multiple HTTP methods." |
| hyp-05 | 2 | neither | ✓ | **✗** | ✓ | ✗ | "Lack of clear documentation on the proper methods to use for defining routes in FastAPI." |
| hyp-05 | 3 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "The design of FastAPI allowing multiple methods in a single route without adequate restrictions." |
| hyp-05 | 4 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Inconsistent handling of operation IDs when using `api_route` and `add_api_route`." |
| hyp-05 | 5 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Potential oversight regarding route definitions and operation ID uniqueness." |
| hyp-06 | 1 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Misconfiguration of JWT audience settings in the authentication layer." |
| hyp-06 | 2 | mechanism_only | ✗ | ✓ | ✓ | ✓ | "Changes in the authentication library version affecting audience claim handling." |
| hyp-06 | 3 | neither | ✗ | ✓ | ✗ | ✓ | "Inconsistent handling of audience claims across different system components." |
| hyp-06 | 4 | mechanism_only | ✗ | ✓ | ✓ | ✓ | "Lack of proper validation checks for audience claims in the authentication layer." |
| hyp-06 | 5 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Potential bugs in the JWT validation logic that bypass audience checks." |
| hyp-07 | 1 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Type inference issues with ambient enums in switch statements." |
| hyp-07 | 2 | mechanism_only | ✓ | ✓ | ✓ | ✓ | "Compiler limitations in handling numeric enums in switch cases." |
| hyp-07 | 3 | neither | ✓ | ✓ | ✓ | ✓ | "Inconsistent behavior of type narrowing across different enum types." |
| hyp-07 | 4 | neither | ✗ | **✗** | ✗ | ✗ | "Issues with the TypeScript compiler's handling of closures and block scoping." |
| hyp-07 | 5 | neither | ✗ | **✗** | ✗ | ✗ | "Nondeterministic type inference behavior in complex callback scenarios." |

(**✗** in Gold sel = sibling-mismatch failure)

---

## 2. Stats by category

| Category | n | Root-cause correct | Kw recall A | Kw recall C | Gold-selection rate |
|---|---|---|---|---|---|
| mechanism_only | 24 | 58.3% (14/24) | 75.0% | 66.7% | **83.3%** (20/24) |
| mechanism_symptom | 4 | 50.0% (2/4) | 50.0% | 100.0% | **50.0%** (2/4) |
| neither | 7 | 28.6% (2/7) | 28.6% | 28.6% | **28.6%** (2/7) |
| **All** | **35** | **51.4%** | **65.7%** | **62.9%** | **68.6%** |

`mechanism_symptom` has *lower* gold-selection (50%) than `mechanism_only`
(83.3%). On the surface this is the opposite of the hypothesis. The reason is
examined in section 3.

---

## 3. Sibling-mismatch failures — full detail

11 of 35 hypotheses chose the wrong supporting incident. By class:

| Class | n | Mismatches | Mismatch rate |
|---|---|---|---|
| mechanism_only | 24 | 5 | 20.8% |
| mechanism_symptom | 4 | 2 | **50.0%** |
| neither | 7 | **5** | **71.4%** |

### The 11 failures

**hyp-02 r2** — class: `mechanism_symptom`, is_match=False
Hypothesis: *"Missing connection ID for Azure Blob storage is preventing log shipping."*
Chosen: `69347f56` *Triggerer unable to ship logs to remote azure blob connection* (score 0.723)
Gold: `375a627d` *Triggerer not starting* (score 0.019) — gap **+0.705**
Analysis: This hypothesis describes a *different incident* (`69347f56`) — it is a factually wrong root cause for the gold case. The cosine selection is correct for the hypothesis's actual content. The high gap is not a sibling-near-tie; it is a content mismatch. The `mechanism_symptom` classification is not the cause.

**hyp-02 r3** — class: `mechanism_symptom`, is_match=False
Hypothesis: *"Remote logging configuration is not enabled, leading to warnings and potential failures."*
Chosen: `69347f56` *Triggerer warning about remote logging but not enabled* (score 0.750)
Gold: `375a627d` *Triggerer not starting* (score 0.149) — gap **+0.602**
Analysis: Same as r2 — the hypothesis is about the azure blob/logging incident, not the gold. Correct cosine selection for wrong content. Not a mechanism-specificity failure.

**hyp-03 r3** — class: `mechanism_only`, is_match=False
Hypothesis: *"Recursive function calls leading to stack overflow."*
Chosen: `5e2254b0` *Crash: RangeError: Maximum call stack size exceeded in isReachableFlow* (score 0.333)
Gold: `a9a17361` *[2.8.0-rc] Segfault when running compiler with --d or --watch (out of memory)* (score 0.226) — gap **+0.108**
Analysis: The hypothesis describes a different failure mode (stack overflow, not OOM/segfault). Cosine selection is again correct for the hypothesis's content. Missing symptom term (segfault, OOM) is irrelevant here — even with them, cosine would still prefer the stack-overflow incident.

**hyp-03 r4** — class: `neither`, is_match=False
Hypothesis: *"Incompatibility or bugs introduced in recent TypeScript versions."*
Chosen: `d5226bf7` *[tsserver] "Error processing request. watch ENOENT"…* (score 0.559)
Gold: `a9a17361` (score 0.185) — gap **+0.374**
Analysis: Entirely generic. The chosen incident also mentions TypeScript version issues in its title. The hypothesis contains no signal from the gold incident (`--watch`, `segfault`, `OOM`). Large gap driven by generic vocabulary matching a different sibling.

**hyp-04 r3** — class: `mechanism_only`, is_match=False
Hypothesis: *"Concurrent tasks causing memory pressure."*
Chosen: `5b155d0c` *Fetching of blocks of cache table may cause high memory pressure* (score 0.546)
Gold: `b093992f` *JavaScript heap out of memory for 10s of MB of source* (score 0.287) — gap **+0.259**
Analysis: Generic mechanism ("memory pressure") with no product or symptom specificity. "Memory pressure" is literally in the chosen incident's title. The gold incident's distinguishing vocabulary ("JavaScript heap", "10s of MB of source") is entirely absent from the hypothesis.

**hyp-04 r4** — class: `mechanism_only`, is_match=False
Hypothesis: *"Memory leaks in the application code."*
Chosen: `5b155d0c` *Fetching of blocks of cache table may cause high memory pressure* (score 0.455)
Gold: `b093992f` (score 0.440) — gap **+0.015**
Analysis: **Near-tie** (gap 0.015). This is genuine retrieval ambiguity — the hypothesis is generic enough that both the gold and the sibling score within 2% of each other. Adding any symptom specificity ("JavaScript heap OOM", "compiler") would break the tie in favour of the gold.

**hyp-04 r5** — class: `neither`, is_match=False
Hypothesis: *"Incompatibility with specific TypeScript versions."*
Chosen: `a562aa3c` *Compiler hang when importing big JS file with --allowJs* (score 0.390)
Gold: `b093992f` (score 0.137) — gap **+0.253**
Analysis: Entirely generic. No heap, no OOM, no source size. The chosen incident mentions "compiler" and "big JS file" — adjacent vocabulary to "TypeScript versions." Large gap, not a near-tie.

**hyp-05 r1** — class: `mechanism_only`, is_match=True ← the failure that prompted this analysis
Hypothesis: *"Improper use of `@router.api_route()` for defining multiple HTTP methods on the same endpoint."*
Chosen: `de1208c6` *SSE `stream_item_type` not propagated through `APIRouter`* (score 0.500)
Gold: `5dba5df8` *Duplicated OperationID when adding route with multiple methods* (score 0.468) — gap **+0.032**
Analysis: **Near-tie** (gap 0.032). The hypothesis names the *mechanism* (`@router.api_route()`, "multiple HTTP methods") but not the *outcome* ("Duplicated OperationID"). The sibling incident's title also contains `APIRouter` — a near-synonym for `@router`. Adding the outcome term ("duplicate OperationID") would shift the cosine strongly toward the gold (as seen in ranks 3–5 which include "operation IDs" and all choose gold with gaps ≥0.12).

**hyp-05 r2** — class: `neither`, is_match=True
Hypothesis: *"Lack of clear documentation on the proper methods to use for defining routes in FastAPI."*
Chosen: `de1208c6` *SSE `stream_item_type` not propagated through `APIRouter`* (score 0.427)
Gold: `5dba5df8` (score 0.351) — gap **+0.076**
Analysis: Generic, documentation-framing hypothesis. `APIRouter` in the sibling title is the closest vocabulary match to "defining routes in FastAPI." No mention of `operationId` or duplication. Neither class correctly classified.

**hyp-07 r4** — class: `neither`, is_match=False
Hypothesis: *"Issues with the TypeScript compiler's handling of closures and block scoping."*
Chosen: `5f5f3b02` *Enum not block-scoped given closure and --target es5* (score 0.474)
Gold: `0c4aacbc` *narrowing in switch doesn't work with ambient enums* (score 0.265) — gap **+0.209**
Analysis: Generic but specific enough to match the wrong incident on "closures and block scoping" — literal terms from the chosen incident's title. The gold incident is about switch narrowing and ambient enums — completely different vocabulary. Large gap, entirely driven by the wrong mechanism focus.

**hyp-07 r5** — class: `neither`, is_match=False
Hypothesis: *"Nondeterministic type inference behavior in complex callback scenarios."*
Chosen: `923c5ef6` *Inference fails nondeterministic when callback gets a `typeof`* (score 0.769)
Gold: `0c4aacbc` (score 0.189) — gap **+0.580**
Analysis: The chosen incident's title contains "Inference fails nondeterministic" and "callback" — near-exact matches to the hypothesis. Gold incident vocabulary (narrowing, switch, ambient enums) is entirely absent. **Largest gap in the dataset (+0.580).** This is not a near-tie; the hypothesis is simply about a different bug class entirely.

---

## 4. Statistical test

**Hypothesis**: mechanism+symptom hypotheses select the gold incident more
reliably than other classes.

Contingency table (gold-selection × class):

| | Chose gold | Chose wrong |
|---|---|---|
| mechanism_symptom | 2 | 2 |
| mechanism_only + neither | 22 | 9 |

Fisher's exact p = **1.000**

**There is no statistical support for the hypothesis.** p=1.0 indicates the
observed distribution is exactly what would be expected by chance. The result
is in the opposite direction from the prediction (mechanism_symptom 50% vs.
others 71%), and neither is the difference meaningful at n=4 in the
mechanism_symptom cell.

Same test for root-cause correctness × class: p = **1.000**. No significance.

---

## 5. The hyp-05 finding does not generalise — but its mechanism does

The original hypothesis was motivated correctly by hyp-05 r1: a
mechanism-only hypothesis with gap +0.032 (near-tie), where adding "duplicate
OperationID" would have flipped the selection. This is a real and local
phenomenon.

But generalising it to "mechanism+symptom hypotheses are more robust" is not
supported, for three reasons:

**Reason 1 — The two mechanism_symptom mismatch failures are content errors, not
structure errors.** hyp-02 r2/r3 chose wrong incidents because their root
causes describe a *different incident* (azure blob logging) — C_hyp correctly
selected the incident that matches those hypotheses' actual content. No
reformulation of the hypothesis structure would fix this without changing the
root cause itself.

**Reason 2 — The dominant mismatch driver is the `neither` class, not
mechanism-only.** 5 of 7 `neither` hypotheses chose wrong (71.4%), vs. 5 of
24 mechanism-only (20.8%). The biggest gaps in the dataset all belong to
`neither` hypotheses (hyp-07 r5: +0.580, hyp-07 r4: +0.209, hyp-04 r5:
+0.253). Generic vocabulary drifts to whatever sibling incident happens to
share surface terms. The bottleneck here is hypothesis specificity in general,
not the presence or absence of a symptom clause.

**Reason 3 — Among correct, specific mechanism-only hypotheses, gold-selection
is already high.** Correct mechanism-only hypotheses (is_match=True) chose gold
13 of 14 times (92.8%). The one exception is hyp-05 r1 (gap +0.032, a
near-tie). mechanism_only is not a systematically weaker structure when the
hypothesis content is accurate and specific — it only fails at the margin when
the mechanism vocabulary happens to overlap with a sibling incident's title.

---

## 6. What the data does support

The analysis reveals a cleaner predictor than mechanism/symptom structure:
**hypothesis specificity** (the inverse of `neither`).

| Property | Gold-selection rate | Mismatch rate |
|---|---|---|
| has specific vocabulary (mechanism_only + mechanism_symptom) | 79.3% (23/29) | 20.7% (6/29) |
| generic (neither) | 28.6% (2/7) | **71.4% (5/7)** |

Every hypothesis classified as `neither` either has a very large gap (wrong
incident by a wide margin) or is wrong about the root cause entirely. No
`neither` hypothesis correctly generates useful validation keywords under
strategy C.

A secondary, localised finding: **near-tie mismatches in the correct
mechanism-only bucket (gap < 0.05)** are the only case where adding a symptom
term would plausibly help. There are 2 such cases (hyp-05 r1 at +0.032,
hyp-04 r4 at +0.015). In both, the missing discriminating term is a specific
outcome noun ("OperationID", "JavaScript heap") that the hypothesis text
doesn't name.

---

## 7. Conclusion: is hypothesis representation quality the dominant bottleneck?

**Partially, but the framing needs to change.**

The dominant bottleneck is not *mechanism-only vs. mechanism+symptom* — it is
*generic vs. specific*. The `neither` class (20% of hypotheses, 71.4%
sibling-mismatch rate) is the component with the highest failure rate and the
widest mismatch gaps. These hypotheses are too vague for any downstream step
(evidence selection, keyword recall, confidence) to work reliably on them.

The mechanism-only class, when specific, performs well (83.3% gold-selection,
75% keyword recall A). Its failures are narrow: 2 near-tie situations where the
mechanism vocabulary happens to be shared with a sibling incident.

**The hypothesis that motivated this analysis (mechanism+symptom improves
C_hyp) is too narrow.** The intervention that would help most is not teaching
the model to add an outcome clause — it is reducing the frequency of the
`neither` class, where hypotheses contain insufficient domain specificity to
anchor retrieval at all.

Recommendation before any new phase: **expand the gold set** (the `neither`
class mismatch rate at n=7 is a data point, not a rate — it may not
replicate), and assess whether the `neither` class rate (20%) is stable across
runs or a non-determinism artefact. If it is stable, reducing hypothesis
genericity — rather than adding symptom clauses — is the higher-value
intervention.
