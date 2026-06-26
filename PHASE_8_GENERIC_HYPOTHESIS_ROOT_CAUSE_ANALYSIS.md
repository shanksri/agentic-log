# Phase 8: Generic Hypothesis Root-Cause Analysis

**Question**: Why are the 7 "neither"-classified hypotheses being generated,
and what is the proximate cause of their genericity in each case?

Source data: `tests/eval/results/hypothesis_v7.json`,
`tests/eval/results/hypothesis_structure_v7.json`.
No production code was changed.

---

## Overview

7 of 35 hypotheses (20%) were classified as `neither` — too vague to assign to
either mechanism or symptom. Mapping them by case and rank:

| Case | Retrieval conf | Gold rank | "Neither" ranks |
|---|---|---|---|
| hyp-03 | MEDIUM (0.463) | 1 | 4 |
| hyp-04 | MEDIUM (0.422) | — (miss) | 5 |
| hyp-05 | MEDIUM (0.452) | 1 | 2 |
| hyp-06 | **HIGH (0.583)** | 1 | 3 |
| hyp-07 | **HIGH (0.731)** | 1 | 3, 4, 5 |

Two patterns visible immediately: "neither" hypotheses are exclusively rank 3+,
and they appear even when retrieval confidence is HIGH and the gold incident is
rank 1 in the retrieved pool.

---

## Per-hypothesis analysis

---

### 1 · hyp-03, rank 4

**Problem**: *the type checker crashes with a memory error while watching files
for changes*

**Retrieval confidence**: MEDIUM — top1 = 0.463

**Retrieved pool**:
1. `a9a17361` — [2.8.0-rc] Segfault when running compiler with --d or --watch (out of memory) **← GOLD**
2. `b093992f` — JavaScript heap out of memory for 10s of MB of source
3. `5e2254b0` — Crash: RangeError: Maximum call stack size exceeded in isReachableFlowNodeWorker…
4. `5497ce0e` — [tsserver] "Error processing request. watch ENOENT" with TypeScript 1.7.3
5. `f350c637` — Crash: RangeError: Maximum call stack size exceeded in addTypesToUnion…

**Generated hypothesis (rank 4)**:
*"Incompatibility or bugs introduced in recent TypeScript versions."*
Keywords: `TypeScript version`, `bugs`, `incompatibility`

**Best correct hypothesis (rank 1)**:
*"Memory leak in the type checker when using --watch flag."*
Keywords: `out of memory`, `--watch`, `memory leak`

**Cosine scores for this hypothesis**:
- 0.559 `[tsserver] "Error processing request. watch ENOENT"…` ← CHOSEN
- 0.327 `Crash: RangeError: Maximum call stack size exceeded in addTypesToUnion…`
- 0.185 gold
- 0.173 `JavaScript heap out of memory…`

**Why it became generic**: The retrieved pool contains four *distinct* crash
failure modes — segfault/OOM on `--watch`, heap OOM on large bundles, two stack
overflow variants, and a tsserver watch ENOENT error. By rank 4 the model has
already generated specific hypotheses anchored to ranks 1–3 (memory leak,
inefficient memory management, recursive stack overflow) and is left without a
remaining specific explanation grounded in the pool. It defaults to the only
unaddressed common thread: "TypeScript version compatibility" — a catch-all
that appears because the tsserver incident (`5497ce0e`, pool rank 4) mentions a
specific version ("TypeScript 1.7.3") and the model's training connects
"TypeScript bugs" to version-specific regressions. The "version incompatibility"
hypothesis is an abstraction across all four failure modes, not a reading of any
one incident.

**Primary cause**: **Model abstraction** — the model exhausted specific
grounded hypotheses at ranks 1–3 and generated a generic fallback at rank 4.
Secondary: **diverse pool** (4 different crash types with no dominant
explanatory thread after rank 3).

---

### 2 · hyp-04, rank 5

**Problem**: *node process runs out of memory when building a large bundle of
files*

**Retrieval confidence**: MEDIUM — top1 = 0.422

**Retrieved pool**:
1. `b093992f` — JavaScript heap out of memory for 10s of MB of source **← GOLD**
2. `5b155d0c` — Fetching of blocks of cache table may cause high memory pressure
3. `a562aa3c` — Compiler hang when importing big JS file with --allowJs
4. `053dc4ca` — Document ingestion into VectorDB does not complete when running inside Celery task
5. `a9a17361` — [2.8.0-rc] Segfault when running compiler with --d or --watch (out of memory)

**Generated hypothesis (rank 5)**:
*"Incompatibility with specific TypeScript versions."*
Keywords: `TypeScript version`, `fixed in TS7`, `compiler issues`

**Best correct hypothesis (rank 2)**:
*"Inefficient memory management in the TypeScript compiler."*
Keywords: `compiler hang`, `maximum call stack size exceeded`, `TypeScript`

**Cosine scores for this hypothesis**:
- 0.390 `Compiler hang when importing big JS file with --allowJs` ← CHOSEN
- 0.165 gold (rank 5 in cosine)
- 0.137 `JavaScript heap out of memory for 10s of MB of source`

**Why it became generic**: Two compounding problems:

First, `053dc4ca` (pool rank 4) is a **completely irrelevant incident** — a
VectorDB/Celery ingestion timeout — that has no relationship to TypeScript OOM.
It is a retrieval false positive that adds incoherent evidence. By the time the
model reaches rank 5 it has seen one VectorDB task, one Java cache OOM, and two
TypeScript OOM variants — a fragmented pool.

Second, the gold incident's resolution text contains *"Try `typescript@next`.
We have fixed all but two of the known issues there"* — a version-specific
resolution note. The model appears to have read this as evidence for a
"version incompatibility" hypothesis, which the resolution text superficially
supports even though the actual root cause is heap exhaustion on large source
bundles.

**Primary cause**: **Retrieval noise** — the irrelevant VectorDB/Celery
incident (`053dc4ca`) diluted the coherence of the pool. Contributing:
**model abstraction** from the resolution text's version reference.

---

### 3 · hyp-05, rank 2

**Problem**: *the generated API docs show the same operation identifier twice
for an endpoint that supports several HTTP verbs*

**Retrieval confidence**: MEDIUM — top1 = 0.452

**Retrieved pool**:
1. `5dba5df8` — Duplicated OperationID when adding route with multiple methods **← GOLD**
2. `1b2c535b` — Add support to lazy initialize `OAuth2` classes
3. `e1f61e5b` — `ActionSearchSource.type` is claimed to be always be `'url'`, but can be `'api'`
4. `539b17a1` — Responses API: parsed_arguments leaks from ParsedResponseFunctionToolCall…
5. `de1208c6` — SSE `stream_item_type` not propagated through `APIRouter` + `include_router`

**Generated hypothesis (rank 2)**:
*"Lack of clear documentation on the proper methods to use for defining routes
in FastAPI."*
Keywords: `documentation`, `FastAPI features`, `operation IDs`

**Best correct hypothesis (rank 1)**:
*"Improper use of `@router.api_route()` for defining multiple HTTP methods on
the same endpoint."*
Keywords: `Duplicated OperationID`, `multiple methods`, `api_route`

**Cosine scores for this hypothesis**:
- 0.427 `SSE stream_item_type not propagated through APIRouter…` ← CHOSEN (wrong)
- 0.351 gold
- 0.271 `ActionSearchSource.type…`
- 0.264 `Add support to lazy initialize OAuth2 classes`

**Why it became generic**: The problem statement says *"the generated API docs
show the same operation identifier twice"* — the phrase "API docs" frames the
symptom as a documentation artifact, which the model at rank 2 picks up and
turns into a documentation-gap hypothesis. This is a **problem-statement framing
effect**: "docs show" implies the issue is in what gets displayed rather than in
the schema-generation code that produces the duplicate.

The retrieved pool amplifies this: only 1 of 5 incidents (`5dba5df8`) is
relevant; the remaining 4 are unrelated FastAPI issues that provide no
counter-signal. The model lacks evidence to anchor a second specific hypothesis,
so it falls back to the problem-statement's surface framing ("documentation").

**Primary cause**: **Problem statement ambiguity** — "API docs show the same
operation identifier twice" is read as a documentation failure. Contributing:
**sparse retrieved context** (only 1 of 5 incidents is relevant, and that
incident has no symptom text beyond its title).

---

### 4 · hyp-06, rank 3

**Problem**: *the keycloak token's audience claim is being ignored by the
authentication layer*

**Retrieval confidence**: **HIGH** — top1 = 0.583

**Retrieved pool**:
1. `1c11adbe` — JWTValidator does not use the configured audience with
   KeycloakAuthManager **← GOLD**
2. `608271f5` — AzureOpenAI with AAD bearer token passed via api_key works in
   2.33.0 but returns 401 in 2.34.0 and after
3. `1b2c535b` — Add support to lazy initialize `OAuth2` classes
4. `0ee80306` — [Airflow 3.2.2] api-server: Azure OAuth button not shown at
   /auth/login/
5. `78d24aee` — `trim_messages` and `ChatAnthropic` token counter with tools

**Generated hypothesis (rank 3)**:
*"Inconsistent handling of audience claims across different components of the
system."*
Keywords: `audience claims`, `inconsistent handling`, `system components`

**Best correct hypothesis (rank 1)**:
*"Misconfiguration of JWT audience settings in the authentication layer."*
Keywords: `jwt_audience`, `configuration`, `KeycloakAuthManager`

**Cosine scores for this hypothesis**:
- 0.205 gold ← CHOSEN (correct despite being "neither")
- 0.087 `trim_messages and ChatAnthropic token counter`
- 0.040 `AzureOpenAI AAD bearer token 401`
- −0.007 `Add support to lazy initialize OAuth2 classes`
- −0.039 `Airflow Azure OAuth button not shown`

**Why it became generic**: This is the cleanest case of **pure model
abstraction**. The retrieval confidence is HIGH, the gold incident is rank 1,
and C_hyp correctly selects the gold for this hypothesis. There is no retrieval
or context problem.

The model at rank 3 — having already generated rank 1 ("misconfiguration of JWT
audience settings") and rank 2 ("changes in authentication library version") —
abstracts upward: it knows the domain (JWT, audience, authentication) but drops
the specific component names (`JWTValidator`, `KeycloakAuthManager`,
`configured audience`) in favour of a broader pattern ("inconsistent handling
across components"). This is the model maintaining output *diversity* by
generalising: rank 3 is a broader umbrella of the same class of issue, not a
different one.

**Primary cause**: **Model abstraction** — rank 3 is an intentional
generalisation of a well-grounded rank-1 hypothesis. The model had full evidence
but chose breadth over specificity at this rank. Note: this hypothesis still
chose gold correctly (cosine 0.205, dominant in its pool) so it caused no
sibling mismatch despite being "neither."

---

### 5 · hyp-07, rank 3

**Problem**: *type inference and switch statement narrowing bugs involving enums*

**Retrieval confidence**: **HIGH** — top1 = 0.731

**Retrieved pool**:
1. `0c4aacbc` — narrowing in switch doesn't work with ambient enums **← GOLD**
2. `f5f33b02` — Enum not block-scoped given closure and --target es5
3. `a21be2b7` — Inference problem
4. `923c5ef6` — Inference fails nondeterministic when callback gets a `typeof`…
5. `d7257140` — Incorrect type inference for array rest assignment…

**Generated hypothesis (rank 3)**:
*"Inconsistent behavior of type narrowing across different enum types."*
Keywords: `type narrowing`, `enum types`, `inconsistent behavior`

**Best correct hypothesis (rank 1)**:
*"Type inference issues with ambient enums in switch statements."*
Keywords: `ambient enums`, `switch statement`, `type inference`

**Cosine scores for this hypothesis**:
- 0.679 gold ← CHOSEN (correct)
- 0.535 `Enum not block-scoped given closure and --target es5`
- 0.325 `Incorrect type inference for array rest assignment…`

**Why it became generic**: Same dynamic as hyp-06 r3. Retrieval is HIGH, gold
is rank 1, C_hyp correctly selects gold (score 0.679). The model at rank 3
drops the discriminating qualifiers — "switch statement" and "ambient" — from
its rank-1 hypothesis ("type inference issues with ambient enums in switch
statements") and generalises to "inconsistent behavior across different enum
types." The root cause is still correct (is_match=True), keywords still work
(kw_A=True, kw_C=True), and gold selection succeeds.

This is **model abstraction at later ranks as deliberate diversity**: the
model's 3rd hypothesis broadens the 1st, maintaining thematic relevance while
avoiding exact repetition.

**Primary cause**: **Model abstraction** — rank-3 diversification broadens a
well-grounded rank-1 hypothesis without adding evidence. In this instance the
abstraction is benign (gold still selected, root cause still correct).

---

### 6 · hyp-07, rank 4

**Problem**: *type inference and switch statement narrowing bugs involving enums*

**Retrieval confidence**: **HIGH** — top1 = 0.731

**Generated hypothesis (rank 4)**:
*"Issues with the TypeScript compiler's handling of closures and block
scoping."*
Keywords: `block scoping`, `closures`, `TypeScript compiler`

**Cosine scores for this hypothesis**:
- 0.474 `Enum not block-scoped given closure and --target es5` ← CHOSEN (wrong)
- 0.442 `Inference fails nondeterministic when callback…`
- 0.271 `Incorrect type inference for array rest assignment…`
- 0.265 gold
- 0.039 `Inference problem`

**Why it became generic**: The retrieved pool contains `f5f33b02` ("Enum not
block-scoped given closure and --target es5") at rank 2 with no direct
connection to the gold incident's failure (switch narrowing / ambient enums).
At rank 4, the model draws on this sibling incident's vocabulary —
"closure", "block scoping" — and constructs a hypothesis from it rather than
continuing to elaborate the ambient-enum/switch-narrowing thread.

This is **context pollution from a competing retrieved incident**. The sibling
incident's title is short, distinctive, and contains unusual terms ("not
block-scoped", "closure") that the model treats as a separate hypothesis class.

Gap to gold: +0.209. The model's chosen sibling score (0.474) dominates gold
(0.265) — this is not a near-tie.

**Primary cause**: **Competing retrieved incident** — `f5f33b02`'s vocabulary
("closure", "block scoping") was absorbed into a distinct hypothesis at rank 4,
pulling the model away from the ambient-enum thread.

---

### 7 · hyp-07, rank 5

**Problem**: *type inference and switch statement narrowing bugs involving enums*

**Retrieval confidence**: **HIGH** — top1 = 0.731

**Generated hypothesis (rank 5)**:
*"Nondeterministic type inference behavior in complex callback scenarios."*
Keywords: `nondeterministic behavior`, `callback`, `type inference`

**Cosine scores for this hypothesis**:
- 0.769 `Inference fails nondeterministic when callback gets a typeof…` ← CHOSEN (wrong)
- 0.407 `Incorrect type inference for array rest assignment…`
- 0.269 `Inference problem`
- 0.189 gold
- 0.186 `Enum not block-scoped given closure and --target es5`

**Why it became generic**: The same mechanism as rank 4 but more extreme. Pool
rank 4 (`923c5ef6`, "Inference fails nondeterministic when callback gets a
`typeof` in an object") supplies both key terms verbatim: "nondeterministic" and
"callback". The model's rank-5 hypothesis is essentially a paraphrase of that
incident's title. The cosine score to the chosen sibling (0.769) is the largest
in the entire dataset — not a near-tie at all. Gold scores only 0.189.

This is the clearest example of **vocabulary absorption from a competing
retrieved incident**. The problem statement's phrase "type inference and switch
statement narrowing" gave the model enough surface relevance to include
inference-related incidents in the pool (rank 4 pool slot), but those incidents
then bleed into the hypothesis at rank 5.

**Primary cause**: **Competing retrieved incident** — rank-4 pool incident
(`923c5ef6`) provides near-literal hypothesis vocabulary.

---

## Cause attribution summary

| Hypothesis | Primary cause | Contributing cause | Gold mismatch? |
|---|---|---|---|
| hyp-03 r4 | Model abstraction (fallback at rank 4) | Diverse pool | Yes |
| hyp-04 r5 | Retrieval noise (irrelevant incident `053dc4ca`) | Model picks up resolution text | Yes |
| hyp-05 r2 | Problem statement framing ("API docs show") | Sparse context (1/5 relevant) | Yes |
| hyp-06 r3 | Model abstraction (rank-3 diversification) | — | **No** |
| hyp-07 r3 | Model abstraction (rank-3 diversification) | — | **No** |
| hyp-07 r4 | Competing retrieved incident (`f5f33b02`) | — | Yes |
| hyp-07 r5 | Competing retrieved incident (`923c5ef6`) | — | Yes |

### Quantified

| Root cause | Count | % |
|---|---|---|
| **Model abstraction / rank-order diversification** | 3 | **43%** |
| **Competing retrieved incident (vocabulary absorption)** | 2 | **29%** |
| **Retrieval noise (irrelevant incident in pool)** | 1 | **14%** |
| **Problem statement framing / ambiguity** | 1 | **14%** |

Regrouped by intervention type:

| Locus of failure | Count | % |
|---|---|---|
| **Hypothesis generation behaviour** (model abstraction) | 3 | 43% |
| **Retrieval context quality** (noise + competing incidents) | 3 | 43% |
| **Problem statement** | 1 | 14% |

---

## Recurring structural patterns

### Pattern 1 — Rank-order gradient (all 7 cases)

Every single "neither" hypothesis is at rank 3, 4, or 5:

| Rank | Neither count | Total | % neither |
|---|---|---|---|
| 1 | 0 | 7 | 0% |
| 2 | 1 | 7 | 14% |
| 3 | 2 | 7 | 29% |
| 4 | 2 | 7 | 29% |
| 5 | 2 | 7 | 29% |

The model generates its most specific, evidence-grounded hypotheses first.
Later ranks are generated under a diversity pressure: the model avoids
repeating earlier hypotheses, so it must reach for broader or differently-framed
alternatives. When the retrieved pool is exhausted of distinct specific threads,
it falls back to:
- Generalised patterns drawn from the same evidence ("inconsistent handling
  across components")
- Sibling incidents in the pool that supply fresh vocabulary
- Generic "version compatibility" catch-alls

This is a **systematic generation-order bias**, independent of retrieval quality.
It would occur even with perfect retrieval if the pool contains any sibling
incidents or if the model generates more than ~2–3 hypotheses per case.

### Pattern 2 — Sibling incident vocabulary absorption (hyp-07 r4, r5)

When the retrieved pool contains semantically-adjacent but incorrect incidents,
the model at later ranks exhausts the gold incident's vocabulary (used at ranks
1–2) and begins drawing from sibling incidents. The result is hypotheses that
are internally coherent and specific to a *different* incident in the pool.

This is most acute in hyp-07 (3 "neither" hypotheses from one case) because
the pool contains 4 TypeScript type-system incidents (all sharing the "inference"
theme), giving the model 4 distinct vocabulary sets to draw on at ranks 3–5.

### Pattern 3 — Retrieval noise amplifies genericity (hyp-04 r5)

A single irrelevant incident in the pool (`053dc4ca`, VectorDB/Celery) does not
directly cause the "neither" hypothesis — its vocabulary is too different to be
absorbed — but it reduces pool coherence and leaves the model with less
evidence to anchor a 5th specific hypothesis. The model falls back to a version-
compatibility guess, picking up on the resolution text's version reference
rather than the retrieved incidents' content.

### Pattern 4 — Problem statement framing (hyp-05 r2)

The phrase *"API docs show the same operation identifier twice"* contains the
word "docs", which primes the model to interpret the issue as a documentation
gap. In a sparse retrieval pool (only 1/5 incidents relevant, with no symptom
text on the gold incident), this framing is not corrected by retrieved evidence.
The problem statement vocabulary overrides the retrieved context when that
context is weak.

---

## Comparison: "neither" vs. best specific hypothesis per case

| Case | Neither hypothesis | Best specific hypothesis | Gap in token specificity |
|---|---|---|---|
| hyp-03 | "Incompatibility or bugs in recent TS versions" | "Memory leak in the type checker using --watch flag" | drops: `--watch`, `memory leak`, `type checker` |
| hyp-04 | "Incompatibility with specific TypeScript versions" | "Inefficient memory management in the TypeScript compiler" | drops: `memory management`, `TypeScript compiler` |
| hyp-05 | "Lack of clear documentation on…defining routes in FastAPI" | "Improper use of `@router.api_route()` for multiple HTTP methods" | drops: `api_route`, `multiple HTTP`, `operationId` |
| hyp-06 | "Inconsistent handling of audience claims across…system" | "Misconfiguration of JWT audience settings in the authentication layer" | drops: `JWT`, `configuration`, `KeycloakAuthManager` |
| hyp-07 | "Nondeterministic type inference in complex callback scenarios" | "Type inference issues with ambient enums in switch statements" | drops: `switch`, `ambient`, `enums` |

In every case the "neither" hypothesis retains the **domain category** (memory,
version, documentation, authentication, type inference) but drops the
**discriminating specifics** (component names, operation names, flags, exact
error class). The abstraction is consistently one level above the specific
incident, not completely uninformed — but one level of abstraction is sufficient
to resolve to the wrong sibling incident in the corpus.

---

## Is hypothesis representation quality the dominant bottleneck?

**Partially yes — but the root cause is model generation behaviour at later
ranks, not retrieval.**

For 3 of the 5 cases that have "neither" hypotheses (hyp-03, hyp-06, hyp-07),
the retrieved pool contained the correct incident at rank 1 with adequate
retrieval confidence (MEDIUM–HIGH). The "neither" hypothesis was generated at a
later rank despite good evidence being available. The model had all the
information it needed to be specific but chose breadth for diversity.

For the other 2 cases (hyp-04, hyp-05), retrieval context quality is a
co-cause: one case has an irrelevant incident in the pool, the other has a
sparse pool (4 irrelevant incidents) combined with an ambiguous problem
statement.

The split is:
- Cases where retrieval is adequate but model still produces generic hypotheses
  at late ranks: **3/5 (60%)**
- Cases where retrieval quality contributes to the generic hypothesis: **2/5
  (40%)**

The 43% attribution to model abstraction and 43% to retrieval context quality
(at the hypothesis level) understates the model side: the two "competing
retrieved incident" failures (hyp-07 r4, r5) are also a consequence of the
generation-order pressure — if the model didn't need to generate 5 hypotheses
per case, it would not have reached ranks 4–5 where it absorbed sibling
vocabulary.

**Conclusion**: The dominant bottleneck before any prompt or architecture
change is the **generation-order diversity pressure** — the model exhausts
specific evidence by rank 2–3 and generates generic or sibling-contaminated
hypotheses at ranks 4–5. Reducing the maximum hypothesis count from 5 to 3
would eliminate 5 of the 7 "neither" hypotheses (those at ranks 4–5), at the
cost of recall depth. The remaining 2 (ranks 3) are model-abstraction artefacts
that would require either prompt changes (require symptoms) or filtering (drop
hypotheses below a specificity threshold) to address.
