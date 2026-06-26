# Validation-Keyword Failure Analysis

Source: [tests/eval/results/hypothesis_v6a.json](tests/eval/results/hypothesis_v6a.json)
(Phase 6A run). For every hypothesis with `validation_keyword_recall_ok ==
false` (across all 5 ranks per case, not just the case-level
`validation_keyword_eval` target), this document shows the gold incident, the
generated hypothesis, its validation keywords, the actual top-5 retrieved
incidents for those keywords (re-run live against the corpus), why the
expected incident was missed, and a failure category.

Categories used:
- **A — keywords too generic**: keywords describe a class of problem shared
  by many incidents, not this one specifically.
- **B — keywords missing discriminating terms**: keywords omit the
  specific terms (error strings, flags, component names) that would
  separate this incident from near-duplicates.
- **C — retrieval ambiguity**: even reasonably specific keywords resolve to
  a near-duplicate/sibling incident because the corpus contains multiple
  very similar issues.
- **D — evaluation issue**: the hypothesis itself is off-topic relative to
  the gold incident, so no keyword set could plausibly retrieve it — this is
  a hypothesis-generation problem surfacing as a keyword-recall failure, not
  a keyword-quality problem per se.

13 hypothesis-level failures were found across 6 of the 7 positive cases
(hyp-01 had zero failures).

---

## hyp-02, rank 2 — "Triggerer not starting"

1. **Gold incident**: `375a627d` — *Triggerer not starting*
2. **Generated hypothesis**: "Missing connection ID for remote logging
   configuration."
3. **Validation keywords**: `["AirflowNotFoundException", "conn_id",
   "logs_to_azure_blob"]`
4. **Top-5 retrieved**:
   - 0.668 `69347f56` Triggerer unable to ship logs to remote azure blob connection
   - 0.541 `0ee80306` [Airflow 3.2.2] api-server: Azure OAuth button not shown at /auth/login
   - 0.485 `67491e2a` Deferrable BeamRunPythonPipelineOperator (DataflowRunner) fails with...
   - 0.476 `e3bfe559` Scheduler crashloops with `ValidationError: UUID input should be a string`...
   - 0.449 `fcbc6109` Document that `airflow db downgrade` from 3.2 to 3.1.x does not revert...
5. **Why retrieval missed it**: The hypothesis itself is about a different
   incident — `69347f56` (Azure Blob log shipping), which is exactly what
   rank1 of the top-5 retrieves with the highest score (0.668). The
   keywords are faithful to the *hypothesis*, but the hypothesis is not
   actually about `375a627d` (Triggerer not starting at all due to a
   startup/cleanup bug) — it's a plausible-sounding but wrong root cause
   drawn from a different retrieved incident in the shared context.
6. **Category**: **D — evaluation issue** (this is `is_match=False` — the
   keyword search is working correctly; it faithfully retrieves the
   incident the hypothesis is actually describing, which is not the gold
   incident).

---

## hyp-03 — "the type checker crashes with a memory error while watching files for changes"

Gold incident: `a9a17361` — *[2.8.0-rc] Segfault when running compiler with
`--d` or `--watch` (out of memory)*. All 5 ranks fail keyword recall —
notably including **rank 1, which is the correct root cause**
(`is_match=True`).

### rank 1 (correct hypothesis, kw fail)

2. **Generated hypothesis**: "Memory leak in the type checker when using
   the `--watch` flag." *(is_match=True)*
3. **Validation keywords**: `["out of memory", "--watch", "memory leak"]`
4. **Top-5 retrieved**:
   - 0.531 `7bc74b8a` WatchList on large collection can cause watch cache to be stale...
   - 0.437 `9c8eac12` Unrestricted caching keyed by generated types causes memory leak in mu...
   - 0.404 `4066fcee` System memory leak when using different input size of torch.nn.Conv3d
   - 0.391 `4c84699f` Simplify watch cache by removing WaitUntilFreshAndGet and using WaitUn...
   - 0.390 `5b155d0c` Fetching of blocks of cache table may cause high memory pressure
5. **Why retrieval missed it**: The keywords are three very common,
   generic terms ("out of memory", "--watch", "memory leak") that match
   *any* memory-leak-plus-watch-mode incident across the entire
   multi-repo corpus (Kubernetes watch caches, PyTorch memory leaks,
   etc.), none of which is `a9a17361`. None of these terms are specific to
   the TypeScript compiler / segfault / `--d` flag that would disambiguate
   this incident from the dozens of other "memory leak" issues.
6. **Category**: **A — keywords too generic.**

### rank 2 (kw fail, is_match=False)

2. **Generated hypothesis**: "Inefficient handling of large files or
   complex code structures."
3. **Validation keywords**: `["JavaScript heap out of memory", "large
   source", "complex for loop"]`
4. **Top-5 retrieved**:
   - 0.610 `b093992f` JavaScript heap out of memory for 10s of MB of source
   - 0.409 `214d9bed` Takes a while with high memory to calculate highly recursive types wit...
   - 0.401 `a562aa3c` Compiler hang when importing big JS file with --allowJs
   - 0.392 `5e2254b0` Crash: RangeError: Maximum call stack size exceeded in isReachableFlow...
   - 0.364 `5b155d0c` Fetching of blocks of cache table may cause high memory pressure
5. **Why retrieval missed it**: The keyword query resolves strongly
   (0.610) to `b093992f` — the *other* near-duplicate TypeScript
   OOM incident from this same corpus (heap OOM on large bundles, vs.
   `a9a17361`'s `--watch`/segfault OOM). "JavaScript heap out of memory" and
   "large source" are specific to `b093992f`'s title, not `a9a17361`'s.
6. **Category**: **C — retrieval ambiguity** (resolves to the
   `a9a17361`/`b093992f` near-duplicate pair, same dynamic Phase 5 flagged
   for rank 1's case-level failure).

### rank 3 (kw fail, is_match=False)

2. **Generated hypothesis**: "Recursive function calls leading to stack
   overflow."
3. **Validation keywords**: `["Maximum call stack size exceeded",
   "recursive", "stack overflow"]`
4. **Top-5 retrieved**:
   - 0.658 `214d9bed` Takes a while with high memory to calculate highly recursive types wit...
   - 0.648 `edfec73e` `symbolToNode` stack overflow when using recursive types and functions
   - 0.610 `0aabf550` Crash: RangeError: Maximum call stack size exceeded during type serial...
   - 0.608 `5e2254b0` Crash: RangeError: Maximum call stack size exceeded in isReachableFlow...
   - 0.573 `e1da6866` Maximum call stack size exceeded in isTypeAssignableTo when using recu...
5. **Why retrieval missed it**: The hypothesis ("recursive calls / stack
   overflow") describes a different failure mode than the gold incident
   (`--watch` segfault from memory exhaustion, not stack-depth recursion).
   The keywords are specific and retrieve a tight, coherent cluster of
   "Maximum call stack size exceeded" incidents — but that cluster simply
   doesn't include `a9a17361`, because the hypothesis is about the wrong
   mechanism.
6. **Category**: **D — evaluation issue** (`is_match=False`; keywords
   correctly retrieve incidents matching this hypothesis's actual claim,
   which isn't the gold root cause).

### rank 4 (kw fail, is_match=False)

2. **Generated hypothesis**: "Incompatibility or bugs in specific
   TypeScript versions."
3. **Validation keywords**: `["TypeScript version", "known issues",
   "compiler crash"]`
4. **Top-5 retrieved**:
   - 0.675 `df5af7de` Generic type inference failed
   - 0.673 `f350c637` Crash: RangeError: Maximum call stack size exceeded in addTypesToUnion...
   - 0.660 `a562aa3c` Compiler hang when importing big JS file with --allowJs
   - 0.657 `0aabf550` Crash: RangeError: Maximum call stack size exceeded during type serial...
   - 0.637 `8dd2a352` Long (infinite?) compile times intersecting with large union
5. **Why retrieval missed it**: The hypothesis is a generic
   "version-compatibility" guess not grounded in the gold incident's actual
   cause. Its keywords ("TypeScript version", "known issues", "compiler
   crash") are broad enough to match almost any compiler-crash issue, and
   indeed retrieve a generic grab-bag of unrelated compiler crashes — none
   close to `a9a17361`.
6. **Category**: **A — keywords too generic** (compounded by **D**: the
   hypothesis itself isn't a real root cause for this incident).

### rank 5 (kw fail, is_match=False)

2. **Generated hypothesis**: "Improper error handling in the type checker."
3. **Validation keywords**: `["Error processing request", "compiler should
   not crash", "error handling"]`
4. **Top-5 retrieved**:
   - 0.452 `5e2254b0` Crash: RangeError: Maximum call stack size exceeded in isReachableFlow...
   - 0.426 `fd6ae144` Scheduler crashes with InvalidStatsNameException for non-ASCII DAG/Tas...
   - 0.407 `0aabf550` Crash: RangeError: Maximum call stack size exceeded during type serial...
   - 0.406 `177f19d2` DAG processor retry surfaces PendingRollbackError and hides the origin...
   - 0.396 `5029c88e` Bug: SSE protocol injection via unvalidated event and id fields in for...
5. **Why retrieval missed it**: Both the hypothesis and its keywords are
   generic boilerplate ("error handling", "compiler should not crash") that
   don't reference memory, `--watch`, or segfaults at all. The retrieved
   top-5 is a low-similarity (all <0.46) grab-bag spanning unrelated repos
   (Airflow scheduler, SSE protocol) — confirming the keywords carry almost
   no discriminating signal.
6. **Category**: **A — keywords too generic.**

---

## hyp-04, rank 2 — "JavaScript heap out of memory for 10s of MB of source" (kw fail, is_match=True)

1. **Gold incident**: `b093992f` — *JavaScript heap out of memory for 10s
   of MB of source*
2. **Generated hypothesis**: "Inefficient memory management in the
   TypeScript compiler when handling large files." *(is_match=True —
   correct root cause)*
3. **Validation keywords**: `["compiler hang", "maximum call stack size
   exceeded", "TypeScript"]`
4. **Top-5 retrieved**:
   - 0.772 `214d9bed` Takes a while with high memory to calculate highly recursive types wit...
   - 0.761 `0aabf550` Crash: RangeError: Maximum call stack size exceeded during type serial...
   - 0.750 `5e2254b0` Crash: RangeError: Maximum call stack size exceeded in isReachableFlow...
   - 0.738 `f350c637` Crash: RangeError: Maximum call stack size exceeded in addTypesToUnion...
   - 0.720 `e1da6866` Maximum call stack size exceeded in isTypeAssignableTo when using recu...
5. **Why retrieval missed it**: The keywords ("compiler hang", "maximum
   call stack size exceeded") describe a *different* symptom
   (stack-overflow crashes during type-checking) than the gold incident's
   actual symptom (heap exhaustion on large source bundles). Even though
   the hypothesis's *root cause text* is correct, its chosen validation
   keywords pull toward the "Maximum call stack size exceeded" cluster — a
   strong, internally-consistent cluster (scores 0.72-0.77) that simply
   doesn't include `b093992f`.
6. **Category**: **B — keywords missing discriminating terms** (the
   keywords should reference "heap out of memory" / "large source" /
   bundle size — terms from the gold incident's own title — rather than
   "maximum call stack size exceeded", which belongs to a sibling cluster).

---

## hyp-04, rank 3 — "JavaScript heap out of memory for 10s of MB of source" (kw fail, is_match=False)

2. **Generated hypothesis**: "Potential memory leaks in the build process
   or dependencies."
3. **Validation keywords**: `["out of memory", "memory pressure",
   "segfault"]`
4. **Top-5 retrieved**:
   - 0.425 `4066fcee` System memory leak when using different input size of torch.nn.Conv3d
   - 0.418 `5b155d0c` Fetching of blocks of cache table may cause high memory pressure
   - 0.362 `9c8eac12` Unrestricted caching keyed by generated types causes memory leak in mu...
   - 0.357 `a9a17361` [2.8.0-rc] Segfault when running compiler with --d or --watch (out of memory)
   - 0.353 `45ce549a` [vllm] [2.12 regression][compile] test_standalone_compile_correctness:...
5. **Why retrieval missed it**: "out of memory", "memory pressure",
   "segfault" are exactly the same generic vocabulary flagged for hyp-03 —
   they spread thinly across every memory-related incident in the corpus
   (all scores <0.43). Notably, `a9a17361` (the *other* near-duplicate)
   actually outranks `b093992f` here — the generic keywords land closer to
   the sibling incident than to the gold one.
6. **Category**: **A — keywords too generic** (with a touch of **C —
   retrieval ambiguity**, since the near-duplicate `a9a17361` appears in
   the top-5 instead).

---

## hyp-04, rank 4 — "JavaScript heap out of memory for 10s of MB of source" (kw fail, is_match=False)

2. **Generated hypothesis**: "Incompatibility or bugs in specific versions
   of TypeScript or Node.js."
3. **Validation keywords**: `["typescript@next", "version issues", "fixed
   in TS7"]`
4. **Top-5 retrieved**:
   - 0.660 `923c5ef6` Inference fails nondeterministic when callback gets a `typeof` in an o...
   - 0.652 `59f56a05` tsc may fail to report error when `file.js` & `file.d.ts` coexist, eve...
   - 0.649 `18c81d79` TypeScript incorrectly reports literal comparison error inside class m...
   - 0.619 `ca1cff60` Type guards not working for indexed types with generics
   - 0.614 `449e6f2b` Issue an error if tsconfig.json results in no files to compile
5. **Why retrieval missed it**: This hypothesis is a generic
   "version-compatibility" guess unconnected to the gold incident's actual
   heap-OOM cause (same pattern as hyp-03 rank 4). Its keywords
   ("typescript@next", "version issues", "fixed in TS7") retrieve a cluster
   of unrelated TypeScript type-checking bug reports with no memory/OOM
   theme at all.
6. **Category**: **D — evaluation issue** (the hypothesis isn't really
   about memory/OOM, so no keyword set derived from it would find
   `b093992f`).

---

## hyp-04, rank 5 — "JavaScript heap out of memory for 10s of MB of source" (kw fail, is_match=True)

2. **Generated hypothesis**: "High memory usage due to large data
   structures or inefficient algorithms in the build process."
   *(is_match=True — correct root cause)*
3. **Validation keywords**: `["high memory pressure", "large data
   structures", "inefficient algorithms"]`
4. **Top-5 retrieved**:
   - 0.425 `5b155d0c` Fetching of blocks of cache table may cause high memory pressure
   - 0.335 `4e512751` [SPARK-56908][SQL] Parquet vectorized reader performance improvements...
   - 0.297 `1aaae8e5` DISABLED test_integer_parameter_serialization_cuda (__main__.TestMulti...
   - 0.266 `4066fcee` System memory leak when using different input size of torch.nn.Conv3d
   - 0.261 `d63e2335` Support Filter pushdown in Spark Structured Streaming
5. **Why retrieval missed it**: Despite a correct root cause, the keywords
   ("high memory pressure", "large data structures", "inefficient
   algorithms") are abstract/textbook phrasing that doesn't reuse any of
   the gold incident's distinctive vocabulary ("JavaScript heap", "10s of
   MB of source", "compiler"/"tsc"). All similarity scores are low
   (<=0.43) and the results are dominated by unrelated Spark/CUDA/cache
   incidents — the keywords are essentially generic English, not
   incident-specific terms.
6. **Category**: **A — keywords too generic.**

---

## hyp-05, rank 2 — "Duplicated OperationID when adding route with multiple methods" (kw fail, is_match=False)

1. **Gold incident**: `5dba5df8` — *Duplicated OperationID when adding
   route with multiple methods*
2. **Generated hypothesis**: "Lack of clear documentation on the correct
   methods to define routes with multiple HTTP verbs."
3. **Validation keywords**: `["documentation", "route definition", "HTTP
   verbs", "FastAPI"]`
4. **Top-5 retrieved**:
   - 0.571 `fe9c0ec3` Automatically support HEAD method for all GET routes, as Starlette doe...
   - 0.540 `50fba191` Malformed Links in Documentation Home
   - 0.519 `481069ab` Automatic OPTIONS request with route schema
   - 0.501 `b74479a8` Implement automatic API documentation generation with interactive exam...
   - 0.472 `ae35b343` Custom APIRoute classes with explicit constructors fail after strict_c...
5. **Why retrieval missed it**: The hypothesis reframes the bug as a
   "documentation gap" rather than the actual technical cause (the OpenAPI
   schema generator producing colliding `operationId`s for multi-method
   routes). Its keywords ("documentation", "route definition", "HTTP
   verbs", "FastAPI") are generic FastAPI/routing/documentation terms that
   match a broad swath of routing- and docs-related issues, none of which
   mention "operationId" or "duplicate" — the two terms that would actually
   identify `5dba5df8`.
6. **Category**: **B — keywords missing discriminating terms** (omits
   "operationId"/"duplicate", the incident's defining vocabulary), compounded
   by **D** since the hypothesis itself reframes the issue as documentation
   rather than schema-generation behavior.

---

## hyp-06, rank 2 — "JWTValidator does not use the configured audience with KeycloakAuthManager" (kw fail, is_match=False)

1. **Gold incident**: `1c11adbe` — *JWTValidator does not use the
   configured audience with KeycloakAuthManager*
2. **Generated hypothesis**: "Incompatibility between different versions of
   the authentication library."
3. **Validation keywords**: `["version 2.33.0", "version 2.34.0",
   "authentication library"]`
4. **Top-5 retrieved**:
   - 0.449 `608271f5` AzureOpenAI with AAD bearer token passed via api_key works in 2.33.0 b...
   - 0.363 `c40e276e` LDAP Giving 500 Error
   - 0.358 `fce49d21` AsyncOpenAI(api_key="") raises OpenAIError in v2.34.0, breaking OpenAI...
   - 0.347 `36e8b575` Docs: update Airflow 3 auth-manager docs that still refer to removed w...
   - 0.341 `0ee80306` [Airflow 3.2.2] api-server: Azure OAuth button not shown at /auth/logi...
5. **Why retrieval missed it**: The hypothesis pivots to a generic
   "version incompatibility" framing, with keywords being specific *version
   numbers* (2.33.0/2.34.0) that happen to coincide with unrelated SDK
   version-bump bugs (Azure/OpenAI client libraries) elsewhere in the
   corpus. None of the keywords reference "JWT", "audience", "Keycloak", or
   "KeycloakAuthManager" — the actual identifying terms for `1c11adbe`. All
   scores are low (<0.45), indicating weak relevance across the board.
6. **Category**: **B — keywords missing discriminating terms** (omits
   "JWT"/"audience"/"Keycloak" entirely in favor of generic version
   numbers), compounded by **D** since the hypothesis itself isn't the
   audience-validation root cause.

---

## hyp-07, rank 5 — "narrowing in switch doesn't work with ambient enums" (kw fail, is_match=True)

1. **Gold incident**: `0c4aacbc` — *narrowing in switch doesn't work with
   ambient enums*
2. **Generated hypothesis**: "The TypeScript compiler's version may have
   unresolved bugs related to type inference and enum handling that have
   been addressed in later releases." *(is_match=True — correct root
   cause)*
3. **Validation keywords**: `["compiler version", "bugs", "type
   inference"]`
4. **Top-5 retrieved**:
   - 0.557 `df5af7de` Generic type inference failed
   - 0.551 `923c5ef6` Inference fails nondeterministic when callback gets a `typeof` in an o...
   - 0.549 `a21be2b7` Inference problem
   - 0.538 `89a29600` Visual Studio 2015 provides inconsistent type info per request order
   - 0.536 `75fa0671` Some overloaded signatures never be choose when explicit type paramete...
5. **Why retrieval missed it**: The keywords ("compiler version", "bugs",
   "type inference") capture only the generic "type inference bug" half of
   the hypothesis and drop the case-specific terms — "narrowing", "switch",
   "ambient enums" — that uniquely identify `0c4aacbc`. The retrieved top-5
   is a coherent but generic "type inference" cluster that doesn't include
   the gold incident anywhere.
6. **Category**: **B — keywords missing discriminating terms** (should
   include "narrowing", "switch", and "enum"/"ambient enum" — all present
   in the hypothesis's own surrounding reasoning and in the gold title, but
   absent from the keyword list).

---

## Summary

| Case/rank | Gold incident | is_match | Category |
|---|---|---|---|
| hyp-02 r2 | 375a627d (Triggerer not starting) | ✗ | D |
| hyp-03 r1 | a9a17361 (--watch segfault/OOM) | ✓ | A |
| hyp-03 r2 | a9a17361 | ✗ | C |
| hyp-03 r3 | a9a17361 | ✗ | D |
| hyp-03 r4 | a9a17361 | ✗ | A (+D) |
| hyp-03 r5 | a9a17361 | ✗ | A |
| hyp-04 r2 | b093992f (heap OOM, large source) | ✓ | B |
| hyp-04 r3 | b093992f | ✗ | A (+C) |
| hyp-04 r4 | b093992f | ✗ | D |
| hyp-04 r5 | b093992f | ✓ | A |
| hyp-05 r2 | 5dba5df8 (Duplicated OperationID) | ✗ | B (+D) |
| hyp-06 r2 | 1c11adbe (JWTValidator audience) | ✗ | B (+D) |
| hyp-07 r5 | 0c4aacbc (narrowing/ambient enums) | ✓ | B |

**Key takeaways**:

1. **The 4 failures on `is_match=True` (correct) hypotheses — hyp-03 r1,
   hyp-04 r2/r5, hyp-07 r5 — are the ones that matter most**, since these
   are cases where a *correct* root cause gets an under-confident composite
   score purely because its own validation keywords don't pan out. All four
   fall into category A or B: the keywords are either generic boilerplate
   ("memory leak", "high memory pressure", "inefficient algorithms",
   "bugs"/"type inference") or omit the specific terms already present in
   the gold incident's title / the hypothesis's own surrounding text
   ("heap out of memory", "narrowing", "switch", "ambient enum"). This is a
   **systematic pattern in `generate_hypotheses()`'s validation-keyword
   output**: it tends to restate the root cause in generic terms rather
   than extracting the literal symptom vocabulary (error strings, flags,
   identifiers) that would actually disambiguate the corpus.
2. **The remaining 9 failures are on `is_match=False` hypotheses** — these
   are largely category D (the hypothesis itself is off-base, so its
   keywords correctly fail to find the gold incident) or B/A (generic
   keywords for an off-base hypothesis). These don't represent a
   keyword-quality problem in isolation so much as a downstream symptom of
   hypothesis-generation quality (Phase 5's finding).
3. **Category C (true retrieval ambiguity from near-duplicate incidents)**
   appears only twice (hyp-03 r2, hyp-04 r3), both involving the
   `a9a17361`/`b093992f` TypeScript-OOM pair — confirming Phase 5's
   observation that this near-duplicate pair is a recurring disambiguation
   challenge, but it is **not** the dominant failure mode at the
   per-hypothesis level (A/B are far more common: 9/13).
4. **No instances of "evaluation issue" in the sense of a harness bug** were
   found — every D-classified failure reflects a genuinely off-topic
   hypothesis whose keywords faithfully (and correctly) fail to retrieve the
   gold incident. The harness's keyword-recall check is behaving as
   designed in all 13 cases.

**Implication for future work**: if validation-keyword quality is to be
improved (out of scope for this analysis, which is measurement-only), the
highest-value fix is making `generate_hypotheses()` include
incident-specific *literal* terms (error message fragments, CLI flags,
class/field names, version-distinguishing identifiers) in
`validation_keywords` for hypotheses that already have the correct root
cause — categories A and B account for 4 of the 4 failures on correct
hypotheses, and all 4 are fixable by keyword specificity alone, without
changing the underlying root-cause text.
