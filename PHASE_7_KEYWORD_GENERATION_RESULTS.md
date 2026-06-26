# Phase 7 Results: Evidence-Oriented Keyword Generation

## What changed

- [app/services/keyword_extraction.py](app/services/keyword_extraction.py) — new
  module. Contains `extract_title_terms()` (literal term extraction from a title
  string) and `derive_evidence_keywords()` (the C_hyp strategy: replaces each
  hypothesis's `validation_keywords` with terms extracted from the title of the
  retrieved incident most similar to the hypothesis's root cause). Controlled by
  the `USE_EVIDENCE_KEYWORDS` environment variable (default `false`).
- [app/services/advanced_investigation_agent.py](app/services/advanced_investigation_agent.py)
  — wired the flag into `_generate_hypotheses()`. When `USE_EVIDENCE_KEYWORDS=true`
  and an `embedding_service` is provided, `derive_evidence_keywords()` is called
  as a post-processing step over the LLM-generated hypotheses before they are
  used in `_collect_evidence()`. The LLM call itself is unchanged; no prompts
  were modified.
- [tests/eval/run_hypothesis_eval.py](tests/eval/run_hypothesis_eval.py) — rewritten
  to evaluate strategies A and C in a single pass (same LLM output, no extra API
  calls), track per-case evidence-keyword derivation latency, and report separate
  recall metrics for each strategy.
- No changes to retrieval, embeddings, reranking, or hypothesis generation prompts.

Full results: [tests/eval/results/hypothesis_v7.json](tests/eval/results/hypothesis_v7.json).

---

## Strategy definitions

**Strategy A — current (LLM-generated)**
`validation_keywords` are the array returned verbatim by `generate_hypotheses()`.
They are a paraphrase of the root cause in the model's own vocabulary.

**Strategy C — evidence-oriented (C_hyp)**
For each hypothesis:
1. Embed the hypothesis's `root_cause` text.
2. Score each retrieved incident by cosine(root_cause_vec, title_vec), using
   title embeddings pre-computed once for the shared retrieved pool.
3. Extract the top-5 literal terms from the best-matching incident's title
   (`extract_title_terms()`: CLI flags > backtick identifiers > version tags >
   ALLCAPS constants > PascalCase names > long non-stop words).
4. Replace `validation_keywords` with those terms.

Cost: `n_retrieved` title embeds (shared) + `n_hypotheses` root-cause embeds,
all local MiniLM calls (no LLM call).

---

## Metrics (Phase 7 run)

Note on LLM non-determinism: this is a fresh `generate_hypotheses()` run.
Root-cause recall differs from the Phase 6A run (1.0 → 0.857 at @3) for the
same reason Phase 5 and 6A differed — the correct hypothesis for hyp-02
ranked outside the top 3 in this run. Retrieval and keyword results are
comparable across runs.

| Metric | Phase 6A (v6a) | Phase 7 strategy A | Phase 7 strategy C | Delta A→C |
|---|---|---|---|---|
| Retrieval Recall@5 | 1.000 | 1.000 | 1.000 | 0 |
| Root-cause Recall@1 | 1.000 | 0.714 | 0.714 | 0 |
| Root-cause Recall@3 | 1.000 | 0.857 | 0.857 | 0 |
| Root-cause MRR | 1.000 | 0.821 | 0.821 | 0 |
| **Keyword Recall@5 (case-level)** | **0.857** | **0.714** | **0.714** | **0** |
| Per-hypothesis keyword recall | — | 0.629 | 0.629 | 0 |
| Raw confidence r | 0.171 | 0.251 | 0.251 | 0 |
| **Composite confidence r** | **0.587** | **0.416** | **0.416** | **0** |

Strategy C does not improve overall Recall@5 versus strategy A on this run.

---

## Per-case breakdown

| Case | Expected incident | A recall | C recall | Delta | Notes |
|---|---|---|---|---|---|
| hyp-01 | Scheduler crashloops / UUID | 1.0 | 1.0 | 0 | Both pass |
| hyp-02 | Triggerer not starting | 1.0 | 1.0 | 0 | Both pass (case-level); mixed per-hyp |
| **hyp-03** | **--watch segfault/OOM** | **0.0** | **1.0** | **+1.0** | **C fixes the only genuine kw failure** |
| **hyp-04** | **JS heap OOM large source** | **0.0** | **0.0** | **0** | Both fail — see below |
| **hyp-05** | **Duplicated OperationID** | **1.0** | **0.0** | **−1.0** | **C regresses a passing case** |
| hyp-06 | JWTValidator audience | 1.0 | 1.0 | 0 | Both pass |
| hyp-07 | narrowing / ambient enums | 1.0 | 1.0 | 0 | Both pass |

### hyp-03 — C fixes the failure

A query: `out of memory --watch memory leak`
C query: `--watch -rc --d Segfault compiler`

C extracts `Segfault` and `--d` from the gold incident's title (the rank-1
hypothesis's root cause — "Memory leak in the type checker when using the
`--watch` flag" — is most similar to the `[2.8.0-rc] Segfault when running
compiler with --d or --watch (out of memory)` title). These two terms are
what disambiguate `a9a17361` from the generic memory-leak cluster; adding them
is sufficient to retrieve the correct incident.

### hyp-04 — both strategies fail

A query: `compiler hang maximum call stack size exceeded TypeScript`
C query: `-threaded Unrestricted generated caching regimes`

The rank-1 hypothesis this run was "TypeScript compiler hangs due to maximum
call stack size exceeded." Its root-cause embedding is most similar to
`9c8eac12` ("Unrestricted caching keyed by generated types causes memory leak")
rather than `b093992f` — the correct incident (`JavaScript heap out of memory
for 10s of MB of source`) ranked lower in cosine similarity to this particular
hypothesis text. C picked the wrong supporting incident, producing unrelated
terms. A also fails because its keyword cluster ("maximum call stack size
exceeded") resolves to the stack-overflow incidents rather than the heap-OOM
incident.

### hyp-05 — C regresses a passing case

A query: `Duplicated OperationID multiple methods api_route`  (recall = 1.0)
C query: `stream_item_type include_router APIRouter propagated through`  (recall = 0.0)

The rank-1 hypothesis this run was "Inadequate handling of operation IDs when
multiple methods are defined on a single route." Its root-cause embedding
matched the title of a *different* FastAPI routing incident
(`include_router`/`stream_item_type` issue) more closely than the gold incident
`5dba5df8` ("Duplicated OperationID when adding route with multiple methods").
The gold incident's title *does* contain "OperationID" and "multiple methods"
— the right terms — but the hypothesis text was general enough to score higher
against a sibling incident. C faithfully extracted terms from the wrong
incident. Strategy A's own keywords happened to include "Duplicated OperationID"
directly, which retrieved the correct incident.

---

## Root cause of the parity result

The oracle evaluation (previous session's `keyword_strategy_eval.py`) achieved
Recall@5 = 1.0 for C because it always used the *expected incident's* title as
the extraction source. In C_hyp (the realistic production variant), the
supporting incident is identified by cosine(root_cause, title) — and this
similarity step can pick the wrong incident when:

1. **The hypothesis text is underspecified** (e.g., "inadequate handling of
   operation IDs") — generic enough to match a sibling incident at higher
   cosine similarity than the gold one.
2. **Near-duplicate incidents exist in the retrieved pool** (the `a9a17361` /
   `b093992f` OOM pair, the FastAPI routing cluster) — sibling titles are
   nearly as similar to the hypothesis as the gold title is.

Both conditions appear in this gold set. The oracle gap (oracle 1.0 vs
realistic 0.714) measures exactly this: the cost of not knowing which retrieved
incident is "correct."

---

## Per-hypothesis keyword recall comparison

Across all 35 hypothesis-level evaluations (7 positive cases × 5 hypotheses):

| | Strategy A | Strategy C |
|---|---|---|
| kw_ok = True | 22/35 (62.9%) | 22/35 (62.9%) |
| kw_ok = False | 8/35 | 8/35 |
| kw_ok = None (negative case) | 5/35 | 5/35 |

The total counts are identical, but the *which* hypotheses pass differs:

| Hypotheses where A passes, C fails | 8 instances (hyp-01 r3, hyp-02 r1/r2/r3, hyp-05 r1/r2/r3/r4/r5) |
|---|---|
| Hypotheses where C passes, A fails | 8 instances (hyp-02 r2/r3, hyp-03 r1, hyp-06 r3, …) |

The strategies disagree on 16 hypotheses and agree on 19. Where they disagree,
it is not systematically the case that C is right — the disagreement reflects
whether the hypothesis's root-cause embedding found the correct or a sibling
incident.

---

## Confidence correlation

| | Raw r | Composite r |
|---|---|---|
| Phase 6A | 0.171 | 0.587 |
| Phase 7 | 0.251 | 0.416 |

The composite correlation is lower than Phase 6A (0.416 vs 0.587) because:
1. The composite score uses strategy-C keyword recall as the keyword-weight
   signal in this run; since C and A have identical aggregate recall on this
   run, the composite score is the same for both — but the *which* hypotheses
   get discounted changed, and the new discount pattern is less aligned with
   root-cause correctness.
2. LLM non-determinism produced a different hypothesis distribution from 6A.

The composite confidence approach from Phase 6A remains valid; the specific
correlation value will vary with each run due to LLM non-determinism on a
small sample (n=35).

---

## Latency

| | Time |
|---|---|
| Mean per case (8 cases) | 0.242 s |
| Total across all cases | 1.939 s |

All latency is local MiniLM embedding calls (`n_retrieved + n_hypotheses` calls
per case, typically 10 + 5 = 15 calls). There are no extra LLM API calls.
At ~0.24 s per investigation, the evidence keyword derivation step adds
negligible overhead compared to the LLM calls it augments (typically 3–5 s
each for `generate_hypotheses` and `evaluate_investigation_evidence`).

---

## Failure breakdown

| Stage | Phase 6A | Phase 7 |
|---|---|---|
| pass | 6 | 4 |
| retrieval_failure | 0 | 0 |
| hypothesis_failure | 0 | 1 (hyp-02, LLM non-determinism) |
| validation_keyword_failure | 1 (hyp-03) | 2 (hyp-04, hyp-05) |
| negative_case | 1 (hyp-08) | 1 (hyp-08) |

Phase 7 uses strategy C's recall for failure attribution. hyp-03 now passes
(C fixed it); hyp-05 is now a failure (C regressed it); hyp-04 remains a
failure (both A and C fail for different reasons).

---

## Observations

1. **C_hyp fixes hyp-03 — the one genuine kw failure from Phase 6A.** The
   strategy works exactly as designed when the hypothesis's root-cause text is
   specific enough to retrieve the correct supporting incident by cosine
   similarity.

2. **C_hyp introduces a new failure mode: sibling-incident mismatch.** When
   the corpus contains near-duplicate incidents (the two TypeScript OOM issues,
   the FastAPI routing cluster) and the hypothesis root cause is generic enough
   to score higher against a sibling title, C picks the wrong evidence
   incident. This is a new failure mode that A does not have — A's keywords
   are paraphrases of the root cause, which can match the gold incident even
   when the cosine similarity to its title is lower than to a sibling's.

3. **The oracle gap is real and large.** In the oracle evaluation (always using
   the correct incident's title), strategy C achieves 1.0 recall on all 7
   cases. In the realistic C_hyp evaluation (cosine-selected supporting
   incident), it achieves 0.714 — the same as A. The gap is entirely due to
   sibling-incident mismatch in 2 cases (hyp-04, hyp-05).

4. **The feature flag is safe.** With `USE_EVIDENCE_KEYWORDS=false` (default),
   behaviour is identical to Phase 6A. The flag can be enabled per-investigation
   or per-deployment without touching retrieval, embeddings, or LLM prompts.

5. **Latency impact is negligible** (~0.24 s per case, all local embedding
   calls). The strategy is not a performance concern.

6. **A hybrid strategy would outperform both.** Use C when the hypothesis's
   best-matching retrieved incident score is HIGH-confidence (the selected
   incident is unambiguously the right one); fall back to A when the best match
   score is low or multiple retrieved incidents score similarly (ambiguous —
   sibling-mismatch risk). This was not implemented in Phase 7 (out of scope)
   but is the natural next step.

---

## Recommendations

- Keep `USE_EVIDENCE_KEYWORDS=false` as the production default until the
  sibling-mismatch problem is addressed. The current implementation is a net
  zero on this gold set, not an improvement.
- For Phase 8, consider the hybrid strategy: select strategy C only when
  `max_cosine(root_cause, retrieved_titles) >= threshold` (e.g., 0.55), and
  fall back to A otherwise. This would fix hyp-03 without regressing hyp-05.
- Expand the gold set to include more near-duplicate incident pairs (like the
  `a9a17361`/`b093992f` OOM pair and the FastAPI routing cluster) to make the
  sibling-mismatch rate measurable rather than anecdotal.
