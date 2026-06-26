# Phase 6A Results: Hypothesis Confidence Propagation

## What changed

- [app/services/confidence.py](app/services/confidence.py): added
  `composite_hypothesis_confidence()` plus two new weight tables. No changes
  to `classify_confidence()` or the existing LOW/MEDIUM/HIGH thresholds from
  Phase 4.
- [tests/eval/run_hypothesis_eval.py](tests/eval/run_hypothesis_eval.py):
  for each hypothesis, the harness now also runs that hypothesis's own
  `validation_keywords` through `search()` (previously only done for the
  first matching hypothesis) and computes a `composite_confidence_score`.
  `confidence_correlation()` now reports both raw and composite correlations.
- No changes to retrieval, prompts, reranking, embeddings, or
  `generate_hypotheses()` itself — this phase is purely a post-processing
  layer over the existing `confidence_score` output.

## Confidence propagation rules

The composite score is a **multiplicative discount** on the LLM's own
`confidence_score`: `composite = raw_confidence * retrieval_weight *
keyword_weight`, clamped to `[0, 1]`. Both weights are `<= 1.0`, so a
hypothesis's composite confidence can only be *lower* than its raw confidence,
never higher — the LLM's self-report is treated as an upper bound, discounted
by how much retrieval evidence actually backs it.

### 1. Retrieval confidence weight

The **initial retrieval confidence** for the investigation (Phase 4's
`classify_confidence(top1_score)` on the seed `search()` results) discounts
*every* hypothesis generated from that context, since all hypotheses share
the same (possibly weak/irrelevant) context:

| Initial retrieval confidence | Weight |
|---|---|
| HIGH | 1.00 |
| MEDIUM | 0.85 |
| LOW | 0.50 |

Rationale: HIGH means the seed context is almost certainly the right incident
— no discount. MEDIUM (a plausible-but-not-certain seed, e.g. paraphrase
matches at 0.42-0.55) gets a light discount. LOW (Phase 4: top1_score < 0.40,
which on the gold set meant *no genuine match exists*) gets a 50% discount —
strong enough to pull even a "confident-sounding" 0.8 raw score down into
MEDIUM/LOW composite territory, without zeroing it out entirely (the
hypothesis might still be useful as a starting point for reasoning, per the
"do not block investigations" requirement from Phase 4).

### 2. Validation-keyword success/failure weight

Independently, **each hypothesis's own `validation_keywords`** are run through
`search()`. If they retrieve a relevant incident in the top 5:

| Validation-keyword outcome | Weight |
|---|---|
| Retrieves a relevant incident (top-5) | 1.00 |
| Does not retrieve a relevant incident, or no keywords generated | 0.60 |

Rationale: a hypothesis whose own validation query can't find supporting
evidence in the corpus is unsupported *regardless* of how confident the model
sounds or how good the initial retrieval was — this is the
`validation_keyword_failure` mode from Phase 5 (hyp-03), where a correct root
cause had a too-generic validation query that resolved to the wrong
near-duplicate incident. A 0.6 weight is a meaningful but non-fatal penalty
(matching the "do not block" principle) — two stacked penalties (LOW
retrieval + failed keywords, e.g. hyp-08) compound to 0.5 x 0.6 = 0.30,
pulling even a 0.8 raw score down to ~0.24.

### 3. Evidence confidence (not separately weighted)

Per-hypothesis "evidence confidence" — i.e. the *result* of the
validation-keyword search (its own top1_score/confidence level via
`IncidentSearchService.confidence_for()`) — was considered as a third factor,
but on inspection it is **redundant with the keyword success/failure check**:
in every observed case, a hypothesis's validation-keyword search either (a)
retrieves the expected incident with a score that classifies HIGH/MEDIUM
(success), or (b) misses it entirely (failure). A separate continuous
"evidence confidence" multiplier was not added in this phase to avoid
double-counting the same signal; this is noted as a future refinement below.

---

## Re-evaluated correlation: raw vs. composite confidence

Re-running [run_hypothesis_eval.py](tests/eval/run_hypothesis_eval.py) (a
fresh LLM run — `generate_hypotheses` is non-deterministic, so absolute
hypothesis text/counts differ slightly from Phase 5's run, but the same 8
cases / corpus / retrieval are used). Full results:
[tests/eval/results/hypothesis_v6a.json](tests/eval/results/hypothesis_v6a.json).

| | n_hypotheses | n_correct | n_incorrect | mean(correct) | mean(incorrect) | **point-biserial r** |
|---|---|---|---|---|---|---|
| Raw `confidence_score` | 35 | 23 | 12 | 0.700 | 0.658 | **0.171** |
| Composite confidence | 35 | 23 | 12 | 0.607 | 0.399 | **0.587** |

**Correlation more than tripled (0.171 -> 0.587)**, and the mean-confidence
gap between correct and incorrect hypotheses widened from 0.042 (barely
separable) to 0.208 (clearly separable). Composite confidence is a
substantially better correctness signal than the raw LLM-reported score,
without touching the model or its prompt.

### Other metrics (this run)

| Metric | Value |
|---|---|
| Retrieval Recall@5 | 1.0 (7/7) |
| Root-cause Recall@1 | 1.0 (7/7) |
| Root-cause Recall@3 | 1.0 (7/7) |
| Root-cause MRR | 1.0 |
| Validation-keyword Recall@5 | 0.857 (6/7) |

(Root-cause recall is higher than Phase 5's run (0.857 -> 1.0) purely due to
LLM non-determinism — hyp-02's correct hypothesis happened to rank #1 this
time. The composite-confidence improvement is independent of this and is the
focus of this phase.)

---

## Worked examples

### hyp-08 (negative case, `init_conf=LOW`, top1_score=0.295)

All 5 hypotheses had raw confidence 0.60-0.80 (Phase 5's headline problem —
confidently-stated, irrelevant hypotheses for a query with no real match).
With composite scoring:

| Rank | Raw | Retrieval weight | Keyword weight | Composite |
|---|---|---|---|---|
| 1 | 0.80 | 0.50 (LOW) | 0.60 (no expected incident) | **0.240** |
| 2 | 0.70 | 0.50 | 0.60 | 0.210 |
| 3 | 0.75 | 0.50 | 0.60 | 0.225 |
| 4 | 0.65 | 0.50 | 0.60 | 0.195 |
| 5 | 0.60 | 0.50 | 0.60 | 0.180 |

All five drop from the 0.60-0.80 "plausible" band into 0.18-0.24 — clearly in
LOW-confidence territory. A downstream consumer that previously saw "5
hypotheses around 70% confidence" now sees "5 hypotheses around 20%
confidence," correctly reflecting that none of them are backed by real
retrieval evidence.

### hyp-03 (`init_conf=MEDIUM`, top1_score=0.463) — validation-keyword failure on the *correct* hypothesis

Rank-1 hypothesis ("Memory leak in the type checker when using the --watch
flag") is the **correct** root cause (is_match=True) but its
validation_keywords fail to retrieve the expected incident (`kw_ok=False`,
the same generic-keyword issue identified in Phase 5):

| | Raw | Retrieval weight | Keyword weight | Composite |
|---|---|---|---|---|
| rank1 (correct) | 0.80 | 0.85 (MEDIUM) | 0.60 (kw fail) | 0.408 |

The composite score (0.408) is now the *lowest-confidence-looking* correct
hypothesis in the whole dataset — appropriately so, since `_collect_evidence`
would surface evidence for the wrong incident if this hypothesis were acted
on. Raw confidence (0.80) gave no hint of this problem.

### hyp-02 (`init_conf=MEDIUM`, top1_score=0.527) — discount widens the correct/incorrect gap

| Rank | Match | Raw | Keyword weight | Composite |
|---|---|---|---|---|
| 1 (correct) | True | 0.80 | 1.00 | 0.680 |
| 2 (incorrect) | False | 0.70 | 0.60 (kw fail) | 0.357 |
| 3 (incorrect) | False | 0.60 | 1.00 | 0.510 |
| 4 (correct) | True | 0.50 | 1.00 | 0.425 |
| 5 (correct) | True | 0.40 | 1.00 | 0.340 |

Raw confidence ranked rank2 (incorrect) above rank3 (incorrect) and rank4/5
(correct) — composite confidence correctly demotes rank2 below rank3 because
its validation keywords don't pan out, narrowing (though not eliminating) the
ranking inversion Phase 5 flagged.

---

## Observations and limitations

1. **The biggest single driver of the correlation improvement is the
   retrieval-confidence weight** (LOW -> x0.5), which uniformly discounts
   every hypothesis in cases like hyp-08 where the whole investigation rests
   on irrelevant context. The keyword-success weight provides a smaller,
   per-hypothesis refinement on top of that.
2. **Composite confidence is still a discount, not a re-ranking** — this
   phase deliberately did not reorder `ranked_hypotheses` or
   `validation_keywords` generation (out of scope: "no hypothesis-generation
   changes"). hyp-02's rank order is unchanged; only the confidence numbers
   attached to each rank changed. A future phase could use composite
   confidence to *re-sort* hypotheses, which would directly fix the Phase 5
   ranking-inversion failure mode.
3. **n=35 hypotheses across 8 cases is still small**, and `generate_hypotheses`
   is non-deterministic (this run's raw metrics differ from Phase 5's). The
   correlation improvement (0.171 -> 0.587) is large enough to be unlikely to
   be pure noise, but should be re-checked on a larger gold set before being
   treated as a precise number.
4. **"Evidence confidence" was folded into the keyword weight** rather than
   added as a third independent factor (see rule 3 above) — if future work
   wants a continuous evidence-strength signal (e.g. the validation-keyword
   search's own top1_score, not just success/failure), that could replace the
   binary keyword weight with a smoother function, likely improving the
   correlation further at the cost of more complexity.
5. **Weights (0.85/0.50 and 1.0/0.60) are reasoned defaults, not fitted** —
   chosen so that (a) HIGH-retrieval cases are unaffected, (b) LOW-retrieval
   cases are discounted enough to visibly separate from MEDIUM/HIGH cases, and
   (c) keyword failure is a meaningful-but-not-fatal penalty. As with Phase
   4's thresholds, these should be revisited as the gold set grows.
