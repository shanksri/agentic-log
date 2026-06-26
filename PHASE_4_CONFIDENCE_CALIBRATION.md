# Phase 4 Results: Retrieval Confidence Calibration

## What changed

- New module [app/services/confidence.py](app/services/confidence.py):
  `classify_confidence(top1_score) -> "LOW" | "MEDIUM" | "HIGH"`, plus the two
  threshold constants `LOW_CONFIDENCE_THRESHOLD = 0.40` and
  `HIGH_CONFIDENCE_THRESHOLD = 0.55`.
- [IncidentSearchService](app/services/search.py): added a static
  `confidence_for(results) -> (top1_score, confidence_level)` helper, and
  `confidence_level` is now included alongside `top1_score` in the
  `retrieval.search` / `retrieval.retrieve` structured logs.
- API (`/search/incidents`, `/search/debug`): `SearchResponse` and
  `SearchDebugResponse` now include `top1_score` and `confidence_level`
  ([app/api/schemas.py](app/api/schemas.py), [app/api/routes/search.py](app/api/routes/search.py)).
- **InvestigationAgent**: `_build_context()` now prefixes the LLM context with
  a `Retrieval confidence: <LEVEL> (top1_score=...)` header. When confidence
  is LOW (including the "no results" case), the header explicitly instructs
  the model to state that no strong historical match was found and to
  separate retrieved-incident evidence from its own general reasoning.
- **AdvancedInvestigationAgent**:
  - Initial retrieval's confidence is computed once and propagated into
    `_build_incident_context()` (same LOW-confidence header as above),
    `_assemble_report()`, and the top-level response as a new
    `retrieval_confidence: {level, top1_score}` field.
  - `_assemble_report()` prepends a `"LOW RETRIEVAL CONFIDENCE: ..."` flag to
    `confidence_assessment` when the initial retrieval confidence is LOW.
  - `_collect_evidence()` now computes `confidence_level`/`top1_score` per
    hypothesis (from its dense-only `search()` call) and includes them in each
    evidence item. `_build_evidence_context()` surfaces this per-hypothesis,
    and adds a note when no strong supporting evidence was found.
- New evaluation: [tests/eval/run_confidence_eval.py](tests/eval/run_confidence_eval.py),
  results in [tests/eval/results/confidence_v4.json](tests/eval/results/confidence_v4.json).

No changes to embeddings, canonical_text, chunking, hybrid search, the
retrieval algorithm, or reranking defaults (reranking remains enabled in the
agents as it was after Phase 2/3A — Phase 4 only adds a confidence signal
alongside it; the separate recommendation to default it off is unchanged but
not implemented here, per "do not re-enable reranking by default" /
keep-algorithm-unchanged scope).

---

## Score distributions (dense-only `search()`, new canonical text)

From the 24-query gold set, top1 similarity score:

| | n | min | max |
|---|---|---|---|
| MATCH (genuine expected incident) | 20 | 0.422 | 0.837 |
| NO_MATCH (no-match-expected) | 4 | 0.232 | 0.344 |

The two ranges do not overlap — a 0.078 gap exists between the highest
NO_MATCH score (0.344, `neg-03`) and the lowest MATCH score (0.422, `para-03`).

---

## Chosen thresholds

```
LOW    : top1_score <  0.40
MEDIUM : 0.40 <= top1_score < 0.55
HIGH   : top1_score >= 0.55
```

### Rationale

- **LOW/MEDIUM boundary (0.40)**: placed inside the 0.344–0.422 gap between
  NO_MATCH and MATCH scores, closer to the MATCH side. This is deliberately
  *not* the midpoint (0.383) — a 0.40 cutoff gives ~0.056 of margin above the
  highest NO_MATCH score (0.344) while still sitting 0.022 below the lowest
  MATCH score (0.422), erring toward correctly flagging borderline/no-match
  queries as LOW rather than risking a false HIGH/MEDIUM read on a
  no-match-expected query. On this gold set it produces a perfect
  confusion matrix (see below): all 4 NO_MATCH queries → LOW, all 20 MATCH
  queries → MEDIUM or HIGH.
- **MEDIUM/HIGH boundary (0.55)**: splits the MATCH range (0.422–0.837) so
  that the weaker paraphrase matches (para-02..05, all 0.42–0.53 — queries
  where the correct incident is semantically close but lexically different)
  fall into MEDIUM, while the lexical-overlap and stronger
  paraphrase/multi-concept matches (0.58–0.84) fall into HIGH. This isn't
  derived from a precision/recall optimum (there's no NO_MATCH data above
  0.40 to optimize against) — it's a round number chosen to make MEDIUM a
  meaningful "plausible but not strongly confident" band rather than an
  empty or near-empty bucket. 4/20 MATCH queries land in MEDIUM with this
  split.
- **Round numbers over fitted values**: 0.40/0.55 were chosen instead of, say,
  0.383 (exact midpoint) or 0.422 (exact MATCH minimum) because the gold set
  has only 24 queries (4 negatives) — thresholds fit tightly to this sample
  would be overconfident. 0.40/0.55 sit comfortably inside the observed gaps
  and are easy to reason about/tune later.

### Confidence level breakdown on the gold set

| | LOW | MEDIUM | HIGH |
|---|---|---|---|
| MATCH (n=20) | 0 | 4 | 16 |
| NO_MATCH (n=4) | 4 | 0 | 0 |

---

## Confusion matrix (LOW vs MATCH/NO_MATCH, threshold = 0.40)

Treating "confidence != LOW" (i.e. `top1_score >= 0.40`) as a prediction of
"MATCH exists":

| | Predicted MATCH | Predicted NO_MATCH |
|---|---|---|
| **Actual MATCH** (n=20) | TP = 20 | FN = 0 |
| **Actual NO_MATCH** (n=4) | FP = 0 | TN = 4 |

| Metric | Value |
|---|---|
| Precision | 1.0 |
| Recall | 1.0 |
| False Positive Rate | 0.0 |
| False Negative Rate | 0.0 |

### Sensitivity across candidate thresholds

From [confidence_v4.json](tests/eval/results/confidence_v4.json) (predicted
MATCH if `top1_score >= threshold`):

| Threshold | TP | FP | TN | FN | Precision | Recall | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 20 | 1 | 3 | 0 | 0.952 | 1.0 | 0.25 | 0.0 |
| 0.35 | 20 | 0 | 4 | 0 | 1.0 | 1.0 | 0.0 | 0.0 |
| **0.40 (chosen LOW boundary)** | **20** | **0** | **4** | **0** | **1.0** | **1.0** | **0.0** | **0.0** |
| 0.42 | 20 | 0 | 4 | 0 | 1.0 | 1.0 | 0.0 | 0.0 |
| 0.45 | 19 | 0 | 4 | 1 | 1.0 | 0.95 | 0.0 | 0.05 |
| 0.55 (chosen MEDIUM/HIGH boundary) | 16 | 0 | 4 | 4 | 1.0 | 0.8 | 0.0 | 0.2 |
| 0.60 | 13 | 0 | 4 | 7 | 1.0 | 0.65 | 0.0 | 0.35 |

The perfect-separation plateau spans 0.35–0.42; 0.40 sits in the middle of
that plateau, with 0.30 already producing a false positive (`neg-03` at
0.344 would be misclassified as MATCH). Anything above ~0.43 starts costing
recall by misclassifying `para-03` (0.422) as LOW.

---

## Examples of LOW/MEDIUM/HIGH outputs

### HIGH — `lex-01`: *"scheduler crashloop ValidationError UUID dag_version_id is NULL"*

top1_score = 0.782 → **HIGH**. `InvestigationAgent._build_context()` produces:

```
Retrieval confidence: HIGH (top1_score=0.782)

Incident 1
Similarity score: 0.782
Title: Scheduler crashloops with ValidationError: UUID input should be a string...
Symptoms: Scheduler crashloops with ValidationError: UUID input should be a…
Severity: unknown
Status: open
Resolution summary: Unknown
...
```

No extra caveats — the agent is told this is a confident match and can
reason from it directly.

### MEDIUM — `para-03`: *"node process runs out of memory when building a large bundle of files"*

top1_score = 0.422 → **MEDIUM**. Context:

```
Retrieval confidence: MEDIUM (top1_score=0.422)

Incident 1
Similarity score: 0.422
Title: JavaScript heap out of memory for 10s of MB of source...
...
```

No LOW-confidence caveat is added (only LOW triggers the explicit
"no strong match" instruction), but the numeric score is visible to the LLM
and to API/log consumers, so a MEDIUM result is distinguishable from a HIGH
one without being treated as "no evidence."

### LOW — `neg-03`: *"spreadsheet export formatting broken when opened in Excel 2010"*

top1_score = 0.344 → **LOW**. Context:

```
Retrieval confidence: LOW (top1_score=0.344)
No strong historical match was found. The incidents below are the closest
available but may not be directly relevant. State explicitly that no strong
historical match was found, and clearly separate evidence drawn from these
incidents from your own general reasoning.

Incident 1
Similarity score: 0.344
Title: ...
...
```

For `AdvancedInvestigationAgent`, this also produces
`retrieval_confidence: {"level": "LOW", "top1_score": 0.344}` at the top level
of the response, and `_assemble_report()` prepends:

```
LOW RETRIEVAL CONFIDENCE: no strong historical match was found for this
problem. The assessment below relies primarily on general reasoning rather
than retrieved incident evidence. <original confidence_assessment text>
```

### LOW — zero results

If `search()`/`retrieve()` returns no candidates at all (e.g. an empty
corpus, or all candidates filtered out by metadata filters),
`confidence_for([])` returns `(None, "LOW")`, and both agents emit:

```
Retrieval confidence: LOW (no similar incidents were retrieved).
No historical evidence is available. Any analysis below must be based on
general reasoning, not retrieved incidents - state this explicitly.
```

The investigation is **not blocked** — `generate_investigation` /
`evaluate_investigation_evidence` still run, just with this framing in their
context.

---

## Observations and recommendations for future refinement

1. **The 0.40/0.55 thresholds perfectly separate this gold set**, but n=24
   (4 negatives) is small and the margin to the nearest NO_MATCH score
   (`neg-03` at 0.344) is only 0.056. As more negative-control queries are
   added to the gold set — especially ones closer to real corpus topics than
   the current 4 (payments, mobile push, spreadsheets, VPN — all quite far
   from this incident corpus's actual domains) — re-run
   `run_confidence_eval.py` and check whether 0.40 still sits inside a
   no-overlap plateau. If new negatives start scoring above ~0.40, the LOW
   boundary will need to move up (at some recall cost on `para-03`-like weak
   paraphrases).
2. **MEDIUM is currently un-validated against negatives** — there's no
   NO_MATCH data in the 0.40–0.55 range to confirm MEDIUM correctly means
   "plausible weak match" rather than "borderline false positive." Adding
   gold queries whose nearest-neighbor incident is topically *related but not
   a true match* (a harder negative class than the current fully-unrelated
   negatives) would let MEDIUM be validated/tuned properly.
3. **`_collect_evidence()` per-hypothesis confidence is new and unused
   downstream beyond the context text** — `evidence[i]["confidence_level"]`
   is now available on the `AdvancedInvestigationAgent.investigate()` return
   value but isn't yet folded into `ranked_hypotheses` ordering or
   `confidence_assessment` beyond the initial-retrieval flag. A natural
   follow-up is to have the final report's `confidence_assessment` reflect a
   combination of (a) initial retrieval confidence and (b) how many
   hypotheses had LOW-confidence supporting evidence.
4. **Thresholds are global constants, not per-query-type** — lexical-overlap
   queries score systematically higher (0.58–0.84) than paraphrase queries
   (0.42–0.63) even when both are genuine matches. A single global threshold
   works here because the NO_MATCH ceiling (0.344) is still below the
   paraphrase floor (0.422), but if paraphrase-style queries become a larger
   share of real traffic, consider whether a single top1_score threshold
   remains the right signal, or whether query-type-aware calibration is
   needed.
5. **No change to reranking defaults in this phase** — both agents still call
   `retrieve(..., expand=True, rerank=True)`. The earlier recommendation
   (Phase 3A/3B analysis) to make reranking optional/off-by-default is
   independent of this phase's confidence work and remains a separate,
   not-yet-implemented decision.
