# 21 — Evaluation Platform Productionization (Phases 21A–21F)

Docs 15 and 20 cover *measuring* retrieval and reasoning quality. This document covers what turns
those measurements into an operable system: **21A** derives structured failure intelligence and
prioritized recommendations from already-computed reports; **21B** validates whether the Phase
20B judges themselves can be trusted; **21C**/**21D** are human-in-the-loop tooling for growing the
gold datasets that docs 15/20 depend on; **21E** wires every earlier phase into one callable
pipeline; **21F** is the on-disk persistence and inspection layer for pipeline runs. Every module in
this document is either a **pure analysis layer** (21A, 21B — never reruns anything, never calls an
LLM, never imports `IncidentSearchService`/agent classes/judge implementations) or **orchestration
only** (21E — introduces no new metrics, retrieval, or reasoning logic). None of 21A–21F is wired
into an API route directly; all are reachable through the `/evaluation/*` endpoints documented in
doc 22, which call into this layer.

---

## Phase 21A — AI Quality Intelligence

Three modules: failure detection, recommendation generation, and report assembly — each consuming
only the output of the one before it.

### Goal

Detect and categorize failures from the immutable reports Phases 16D/16E/16F (retrieval), 20A
(reasoning), and 20B (judge) already produce; cluster related failures; turn clusters into
priority-ordered, evidence-traceable engineering recommendations; and assemble all of it into one
`AIQualityReport` answering "what is system quality right now, and what should we do about it."

### Motivation

Every earlier evaluation phase produces a report, but nothing synthesizes *why* things failed or
*what to do next* across all three report types at once. This phase is explicitly "a PURE analysis
layer over evaluation artifacts already produced by every earlier phase... never reruns retrieval,
never reruns an investigation, never makes an LLM call, and never imports `IncidentSearchService`,
`MultiAgentInvestigationOrchestrator`, `LLMService`, or any agent class." The recommendation
engine's own docstring states the guiding constraint even more sharply: "never hard-codes a
recommendation unrelated to an observed failure... `generate_recommendations` produces zero
recommendations when given zero clusters."

### Architecture

**Failure analysis** (`app/evaluation/failure_analysis.py`):

```python
Component(str, Enum): RETRIEVAL, PLANNER, HYPOTHESIS_GENERATOR, EVIDENCE_EVALUATOR, DECISION,
    CRITIC, ORCHESTRATOR, JUDGE
FailureCategory(str, Enum): SEARCH_FAILURE, INCOMPLETE_RECALL, UNRESOLVED_GOLD_ENTRY,      # retrieval
    STRATEGY_MISMATCH, MISSING_HYPOTHESIS, DUPLICATE_HYPOTHESIS, INCORRECT_DECISION,
    INCORRECT_CRITIQUE, NO_CONVERGENCE,                                                     # reasoning
    LOW_CONFIDENCE, MALFORMED_EVALUATION, RULE_DISAGREEMENT                                  # judge
CauseLevel(str, Enum): IMMEDIATE, UNDERLYING, SYSTEMIC   # ordered
Severity(str, Enum): LOW, MEDIUM, HIGH, CRITICAL

CauseStep(frozen): level: CauseLevel, description: str
FailureRecord(frozen): component, stage: str, category, severity, subject_id: str, description: str,
    evidence: tuple[str,...], metrics_involved: tuple[str,...],
    cause_chain: tuple[CauseStep,...]   # always exactly 3 steps: IMMEDIATE, UNDERLYING, SYSTEMIC
FailureCluster(frozen): component, category, failures: tuple[FailureRecord,...],
    severity: Severity (max among members), common_cause: str (mode of systemic causes in cluster)
FailureSummary(frozen): total_failures, by_component/by_category/by_severity: tuple[...Count,...]

classify_severity(deviation: float) -> Severity    # deviation in [0,1], clamped:
    #  >= 0.75 -> CRITICAL, >= 0.50 -> HIGH, >= 0.25 -> MEDIUM, else LOW

analyze_retrieval_failures(report: EvaluationReport) -> tuple[FailureRecord,...]
analyze_reasoning_failures(report: InvestigationEvaluationReport) -> tuple[FailureRecord,...]
analyze_judge_failures(evaluations, *, judge_errors=()) -> tuple[FailureRecord,...]
cluster_failures(failures) -> tuple[FailureCluster,...]     # grouped by (component, category)
summarize_failures(failures) -> FailureSummary
```

**Recommendation engine** (`app/evaluation/recommendation_engine.py`):

```python
Priority(str, Enum): LOW, MEDIUM, HIGH, CRITICAL     # mirrors Severity vocabulary 1:1
CONFIDENCE_BASE, CONFIDENCE_PER_FAILURE, CONFIDENCE_MAX = 0.3, 0.1, 1.0

Recommendation(frozen): problem: str, root_cause: str (cluster.common_cause, verbatim),
    estimated_impact: int (len(cluster.failures)), confidence: float ([0,1], rounded to 4dp),
    recommended_action: str, priority: Priority

_ACTION_BY_CATEGORY: dict[FailureCategory, str]   # one deterministic action string per category,
    # e.g. SEARCH_FAILURE -> "investigate retrieval infrastructure/connectivity stability",
    #      INCORRECT_DECISION -> "recalibrate decision acceptance threshold or evidence weighting"

generate_recommendations(clusters: Sequence[FailureCluster]) -> tuple[Recommendation,...]
    # confidence = min(1.0, 0.3 + 0.1 * len(cluster.failures)) -- more repetition = more confidence
    # this is a real pattern, not noise (same reasoning doc 19's orchestrator applies to repeated signals)
    # sorted by (-priority, -estimated_impact, problem) for deterministic ordering
```

**AI quality report** (`app/evaluation/ai_quality_report.py`):

```python
ComponentSummary(frozen): component, total_failures, severity_breakdown: tuple[SeverityFailureCount,...]
TrendSummary(frozen): failure_count_trend: tuple[int,...] (oldest first),
    regression_verdict: str | None   # carried through verbatim, NEVER re-derived
AIQualityReport(frozen): generated_at, overall_summary: str, failure_summary: FailureSummary,
    component_summaries: tuple[ComponentSummary,...], failure_clusters: tuple[FailureCluster,...],
    recommendations: tuple[Recommendation,...], trend_summary: TrendSummary | None
    # trend_summary is None unless >= 2 report histories are supplied, or a regression_verdict is
    # explicitly passed in -- never fabricates a one-point trend

build_quality_report(retrieval_reports=(), reasoning_reports=(), judge_evaluations=(),
                      judge_errors=(), regression_verdict=None) -> AIQualityReport
build_quality_report_from_benchmarks(...)   # convenience: reads Phase 16F/20A/20B repositories directly
```

### Lifecycle

**Retrieval failures** (`analyze_retrieval_failures`): for each `QueryEvaluationOutcome`, a skipped
query becomes a CRITICAL `SEARCH_FAILURE` (deviation=1.0); a resolved query with `recall_at_k < 1.0`
becomes an `INCOMPLETE_RECALL` failure with `severity = classify_severity(1.0 - recall)`, whose
systemic cause names the query's *category* if that category's mean also underperforms the dataset
mean, or says "isolated query-level miss" otherwise; unresolved gold entries become
`UNRESOLVED_GOLD_ENTRY` at a fixed severity. A perfect query (recall=1.0, not skipped, nothing
unresolved) produces **no** record — only deviations are recorded.

**Reasoning failures** (`analyze_reasoning_failures`): for each `InvestigationResult`, every failing
check (`planner_correct`, `hypothesis_recall_hit`, `decision_correct`, `critic_correct`,
`stopping_correct`) produces its own `FailureRecord` — not one aggregated record — but the
**cause chain is anchored on the most upstream failing stage** (planner &gt; hypotheses &gt; decision
&gt; critic &gt; stopping), since "a downstream failure is frequently a consequence of an upstream
one." Deviation increases with the number of co-occurring failures:
`min(1.0, 0.5 + 0.15 * extra_failing_checks)`.

**Judge failures** (`analyze_judge_failures`): any `JudgeEvaluation` whose `score.band` is "Poor" or
"Weak" becomes a `LOW_CONFIDENCE` record with `severity = classify_severity((SCORE_MAX -
score.value) / (SCORE_MAX - SCORE_MIN))`, evidence drawn from the judge's own flagged weaknesses;
pre-caught `JudgeResponseError` text passed in as `judge_errors` becomes `MALFORMED_EVALUATION`
records without raising a new exception.

**Clustering** groups by `(component, category)`, computing cluster severity as the max among
members and `common_cause` as the most-frequent systemic-cause description across the cluster.
**Recommendation generation** builds exactly one `Recommendation` per cluster — `root_cause` is the
cluster's `common_cause` verbatim, `recommended_action` a fixed lookup by category, `confidence`
scaling with cluster size (capped at 1.0), `priority` mapped 1:1 from the cluster's severity — then
sorts by priority descending, impact descending, problem text alphabetically for full determinism.
**Report assembly** accumulates failures across every supplied retrieval/reasoning report plus judge
evaluations/errors, clusters and summarizes them, and populates `trend_summary` only when at least
two report histories exist (or a regression verdict was explicitly supplied) — the regression
verdict itself is always carried through from an already-computed `RegressionReport`/
`ReasoningRegressionReport`, never recomputed.

### Design decisions

- **Read-only, evidence-traceable by construction** — every `Recommendation` traces to exactly one
  `FailureCluster`, which traces to `FailureRecord`s, which trace to fields already present on
  input reports; nothing is invented.
- **Upstream-anchored cause chains for multi-failure investigations** is an explicit heuristic, not
  a verified causal claim — documented as such in Risks.
- **Confidence (recommendation) is independent of severity** — a single CRITICAL failure has *low*
  confidence (0.4, could be coincidence); ten LOW failures in one cluster have *high* confidence
  (capped at 1.0) — repetition, not severity, is evidence of a real pattern.
- **No fallback recommendation text for unknown categories** — `_ACTION_BY_CATEGORY` covers every
  current `FailureCategory` member; a fallback exists only as a defensive measure for a category
  added later without an update to this dict.
- **Deterministic ordering everywhere** — recommendation sort order, cluster iteration order (first-
  seen), and component-summary order are all fixed so identical inputs always produce identical
  reports.

### Interfaces

`failure_analysis.py` imports report types from Phase 16D (`EvaluationReport`,
`QueryEvaluationOutcome`), Phase 20A (`InvestigationEvaluationReport`, `InvestigationResult`), and
Phase 20B (`JudgeEvaluation`, `SCORE_MIN`/`SCORE_MAX`) — and explicitly never imports
`IncidentSearchService`, any agent class, or any judge implementation. `recommendation_engine.py`
imports only `failure_analysis.py`'s types. `ai_quality_report.py` imports both, plus (via the
`_from_benchmarks` convenience builder) Phase 16F/20A/20B's repository types. None of the three is
imported by any API route directly; all three are consumed by the `/evaluation/full` endpoint (doc
22) via `build_quality_report`.

### Testing

`test_failure_analysis.py`: severity-band thresholds and out-of-range clamping; a perfect query
producing no record; a skipped query producing a CRITICAL search failure with a full 3-level cause
chain; incomplete-recall severity matching the deviation exactly; unresolved gold entries detected;
the category-vs-dataset-mean systemic-cause distinction; multi-failure reasoning cases and
upstream-anchored cause chains; clustering and summary aggregation behavior. `test_recommendation_engine.py`:
zero clusters produce zero recommendations; every recommendation field traces back to its cluster;
confidence increases monotonically with cluster size up to the cap; priority matches severity
1:1; every `FailureCategory` has a defined, non-fallback action; recommendations sort by
priority-then-impact-then-problem-text; ordering is deterministic across repeated calls; and
`Recommendation` is frozen. `test_ai_quality_report.py`: report assembly from single vs. multiple
report histories; trend-summary population only when warranted; component-summary breakdowns;
graceful behavior with zero reports supplied; and regression-verdict pass-through.

### Risks

Upstream-anchored cause chains for multi-failure investigations are a documented heuristic, not
verified causal proof — a downstream failure with a fully-correct upstream chain still gets a
shallower cause chain than it might deserve. Judge cause chains are inherently shallow (a judge
score is already a holistic semantic judgment; there's no further breakdown available without
re-calling the judge). The category-mean systemic-cause check is a simplification — it only
distinguishes "category underperforms" from "isolated miss," nothing finer-grained.

### Future work

None named explicitly in the module docstrings.

---

## Phase 21B — Judge Validation

Four modules answering one question, per the module docstrings: "can this judge be trusted?" — all
pure statistics over already-collected scores; none calls a judge, reruns reasoning, or makes an
LLM call.

### Goal

Correlate judge scores against external quality signals, measure agreement between judge pairs
(human/rule/LLM) and consistency of repeated evaluations, detect systematic bias, and roll all of
it into one `JudgeValidationReport` with a production-usable trustworthiness verdict.

### Motivation

Phase 20B ships two judge implementations with no way to know whether either can actually be
trusted. Per the calibration module's docstring: "Do not invent calibration curves beyond available
data — this module computes exactly one number (Pearson's r) per comparison and reports `None`...
whenever fewer than two points exist or either series has zero variance, rather than fitting any
curve." Per the agreement module: "treat judges exactly like ML models... only validation" — this
is statistics over outputs, never new reasoning.

### Architecture

**Calibration &amp; correlation** (`app/evaluation/judge_calibration.py`):

```python
CORRELATION_WEAK_THRESHOLD = 0.2   # Cohen's conventional "small effect" threshold
_REGRESSION_VERDICT_TO_NUMBER = {"improved": 1.0, "unchanged": 0.0, "mixed": 0.0,
                                  "regressed": -1.0, "incompatible": None}

CalibrationPoint(frozen): subject_id, judge_score: float, quality_metric: float
CalibrationResult(frozen): metric_name, n: int, correlation: float | None,
    direction: str ("positive"/"negative"/"weak"/"undefined"), points: tuple[CalibrationPoint,...]
CorrelationResult(frozen): series_a_name, series_b_name, n, correlation: float | None, direction: str

pearson_correlation(xs, ys) -> float | None   # None if n<2 or either series has zero variance
classify_direction(correlation) -> str        # >=0.2 positive, <=-0.2 negative, else weak/undefined
regression_verdict_to_number(verdict: str) -> float | None   # string-keyed, never imports the Verdict enum
analyze_calibration(metric_name, points) -> CalibrationResult
analyze_correlation(series_a_name, series_a, series_b_name, series_b) -> CorrelationResult
```

**Agreement &amp; bias** (`app/evaluation/judge_agreement.py`):

```python
BIAS_THRESHOLD = 0.5        # half a rubric point on the 1-10 scale
DEFAULT_TOLERANCE = 1.0     # "within one point" default agreement tolerance
AgreementPair(str, Enum): HUMAN_VS_LLM, HUMAN_VS_RULE, RULE_VS_LLM
BiasDirection(str, Enum): FIRST_HIGHER, SECOND_HIGHER

ScoredRecord(frozen): record_id, stage, human_score/rule_score/llm_score: float | None each
AgreementResult(frozen): pair, stage, n, differences: tuple[float,...],
    mean_absolute_difference: float | None, agreement_within_tolerance: float | None, tolerance
ConsistencyResult(frozen): stage, n, scores: tuple[float,...], mean, variance (population), std_dev, min, max
PromptVariantResult(frozen): variant, stage, score
StageDrift(frozen): stage, drift: float (max-min across variants), scores_by_variant
PromptSensitivityReport(frozen): stage_drifts, mean_drift, max_drift
BiasFinding(frozen): pair, stage, mean_signed_difference (first - second, NOT absolute),
    direction: BiasDirection, n, description: str

compute_agreement(records, pair, *, tolerance=1.0) -> tuple[AgreementResult,...]
analyze_consistency(stage, scores) -> ConsistencyResult
collect_repeated_scores(evaluate_fn, *, n) -> tuple[float,...]   # the ONLY place this module touches a Judge
analyze_prompt_sensitivity(results) -> PromptSensitivityReport
analyze_bias(records, pair) -> tuple[BiasFinding,...]   # only reports if |mean_signed_diff| >= 0.5
```

**Human evaluation dataset** (`app/evaluation/human_eval_dataset.py`):

```python
HumanEvaluationRecord(frozen): record_id,
    human_planner_score/human_hypotheses_score/human_decision_score/human_critique_score/
    human_overall_score: float | None = None (each independently optional, [0,10] if present),
    notes: str = ""
    .issues() -> list[str]
HumanEvaluationDataset(frozen): version, description, created_at, records=(), author=None
    .issues() -> list[str]; .is_valid() -> bool; .get(record_id) -> HumanEvaluationRecord | None
```

**Validation report** (`app/evaluation/judge_validation_report.py`):

```python
CONFIDENCE_BASELINE_N = 20      # "enough data points to draw a basic conclusion"
PENALTY_PER_FINDING = 0.2
LOW_AGREEMENT_THRESHOLD = 0.6   # same "majority" reasoning as doc 19C's CONTRADICTION_RATIO_THRESHOLD
HIGH_STD_DEV_THRESHOLD = 1.0    # full rubric point

Trustworthiness(str, Enum): HIGH, MEDIUM, LOW, VERY_LOW, INSUFFICIENT_DATA
_RECOMMENDATION_BY_TRUSTWORTHINESS: dict[Trustworthiness, str]   # fixed sentence per band

JudgeValidationReport(frozen): generated_at, agreement, consistency, calibration, bias, correlation
    (each a tuple of the corresponding Phase 21B result type), overall_trustworthiness,
    recommended_production_usage: str, confidence_level: float

assemble_validation_report(agreement=(), consistency=(), calibration=(), bias=(), correlation=()) -> JudgeValidationReport
build_validation_report_from_benchmarks(judged_repo, experiment_name=None, existing_*=()) -> JudgeValidationReport
```

### Lifecycle

**Calibration**: `pearson_correlation` computes the standard `r` formula and returns `None` — never
a guessed value — whenever fewer than two points exist or either series is constant.
`classify_direction` applies the 0.2 threshold in both directions, else "weak," or "undefined" for
`None`. `regression_verdict_to_number` maps a plain string (never the `Verdict` enum, to avoid
coupling to Phase 16E/20A) to a signed number, with `"incompatible"` mapping explicitly to `None`.

**Agreement/consistency/bias**: `compute_agreement` groups by stage and, for each stage, computes
the absolute-difference distribution between the two named judges in a pair, the mean absolute
difference, and the fraction within `tolerance` — records missing a score on either side are simply
excluded, never treated as zero. `analyze_consistency` is pure statistics over a caller-supplied
list of repeated scores (population variance, std dev, min, max) — this module never collects the
scores itself; `collect_repeated_scores` is the one explicit exception, a thin wrapper that calls a
caller-supplied zero-argument closure `n` times. `analyze_bias` computes the **signed** (not
absolute) mean difference per stage and only emits a `BiasFinding` if `|mean_signed_difference| >=
0.5` — a difference below that is treated as noise, not a documented finding.

**Validation report assembly**: trust starts at a perfect 1.0 and is penalized 0.2 per: each
`BiasFinding`; each `AgreementResult` with `agreement_within_tolerance < 0.6`; each
`ConsistencyResult` with `std_dev > 1.0`; each `CalibrationResult` with `direction == "negative"`;
each `CorrelationResult` with `direction == "negative"`. The penalized score is clamped to `[0, 1]`
and banded: `>=0.75` HIGH, `>=0.5` MEDIUM, `>=0.25` LOW, else VERY_LOW — unless the total sample
size across every input is zero, in which case the verdict is `INSUFFICIENT_DATA` regardless of the
penalty math (a fabricated score is never reported from zero data).
`confidence_level = min(1.0, total_n / 20)`. `build_validation_report_from_benchmarks` loads a Phase
20B judged-run history, extracts mean session-level judge score alongside decision accuracy and
regression verdict per run, builds two correlation analyses from that (judge-score-vs-accuracy,
judge-score-vs-regression), folds in any separately-computed agreement/consistency/calibration/bias
(e.g. against a `HumanEvaluationDataset`), and calls `assemble_validation_report`.

### Design decisions

- **`None` over a guessed number, everywhere in 21B** — insufficient data always produces `None`/
  `INSUFFICIENT_DATA`, never a fabricated zero or default score.
- **Bias is signed, not absolute** — direction (which judge scores systematically higher) is
  reported, not just magnitude, because "different by 2 points, always in the same direction" is a
  materially different finding from "different by 2 points, randomly either way."
- **Human scores are entirely optional** in `HumanEvaluationDataset` — a record with every score
  `None` (only `notes` populated) is valid, "since human labels should remain optional so synthetic
  datasets are still usable." The schema deliberately does not import `app.evaluation.judge` or its
  SCORE_MIN/MAX constants, so a future change to the judge's scale can't silently break this schema.
- **Penalty-based trustworthiness, not a hidden formula** — starts at perfect trust, subtracts a
  fixed, documented amount per piece of negative evidence; every threshold (0.6 agreement, 1.0 std
  dev, 0.2 correlation) traces to a stated rationale.
- **Confidence is reported separately from the trustworthiness verdict** — two judges could land in
  the same trustworthiness band from very different amounts of supporting data (20 points vs. 200);
  conflating the two would hide that difference.
- **Regression verdict encoded numerically, never re-derived** — `regression_verdict_to_number`
  reads only the plain string a regression report already produced.

### Interfaces

`judge_calibration.py` and `judge_agreement.py` import nothing from the rest of the evaluation
system (pure statistics, standard library only) and never import `RuleJudge`/`LLMJudge`.
`human_eval_dataset.py` imports nothing from Phase 20B either. `judge_validation_report.py` imports
`judge_calibration`/`judge_agreement`'s result types plus Phase 20B's
`JudgedReasoningBenchmarkRepository` (for the convenience builder only). None imported directly by
any route; `build_validation_report_from_benchmarks` is called by the `/evaluation/full` pipeline
(doc 21E, doc 22).

### Testing

`test_judge_calibration.py`: hand-verified Pearson r; `None` for `n<2` and for zero-variance series;
direction-classification boundary cases at ±0.2; all five regression-verdict mappings including
`"incompatible" -> None`; correlation on arbitrary series pairs. `test_judge_agreement.py`:
agreement-distribution and tolerance-fraction computation; consistency mean/variance/std-dev
correctness; prompt-sensitivity drift (max-min per stage); bias detection above/below the 0.5
threshold and correct direction classification; missing-score records excluded rather than imputed.
`test_human_eval_dataset.py`: records with all-None, some-populated, and all-populated scores;
score-range validation; duplicate-record-id rejection; `.get()` lookup;
`.is_valid()` edge cases. `test_judge_benchmark_validation.py`: trustworthiness penalty accumulation
and banding; confidence as `min(1.0, n/20)`; `INSUFFICIENT_DATA` when `n=0`; the fixed
recommendation-string mapping; and benchmark-history integration (loading runs, extracting metrics,
computing correlations).

### Risks

No confidence interval or hypothesis-testing rigor is applied to the Pearson correlations —
`n>=2` is the only bar before a correlation is reported, which can be a very small, noisy sample.
The 0.5-point bias threshold and 1.0-point consistency threshold are reasoned defaults on this
framework's 1–10 scale, not independently validated. The trustworthiness penalty schedule (flat 0.2
per finding, regardless of finding severity) treats a mild negative correlation the same as a severe
one.

### Future work

None named explicitly in the module docstrings.

---

## Phase 21C — Gold Dataset Authoring

### Goal

A human-in-the-loop workflow where an LLM proposes diverse candidate queries and investigation
scenarios from incident descriptions, a human reviewer accepts/edits/rejects each candidate, and
accepted items export directly into the existing `GoldDataset`/`ReasoningGoldDataset` schemas (doc
15, doc 20).

### Motivation

Growing the gold datasets docs 15/20 depend on has always been manual authoring. Per the module
docstring: "No retrieval. No evaluation. No benchmark execution. Only dataset creation." — this
phase is scoped strictly to producing candidates for a human to review, never to judging or scoring
anything.

### Architecture

```python
DEFAULT_N_QUERIES = 5
GENERATION_METHODS = ("exact_keyword", "paraphrase", "symptom_description", "novice_wording", "multi_concept")
_METHOD_TO_CATEGORY = {"exact_keyword": "lexical-overlap", "paraphrase": "paraphrase",
    "symptom_description": "paraphrase", "novice_wording": "paraphrase", "multi_concept": "multi-concept"}
_METHOD_TO_DIFFICULTY = {"exact_keyword": "easy", "paraphrase": "medium",
    "symptom_description": "medium", "novice_wording": "medium", "multi_concept": "hard"}

class AuthorLLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...   # no SDK coupling, same pattern as 20B's JudgeLLMClient

IncidentSummary(frozen): incident_id, title, description, source_type="", source_external_id=""
ReviewDecision(str, Enum): PENDING, ACCEPTED, EDITED, REJECTED

CandidateQuery(frozen): id (uuid), incident_id, query, category, difficulty, rationale,
    generation_method, status=PENDING, edited_query: str | None = None
    .effective_query -> edited_query if set else query   # original always preserved even after edit
CandidateScenario(frozen): id, incident_id, problem, expected_root_causes, suggested_strategy,
    rationale, status=PENDING, edited_problem: str | None = None
    .effective_problem -> edited_problem if set else problem

AuthoringStats(frozen): total_generated, accepted, edited, rejected, pending (queries),
    acceptance_rate = (accepted+edited)/total, edit_rate = edited/max(1, accepted+edited),
    mean_queries_per_incident, total_scenarios/accepted_scenarios/... (scenarios, same shape)

AuthorResponseError, VersionAlreadyExportedError, CandidateNotFoundError   # exceptions

LLMDatasetAuthor(llm_client):
    .generate_queries(incident, n=5) -> tuple[CandidateQuery,...]
    .generate_investigation(incident) -> CandidateScenario
    .review(candidate_id, decision, edited_text=None) -> updated candidate
    .export_gold_dataset(version, description, author=None) -> GoldDataset
    .export_reasoning_dataset(...) -> ReasoningGoldDataset
    .stats() -> AuthoringStats
```

### Lifecycle

**Generation**: `generate_queries` prompts for exactly `n` diverse variants covering the five
`GENERATION_METHODS`, requesting a strict JSON shape (`{"candidates": [{"generation_method",
"query", "rationale"}, ...]}`); each parsed item is looked up in `_METHOD_TO_CATEGORY`/
`_METHOD_TO_DIFFICULTY`, wrapped in a `CandidateQuery` with a fresh UUID and `status=PENDING`, and
stored. `generate_investigation` follows the identical pattern for a single `CandidateScenario`,
requesting `{"problem", "expected_root_causes", "suggested_strategy", "rationale"}`. Malformed JSON
or missing/invalid fields on either path raise `AuthorResponseError` immediately — no automatic
retry. **Review**: `.review(candidate_id, decision, edited_text=None)` transitions
PENDING→ACCEPTED/EDITED/REJECTED; on EDITED, `edited_text` is required non-empty and stored
separately from the original AI-generated text (both survive — full audit trail), and a **new**
frozen object replaces the dict entry (no in-place mutation). **Export**: filters to
`status in (ACCEPTED, EDITED)`, uses `effective_query`/`effective_problem` (edited text if present,
else original), and raises `VersionAlreadyExportedError` if the same version string was already
exported from this author instance. Exported `GoldQuery.expected_incidents` is left empty — Phase
21D is what populates it. **Stats** are computed fresh from the live candidate pool on every call,
never accumulated separately.

### Design decisions

- **Original AI text is always preserved alongside a human edit** — `edited_query`/`edited_problem`
  sit next to the untouched `query`/`problem`, so an edit is auditable, not a silent overwrite.
- **Version-collision tracking is in-process only** — a fresh `LLMDatasetAuthor` instance can
  re-export any version; this is a session-level safety check, not a database constraint.
- **Acceptance rate counts edits as accepted** — the numerator is `accepted + edited`, since an
  edited candidate was still judged usable, just imperfect as generated.
- **No automatic parse-failure fallback** — an `AuthorResponseError` always propagates to the
  caller; retry/repair policy is explicitly left to the caller.
- **Explicit forbidden-import list** — the module must not import `IncidentSearchService`, the
  evaluation harness, `Planner`, `Judge`, `Benchmark`, or `Regression`; its only real dependencies
  are the `GoldDataset`/`ReasoningGoldDataset` schemas and the `AuthorLLMClient` protocol.

### Interfaces

Imports `GoldDataset`/`GoldQuery` (Phase 16B) and `ReasoningGoldDataset`/`InvestigationScenario`
(Phase 20A) schemas only, plus the standard library. Public surface: `LLMDatasetAuthor` and its
supporting dataclasses/enums/exceptions. Not imported by any API route.

### Testing

`test_dataset_authoring.py` covers: query-generation diversity (all five methods produced and
correctly mapped to category/difficulty); JSON parsing (valid, invalid, missing fields, invalid
enum values, each raising `AuthorResponseError`); review state transitions and edit-text
preservation; version-export uniqueness (`VersionAlreadyExportedError` on a second export of the
same version); statistics formulas (`acceptance_rate`, `edit_rate`, `mean_queries_per_incident`);
and candidate validation via `.issues()`/`.is_valid()`.

### Risks

No automated quality check on LLM-generated candidates beyond schema validation — a candidate can
be well-formed JSON but a poor or nonsensical query/scenario; catching that is entirely the human
reviewer's job. Version-collision protection resets with every new `LLMDatasetAuthor` instance, so
it cannot prevent a genuinely duplicate export across separate sessions/processes.

### Future work

None named explicitly in the module docstring.

---

## Phase 21D — Assisted Gold Labeling

### Goal

Assist a human reviewer in populating `GoldQuery.expected_incidents` for the `CandidateQuery`
objects Phase 21C's `export_gold_dataset()` produces: run retrieval automatically for each
candidate, present the top-K results in rank order, and let the reviewer simply *select* the
correct incident id(s) rather than manually searching and copying UUIDs.

### Motivation

Per the module docstring, without this framework "a reviewer must manually run a search, scan
results, and copy UUIDs — slow and error-prone." This phase automates the retrieval and
presentation steps while leaving the actual relevance judgment entirely to the human.

### Architecture

```python
DEFAULT_LIMIT = 10
RETRIEVAL_STRATEGY_DENSE, RETRIEVAL_STRATEGY_HYBRID = "dense", "hybrid"
DEFAULT_RELEVANCE = RELEVANCE_MAX = 3   # Phase 16B's 1-3 scale; every reviewer selection gets the max

CandidateIncident(frozen): incident_id, title, source (source_type:source_external_id), score,
    rank (1-based), explanation="" (e.g. "dense similarity 0.8234")
LabelDecision(str, Enum): PENDING, LABELED, SKIPPED
GoldLabelSession(frozen): session_id, query_id, query, retrieval_strategy,
    candidates: tuple[CandidateIncident,...], selected_incident_ids: tuple[str,...] = (), status=PENDING
LabelingProvenance(frozen): query_id, original_query (pre-21C-edit), retrieval_strategy,
    selected_incident_ids, labeled_at
LabeledGoldQuery(frozen): gold_query: GoldQuery, provenance: LabelingProvenance
LabelingStats(frozen): total_sessions, labeled, skipped, pending, avg_candidates_presented,
    avg_selected_per_labeled, single_label_pct, multi_label_pct

class GoldLabelRetriever(ABC):
    retrieve_candidates(query, limit=10) -> tuple[CandidateIncident,...]
    strategy_name: str
DenseGoldLabelRetriever   # wraps IncidentSearchService.search(); strategy_name="dense"
HybridGoldLabelRetriever  # wraps HybridRetriever.retrieve(); strategy_name="hybrid"

GoldLabelingWorkflow(retriever, limit=10):
    .add_query(candidate_query) -> GoldLabelSession   # fires retrieval eagerly
    .label(session_id, selected_ids) -> GoldLabelSession
    .skip(session_id) -> GoldLabelSession
    .export_labeled_queries() -> tuple[LabeledGoldQuery,...]   # only status=LABELED
    .stats() -> LabelingStats
```

### Lifecycle

`.add_query(candidate_query)` immediately calls `retriever.retrieve_candidates(candidate_query.query,
limit)`, wraps the results as a new `GoldLabelSession` with `status=PENDING`, and stores it.
`.label(session_id, selected_ids)` replaces that session with a new frozen object carrying
`selected_incident_ids=tuple(selected_ids)` and `status=LABELED` — zero, one, or many ids are all
valid. `.skip(session_id)` sets `status=SKIPPED` without discarding the session (it stays available
for a future pass). `.export_labeled_queries()` filters to `LABELED` sessions only and, for each,
determines a category (`"lexical-overlap"` for exactly one selection, `"multi-concept"` for more
than one, `"no-match-expected"` for zero), builds a `GoldQuery` with every selected id wrapped as an
`ExpectedIncident(relevance=RELEVANCE_MAX)`, and pairs it with a `LabelingProvenance` recording the
pre-edit original query, retrieval strategy, and selections for audit purposes. `.stats()` computes
every figure fresh from current session state.

### Design decisions

- **No automatic incident selection anywhere** — the framework only ever proposes candidates; the
  human always makes the final call.
- **Retrieval fires eagerly on `.add_query()`** — a deliberate simplicity choice; the docstring
  notes this could become async in a future phase without changing the public API shape.
- **All reviewer selections get `RELEVANCE_MAX` (3)** — the safest uniform choice on Phase 16B's
  1–3 graded scale; per-selection grading is left to a future phase.
- **Zero-selection sessions are fully supported** (`"no-match-expected"`) — an explicit,
  first-class outcome, not an edge case to work around.
- **Provenance records the *original*, pre-21C-edit query** — so a future auditor can reproduce
  exactly the candidate set the reviewer saw, independent of any later query editing.
- **Explicit forbidden-import list** — must not import the evaluation harness, regression, `Judge`,
  `Planner`, `Critic`, or `Benchmark`; only `IncidentSearchService`, `HybridRetriever`, and the gold
  schemas.

### Interfaces

Imports `IncidentSearchService`/`IncidentSearchResult` and `HybridRetriever`/`HybridSearchResult`
(for the two concrete retrievers), plus `GoldDataset`/`GoldQuery`/`ExpectedIncident` (Phase 16B) and
`CandidateQuery` (Phase 21C). Public surface: `GoldLabelingWorkflow`, `GoldLabelRetriever` and its
two concrete implementations. Not imported by any API route.

### Testing

`test_gold_labeling.py` covers: retriever-adapter correctness (translating both
`IncidentSearchResult` and `HybridSearchResult` into `CandidateIncident`); session creation and
labeling-state transitions; multi-selection handling; provenance audit-trail preservation; the
zero-selection edge case; statistics formulas (averages and single/multi-label percentages); and
export filtering to `LABELED`-only sessions.

### Risks

Session state is entirely in-process — a fresh `GoldLabelingWorkflow` instance has no memory of a
previous labeling pass, so there is no persistence across process restarts. Uniform
`RELEVANCE_MAX` for every selection means the framework cannot currently distinguish "this is the
one exact match" from "this is also relevant but secondary" without a future per-selection grading
extension.

### Future work

The docstring notes retrieval could become asynchronous in a future phase without changing the
public API; per-selection relevance grading (beyond the uniform `RELEVANCE_MAX`) is an implied, not
stated, next step.

---

## Phase 21E — End-to-End Evaluation Pipeline

### Goal

Wire every existing evaluation component — retrieval evaluation (16D/16E/16F), reasoning evaluation
(20A), judging (20B), and AI quality intelligence/validation (21A/21B) — into one callable sequence.

### Motivation

Per the module docstring: "This module introduces NO new metrics, NO new retrieval algorithms, NO
reasoning logic, and NO prompt tuning — it only calls the existing public APIs in the correct
order." Every prior phase up to 21D is a library call a caller could sequence manually; this phase
is purely the sequencing, with per-stage skippability and resilient error handling.

### Architecture

```python
EvaluationPipelineConfig(frozen): experiment_name="default",
    run_retrieval/run_reasoning/run_judge/run_failure_analysis/run_validation: bool = True,
    persist_results: bool = True, retrieval_k=10, retrieval_expand=False, retrieval_rerank=False,
    n_hypotheses=3

PipelineRepositories: retrieval_repo: BenchmarkRepository|None,        # Phase 16F
    reasoning_repo: ReasoningBenchmarkRepository|None,                  # Phase 20A
    judged_repo: JudgedReasoningBenchmarkRepository|None                # Phase 20B

PipelineInputs: gold_dataset: GoldDataset|None, search_service: object = None,   # typed loosely to
    reasoning_dataset: ReasoningGoldDataset|None, orchestrator: object = None,    # avoid importing
    judge: Judge|None                                                             # concrete classes

ExecutionSummary(frozen): start_time, end_time, duration_seconds, retrieval_queries,
    reasoning_scenarios, judge_evaluations, warnings: tuple[str,...], errors: tuple[str,...]

EvaluationPipelineResult(frozen): retrieval_report/retrieval_regression/retrieval_benchmark,
    reasoning_report/reasoning_regression/reasoning_benchmark, judge_report,
    judge_validation_report, quality_report, execution_summary (always populated)

EvaluationPipeline(config, repositories):
    .run(inputs: PipelineInputs) -> EvaluationPipelineResult
```

### Lifecycle

Ten sequential steps, each independently skippable and individually resilient to failure:

1–4. **Retrieval**: if `run_retrieval` and both `gold_dataset`/`search_service` are supplied, call
`harness.evaluate(gold_dataset, search_service, k=, expand=, rerank=)`; if `retrieval_repo` is
configured, fetch the previous run and `compare()` for a `retrieval_regression`; wrap in a
`BenchmarkRun` via `create_benchmark_run` and save if `persist_results`. Any exception here is caught
and appended to `errors`; the pipeline continues.

5–7. **Reasoning**: same shape — `evaluate_reasoning_dataset(reasoning_dataset, orchestrator,
n_hypotheses=)`, optional `compare_reasoning()` regression against a prior `reasoning_repo` entry,
wrap via `create_reasoning_benchmark_run`, save if configured. Same resilient error handling.

8. **Judge**: if `run_judge` and a `judge` was supplied and a reasoning benchmark exists, call
`judge.evaluate_session(result.problem, result.session)` for every `InvestigationResult` in the
reasoning report — a **per-scenario** try/except, so one judge failure is recorded in `errors` and
skipped without aborting the rest — then wrap the successful evaluations via
`create_judged_benchmark_run` and save if configured.

9. **Failure analysis / quality report**: if `run_failure_analysis`, accumulate whatever reports
actually ran (`[retrieval_report]`/`[reasoning_report]` if present, judge evaluations if present),
derive a `regression_verdict` from whichever regression (retrieval or reasoning) actually ran, and
call `build_quality_report(...)`.

10. **Judge validation**: if `run_validation` and a `judged_repo` was supplied, call
`build_validation_report_from_benchmarks(judged_repo, experiment_name)`.

Every stage that is disabled, or whose required input is missing, adds a `warnings` entry instead
of an error and is simply skipped. `execution_summary` (timing, per-stage counts, warnings, errors)
is always populated regardless of how many stages actually ran.

### Design decisions

- **Loosely-typed `search_service`/`orchestrator` inputs (`object`)** — avoids importing
  `IncidentSearchService` or any agent class into this orchestration-only module; a type mismatch
  surfaces as a runtime `AttributeError` from the called harness, not a static import dependency
  here.
- **Every stage independently skippable, with a warning rather than a silent no-op** — a caller can
  tell from `execution_summary.warnings` exactly why a section of the result is empty.
- **Per-scenario judge error isolation** — one bad judge call must not discard every other
  scenario's judge evaluation in the same run.
- **`persist_results=False` still builds full report objects in memory** — so failure analysis and
  validation can run against them in the same process even when a caller doesn't want anything
  written to a repository yet.
- **Regression verdict pass-through, not re-derivation** — whichever regression comparison actually
  ran (retrieval's or reasoning's) supplies the verdict string fed into the quality report.

### Interfaces

Imports `evaluate` (Phase 16D), `compare` (Phase 16E), benchmark-run construction (Phase 16F),
`evaluate_reasoning_dataset` (Phase 20A), the `Judge` interface and benchmark helpers (Phase 20B),
and `build_quality_report`/`build_validation_report_from_benchmarks` (Phase 21A/21B). Public surface:
`EvaluationPipeline`, `EvaluationPipelineConfig`, `PipelineRepositories`, `PipelineInputs`,
`EvaluationPipelineResult`, `ExecutionSummary`. Not imported by any route directly — the
`/evaluation/full` endpoint (doc 22) constructs and runs it, and `scripts/run_full_evaluation.py`
is a CLI entry point over the same class.

### Testing

`test_evaluation_pipeline.py` covers: the full pipeline with every stage enabled; each stage
individually disabled via its config flag; missing-input handling (dataset/service/orchestrator all
`None`); error recording and continued execution after a stage failure; regression computation and
verdict carry-through into the quality report; quality-report assembly with multiple input
histories; and `execution_summary` timing/counting correctness.

### Risks

None named explicitly beyond what the design decisions already surface: a caller relying on
`persist_results=False` to mean "nothing happens" must still account for the full in-memory report
construction cost.

### Future work

None named explicitly in the module docstring.

---

## Phase 21F — Persistent Evaluation &amp; Experiment Tracking

### Goal

A production-grade, on-disk persistence layer around `EvaluationPipelineResult` (Phase 21E): every
report the pipeline already computed is written to disk in full, with convenience failure-filter
files and cross-run statistics — nothing is recomputed here.

### Motivation

Phase 16F's `BenchmarkRepository` (doc 20A) already persists retrieval-only runs in a flat,
one-file-per-run layout with a typed `BenchmarkRun` dataclass. Phase 21F needs to persist the
*whole pipeline's* output — retrieval, reasoning, judge, quality, and validation reports together —
in a form a human can browse without importing every evaluation module's types. Rather than extend
16F (forbidden — would modify already-shipped code), this phase is an additional, orthogonal
persistence layer store reports as plain JSON dicts.

### Architecture

```
.evaluation_runs/
  latest/                     # always the most recent run — a directory COPY, not a symlink
    metadata.json, summary.json, retrieval_report.json, reasoning_report.json, judge_report.json,
    quality_report.json, validation_report.json, regression_report.json,
    failed_queries.json, failed_reasoning.json, judge_disagreements.json
  history/
    20260701_213015_nightly/    # YYYYMMDD_HHMMSS_<sanitized experiment_name>
      (same file set as latest/)
```

```python
make_run_id(experiment_name, now=None) -> str   # "20260701_213015_nightly"; spaces/slashes sanitized to underscores

_to_jsonable(value) -> Any   # recursive dataclass-tree -> JSON-safe dict/list conversion (enums -> .value)

RunMetadata(frozen): run_id, timestamp, git_commit: str|None (best-effort `git rev-parse --short HEAD`),
    experiment_name, retrieval_dataset_version, reasoning_dataset_version, judge: str|None,
    duration: float, configuration: dict[str, Any]

ExperimentRun(frozen): metadata, summary: dict, retrieval_report/reasoning_report/judge_report/
    quality_report/validation_report/regression_report: dict | None (JSON-parsed, not re-typed),
    failed_queries/failed_reasoning/judge_disagreements: tuple[dict,...]   # convenience filters

ExperimentStats(frozen): total_runs, best_mrr, best_ndcg, best_reasoning_accuracy: float|None,
    latest_run: str|None, trend: tuple[str,...]   # run_ids oldest -> newest

class ExperimentRepository:
    def __init__(self, base_dir=Path(".evaluation_runs")): ...
    def save(result: EvaluationPipelineResult, *, experiment_name="default", git_commit=None,
             retrieval_dataset_version=None, reasoning_dataset_version=None, judge_name=None,
             run_id=None) -> str: ...
    def load(run_id) -> ExperimentRun | None
    def latest() -> ExperimentRun | None
    def list_runs() -> tuple[str, ...]      # sorted by metadata.json timestamp, not filesystem mtime
    def delete(run_id) -> bool              # repopulates latest/ from the newest remaining run if needed
    def stats() -> ExperimentStats
```

Convenience filters, computed once at save time and never recomputed at read time:
`_failed_queries` (per-query entries where `skipped=True` or `recall_at_k < 1.0`);
`_failed_reasoning` (`InvestigationResult` entries where `converged=False` or `decision_correct=False`
or `planner_correct=False`); `_judge_disagreements` (`JudgeEvaluation` entries with `score.value <
5.0`, the midpoint of the 1–10 scale).

### Lifecycle

`repo.save(result, experiment_name=...)` generates a run id (or accepts one), builds `RunMetadata`
(attempting a best-effort `git rev-parse --short HEAD` if `git_commit` wasn't supplied — pass `""`
to suppress), writes every report as its own JSON file plus the three convenience filters into a
fresh `history/<run_id>/` directory, then atomically replaces `latest/` with a copy of the same
directory (`shutil.rmtree` + `copytree`, chosen over a symlink for cross-platform compatibility) and
returns `run_id`. `repo.load(run_id)`/`repo.latest()` both delegate to an internal loader that reads
every JSON file back into an `ExperimentRun` of plain dicts (never re-typed to the original
dataclasses — this avoids importing dozens of evaluation modules just to load a run for
inspection). `repo.list_runs()` sorts by the timestamp recorded inside each run's `metadata.json`,
not directory mtime, so ordering survives a copy operation. `repo.stats()` scans every persisted
run's `retrieval_report`/`reasoning_report` dicts for `mean_reciprocal_rank`, NDCG, and
`decision_accuracy` to compute running bests, plus a full oldest-to-newest `trend` of run ids.

### Design decisions

- **Pure filtering, no re-derivation** — the three convenience-filter predicates are fixed and
  documented; they read already-computed report fields, they never recompute anything.
- **JSON, not pickle** — human-readable, safe to round-trip, and reports are stored as plain dicts
  rather than re-typed dataclasses specifically so this module doesn't need to import every
  evaluation module's types just to persist/load a run.
- **Directory copy, not symlink, for `latest/`** — maximizes cross-platform compatibility
  (symlinks are inconsistently supported/permissioned on Windows); the copy is atomic from the
  caller's perspective (`rmtree` then `copytree`).
- **Chronological ordering from `metadata.json`, not filesystem timestamps** — filesystem mtimes
  can be altered by the copy operation that populates `latest/`; the run's own recorded timestamp
  is the only reliable ordering key.
- **Git commit lookup is best-effort and silently absent on failure** — the module never fails a
  save because `git` isn't available or the working tree isn't a repo.

### Interfaces

Imports only `EvaluationPipelineResult` (Phase 21E) for typing at the `save()` boundary; otherwise
pure JSON/filesystem handling with no other evaluation-module imports. Public surface:
`ExperimentRepository`, `ExperimentRun`, `ExperimentStats`, `make_run_id`. Not imported by any route
directly — consumed by `scripts/inspect_evaluation_run.py` (a CLI over this repository) and,
separately, `app/api/routes/evaluation.py`'s own `ExperimentRepository` dependency (doc 22) —
though note doc 22 constructs its **own** `ExperimentRepository` pointed at `.evaluation_runs/`
directly, not via this module's CLI script.

### Testing

`test_experiment_tracking.py` covers: `make_run_id` format and sanitization; save/load round-trip
correctness; atomic overwrite behavior of `latest/`; chronological ordering of `list_runs()`; the
three convenience filters (failed queries by skip/recall, failed reasoning by
convergence/decision/planner correctness, judge disagreements by score threshold); `stats()`
computing correct bests across a multi-run history; and `delete()` correctly repopulating `latest/`
when the deleted run was the most recent one.

### Risks

None named explicitly in the module docstring; implicit ones: storing every report as an untyped
dict means a future schema change to any upstream report type requires no changes here (a
deliberate benefit) but also means this layer cannot validate that a loaded report still matches
its originating dataclass shape — malformed or stale JSON would only surface as a `KeyError` in
whatever code reads the loaded dict later.

### Future work

None named explicitly in the module docstring.

---

## CLI entry points

**`scripts/run_full_evaluation.py`** (Phase 21E) — a CLI wrapper over `EvaluationPipeline`. Flags:
`--retrieval-dataset PATH`, `--reasoning-dataset PATH`, `--judge {rule|none}`, `--experiment NAME`,
`--k K`, `--no-persist`, `--skip-retrieval`/`--skip-reasoning`/`--skip-validation`. Attempts to build
`IncidentSearchService`/`MultiAgentInvestigationOrchestrator` from the environment and gracefully
skips stages if unavailable; uses in-memory repositories (results are **not** written to
`.evaluation_runs/` from this script — a caller wanting Phase 21F persistence uses
`ExperimentRepository` directly). Prints a summary table of retrieval/reasoning/judge/quality
results plus warnings/errors.

**`scripts/inspect_evaluation_run.py`** (Phase 21F) — a CLI wrapper over `ExperimentRepository`.
Commands: `latest`, `<run_id>`, `--list`, `--stats`, `--failed-queries`, `--dir DIR` (custom storage
location, default `.evaluation_runs`). Pretty-prints metadata, every report section, failed items,
trustworthiness verdict, and warnings/errors for a persisted run.

---

## Relationship between Phase 16F and Phase 21F ("experiment tracking" appears twice)

These are complementary layers, not duplicates, despite both being named "experiment tracking" in
their docstrings:

| | Phase 16F `benchmark.py` | Phase 21F `experiment_tracking.py` |
|---|---|---|
| Scope | Retrieval only (`EvaluationReport`) | Whole pipeline (retrieval + reasoning + judge + quality + validation) |
| Data model | Typed dataclass (`BenchmarkRun` holding a real `EvaluationReport`) | Loaded plain dicts (`ExperimentRun`) |
| Storage layout | Flat directory, one `{run_id}.json` per run | `history/<run_id>/` subdirectories + a `latest/` copy |
| Convenience filters | None | Failed queries / failed reasoning / judge disagreements |
| Statistics | None | `ExperimentStats` (best MRR/NDCG/accuracy, trend) |

Phase 21E's pipeline uses Phase 16F's `BenchmarkRepository` (and 20A's/20B's equivalents) internally
for regression comparisons and per-domain storage; Phase 21F's `ExperimentRepository` is an
*additional*, optional layer a caller adds on top for human-readable inspection and cross-run
statistics — the two are not alternatives to choose between, they compose.
</content>
