# 20 — Reasoning Evaluation &amp; LLM-as-Judge (Phases 20A–20B)

Doc 15 (evaluation framework) covers *retrieval* evaluation only — gold datasets, a harness, a
metric engine, regression, and benchmark history, all scoped to "did the right incidents come
back." This document covers the parallel evaluation system built for the *reasoning* layer (docs
19's four agents): **20A** is a heuristic, gold-answer-based harness that judges an
`InvestigationSession` against expected strategy/hypotheses/verdict/stopping-reason strings, with
its own benchmark-history and regression tooling; **20B** is an orthogonal, semantic
`Judge` abstraction (`RuleJudge`, deterministic, and `LLMJudge`, provider-agnostic) that scores
open-ended reasoning quality on a 1–10 rubric with no gold answer required. Both read the same
immutable `InvestigationSession`/`InvestigationDecision` types 19A–19D produce; neither modifies
anything in doc 19, and neither is wired into any API route as a standalone concern — both are
consumed indirectly through the Phase 21 evaluation pipeline and API (docs 21, 22).

---

## Phase 20A — Reasoning Evaluation (Heuristic, Gold-Answer-Based)

This phase bundles four modules — dataset schema, harness, benchmark-history integration, and
regression — each mirroring a specific Phase 16 retrieval-evaluation precursor.

### Goal

Define a gold-scenario schema for investigations (mirroring Phase 16B's retrieval gold format),
run an orchestrator against those scenarios and judge each result by plain string/heuristic
comparison (mirroring Phase 16D's harness), store and version those runs (mirroring Phase 16F), and
compare two runs for regression (mirroring Phase 16E) — all without touching any Phase 16 code or
importing its private helpers.

### Motivation

Phase 16B/16D/16E/16F built a mature evaluation-platform pattern for retrieval; reasoning needs the
identical *shape* of pattern (gold dataset → harness → benchmark storage → regression) but scoped
to entirely different fields (strategy, hypotheses, verdict, stopping reason instead of expected
incidents). Rather than generalize Phase 16's types to cover both domains — which would require
touching already-shipped, already-tested code — Phase 20A duplicates the small, already-simple
interfaces locally. The module docstrings are explicit that this is a deliberate trade-off: e.g.
`judge_benchmark.py`'s docstring notes "Phase 20A duplicates Phase 16F's [pattern] because no
shared base class exists to extend."

### Architecture

**Dataset schema** (`app/evaluation/reasoning_dataset.py`):

```python
VALID_STRATEGIES = frozenset({"infrastructure_failure", "configuration", "authentication",
                               "network", "application_failure", "unknown"})
VALID_VERDICTS = frozenset({"approved", "need_more_evidence",
                             "alternative_hypothesis_plausible", "inconclusive"})
VALID_STOPPING_REASONS = frozenset({"critic_approved", "max_iterations",
                                     "no_progress", "no_new_hypotheses"})

InvestigationScenario(frozen): id, problem, expected_strategy: str, expected_root_causes: tuple[str,...],
    expected_verdict: str, expected_stopping_reason: str, notes: str = ""
    .issues() -> list[str]   # per-scenario validation, including coherence (an "approved" verdict
                              # requires non-empty expected_root_causes)

ReasoningGoldDataset(frozen): version, description, created_at, scenarios: tuple[InvestigationScenario,...],
    author: str | None = None
    .issues() -> list[str]   # dataset metadata + unique scenario ids + all scenarios valid
    .is_valid() -> bool
```

Every "expected" field is a **plain string**, never `PlanningStrategy`/`CritiqueVerdict`/
`StoppingReason` (docs 19B/19C/19D's runtime enums) — so a dataset file is readable and loadable
without importing any reasoning implementation class, mirroring Phase 16B's identical principle for
retrieval gold data. An empty `expected_root_causes` tuple is intentionally valid — it represents a
negative-control scenario (e.g. an unrelated problem) where the correct behavior is to remain
uncertain, not to name a cause.

**Harness** (`app/evaluation/reasoning_harness.py`):

```python
class _Orchestrator(Protocol):     # duck-typed, no import of the concrete orchestrator required
    def investigate(self, problem: str, *, n_hypotheses: int = ..., routing_observation=...) -> InvestigationSession

ReasoningMetrics(frozen): num_scenarios, planner_accuracy, hypothesis_recall, hypothesis_precision,
    decision_accuracy, critic_accuracy, stopping_accuracy: float | None each,
    convergence_rate: float | None, mean_iteration_count: float | None

InvestigationResult(frozen): scenario_id, problem, expected_*, actual_strategy, actual_root_causes,
    actual_verdict, actual_stopping_reason, total_iterations,
    planner_correct, hypothesis_recall_hit: bool, hypothesis_precision: float | None,
    decision_correct, critic_correct, stopping_correct, converged: bool,
    session: InvestigationSession, explanation: tuple[str, ...]

InvestigationEvaluationReport(frozen): dataset_version, dataset_description, n_hypotheses,
    results: tuple[InvestigationResult,...], metrics: ReasoningMetrics,
    started_at, finished_at, duration_seconds

evaluate_scenario(scenario, orchestrator, *, n_hypotheses) -> InvestigationResult
evaluate_reasoning_dataset(dataset, orchestrator, *, n_hypotheses) -> InvestigationEvaluationReport
_root_cause_matches(root_cause, expected) -> bool   # case-insensitive, bidirectional substring match
```

**Benchmark integration** (`app/evaluation/reasoning_benchmark.py`):

```python
ReasoningBenchmarkRun(frozen): run_id, timestamp, experiment_name,
    report: InvestigationEvaluationReport, regression: ReasoningRegressionReport | None,
    git_commit_sha: str | None, notes: str | None

CombinedBenchmarkRun(frozen): run_id, timestamp, experiment_name,
    retrieval: BenchmarkRun | None,       # Phase 16F, unmodified
    reasoning: ReasoningBenchmarkRun | None    # at least one of the two must be non-None

ReasoningBenchmarkRepository (ABC): save, get, list_runs(experiment_name=None), latest(), delete
InMemoryReasoningBenchmarkRepository / FileReasoningBenchmarkRepository (concrete)

create_reasoning_benchmark_run(...) -> ReasoningBenchmarkRun
combine_benchmark_runs(*, retrieval=None, reasoning=None, ...) -> CombinedBenchmarkRun   # raises if both None
compare_reasoning_runs(baseline, candidate) -> ReasoningRegressionReport   # delegates to reasoning_regression.compare_reasoning
reasoning_regression_history(repository, ...) -> tuple[ReasoningRegressionReport, ...]
```

**Regression** (`app/evaluation/reasoning_regression.py`):

```python
EPSILON = 1e-9   # identical threshold to Phase 16E's own EPSILON

DeltaClassification(str, Enum): IMPROVED, REGRESSED, UNCHANGED, UNDEFINED
Verdict(str, Enum): IMPROVED, REGRESSED, UNCHANGED, MIXED, INCOMPATIBLE

CompatibilityCheck(frozen): compatible: bool, reasons: tuple[str,...]
MetricDelta(frozen): baseline, candidate: float|None, delta: float|None, classification
CategoryDelta(frozen): category: str, metrics: dict[str, MetricDelta], verdict: Verdict
ReasoningRegressionReport(frozen): baseline, candidate: InvestigationEvaluationReport,
    compatibility, verdict, planner/hypothesis/decision/critic/iteration: CategoryDelta|None, summary: str

compare_reasoning(baseline, candidate) -> ReasoningRegressionReport
```

### Lifecycle

**Harness**: for each scenario, call `orchestrator.investigate(scenario.problem, n_hypotheses=...)`
— the *only* reasoning that happens; this phase makes zero LLM/planning/hypothesis calls itself.
Extract `actual_strategy` from `session.iterations[0].plan.strategy.value` (first iteration only —
planning is a deterministic pure function of problem/retrieved_incidents per 19B, so every
iteration replans identically). Collect all hypotheses across every iteration into
`actual_root_causes`. Extract `actual_verdict` from `session.final_report.critique.verdict.value`
and `actual_stopping_reason`/`total_iterations` from the session directly. Compute, via pure string
comparison — no LLM: planner accuracy (`actual_strategy == expected_strategy`); hypothesis recall
(vacuously true if `expected_root_causes` is empty, else true iff at least one generated hypothesis
substring-matches at least one expected cause); hypothesis precision (fraction of *all* generated
hypotheses matching an expected cause, `None` if zero hypotheses or empty expected list); decision
accuracy (for an empty expected list, correct iff `selected_hypothesis is None`; otherwise correct
iff a hypothesis was accepted **and** its root cause matches an expected one); critic accuracy and
stopping accuracy (direct string equality); convergence (`stopping_reason !=
StoppingReason.MAX_ITERATIONS`). `evaluate_reasoning_dataset` runs every scenario and aggregates via
`_mean()` over defined (non-`None`) values only — same "mean over defined values" convention as
Phase 16D's `AggregateMetrics`.

**Regression**: `compare_reasoning` first checks compatibility (`dataset_version` match, identical
scenario-id coverage); if incompatible, returns immediately with `verdict=INCOMPATIBLE` and no
category deltas. Otherwise computes five category deltas: **planner** (single metric); **hypothesis**
(recall + precision folded — both must not regress for the category to read IMPROVED/UNCHANGED);
**decision** and **critic** (single metric each); **iteration** (three diagnostic metrics —
`mean_iteration_count` with `higher_is_better=False`, `convergence_rate`, `stopping_accuracy` — never
counted in the overall verdict). The overall verdict rolls up only planner/hypothesis/decision/critic.

### Design decisions

- **Duplicated, not imported, from Phase 16E** — the regression logic here is "byte-for-byte
  identical" to Phase 16E's `_classify`/`_metric_delta`/`_verdict_from_classifications`, but
  reimplemented locally because importing another module's underscore-prefixed internals "is not a
  stable contract."
- **Duplicated, not subclassed, from Phase 16F** — `BenchmarkRepository` is hard-typed to a
  retrieval `BenchmarkRun`; a `ReasoningBenchmarkRepository` subclass can't satisfy that same ABC
  for a different run type without either modifying the ABC (forbidden) or violating Liskov
  substitution, so duplication (all three implementations are small) is the only clean option.
  `CombinedBenchmarkRun` composes rather than adds fields to either side.
- **Substring matching, not exact equality, for root-cause comparison** — acknowledges generated
  causes may be full sentences containing (or contained in) the expected keyword phrase; a
  deliberate heuristic trade-off (see Risks).
- **First iteration's plan is representative** for planner accuracy, since planning is deterministic.
- **Negative-control scenarios are first-class**, not an error case — empty `expected_root_causes`
  paired with `inconclusive`/`need_more_evidence` is a valid, intentional scenario shape.
- **Session embedded by reference in `InvestigationResult`** — denormalized (breaks normalization)
  but lets a reader drill into full iteration history without re-running, the same choice Phase
  16E/16F made for their own report types.

### Interfaces

`reasoning_dataset.py` has zero phase dependencies (schema-only). `reasoning_harness.py` imports
from `reasoning_dataset` and duck-types the orchestrator via a `Protocol` (never imports the
concrete `MultiAgentInvestigationOrchestrator` class). `reasoning_benchmark.py` imports
`app.evaluation.benchmark` (`BenchmarkRun`, for `CombinedBenchmarkRun`), `reasoning_harness`, and
`reasoning_regression`. `reasoning_regression.py` imports only `reasoning_harness`'s types. All four
consumed by the `/evaluation/reasoning` and `/evaluation/full` endpoints (doc 22) via
`evaluate_reasoning_dataset`, `create_reasoning_benchmark_run`, and `compare_reasoning_runs`.

### Testing

`test_reasoning_dataset.py`: valid scenario has no issues; a negative-control scenario (empty
root_causes) is valid; empty id/problem invalid; unknown strategy/verdict/stopping-reason invalid
(closed-set checks); an approved verdict with no expected root causes is invalid (coherence check);
dataset-level duplicate-id rejection and non-empty-scenarios requirement. `test_reasoning_harness.py`:
a fully-correct investigation produces no explanation text; planner mistakes, missing hypotheses,
incorrect acceptance/rejection, and negative-control correctness/incorrectness are each detected and
explained; critic-verdict and stopping-reason mismatches are detected; a "missing convergence"
explanation is added when max-iterations was hit but not expected; first-iteration-plan usage is
confirmed with a hypothesis-precision test (0.5, one matching one not); dataset-wide aggregation
across a perfect and a poor scenario yields 0.5 for planner/critic accuracy and convergence rate.
`test_reasoning_benchmark.py`: run-id/timestamp auto-generation; in-memory repo save/get/list/latest/
delete and duplicate-id rejection; file-repo JSON round-trip; `compare_reasoning_runs` delegation;
regression-history generation; `combine_benchmark_runs` requiring at least one side and working with
reasoning-only. `test_reasoning_regression.py`: identical reports are UNCHANGED; single-metric
improvement/regression detected per category; hypothesis category folding (recall improves,
precision unchanged → category IMPROVED); mixed planner+decision changes yield overall MIXED;
iteration category regressing does not move the overall verdict; fewer iterations classified as
improved (confirms `higher_is_better=False`); incompatible dataset versions and incompatible
scenario coverage both rejected; both-sides-`None` metrics counted as UNCHANGED, not a regression.

### Risks

Substring matching can over-match (unrelated causes sharing a common word) or under-match
(synonymous phrasing with no literal overlap). The first-iteration-plan assumption breaks if
planning is ever made non-deterministic. Scoring is binary correct/incorrect with no confidence or
partial-credit signal. Negative-control interpretation depends on the scenario's exact
`expected_root_causes` shape — an investigation that rejects everything could be correct (expected)
or wrong (a specific hypothesis was expected to be accepted), and only the dataset authoring
distinguishes the two.

### Future work

None named explicitly in the docstrings beyond the fact that Phase 20B (semantic judgment) is
introduced *alongside*, not in place of, this heuristic layer.

---

## Phase 20B — Judge Framework (Semantic, LLM-Backed or Rule-Based)

This phase bundles four modules: the `Judge` interface and report model, `RuleJudge` (deterministic),
`LLMJudge` (semantic), and judge-benchmark integration.

### Goal

Provide a pluggable evaluation abstraction with five stage-specific methods
(`evaluate_plan`/`evaluate_hypotheses`/`evaluate_decision`/`evaluate_critique`/`evaluate_session`),
each producing a 1–10 score with a mandatory written explanation and typed strengths/weaknesses/
recommendations — assessing subjective reasoning *quality* on problems with no single correct
answer, which Phase 20A's gold-answer comparison structurally cannot do.

### Motivation

Phase 20A judges correctness through string equality — useful for regression-testing known
scenarios, but blind to "was this actually a reasonable plan" or "were these hypotheses plausible"
on open-ended problems. Phase 20B's `Judge` interface reads the same immutable
`InvestigationSession` data Phase 20A reads, but produces explainable, subjective scores instead of
pass/fail. Two implementations ship — `RuleJudge` (zero LLM calls, fast, reproducible, required so
"unit tests must never require OpenAI") and `LLMJudge` (semantic, via an abstract client protocol,
provider-agnostic) — with the interface explicitly left open for a future `HumanJudge`.

### Architecture

**Interface &amp; report model** (`app/evaluation/judge.py`):

```python
SCORE_MIN, SCORE_MAX = 1.0, 10.0
RUBRIC_BANDS = ((1.0, 2.0, "Poor"), (2.0, 4.0, "Weak"), (4.0, 6.0, "Acceptable"),
                (6.0, 8.0, "Good"), (8.0, 10.0, "Excellent"))   # fixed, documented mapping

STAGE_PLAN, STAGE_HYPOTHESES, STAGE_DECISION, STAGE_CRITIQUE, STAGE_SESSION   # stage-name constants
CRITERIA: dict[stage, tuple[str, ...]]   # single source of truth for per-stage rubric criteria, e.g.
    STAGE_PLAN: ("chosen_strategy", "investigation_objective", "prioritization", "appropriateness")
    STAGE_HYPOTHESES: ("correctness", "diversity", "completeness", "plausibility")

classify_score(value) -> str            # clamps then maps to a RUBRIC_BANDS label
make_judge_score(value) -> JudgeScore   # the only legal constructor — value and band can never disagree

JudgeScore(frozen): value: float, band: str
JudgeFinding(frozen): criterion: str, detail: str
JudgeEvaluation(frozen): stage, score: JudgeScore, explanation: str (required, non-empty — enforced
    in __post_init__), strengths/weaknesses/recommendations: tuple[JudgeFinding,...] = ()

class Judge(ABC):    # five separate typed methods, not one generic evaluate(stage, **kwargs)
    evaluate_plan(problem, plan) -> JudgeEvaluation
    evaluate_hypotheses(problem, plan, hypotheses) -> JudgeEvaluation
    evaluate_decision(problem, hypotheses, decision, evaluations) -> JudgeEvaluation
    evaluate_critique(problem, decision, critique) -> JudgeEvaluation
    evaluate_session(problem, session) -> JudgeEvaluation
```

**RuleJudge** (`app/evaluation/rule_judge.py`) — deterministic heuristics per stage:

- **Plan**: baseline 6.0 ("Acceptable"); +2.0 if strategy is not `UNKNOWN`; +1.0 if `objective` is
  non-empty; +1.0 if `priority_list` has more than one entry. Max 10.0.
- **Hypotheses**: zero hypotheses → immediate 1.0 ("Poor"); else baseline 5.0, +2.0 if more than one
  distinct root cause (diversity), +2.0 if every hypothesis has non-empty `validation_keywords`
  (completeness). Max 9.0.
- **Decision**: uncertain (no accepted hypothesis) → 4.0 ("Weak"); accepted → `SCORE_MIN +
  accepted_score.composite_score * (SCORE_MAX - SCORE_MIN)` — linearly rescales 19A's already-computed
  composite score rather than inventing a new number.
- **Critique**: fixed mapping from `CritiqueVerdict` — APPROVED→9.0,
  ALTERNATIVE_HYPOTHESIS_PLAUSIBLE→6.0, NEED_MORE_EVIDENCE→5.0, INCONCLUSIVE→4.0.
- **Session**: `mean(plan, hypotheses, decision, critique scores) - 0.5 * max(0, total_iterations - 1)`,
  floored at `SCORE_MIN` — one free iteration, each additional costs half a point.

**LLMJudge** (`app/evaluation/llm_judge.py`) — provider-agnostic semantic scoring:

```python
class JudgeLLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...   # the ONLY coupling point; no openai/LLMService import

class JudgeResponseError(ValueError): ...   # raised on any malformed response, no retry

LLMJudge(client: JudgeLLMClient):
    # each evaluate_* method builds ONE plain-text prompt (problem + artifact rendered from existing
    # dataclass fields + CRITERIA[stage] + RUBRIC_BANDS text), calls client.complete() exactly once,
    # and parses a required JSON object: {"score": 1-10, "explanation": str (required, non-empty),
    #  "strengths"/"weaknesses"/"recommendations": [{"criterion": str, "detail": str}, ...]}
```

**Judge-benchmark integration** (`app/evaluation/judge_benchmark.py`):

```python
JudgeAggregateMetrics(frozen): num_evaluations, mean_plan_score, mean_hypotheses_score,
    mean_decision_score, mean_critique_score, mean_session_score: float | None each

JudgedReasoningBenchmarkRun(frozen): run_id, timestamp, experiment_name,
    reasoning_run: ReasoningBenchmarkRun,       # Phase 20A, unmodified, by reference
    judge_evaluations: tuple[JudgeEvaluation,...],
    judge_aggregate: JudgeAggregateMetrics | None

aggregate_judge_evaluations(evaluations) -> JudgeAggregateMetrics    # groups by stage, means each
create_judged_benchmark_run(...) -> JudgedReasoningBenchmarkRun      # auto-computes aggregate

JudgedReasoningBenchmarkRepository (ABC) / InMemory... / File...     # mirrors Phase 20A's repository shape
compare_judge_aggregates(baseline, candidate) -> dict[str, JudgeMetricDelta]   # per-stage deltas,
    # deliberately NO overall verdict — preserves stage-level granularity
```

### Lifecycle

A caller (the reasoning harness, or an evaluation script) depends only on the `Judge` interface,
never a concrete class. Each `evaluate_*` call receives already-computed 19A–19D artifacts and
returns one `JudgeEvaluation`. For `RuleJudge`, scoring is pure arithmetic over presence/absence
and counts — deterministic, zero I/O. For `LLMJudge`, one prompt is built per stage (naming the
problem, the artifact rendered as plain text, that stage's `CRITERIA`, and the `RUBRIC_BANDS`),
`client.complete(prompt)` is called exactly once, and the JSON response is parsed strictly —
malformed JSON, a missing `score`/`explanation` key, a non-numeric score, or a finding missing
`criterion`/`detail` all raise `JudgeResponseError` immediately, with no retry (an explicit stop
condition: "do not implement automatic self-improvement"). The parsed score is clamped via
`make_judge_score` before being returned. For benchmark integration: a `ReasoningBenchmarkRun`
(Phase 20A) is wrapped, per-scenario judge evaluations are attached, `judge_aggregate` is
auto-computed on construction, and `compare_judge_aggregates` reports one `JudgeMetricDelta` per
stage independently — with no rollup to a single headline verdict.

### Design decisions

- **Five separate typed methods, not one generic `evaluate(stage, **kwargs)`** — forces
  stage-specific logic and makes it a type error to apply the wrong criteria to the wrong stage;
  matches the `PlannerAgent`/`CriticAgent`/`RoutingPolicy` pattern already established in the
  codebase.
- **No gold answer required, anywhere in the interface** — `evaluate_hypotheses` is judged on
  correctness/diversity/completeness/plausibility, not "correctness relative to expected root
  causes"; this is what lets Judge assess open-ended problems Phase 20A's harness cannot score.
  `RuleJudge`'s heuristics reflect this limitation honestly: it can verify a hypothesis list is
  diverse and has keywords, but has no way to check plausibility (see Risks).
  `LLMJudge` is the intended answer to that specific gap.
- **`JudgeEvaluation.explanation` is mandatory** — `__post_init__` raises on an empty or
  whitespace-only string; every score must say why.
- **Findings are typed (`criterion` + `detail`), never bare strings** — matches the project's
  general "avoid free-form dictionaries" convention.
- **`JudgeScore` can only be constructed via `make_judge_score`** — guarantees `value` and `band`
  can never disagree.
- **`RuleJudge`'s decision score reuses Phase 19A's `composite_score`** rather than inventing new
  math, trusting an already-justified number (doc 19's Phase 19A design decisions).
- **`LLMJudge` depends only on a two-method `Protocol`** (`JudgeLLMClient.complete`), never on
  `openai`, `LLMService`, or any provider SDK — this phase builds the framework; wiring an actual
  provider adapter is explicitly deferred.
- **Judge-benchmark composition, not modification** — `JudgedReasoningBenchmarkRun` wraps a Phase
  20A `ReasoningBenchmarkRun` by reference rather than adding fields to it, the same
  composition-over-inheritance choice 20A itself made relative to Phase 16F.
- **No overall judge verdict in regression** — unlike Phase 20A's reasoning regression (one rolled-up
  `Verdict`), judge-score regression reports five independent per-stage deltas; averaging five
  semantically different rubrics into one number was judged to lose too much information.

### Interfaces

`judge.py` imports only type hints from `app.services.*` (no behavioral coupling). `rule_judge.py`
and `llm_judge.py` both import `judge.py`'s ABC, constants, and constructors, and read-only types
from `app.services.*`; neither modifies anything in doc 19. `judge_benchmark.py` imports `judge.py`
(`JudgeEvaluation`) and `reasoning_benchmark.py` (`ReasoningBenchmarkRun`). None of the four modules
is imported directly by any API route; all are consumed indirectly via the `/evaluation/reasoning`
and `/evaluation/full` endpoints (doc 22), which construct a `RuleJudge()` when `judge="rule"` is
requested — `LLMJudge` has no concrete client wired anywhere yet.

### Testing

`test_judge.py`: rubric-band mapping for every integer 1–10 and out-of-range clamping (both
directions); `JudgeEvaluation` rejecting empty and whitespace-only explanations; immutability;
default-empty finding tuples; `CRITERIA` covering every stage with at least one criterion each;
`Judge` being uninstantiable directly and rejecting a subclass that implements only one of the five
methods. `test_rule_judge.py`: plan scoring at max (known strategy + multiple priorities), reduced
score with a recommendation for `UNKNOWN` strategy, reduced score for a single priority; hypotheses
scoring at minimum for zero hypotheses, higher for diverse well-keyed hypotheses, reduced for
missing keywords; decision scoring 4.0 when uncertain, exact rescaling of a 0.8 composite score to
8.2, a flagged weakness when no supporting evidence exists; critique scores following verdict
severity in order and flagging a missing explanation; session scoring with no efficiency penalty at
one iteration and a documented penalty at three; and determinism across repeated calls with
identical input. `test_llm_judge.py`: exactly one `complete()` call per `evaluate_*` invocation with
the correct stage; hypothesis root causes, accepted-hypothesis text, critique verdict, and full
iteration timeline all appearing in their respective prompts; graceful "(no hypotheses were
generated)"/"(no hypothesis was accepted)" rendering for empty/uncertain cases; and strict-parsing
failures for malformed JSON, a missing score field, a non-numeric score, a non-list findings field,
and a finding missing `criterion`/`detail` — each raising `JudgeResponseError` (itself a `ValueError`
subclass) with a specific message; plus confirmation an out-of-range score (99.0) is clamped to 10.0.
`test_judge_benchmark.py`: aggregation grouping by stage with correct means and `None` for absent
stages, and correct handling of empty input; automatic aggregate computation on construction, and
`None` aggregate when no evaluations are supplied; heuristic `ReasoningMetrics` on the wrapped run
remaining untouched; in-memory repo operations and duplicate-id rejection; file-repo JSON round-trip
both with and without judge evaluations; and `compare_judge_aggregates` detecting improvement,
regression, and `UNDEFINED` when one side is missing data.

### Risks

`RuleJudge` cannot assess semantic correctness or plausibility at all — a plan scores 10.0 purely
because a strategy was identified and priorities are listed, regardless of whether the plan
actually fits the problem; a hypothesis list scores well purely on diversity/keyword-presence,
so implausible hypotheses ("aliens did it") score identically to plausible ones as long as both
have keywords and distinct root causes. `RuleJudge`'s decision score inherits any miscalibration in
19A's `composite_score`. `RuleJudge` never reads the problem text at all, so it cannot detect
misalignment between problem and plan/hypotheses. The session efficiency penalty (0.5/iteration) is
a reasoned default, not tuned. For `LLMJudge`: identical sessions can score differently across calls
due to LLM non-determinism; artifact text is embedded directly into prompts with no injection
mitigation; malformed responses raise immediately with no retry/repair, so callers must add
resilience if needed; there is no cross-stage reasoning (a critique score is not informed by the
decision score that produced it, even though logically it should be); and no concrete
`JudgeLLMClient` implementation ships with this phase — a future phase must adapt it to whichever
provider is in use. For judge-benchmark integration: averaging five stage scores into one aggregate
per run loses per-scenario granularity, and there is deliberately no single "judge verdict" to
summarize a regression comparison.

### Future work

The `judge.py` docstring mentions a `HumanJudge` as "architecture only; not implemented this
phase." `llm_judge.py`'s docstring states "a future phase may implement and wire a concrete
`JudgeLLMClient` adapter for the provider in use at that time" — no such adapter exists yet.
</content>
