# 19 — Multi-Agent Investigation Framework (Phases 19A–19D)

This document covers the four phases that replace the platform's single-pass, single-LLM-call
investigation shape (the pre-existing `InvestigationAgent` and `AdvancedInvestigationAgent`, both
of which remain untouched) with an explicit, composable multi-agent pipeline: **19A** generates and
evidence-scores hypotheses deterministically, **19B** plans investigation strategy ahead of
generation, **19C** audits the resulting decision without overturning it, and **19D** coordinates
all three into an iterative loop with explicit stopping conditions. Every phase is additive — none
of the prior three phases, nor either pre-existing agent, is modified by a later one.

**Update:** Phase 19D (`MultiAgentInvestigationOrchestrator`) is now wired into
`app/api/routes/agent.py` as `POST /agent/investigate` — see **Phase 23A** below, the single
canonical investigation endpoint — and its default `search_service` construction was switched from
a plain, dense-only `IncidentSearchService` to a fully-wired `RoutedSearchService` (doc 18, Phase
18E) — investigations now benefit from adaptive routing (Dense/BM25/Hybrid) the same way
`/search/incidents` does. Phases 19A/19B/19C's own narrower wrapper agents
(`HypothesisDrivenInvestigationAgent`, `PlannedInvestigationAgent`,
`CriticReviewedInvestigationAgent`) remain unwired and still default to plain dense
`IncidentSearchService` — only Phase 19D's orchestrator was adopted. See "Integration status" at
the end of this document for the current picture.

**Phase 23A (API surface consolidation):** `POST /agent/investigate` (single-shot
`InvestigationAgent`), `POST /agent/investigate-advanced` (single-shot `AdvancedInvestigationAgent`),
and `POST /agent/investigate-orchestrated` (this orchestrator) used to coexist as three routes for
one business capability — "investigate this problem and report a root cause" — at three successive
levels of sophistication, with the orchestrator already documented as canonical. The two narrower
routes were removed; `POST /agent/investigate` now serves the orchestrator directly (the path
previously used by the single-shot agent was reassigned, not duplicated). `InvestigationAgent` and
`AdvancedInvestigationAgent` themselves are unmodified and still directly unit-tested — only their
public HTTP routes were retired. See `app/api/routes/agent.py`'s module docstring for the full
rationale.

---

## Phase 19A — Hypothesis-Driven Investigation Framework

### Goal

Replace the single-pass reasoning shape — one LLM call producing a final answer, or
`AdvancedInvestigationAgent`'s two-LLM-call "Policy B" (hypothesis generation followed by a second,
synthesizing LLM call) — with an architecture that generates competing hypotheses, evaluates each
one independently against retrieved evidence, scores them programmatically, and reaches an
accept/reject/uncertain decision **without** a second LLM judgment call.

### Motivation

The existing pipeline is either single-shot or two-LLM-call. Phase 19A trades the second LLM call
for **evidence-driven evaluation**: hypotheses are generated once (via the unmodified
`LLMService.generate_hypotheses`), but acceptance is decided deterministically by comparing each
hypothesis's own evidence search against a fixed floor — no LLM re-judges the candidates. This
buys an auditable, reproducible decision step in place of a second opaque LLM call.

### Architecture

Every output type is a frozen (immutable) dataclass:

```python
InvestigationHypothesis:  id, root_cause, rationale, validation_keywords: tuple[str,...], raw_confidence: float
EvidenceEvaluation:       hypothesis_id, query, supporting_evidence, contradicting_evidence,
                          missing_evidence: tuple[str,...], evidence_confidence_level: str,
                          evidence_top1_score: float | None
HypothesisScore:          hypothesis_id, raw_confidence, retrieval_confidence_level,
                          evidence_confidence_level, supporting_count, contradicting_count,
                          missing_count, composite_score: float
InvestigationDecision:    accepted: InvestigationHypothesis | None, accepted_score,
                          rejected: tuple[tuple[hypothesis, score], ...], is_uncertain: bool, rationale
InvestigationReport:      problem, selected_hypothesis, confidence, confidence_level,
                          supporting_evidence, contradicting_evidence, remaining_uncertainty,
                          is_uncertain, rejected_hypotheses
```

**Components**, each owning exactly one stage:

- `HypothesisGenerator(llm_service)` — wraps `LLMService.generate_hypotheses()` (Phase 6A,
  unmodified) and converts raw dicts into immutable `InvestigationHypothesis` records. The **only**
  LLM call in the whole phase.
- `HypothesisEvaluator(search_service)` — evaluates one hypothesis at a time: runs a single
  `IncidentSearchService.search()` (never `.retrieve()`) against the hypothesis's own
  `validation_keywords` (falling back to `root_cause` if none), then classifies each retrieved
  incident by similarity: `>= LOW_CONFIDENCE_THRESHOLD` (0.40, imported from `app.services.confidence`,
  Phase 6A) → supporting; `< 0.40` → contradicting; no results → missing.
- `score_hypothesis(...)` — delegates entirely to the unmodified Phase 6A
  `composite_hypothesis_confidence()` formula (`raw_confidence × retrieval_weight × keyword_weight`).
  No new scoring math is introduced.
- `make_investigation_decision(scored)` — pure and deterministic: hypotheses with
  `composite_score >= ACCEPTANCE_COMPOSITE_FLOOR` (0.60) are "eligible"; if none are eligible, or
  there are no hypotheses at all, the decision is `is_uncertain=True` with everything rejected;
  otherwise the eligible hypothesis with the highest composite score is accepted (ties broken by
  generation order) and every other hypothesis, eligible or not, is rejected.
- `build_investigation_report(...)` — pure assembly of the final `InvestigationReport` from
  already-computed objects; no LLM call, no retrieval.
- `HypothesisDrivenInvestigationAgent` — public entry point:
  `investigate(problem, n_hypotheses=DEFAULT_HYPOTHESIS_COUNT=3) -> InvestigationReport`.

### Lifecycle

```
problem
  → search_service.retrieve(problem, limit=10, expand=True, rerank=True)   [pre-16, unmodified]
  → initial_results, retrieval_confidence_level
  → HypothesisGenerator.generate(problem, context, n=3, existing_root_causes=None)
      [ONE LLM call]
  → tuple[InvestigationHypothesis, ...]
  → for each hypothesis independently:
        HypothesisEvaluator.evaluate(hypothesis, limit=5)   [one search() call each]
      → EvidenceEvaluation
      → score_hypothesis(hypothesis, evaluation, retrieval_confidence_level)
      → HypothesisScore
  → make_investigation_decision(scored)         [deterministic, no LLM, no retrieval]
  → InvestigationDecision
  → build_investigation_report(problem, decision, evaluations)   [pure assembly]
  → InvestigationReport
```

Total LLM calls per investigation: **1**. Total retrieval calls: **1 + N** (initial retrieve, plus
one evidence search per hypothesis).

### Design decisions

- **Neither `InvestigationAgent` nor `AdvancedInvestigationAgent` is modified.** Phase 19A is a new,
  parallel agent — an optional future adoption path, not a replacement.
- **Exactly one LLM call.** Hypothesis *selection* is deterministic and evidence-driven, unlike
  `AdvancedInvestigationAgent`'s second synthesizing LLM call.
- **Each hypothesis is evaluated independently** — not scored as a set — per an explicit
  requirement to "capture the reasoning inputs," not just assign a number.
- **Scoring reuses Phase 6A's formula rather than inventing new math**, keeping the decision
  grounded in an already-documented, already-justified calculation (doc 14).
- **`ACCEPTANCE_COMPOSITE_FLOOR = 0.60` is carried over from Phase 6A's escalation floor by
  analogy**, not independently re-derived for this specific accept/reject decision.
- **"Contradicting evidence" is a retrieval-strength heuristic, not semantic contradiction** — a
  low-similarity result means the hypothesis's own keyword search didn't find strong grounding, not
  that the evidence logically argues against the hypothesis.
- **Ties are broken by generation order** — deterministic but arbitrary among equally-scored
  hypotheses.
- **"Uncertain" is a first-class, intended outcome**, not an error path — the agent is explicitly
  designed not to force a confident answer when none is warranted.

### Interfaces

Public entry point: `HypothesisDrivenInvestigationAgent.investigate(problem, n_hypotheses=3) ->
InvestigationReport`. Depends read-only on: `LLMService.generate_hypotheses()` (Phase 6A),
`IncidentSearchService.retrieve()` / `.search()` / `.confidence_for()` (pre-16),
`composite_hypothesis_confidence()` and `classify_confidence()` (Phase 6A),
`LOW_CONFIDENCE_THRESHOLD = 0.40`. **Not imported anywhere under `app/api/routes/`** — confirmed
absent from `agent.py`, which only wires `InvestigationAgent`/`AdvancedInvestigationAgent`.

### Testing

`tests/unit/test_hypothesis_investigation.py` covers, among others: raw-dict-to-dataclass
conversion and sequential ID assignment; confidence clamping to `[0, 1]` and defaulting malformed
values to `0.0`; keyword-list coercion; threading `existing_root_causes` through; evidence
classification at, above, and below the 0.40 boundary (boundary itself counts as supporting);
falling back to `root_cause` when no keywords exist; joining multiple keywords into one query;
scoring delegating exactly to `composite_hypothesis_confidence`; decision logic for zero
hypotheses, exactly one eligible hypothesis, zero eligible hypotheses (all rejected, uncertain),
selecting the max composite score among several eligible, and tie-breaking by generation order;
report assembly for both the accepted and uncertain cases, including that rejected alternatives and
missing evidence both surface in `remaining_uncertainty`; dataclass immutability; and an end-to-end
test confirming the agent calls `retrieve()` with `expand=True, rerank=True` and evaluates every
hypothesis independently.

### Risks

- Contradiction detection is a retrieval-strength proxy, not a logical check — a weak match could
  be irrelevant rather than actually contradictory.
- Tie-breaking by generation order has no semantic justification beyond determinism.
- `ACCEPTANCE_COMPOSITE_FLOOR = 0.60` is reused from a different decision (escalation, Phase 6A)
  and was not independently validated for this accept/reject decision.

### Future work

The module is explicitly built as a foundation for later phases without further interface changes:
`HypothesisGenerator.generate()`'s existing `context` parameter is where a planner (19B) injects
richer context; `HypothesisEvaluator.evaluate()`'s `(hypothesis) -> EvidenceEvaluation` signature is
stable enough for a future LLM-based evidence evaluator to replace its internals;
`make_investigation_decision()` is where a critic (19C) would attach additional signals to a richer
score without changing the decision function's shape; `build_investigation_report()` is the seam
for a future synthesizer.

---

## Phase 19B — Planner Agent

### Goal

Introduce the first true reasoning agent ahead of hypothesis generation. Given a problem (and
optionally already-retrieved incidents and a Phase 18B `RoutingObservation`), `PlannerAgent`
decides **how** the investigation should proceed — objective, strategy, priorities, evidence
priorities, assumptions, expected difficulty — recorded as an immutable `InvestigationPlan`. The
planner never generates hypotheses (19A's `HypothesisGenerator` remains the only thing that does)
and hypothesis generation stays strategy-independent: the LLM is never shown a `PlanningStrategy`
enum value, only plain text.

### Motivation

The pre-19B pipeline is strategy-blind — every problem gets the same retrieval-then-generate
treatment regardless of category (authentication vs. network vs. infrastructure, etc.). Phase 19B
is fully deterministic and rule-based (keyword matching), explicitly following the same
architectural choice Phase 18A's `DefaultRuleBasedRoutingPolicy` made for the same reason: a
simple, explainable first version, with ML/LLM-based planning deferred to a future phase.

### Architecture

```python
class PlanningStrategy(str, Enum):
    AUTHENTICATION, NETWORK, INFRASTRUCTURE_FAILURE, CONFIGURATION,
    APPLICATION_FAILURE, UNKNOWN

InvestigationPlan(frozen):
    problem, strategy: PlanningStrategy, objective: str,
    priority_list: tuple[str,...], evidence_priorities: tuple[str,...],
    assumptions: tuple[str,...], expected_difficulty: str, strategy_rationale: str
```

Each strategy has a fixed template (objective / priorities / evidence priorities / assumptions /
static difficulty) and a keyword set, e.g. AUTHENTICATION matches `auth, login, token, credential,
permission, 403, 401, unauthorized, oauth, jwt, ssl, certificate, tls`; NETWORK matches `timeout,
connection refused, dns, network, firewall, latency, socket, tcp, proxy, unreachable`;
INFRASTRUCTURE_FAILURE matches `kubernetes, pod, node, cluster, container, crashloopbackoff, oom,
disk, cpu, memory pressure, kubelet, docker`; CONFIGURATION matches `config, yaml, environment
variable, env var, misconfigured, settings, feature flag, helm values`; APPLICATION_FAILURE matches
`exception, stack trace, null pointer, crash, bug, panic, segfault, null reference, unhandled`.
Matching is case-insensitive and word-boundary aware (so "room" cannot match the OOM keyword).

- `PlannerAgent` (ABC) — `plan(problem, *, retrieved_incidents=(), routing_observation=None) ->
  InvestigationPlan`. Swappable by design.
- `RuleBasedPlanner(PlannerAgent)` — zero LLM calls. Checks, in fixed priority order (**first
  match wins**, mirroring Phase 18A): (1) `routing_observation.signals.has_stack_trace` if
  supplied → `APPLICATION_FAILURE`; (2) AUTHENTICATION keywords; (3) NETWORK; (4)
  INFRASTRUCTURE_FAILURE; (5) CONFIGURATION; (6) APPLICATION_FAILURE keywords; (7) no match →
  `UNKNOWN`. Order runs narrowest/most-specific first, broadest last — deliberately, since
  application-failure terms ("crash", "exception") are broad enough to co-occur with almost any
  category, so a problem mentioning both "401" and "crash" resolves to AUTHENTICATION.
- `plan_then_generate_hypotheses(plan, generator, retrieval_context="", n=3,
  existing_root_causes=None)` — renders the plan to plain text and **prepends** it to the context
  string `HypothesisGenerator.generate()` already accepted; the generator's signature and behavior
  are otherwise unchanged.
- `PlannedInvestigationAgent` — `investigate(problem, n_hypotheses=3, routing_observation=None) ->
  (InvestigationPlan, InvestigationReport)`, planner injectable, defaults to `RuleBasedPlanner`.

### Lifecycle

```
problem
  → search_service.retrieve(...)                                   [pre-16, unmodified]
  → PlannerAgent.plan(problem, retrieved_incidents=initial_results, routing_observation=...)
  → InvestigationPlan
  → plan_block = render(plan)  (strategy, rationale, objective, priorities, assumptions, difficulty)
  → plan_then_generate_hypotheses(plan, generator, retrieval_context=..., n=3)
      [prepends plan_block to context; HypothesisGenerator itself is Phase 19A, UNMODIFIED]
  → [rest of 19A pipeline: evaluate, score, decide, report — unmodified]
  → (InvestigationPlan, InvestigationReport)
```

Inside `RuleBasedPlanner.plan()`: check `routing_observation.signals.has_stack_trace` first if
supplied; else lowercase `problem + incident_texts` and match keyword sets in the fixed priority
order above; look up the winning strategy's template; assemble `InvestigationPlan` with
`strategy_rationale` naming exactly which signal or keyword fired.

### Design decisions

- **19A is untouched**; every type/field read from it is read-only.
- **Zero LLM calls**, matching Phase 18A's rationale for a simple, explainable first version.
- **Hypothesis generation stays strategy-independent at the type level** — the integration seam is
  a plain-text prepend, not a branch on `PlanningStrategy` inside `HypothesisGenerator`.
- **Priority order is narrowest-first, broadest-last**, matching Phase 18A's routing philosophy so
  a problem with mixed signals resolves to the more specific, more actionable category.
- **`routing_observation` is best-effort optional** — Phase 18B's `RoutedSearchService` is not yet
  wired into the investigation pipeline, so in production today keyword matching is the only signal
  actually exercised.
- **`expected_difficulty` is a static per-strategy label**, not computed from the specific problem.
- **`strategy_rationale` mirrors Phase 18A's `RoutingDecision.reason` explainability contract** —
  always names the exact signal or keyword that fired.

### Interfaces

Public: `PlannerAgent.plan(...)`, `RuleBasedPlanner.plan(...)`,
`PlannedInvestigationAgent.investigate(...)`, `plan_then_generate_hypotheses(...)`. Depends
read-only on all of Phase 19A (`HypothesisGenerator`, `HypothesisEvaluator`, `score_hypothesis`,
`make_investigation_decision`, `build_investigation_report`, `InvestigationReport`,
`InvestigationHypothesis`), on Phase 18A/18B's `RoutingObservation` (type hint only — no behavioral
coupling), and on `IncidentSearchService.retrieve()`. **Not wired into `app/api/routes/agent.py`.**

### Testing

`tests/unit/test_planner_agent.py` covers: parametrized keyword-matching across 10+ example
problems per strategy; unrelated problems resolving to UNKNOWN; word-boundary matching (no bare
substring false positives); narrower-category precedence over broader ones on mixed-signal input;
rationale text naming the matched keyword (or explicitly stating no match for UNKNOWN); routing
observation with `has_stack_trace=True` forcing APPLICATION_FAILURE, and `False` falling through to
keywords; retrieved incident text contributing to the keyword haystack; every strategy having a
complete, non-empty template; problem text preserved verbatim on the plan; dataclass immutability;
determinism across repeated calls with identical input; empty-problem, no-incidents, and
no-routing-observation edge cases not crashing; planner swappability via a stub implementation;
planner injection into `PlannedInvestigationAgent`; confirmation that `plan_then_generate_hypotheses`
folds problem and plan context (including retrieval confidence) into the generator's context
string and that the generator never receives the enum directly; and an end-to-end test including
the all-uncertain (empty hypotheses) case.

### Risks

- Keyword sets are hand-written by inspection, not derived from a gold dataset, and may over- or
  under-match real production text.
- The narrowest-first/broadest-last ordering can mis-route a problem with genuinely mixed signals
  when no `routing_observation` is present to disambiguate.
- `routing_observation` is rarely populated in practice since Phase 18B isn't wired into the
  investigation pipeline yet, so the stack-trace override will rarely fire.
- `expected_difficulty` doesn't vary within a strategy, so two very different problems in the same
  category get an identical difficulty label.

### Future work

The `PlannerAgent` ABC is explicitly designed so a future `MLPlanner`, `LLMPlanner`, or
`HybridPlanner` can replace `RuleBasedPlanner` without changing any caller; a planner driven by
`routing_observation` signals becomes possible once Phase 18B is actually wired into the
investigation pipeline.

---

## Phase 19C — Critic Agent

### Goal

Add an adversarial review stage **after** the decision stage, without replacing it. Given the
`InvestigationPlan` (19B), the `InvestigationDecision`, and its `EvidenceEvaluation` map (both 19A,
unmodified), `CriticAgent` independently judges whether the accepted hypothesis is actually
well-supported or whether the pipeline should keep looking. It never generates hypotheses, never
retrieves new evidence (it only re-reads already-computed `EvidenceEvaluation` objects), and never
overturns `make_investigation_decision`'s output — its verdict is an additional signal layered on
top of an already-complete decision.

### Motivation

A deterministic decision (19A) is not automatically a correct one. The critic is an audit layer:
given an already-made decision, it applies independent heuristics and, if it finds the decision
under-justified, flags it (without changing it) so a future iterative caller (19D) can act on the
flag. This phase deliberately chooses zero LLM calls for the same reason 19A/19B did, but the
`CriticAgent` ABC is built so a future LLM-based critic can drop in behind the same signature.

### Architecture

```python
class CritiqueVerdict(str, Enum):
    APPROVED, NEED_MORE_EVIDENCE, ALTERNATIVE_HYPOTHESIS_PLAUSIBLE, INCONCLUSIVE

CritiqueResult(frozen):
    verdict, confidence: float, findings: tuple[str,...], unresolved_questions: tuple[str,...],
    missing_evidence: tuple[str,...], recommended_actions: tuple[str,...], explanation: str

CritiquedInvestigationReport(frozen):
    investigation: InvestigationReport   # 19A, unchanged
    critique: CritiqueResult             # this phase
```

Constants: `CONTRADICTION_RATIO_THRESHOLD = 0.5`, `MARGIN_THRESHOLD = 0.10`.

- `CriticAgent` (ABC) — `critique(plan, decision, evaluations) -> CritiqueResult`.
- `HeuristicCriticAgent(CriticAgent)` — zero LLM calls, checks in fixed priority order (most
  serious objection wins):
  1. **No accepted hypothesis at all** (`decision.is_uncertain` or `decision.accepted is None`) →
     `INCONCLUSIVE` — critic has nothing to approve or challenge.
  2. **Evidence completeness** — accepted hypothesis's own `EvidenceEvaluation.missing_evidence` is
     non-empty (or the evaluation is absent from the map entirely) → `NEED_MORE_EVIDENCE`,
     `confidence=1.0` — absence of grounding is the strongest objection.
  3. **Contradiction strength** — `contradiction_ratio = contradicting_count / (supporting_count +
     contradicting_count)`; if `>= 0.5` → `NEED_MORE_EVIDENCE`, `confidence = min(1.0,
     contradiction_ratio)`.
  4. **Score margin vs. runner-up** — `margin = accepted_score.composite_score - runner_up_score`
     (0.0 if nothing was rejected); if `margin < 0.10` → `ALTERNATIVE_HYPOTHESIS_PLAUSIBLE`,
     `confidence = round(max(0.0, 1.0 - margin / 0.10), 4)`.
  5. **Otherwise** → `APPROVED`, `confidence = accepted_score.composite_score`.
- `CriticReviewedInvestigationAgent` — `investigate(problem, n_hypotheses=3,
  routing_observation=None) -> (InvestigationPlan, CritiquedInvestigationReport)`, critic
  injectable, defaults to `HeuristicCriticAgent`.

### Lifecycle

```
problem
  → PlannedInvestigationAgent's pipeline [19B plan → 19A generate/evaluate/decide]
  → (InvestigationPlan, InvestigationDecision, evaluations)
  → CriticAgent.critique(plan, decision, evaluations)     [zero LLM calls]
  → CritiqueResult
  → build_investigation_report(problem, decision, evaluations)   [19A, unmodified, reused]
  → CritiquedInvestigationReport(investigation=..., critique=...)   [composition, not mutation]
```

`HeuristicCriticAgent.critique()` runs the four checks above strictly in order and returns on the
first one that fires; `APPROVED` is the fallthrough when none of the objections apply.

### Design decisions

- **19A and 19B are untouched.** `CritiquedInvestigationReport` composes the unmodified 19A
  `InvestigationReport` rather than adding fields to it.
- **Zero LLM calls**, explicitly deferring an LLM-based critic to a future phase behind the same
  ABC.
- **Four distinct verdicts, never collapsed to a boolean** — each carries different actionable
  meaning: nothing to approve/challenge (INCONCLUSIVE), gather more evidence
  (NEED_MORE_EVIDENCE), don't rule out the runner-up yet (ALTERNATIVE_HYPOTHESIS_PLAUSIBLE), or
  no objection found (APPROVED).
- **Checks run in fixed priority order, most serious objection wins** — mirroring 19A's own
  decision-making structure.
- **`CONTRADICTION_RATIO_THRESHOLD = 0.5`** is chosen as the plain-language "majority of this
  hypothesis's own evidence contradicts it" cut point — not tuned against a dataset.
- **`MARGIN_THRESHOLD = 0.10`** reuses the same 0–1 composite-score scale
  `ACCEPTANCE_COMPOSITE_FLOOR` (0.60) already operates on; a gap under one tenth of that scale is
  read as "too close to confidently rule out."
- **The critic cannot overturn the decision** — by design, so downstream logic (19D) decides what
  to do with a non-APPROVED verdict, not the critic itself.
- **Only the accepted hypothesis's own evidence is re-examined** — no cross-check against rejected
  hypotheses' evidence; the score-margin heuristic is a proxy for that, not a direct check.
- **`explanation` always names the specific heuristic and numeric signal that fired**, mirroring
  Phase 18A/19B's explainability contract.

### Interfaces

Public: `CriticAgent.critique(...)`, `HeuristicCriticAgent.critique(...)`,
`CriticReviewedInvestigationAgent.investigate(...)`. Depends read-only on all of 19A
(`InvestigationDecision`, `EvidenceEvaluation`, `InvestigationReport`, `build_investigation_report`,
`make_investigation_decision`, `score_hypothesis`, `HypothesisEvaluator`, `HypothesisGenerator`) and
19B (`InvestigationPlan`, `PlannerAgent`, `RuleBasedPlanner`, `plan_then_generate_hypotheses`).
**Not wired into `app/api/routes/agent.py`.**

### Testing

`tests/unit/test_critic_agent.py` covers: APPROVED when evidence is strong with no close
competitor, and when margin is exactly at or above the threshold; NEED_MORE_EVIDENCE when the
accepted hypothesis has no evidence at all, when its evaluation is entirely absent from the map,
and when the contradiction ratio is at or above 0.5 (with confidence tracking the ratio);
ALTERNATIVE_HYPOTHESIS_PLAUSIBLE when the margin is under 0.10, and confirmation that a margin of
exactly 0.10 is treated as APPROVED (not "plausible" — strict inequality); INCONCLUSIVE both when
the decision itself is uncertain and when there are no hypotheses at all; determinism across
repeated calls; dataclass immutability; all four verdict values present as distinct enum members;
critic swappability via a stub; an end-to-end APPROVED case and an end-to-end INCONCLUSIVE case
(empty hypotheses); an explicit test that a NEED_MORE_EVIDENCE verdict does **not** change
`selected_hypothesis` (i.e., the critic really doesn't overturn anything); and injected
planner/critic dependency wiring.

### Risks

- `CONTRADICTION_RATIO_THRESHOLD` and `MARGIN_THRESHOLD` are reasoned defaults, not validated
  against any gold dataset — both could over- or under-trigger on real data.
- The critic only re-examines the accepted hypothesis's own evidence; it never directly checks
  whether a rejected hypothesis's evidence actually argues against the accepted one.
- A single-hypothesis investigation can never produce `ALTERNATIVE_HYPOTHESIS_PLAUSIBLE` — there is
  no rejected hypothesis to compare against, so this verdict's coverage depends on the caller
  requesting `n_hypotheses >= 2`.
- Even a `NEED_MORE_EVIDENCE` or `ALTERNATIVE_HYPOTHESIS_PLAUSIBLE` verdict leaves
  `InvestigationReport.selected_hypothesis` exactly as 19A computed it — nothing acts on the verdict
  until 19D.

### Future work

The `CriticAgent` ABC is built so a future LLM-based critic can implement the same
`critique(plan, decision, evaluations) -> CritiqueResult` signature without changing callers; an
iterative orchestrator (19D) is the explicitly anticipated consumer of non-APPROVED verdicts.

---

## Phase 19D — Multi-Agent Investigation Orchestrator

### Goal

Coordinate the independent agents from 19A (hypothesis generation, evidence evaluation, decision),
19B (planning), and 19C (critique) into an **iterative** investigation workflow: run one pass, ask
the critic whether it's good enough, and if not, run another — up to a configured limit, or until
further passes stop making measurable progress. The orchestrator is coordination-only: it never
retrieves evidence, plans, scores, or critiques itself; it only decides **when** to call the
existing agents again and **when** to stop.

### Motivation

19A–19C build a single-pass pipeline with quality control (the critic), but a single pass often
isn't enough — the first hypothesis generation might miss good candidates, or the critic might
flag a need for more investigation. 19D adds iteration **without changing any single-pass logic**:
the planner, generator, evaluator, decision-maker, and critic are all called exactly as before,
just repeatedly, with each iteration told what root causes were already tried
(`existing_root_causes`, an existing, unmodified parameter of 19A's `HypothesisGenerator.generate()`).
The orchestrator threads iteration state forward and evaluates stopping conditions deterministically.

### Architecture

```python
class StoppingReason(str, Enum):
    CRITIC_APPROVED, MAX_ITERATIONS, NO_PROGRESS, NO_NEW_HYPOTHESES

InvestigationIteration(frozen):
    iteration_number, plan: InvestigationPlan, hypotheses: tuple[InvestigationHypothesis,...],
    evaluations: Mapping[str, EvidenceEvaluation], decision: InvestigationDecision,
    critique: CritiqueResult, progress_note: str, rationale: str

InvestigationState(frozen):     # functional "current position", never mutated
    iteration, plan, hypotheses, evaluations, decision, critique,
    previous_iterations: tuple[InvestigationIteration, ...]

OrchestratorConfig(frozen):
    max_iterations: int = DEFAULT_MAX_ITERATIONS (3)
    stop_on_approval: bool = True
    require_progress: bool = True

InvestigationSession(frozen):
    final_report: CritiquedInvestigationReport, iterations: tuple[InvestigationIteration,...],
    stopping_reason: StoppingReason, total_iterations: int, stop_explanation: str
```

Verdict rank ordering used for progress detection (how close each verdict is to full approval):

```python
_VERDICT_RANK = {INCONCLUSIVE: 0, NEED_MORE_EVIDENCE: 1,
                 ALTERNATIVE_HYPOTHESIS_PLAUSIBLE: 2, APPROVED: 3}
```

`detect_progress(previous_iteration, current_iteration) -> (bool, str)` — a pure function checking
four independent signals, returning `True` if **any** improved: (1) composite score of the accepted
hypothesis strictly increased (0.0 stands in for an uncertain iteration); (2) the accepted
hypothesis changed **and** the new composite score is `>=` the old one (a swap to a strictly weaker
hypothesis does not count as progress); (3) the sum of supporting-evidence counts across all
hypotheses in the current iteration exceeds the previous iteration's sum (each iteration's count is
independent — there is no cross-iteration accumulation); (4) the critique verdict's rank strictly
increased. Iteration 1 always returns `progress_made=True` as a baseline with nothing to compare.

`MultiAgentInvestigationOrchestrator(db, *, config=None, planner=None, critic=None,
search_service=None, llm_service=None)` — public entry point:
`investigate(problem, n_hypotheses=3, routing_observation=None) -> InvestigationSession`.

### Lifecycle

```
problem
  → search_service.retrieve(problem, limit=10, expand=True, rerank=True)   [called ONCE, pre-16]
  → initial_results, retrieval_confidence_level
  → iterations = [], seen_root_causes = set()
  → for iteration_number in 1..config.max_iterations:
        plan = planner.plan(problem, retrieved_incidents=initial_results)            [19B]
        hypotheses = plan_then_generate_hypotheses(plan, generator, ...,
                       existing_root_causes=seen_root_causes)                        [19B/19A]
        new_root_causes = {h.root_cause for h in hypotheses} - seen_root_causes
        if iteration_number > 1 and not new_root_causes:
            → STOP, reason=NO_NEW_HYPOTHESES   (attempt discarded; LLM call already happened)
        seen_root_causes |= new_root_causes
        evaluations = {h.id: evaluator.evaluate(h) for h in hypotheses}              [19A]
        decision = make_investigation_decision(scored)                               [19A]
        critique = critic.critique(plan, decision, evaluations)                      [19C]
        if iteration_number == 1: progress_made = True   # baseline
        else: progress_made, progress_note = detect_progress(previous, current)
        record InvestigationIteration; append to iterations
        # stopping checks, fixed priority order:
        if critique.verdict == APPROVED and config.stop_on_approval:
            → STOP, reason=CRITIC_APPROVED
        if iteration_number >= config.max_iterations:
            → STOP, reason=MAX_ITERATIONS
        if config.require_progress and not progress_made:
            → STOP, reason=NO_PROGRESS
        # else continue to iteration_number + 1
  → final_iteration = iterations[-1]
  → final_report = CritiquedInvestigationReport(
        investigation=build_investigation_report(problem, final_iteration.decision,
                                                   final_iteration.evaluations),      [19A, reused]
        critique=final_iteration.critique)                                            [19C]
  → InvestigationSession(final_report, tuple(iterations), stopping_reason,
                          len(iterations), stop_explanation)
```

Each iteration calls exactly the same five agent operations 19C's `CriticReviewedInvestigationAgent`
already calls once — `plan()`, `plan_then_generate_hypotheses()`, `evaluate()` per hypothesis,
`make_investigation_decision()`, `critique()` — the only difference is repetition plus
`existing_root_causes` threading.

**Stopping conditions**, checked in this fixed order every iteration: `CRITIC_APPROVED` (verdict
was `APPROVED` this iteration and `stop_on_approval=True`, the default) → success; `MAX_ITERATIONS`
(the configured budget was exhausted without ever reaching `APPROVED`; the session's `final_report`
honestly carries whatever the last iteration's critique actually said); `NO_PROGRESS`
(`require_progress=True`, the default, and `detect_progress` found no improvement over the prior
iteration) → stop rather than spend further LLM calls on a converged state. `NO_NEW_HYPOTHESES` is
checked separately, earlier in the loop body, right after generation: if a later iteration's
hypothesis generation produces no root cause not already seen, the attempt is discarded (not
recorded as an `InvestigationIteration`) — but the LLM call for that discarded attempt has already
happened.

### Design decisions

- **19A, 19B, and 19C are all untouched** — every imported type/function is read-only.
- **A single initial `retrieve()` call is reused by every iteration's planner** — no per-iteration
  re-retrieval of the initial result set.
- **No new retrieval algorithm and no new LLM calls** beyond the one 19A already makes per
  iteration inside hypothesis generation.
- **Deterministic execution given deterministic agent outputs** — the only source of
  non-determinism is the LLM call inside generation itself, identical to 19A/19B/19C.
- **`NO_NEW_HYPOTHESES` is detected only after the LLM call has already run** — the orchestrator
  cannot know a generation attempt is stale without first generating it; only the subsequent
  evidence-evaluation and critique work is skipped for that discarded attempt.
- **Progress signal 2 is conservative by design** — swapping to a weaker accepted hypothesis is
  never counted as progress on its own, to avoid rewarding thrashing between candidates.
- **`DEFAULT_MAX_ITERATIONS = 3` mirrors 19A's `DEFAULT_HYPOTHESIS_COUNT = 3` in spirit** (bounding
  the LLM-call budget per investigation) but is not tuned against data on how many iterations real
  investigations need to converge.
- **Evidence accumulation is per-iteration, not cross-iteration** — progress signal 3 compares each
  iteration's own evidence total, it does not track a running cumulative count.
- **State is threaded functionally** — `InvestigationState`/each `InvestigationIteration` is never
  mutated; every pass constructs a new immutable record appended to history.

### Interfaces

Public: `MultiAgentInvestigationOrchestrator.investigate(...)`, and the pure function
`detect_progress(previous_iteration, current_iteration)`. Depends read-only on all of 19A
(`HypothesisGenerator`, `HypothesisEvaluator`, `score_hypothesis`, `make_investigation_decision`,
`build_investigation_report`, `InvestigationDecision`, `EvidenceEvaluation`,
`InvestigationHypothesis`, `DEFAULT_HYPOTHESIS_COUNT`), 19B (`PlannerAgent`, `RuleBasedPlanner`,
`plan_then_generate_hypotheses`, `InvestigationPlan`), 19C (`CriticAgent`, `HeuristicCriticAgent`,
`CritiqueVerdict`, `CritiqueResult`, `CritiquedInvestigationReport`), and
`IncidentSearchService.retrieve()`/`.confidence_for()`.

**Wired into `app/api/routes/agent.py`** as `POST /agent/investigate` (Phase 23A: the single
canonical investigation route — see doc 22's sibling API docs and this document's "Integration
status"). `__init__`'s `search_service` parameter
now accepts `IncidentSearchService | RoutedSearchService | None` and, when not explicitly passed,
defaults to `app.services.search_factory.build_routed_search_service(db, llm_service=self.llm_service)`
(doc 18, Phase 18E) rather than a plain `IncidentSearchService(db)` — see doc 18's Phase 18E section
for the full design rationale. `HypothesisEvaluator` (19A) received the same type-hint widening
since it holds a reference to `self.search_service` and calls `.search()` on it directly; both
`IncidentSearchService` and `RoutedSearchService` expose that method with identical semantics.

### Testing

`tests/unit/test_investigation_orchestrator.py` covers (Phase 18E additions, at the end of the
file): the default `search_service` is a real `RoutedSearchService` instance, not a plain
`IncidentSearchService`, when none is explicitly passed; an explicitly-passed `search_service`
bypasses the routed default entirely and the routed factory is never even called (proving
`app/api/routes/evaluation.py`'s pinned-dense `_build_orchestrator` is unaffected); and an
end-to-end test with a real `RoutedSearchService`/`RoutingEngine` showing both the initial
`retrieve()` call and a hypothesis's evidence `search()` call route to BM25 for a short problem
statement, with `RoutedSearchService.last_observation` populated accordingly. Pre-existing coverage
(unchanged): immediate stop on first-iteration approval
(`total_iterations=1`); multiple iterations when the first pass isn't approved, with
`existing_root_causes` correctly threaded to the second attempt; `MAX_ITERATIONS` stopping when the
verdict never reaches APPROVED even as the composite score keeps improving, both at the default
budget of 3 and at a custom smaller budget; `NO_PROGRESS` stopping when repeated passes are
identical on every signal (with iteration 1 correctly excluded as the baseline), and confirmation
that `require_progress=False` disables that stop and lets the loop run to `MAX_ITERATIONS` instead;
`NO_NEW_HYPOTHESES` stopping both when the generator repeats an identical root cause and when it
returns nothing on two consecutive attempts (in the latter case the final report is
correctly `is_uncertain=True`); determinism of the whole session given identical fake-agent
outputs; immutability of `InvestigationIteration` and `InvestigationSession`; each of the four
`detect_progress` signals individually (`composite_score` improving, evidence count increasing,
verdict rank improving, and the "swap to lower score is not progress" negative case); a
"nothing changed" case returning `False`; confirmation all four `StoppingReason` values are
distinct; that the orchestrator defaults to `RuleBasedPlanner`/`HeuristicCriticAgent`; and injected
planner/critic dependency wiring.

### Risks

- `NO_NEW_HYPOTHESES` can only be detected after the (costed) LLM call for that iteration has
  already run — the check saves the downstream evaluate/critique work, not the generation call
  itself.
- Progress signal 2's conservatism means a genuinely-worse swap and a genuinely-unchanged state
  both simply fail to register as progress; the orchestrator's `detect_progress` explanation text
  is the only place the two are distinguished.
- `DEFAULT_MAX_ITERATIONS = 3` is a reasoned default carried over by analogy from 19A's hypothesis
  count, not validated against data on real investigation convergence behavior.
- Evidence accumulation is not cross-iteration, so a later iteration with fewer, better-supported
  hypotheses can show a lower raw evidence-count signal than an earlier, noisier iteration —
  progress signal 3 can understate genuine improvement in that case.

### Future work

No "future phase" language appears in this module's docstring beyond what's implied by its own
non-goals — it is presented as the terminal phase in the 19A–19D sequence. It has since been wired
into `app/api/routes/agent.py` and adopted RoutedSearchService as its default retrieval backend
(Phase 18E) — see "Integration status" below; the three narrower 19A/19B/19C wrapper agents remain
unwired, which is the next natural step if a use case for them (short of the full orchestrator)
emerges. See doc 22 for the equivalent evaluation-side API surface and doc 17 for the
platform-level roadmap.

---

## Integration status

Phase 19D (`MultiAgentInvestigationOrchestrator`) is wired into `app/api/routes/agent.py` as the
single canonical investigation route (Phase 23A):

```python
POST /agent/investigate  → MultiAgentInvestigationOrchestrator   (planner + evidence-driven
                                                                    hypotheses + critic + iterative loop)
```

Before Phase 23A, three routes coexisted for this one capability:

```python
POST /agent/investigate               → InvestigationAgent                    (single-pass, one LLM call)
POST /agent/investigate-advanced      → AdvancedInvestigationAgent            (single-pass, two LLM calls)
POST /agent/investigate-orchestrated  → MultiAgentInvestigationOrchestrator   (this orchestrator)
```

`/agent/investigate-orchestrated` was already documented as canonical; the other two were earlier,
narrower implementations of the same capability, not distinct ones, so Phase 23A retired their
routes and reassigned the plain `/agent/investigate` path to the orchestrator. `InvestigationAgent`
and `AdvancedInvestigationAgent` remain unmodified in `app/services/` and are still directly
unit-tested — only their HTTP routes were removed. The orchestrator's default retrieval backend is
`RoutedSearchService` (doc 18, Phase 18E) rather than plain dense `IncidentSearchService`, so both
the orchestrator's initial retrieval and every hypothesis's evidence search benefit from adaptive
routing when `Settings.search_routing_enabled` is set.

Phases 19A/19B/19C's own narrower wrapper agents — `HypothesisDrivenInvestigationAgent`
(`app.services.hypothesis_investigation`), `PlannedInvestigationAgent`
(`app.services.planner_agent`), `CriticReviewedInvestigationAgent` (`app.services.critic_agent`) —
remain independently importable and testable but have no route of their own and still default to
plain dense `IncidentSearchService`; only Phase 19D's full orchestrator was adopted into production.

**Since Phase 23B/23C:** `POST /agent/investigate` requires `Authorization: Bearer <API_KEY>`
(missing/malformed/wrong key → `401`) and is capped at `RATE_LIMIT_AGENT_PER_MINUTE` (default
20/min) per caller — the platform's most expensive single-request capability short of
`/evaluation/full`, since one investigation can run several LLM calls across multiple orchestrator
iterations. Neither changes anything documented above; both are cross-cutting API concerns, not
orchestrator behavior. See doc 23.
</content>
