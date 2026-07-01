"""Multi-Agent Investigation Orchestrator (Phase 19D).

Coordinates the existing, independent agents from Phase 19A (Hypothesis
Generator, Evidence Evaluator, Decision Engine), Phase 19B (Planner Agent),
and Phase 19C (Critic Agent) into an ITERATIVE investigation workflow: run
one pass, ask the Critic whether it is good enough, and if not, run
another pass — up to a configured limit, or until further passes stop
making measurable progress.

The orchestrator is coordination-only. It never retrieves evidence itself
(``IncidentSearchService.retrieve``/``.search()`` are only ever called
indirectly, through the unmodified agents that already wrap them), never
plans (``PlannerAgent.plan``, Phase 19B, untouched), never scores
(``score_hypothesis``/``make_investigation_decision``, Phase 19A,
untouched), and never critiques (``CriticAgent.critique``, Phase 19C,
untouched). Every reasoning decision still belongs to the agent that
already owned it; this module only decides WHEN to call them again and
WHEN to stop.

This phase does NOT modify ``app.services.hypothesis_investigation`` (19A),
``app.services.planner_agent`` (19B), or ``app.services.critic_agent``
(19C) — every type/function imported from those modules is read-only and
unmodified. ``InvestigationReport`` (19A) and ``CritiquedInvestigationReport``
(19C) are both left exactly as they were; this phase's ``InvestigationSession``
composes a ``CritiquedInvestigationReport`` as its final report rather than
adding fields to it. It introduces no new retrieval algorithm (the SAME
single ``search_service.retrieve()`` call from Phase 19B/19C's
orchestrators is reused, called once per investigation, not once per
iteration — see "Orchestration lifecycle") and no LLM call beyond the one
``HypothesisGenerator.generate()`` already makes per iteration.

# Updated architecture

```
                                  Problem
                                     │
                                     ▼
                  search_service.retrieve(problem, ...)    [pre-16,
                                     │                       unmodified;
                                     │                       called ONCE,
                                     │                       reused by every
                                     │                       iteration's plan]
                                     ▼
        ┌──────────────────────── iteration N ────────────────────────┐
        │                                                              │
        │   PlannerAgent.plan(...)                    [19B, unmodified]│
        │            │                                                 │
        │            ▼                                                 │
        │   plan_then_generate_hypotheses(...)        [19B, unmodified]│
        │   (ONE LLM call: HypothesisGenerator.generate, 19A)          │
        │            │                                                 │
        │            ▼                                                 │
        │   HypothesisEvaluator.evaluate() per hyp.   [19A, unmodified]│
        │            │                                                 │
        │            ▼                                                 │
        │   score_hypothesis() per hypothesis          [19A, unmodified]│
        │            │                                                 │
        │            ▼                                                 │
        │   make_investigation_decision(scored)        [19A, unmodified]│
        │            │                                                 │
        │            ▼                                                 │
        │   CriticAgent.critique(plan, decision, evals) [19C, unmodified]│
        │            │                                                 │
        └────────────┼─────────────────────────────────────────────────┘
                      ▼
         Orchestrator: evaluate stopping conditions
           (THIS PHASE — the only new reasoning: when to stop/continue)
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
         Finish            Iterate (loop back to "iteration N+1")
            │
            ▼
   InvestigationSession(final_report=, iterations=, stopping_reason=,
                         total_iterations=)
```

# Orchestration lifecycle

```
MultiAgentInvestigationOrchestrator.investigate(problem, ...)
  1. search_service.retrieve(problem, expand=True, rerank=True)  [ONCE,
     pre-16, unmodified] -> initial_results, retrieval_confidence_level
  2. for iteration_number in 1..config.max_iterations:
       a. plan = planner.plan(problem, retrieved_incidents=initial_results,
          routing_observation=)                                  [19B]
       b. hypotheses = plan_then_generate_hypotheses(plan, generator,
          retrieval_context=, n=, existing_root_causes=<every root_cause
          seen in a PRIOR iteration>)                             [19B/19A]
       c. if iteration_number > 1 and every hypothesis's root_cause was
          already seen in a prior iteration (including the degenerate
          case of zero hypotheses) -> STOP, reason=NO_NEW_HYPOTHESES,
          using the LAST iteration that produced a recorded
          ``InvestigationIteration`` as the final one (this attempt is
          discarded, not recorded — see "Risks discovered" for the one
          cost this does not avoid)
       d. evaluations = {evaluator.evaluate(h) for h in hypotheses}  [19A]
       e. decision = make_investigation_decision(scored)             [19A]
       f. critique = critic.critique(plan, decision, evaluations)    [19C]
       g. compute progress vs. the previous recorded iteration (iteration
          1 is always the deterministic baseline - see "Progress
          detection")
       h. record InvestigationIteration(iteration_number, plan,
          hypotheses, evaluations, decision, critique, progress_note,
          rationale)
       i. stopping checks, in this fixed priority order (see "Stopping
          conditions"):
            - critique.verdict == APPROVED and config.stop_on_approval
              -> STOP, reason=CRITIC_APPROVED
            - iteration_number >= config.max_iterations
              -> STOP, reason=MAX_ITERATIONS
            - config.require_progress and not progress_made
              -> STOP, reason=NO_PROGRESS
            - else -> continue to iteration_number + 1
  3. assemble InvestigationSession from the last recorded iteration's
     decision/evaluations (via the unmodified ``build_investigation_report``)
     and critique, plus the full ``iterations`` history.
```

# Investigation state model

``InvestigationIteration`` (frozen) is the immutable record of ONE pass:
``iteration_number``, ``plan`` (19B), ``hypotheses`` (19A, the tuple this
pass generated), ``evaluations`` (19A, ``{hypothesis_id: EvidenceEvaluation}``
for this pass only), ``decision`` (19A), ``critique`` (19C), plus two
fields this phase adds for explainability (see "Explainability"):
``progress_note`` (how this iteration compares to the previous one) and
``rationale`` (why this iteration ran at all).

``InvestigationState`` (frozen) is the functional "current position in the
loop" snapshot the orchestrator threads through ``investigate()``:
``iteration`` (the current iteration number), ``plan``, ``hypotheses``,
``evaluations``, ``decision``, ``critique`` (all mirroring the
just-completed iteration), and ``previous_iterations`` (every
``InvestigationIteration`` recorded so far, INCLUDING the current one —
per the brief's "each iteration should produce a NEW state rather than
mutating the previous one," ``investigate()`` never mutates a
``InvestigationState`` in place; each loop pass constructs a brand new one
appending to ``previous_iterations``).

# Iteration workflow

The orchestrator does not invent any new per-iteration logic: each
iteration calls, in order, exactly the same five agent operations Phase
19C's ``CriticReviewedInvestigationAgent`` already calls once —
``PlannerAgent.plan`` -> ``plan_then_generate_hypotheses`` ->
``HypothesisEvaluator.evaluate`` -> ``make_investigation_decision`` ->
``CriticAgent.critique`` — the only difference is that this module calls
that same sequence repeatedly and feeds ``existing_root_causes`` forward
(``HypothesisGenerator.generate``'s existing, unmodified parameter,
Phase 19A) so each new iteration's hypothesis generation is explicitly
told what was already tried.

# Stopping conditions

``StoppingReason`` (str enum, never collapsed to a boolean):

- **``CRITIC_APPROVED``** — the critic's verdict for this iteration was
  ``CritiqueVerdict.APPROVED`` (19C, unmodified) and
  ``config.stop_on_approval`` is ``True`` (the default). The investigation
  succeeded.
- **``MAX_ITERATIONS``** — ``config.max_iterations`` passes ran without
  ever reaching ``CRITIC_APPROVED``. The investigation stops with
  whatever its last iteration's critique said (often
  ``NEED_MORE_EVIDENCE``/``ALTERNATIVE_HYPOTHESIS_PLAUSIBLE`` — the
  session's ``final_report.critique`` still records that signal
  honestly).
- **``NO_PROGRESS``** — ``config.require_progress`` is ``True`` (the
  default) and the most recent iteration showed no measurable
  improvement over the one before it (see "Progress detection"). Stops
  rather than burning further iterations/LLM calls on a converged state.
- **``NO_NEW_HYPOTHESES``** — a later iteration's hypothesis generation
  produced no root cause that had not already been generated in an
  earlier iteration (including generating nothing at all). There is
  nothing new to evaluate, so iterating further cannot help.

Checked in that fixed priority order every iteration — ``CRITIC_APPROVED``
first because it is unconditional success; ``MAX_ITERATIONS`` next because
it is a hard budget regardless of trend; ``NO_PROGRESS`` last among the
three because it only matters once there IS a trend to evaluate (it
requires a previous iteration to compare against). ``NO_NEW_HYPOTHESES``
is checked separately, earlier in the loop body (see "Orchestration
lifecycle" step 2c), because it short-circuits before evidence evaluation
even runs for that attempt.

# Progress detection

``detect_progress(previous, current)`` is a pure, deterministic function
over two already-computed ``InvestigationIteration`` records. It checks
four independent signals (matching the brief's examples) and returns
``True`` ("made progress") if ANY of them improved:

1. **composite score improved** — the accepted hypothesis's
   ``HypothesisScore.composite_score`` (0.0 if the decision was
   uncertain) is strictly greater than the previous iteration's.
2. **accepted hypothesis changed for the better** — the accepted
   hypothesis's id differs from the previous iteration's AND the new
   composite score is not lower than the previous one's. (Changing to a
   *different but weaker* accepted hypothesis is deliberately NOT counted
   as progress on its own — see "Risks discovered" — only a change that
   does not regress the score counts.)
3. **evidence increased** — the total supporting-evidence count summed
   across every hypothesis evaluated this iteration is strictly greater
   than the previous iteration's total.
4. **critique improved** — the critique verdict's rank in the fixed order
   ``INCONCLUSIVE(0) < NEED_MORE_EVIDENCE(1) <
   ALTERNATIVE_HYPOTHESIS_PLAUSIBLE(2) < APPROVED(3)`` strictly increased.
   This ordering reflects how close each verdict is to a fully-approved
   investigation, not severity of any single heuristic.

If none of the four signals improved, ``detect_progress`` returns
``False`` with an explanation naming exactly which values stayed flat.
The very first iteration is always treated as the deterministic baseline
(``progress_made=True``, since there is nothing to compare it against) —
it can never trigger ``NO_PROGRESS`` on its own.

# Final report

``InvestigationSession`` (frozen) composes, rather than modifies, the
unmodified Phase 19A/19C types: ``final_report``
(``CritiquedInvestigationReport``, 19C, built from the LAST recorded
iteration's ``decision``/``evaluations``/``critique`` via the unmodified
``build_investigation_report``), ``iterations`` (the full, ordered
``tuple[InvestigationIteration, ...]`` history), ``stopping_reason``
(``StoppingReason``), ``total_iterations`` (``len(iterations)``), and
``stop_explanation`` (the plain-text reason the loop actually stopped,
for the same explainability contract every rule-based component in this
codebase has followed since Phase 18A).

# Determinism

Given identical agent outputs (i.e. an identical, deterministic
``LLMService``/fake), ``investigate()`` always executes the exact same
sequence of calls in the exact same order, with the exact same stopping
checks in the exact same priority — the only source of non-determinism is
the LLM call inside ``HypothesisGenerator.generate()`` itself, exactly as
upstream in 19A/19B/19C.

# File-by-file summary

- ``app/services/investigation_orchestrator.py`` (this file) —
  ``StoppingReason``, ``InvestigationIteration``, ``InvestigationState``,
  ``OrchestratorConfig``, ``InvestigationSession``, ``detect_progress``,
  ``MultiAgentInvestigationOrchestrator``.
- ``tests/unit/test_investigation_orchestrator.py`` — comprehensive tests
  (see the test module for the full list).

# Risks discovered

- **A ``NO_NEW_HYPOTHESES``-terminated attempt still spends its LLM
  call.** Detecting "no new root causes" happens AFTER
  ``HypothesisGenerator.generate()`` has already run for that iteration —
  the orchestrator cannot know the hypotheses are stale without first
  generating them. Only the subsequent evidence-evaluation/critique work
  is skipped for that discarded attempt, not the LLM call itself.
- **Progress signal 2 (accepted hypothesis changed) is conservative by
  design.** A hypothesis swap that *lowers* the composite score is never
  counted as progress on its own, but it also is not actively penalized —
  if no other signal improved either, the iteration correctly stops via
  ``NO_PROGRESS``, but the orchestrator does not distinguish "got worse"
  from "stayed exactly the same" in its explanation beyond what
  ``detect_progress``'s text already states.
- **``DEFAULT_MAX_ITERATIONS = 3`` is a reasoned default, not a validated
  constant.** It mirrors Phase 19A's ``DEFAULT_HYPOTHESIS_COUNT = 3`` in
  spirit (bound the per-investigation LLM-call budget to a small, fixed
  number) but was not tuned against any dataset measuring how many
  iterations real investigations typically need to converge.
- **Evidence accumulation is NOT cross-iteration.** Each iteration's
  ``HypothesisEvaluator.evaluate()`` call is independent — evidence found
  in iteration 1 is not merged into iteration 2's evidence count; "evidence
  increased" (progress signal 3) compares iteration totals, not a running
  cumulative total, so a later iteration with fewer hypotheses can show a
  drop in raw evidence count even if its hypotheses are individually
  better-supported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from app.services.critic_agent import (
    CriticAgent,
    CritiqueResult,
    CritiqueVerdict,
    CritiquedInvestigationReport,
    HeuristicCriticAgent,
)
from app.services.hypothesis_investigation import (
    DEFAULT_HYPOTHESIS_COUNT,
    EvidenceEvaluation,
    HypothesisEvaluator,
    HypothesisGenerator,
    InvestigationDecision,
    InvestigationHypothesis,
    build_investigation_report,
    make_investigation_decision,
    score_hypothesis,
)
from app.services.llm_service import LLMService
from app.services.planner_agent import (
    InvestigationPlan,
    PlannerAgent,
    RuleBasedPlanner,
    plan_then_generate_hypotheses,
)
from app.services.search import IncidentSearchService

# Bounds the per-investigation LLM-call budget (one HypothesisGenerator
# call per iteration) to a small, fixed number, in the same spirit as
# Phase 19A's DEFAULT_HYPOTHESIS_COUNT — see module docstring's "Risks
# discovered" for why this is a reasoned default, not a validated one.
DEFAULT_MAX_ITERATIONS = 3

_VERDICT_RANK: dict[CritiqueVerdict, int] = {
    CritiqueVerdict.INCONCLUSIVE: 0,
    CritiqueVerdict.NEED_MORE_EVIDENCE: 1,
    CritiqueVerdict.ALTERNATIVE_HYPOTHESIS_PLAUSIBLE: 2,
    CritiqueVerdict.APPROVED: 3,
}


class StoppingReason(str, Enum):
    CRITIC_APPROVED = "critic_approved"
    MAX_ITERATIONS = "max_iterations"
    NO_PROGRESS = "no_progress"
    NO_NEW_HYPOTHESES = "no_new_hypotheses"


@dataclass(frozen=True)
class InvestigationIteration:
    """One immutable, recorded pass through Planner -> Generator ->
    Evaluator -> Decision -> Critic. See module docstring's "Investigation
    state model".
    """

    iteration_number: int
    plan: InvestigationPlan
    hypotheses: tuple[InvestigationHypothesis, ...]
    evaluations: Mapping[str, EvidenceEvaluation]
    decision: InvestigationDecision
    critique: CritiqueResult
    progress_note: str
    rationale: str


@dataclass(frozen=True)
class InvestigationState:
    """The functional "current position" snapshot threaded through the
    loop — see module docstring's "Investigation state model". Never
    mutated; each iteration produces a brand new one.
    """

    iteration: int
    plan: InvestigationPlan
    hypotheses: tuple[InvestigationHypothesis, ...]
    evaluations: Mapping[str, EvidenceEvaluation]
    decision: InvestigationDecision
    critique: CritiqueResult
    previous_iterations: tuple[InvestigationIteration, ...]


@dataclass(frozen=True)
class OrchestratorConfig:
    """No magic constants - every threshold here is named and has a
    documented default (see module-level ``DEFAULT_MAX_ITERATIONS``).
    """

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    stop_on_approval: bool = True
    require_progress: bool = True


@dataclass(frozen=True)
class InvestigationSession:
    """The final, immutable output — see module docstring's "Final
    report". Composes ``CritiquedInvestigationReport`` (19C, unmodified)
    rather than adding fields to it.
    """

    final_report: CritiquedInvestigationReport
    iterations: tuple[InvestigationIteration, ...]
    stopping_reason: StoppingReason
    total_iterations: int
    stop_explanation: str


# ── Progress detection ────────────────────────────────────────────────────────────


def _composite_of(decision: InvestigationDecision) -> float:
    return decision.accepted_score.composite_score if decision.accepted_score else 0.0


def _total_supporting(evaluations: Mapping[str, EvidenceEvaluation]) -> int:
    return sum(len(evaluation.supporting_evidence) for evaluation in evaluations.values())


def detect_progress(
    previous: InvestigationIteration, current: InvestigationIteration
) -> tuple[bool, str]:
    """See module docstring's "Progress detection". Pure, deterministic,
    no agent calls.
    """
    prev_id = previous.decision.accepted.id if previous.decision.accepted else None
    curr_id = current.decision.accepted.id if current.decision.accepted else None
    prev_composite = _composite_of(previous.decision)
    curr_composite = _composite_of(current.decision)
    prev_evidence = _total_supporting(previous.evaluations)
    curr_evidence = _total_supporting(current.evaluations)
    prev_rank = _VERDICT_RANK[previous.critique.verdict]
    curr_rank = _VERDICT_RANK[current.critique.verdict]

    composite_improved = curr_composite > prev_composite
    accepted_changed_for_better = curr_id != prev_id and curr_composite >= prev_composite
    evidence_increased = curr_evidence > prev_evidence
    critique_improved = curr_rank > prev_rank

    reasons = []
    if composite_improved:
        reasons.append(f"composite score improved ({prev_composite:.2f} -> {curr_composite:.2f})")
    if accepted_changed_for_better:
        reasons.append(f"accepted hypothesis changed ({prev_id!r} -> {curr_id!r})")
    if evidence_increased:
        reasons.append(f"supporting evidence increased ({prev_evidence} -> {curr_evidence})")
    if critique_improved:
        reasons.append(
            f"critique verdict improved "
            f"({previous.critique.verdict.value} -> {current.critique.verdict.value})"
        )

    made_progress = composite_improved or accepted_changed_for_better or evidence_increased
    made_progress = made_progress or critique_improved
    if made_progress:
        return True, "; ".join(reasons)
    return False, (
        f"no measurable improvement over iteration {previous.iteration_number}: "
        f"composite_score unchanged at {curr_composite:.2f}, accepted hypothesis unchanged "
        f"({curr_id!r}), supporting evidence unchanged at {curr_evidence}, critique verdict "
        f"unchanged at {current.critique.verdict.value}"
    )


# ── Orchestrator ───────────────────────────────────────────────────────────────────


class MultiAgentInvestigationOrchestrator:
    """Coordinates Phase 19A/19B/19C's agents into an iterative loop. See
    module docstring's "Orchestration lifecycle". Performs no retrieval,
    planning, scoring, or critique itself.
    """

    def __init__(
        self,
        db: Session,
        *,
        config: OrchestratorConfig | None = None,
        planner: PlannerAgent | None = None,
        critic: CriticAgent | None = None,
        search_service: IncidentSearchService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.search_service = search_service or IncidentSearchService(db)
        self.llm_service = llm_service or LLMService()
        self._planner = planner or RuleBasedPlanner()
        self._critic = critic or HeuristicCriticAgent()
        self._generator = HypothesisGenerator(self.llm_service)
        self._evaluator = HypothesisEvaluator(self.search_service)

    def investigate(
        self,
        problem: str,
        *,
        n_hypotheses: int = DEFAULT_HYPOTHESIS_COUNT,
        routing_observation=None,
    ) -> InvestigationSession:
        initial_results = self.search_service.retrieve(
            problem, limit=10, expand=True, rerank=True,
            call_site="investigation_orchestrator.investigate",
        )
        _, retrieval_confidence_level = IncidentSearchService.confidence_for(initial_results)
        retrieval_context = f"Retrieval confidence: {retrieval_confidence_level}"

        iterations: list[InvestigationIteration] = []
        seen_root_causes: set[str] = set()
        stopping_reason: StoppingReason
        stop_explanation: str

        for iteration_number in range(1, self.config.max_iterations + 1):
            plan = self._planner.plan(
                problem, retrieved_incidents=initial_results,
                routing_observation=routing_observation,
            )
            existing = tuple(seen_root_causes) if seen_root_causes else None
            hypotheses = plan_then_generate_hypotheses(
                plan, self._generator, retrieval_context=retrieval_context,
                n=n_hypotheses, existing_root_causes=existing,
            )

            new_root_causes = {h.root_cause for h in hypotheses} - seen_root_causes
            if iteration_number > 1 and not new_root_causes:
                stopping_reason = StoppingReason.NO_NEW_HYPOTHESES
                stop_explanation = (
                    f"iteration {iteration_number} produced no root cause not already seen "
                    f"in a prior iteration (seen so far: {sorted(seen_root_causes) or 'none'})"
                )
                break

            seen_root_causes.update(new_root_causes)
            evaluations = {
                hypothesis.id: self._evaluator.evaluate(hypothesis) for hypothesis in hypotheses
            }
            scored = [
                (
                    hypothesis,
                    score_hypothesis(
                        hypothesis, evaluations[hypothesis.id],
                        retrieval_confidence_level=retrieval_confidence_level,
                    ),
                )
                for hypothesis in hypotheses
            ]
            decision = make_investigation_decision(scored)
            critique = self._critic.critique(plan, decision, evaluations)

            previous_iteration = iterations[-1] if iterations else None
            if previous_iteration is None:
                progress_made, progress_note = True, "baseline iteration (no prior iteration)"
                rationale = f"iteration {iteration_number} is the baseline investigation pass"
            else:
                candidate = InvestigationIteration(
                    iteration_number=iteration_number, plan=plan, hypotheses=hypotheses,
                    evaluations=evaluations, decision=decision, critique=critique,
                    progress_note="", rationale="",
                )
                progress_made, progress_note = detect_progress(previous_iteration, candidate)
                rationale = (
                    f"iteration {iteration_number} ran because iteration "
                    f"{previous_iteration.iteration_number}'s critique verdict was "
                    f"{previous_iteration.critique.verdict.value}, which does not satisfy the "
                    f"stopping condition (critic approved)"
                )

            current_iteration = InvestigationIteration(
                iteration_number=iteration_number, plan=plan, hypotheses=hypotheses,
                evaluations=evaluations, decision=decision, critique=critique,
                progress_note=progress_note, rationale=rationale,
            )
            iterations.append(current_iteration)

            if critique.verdict == CritiqueVerdict.APPROVED and self.config.stop_on_approval:
                stopping_reason = StoppingReason.CRITIC_APPROVED
                stop_explanation = (
                    f"critic approved the accepted hypothesis on iteration {iteration_number}"
                )
                break
            if iteration_number >= self.config.max_iterations:
                stopping_reason = StoppingReason.MAX_ITERATIONS
                stop_explanation = (
                    f"reached configured max_iterations={self.config.max_iterations} without "
                    f"an APPROVED critique (last verdict: {critique.verdict.value})"
                )
                break
            if self.config.require_progress and not progress_made:
                stopping_reason = StoppingReason.NO_PROGRESS
                stop_explanation = progress_note
                break
        else:
            # unreachable: the max_iterations check above always breaks on
            # the final loop pass, but kept for defensive completeness.
            stopping_reason = StoppingReason.MAX_ITERATIONS
            stop_explanation = f"reached configured max_iterations={self.config.max_iterations}"

        final_iteration = iterations[-1]
        final_investigation_report = build_investigation_report(
            problem, final_iteration.decision, final_iteration.evaluations
        )
        final_report = CritiquedInvestigationReport(
            investigation=final_investigation_report, critique=final_iteration.critique
        )
        return InvestigationSession(
            final_report=final_report,
            iterations=tuple(iterations),
            stopping_reason=stopping_reason,
            total_iterations=len(iterations),
            stop_explanation=stop_explanation,
        )
