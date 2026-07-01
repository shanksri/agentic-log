"""Critic Agent (Phase 19C).

Introduces an adversarial review stage AFTER the decision stage: given the
``InvestigationPlan`` (Phase 19B), the ``InvestigationDecision`` and its
``EvidenceEvaluation`` map (Phase 19A, both unmodified), the
``CriticAgent`` independently asks whether the accepted hypothesis is
actually well-supported, or whether the pipeline should have kept looking.

The Critic does NOT generate hypotheses (``HypothesisGenerator``, Phase
19A, is never called here), does NOT retrieve evidence (no
``IncidentSearchService`` call is made; it only re-reads the
``EvidenceEvaluation`` objects the evaluator already produced), and does
NOT make the final investigation decision (``make_investigation_decision``,
Phase 19A, is untouched and still owns "accepted vs rejected"). Its output
is an independent, additional signal layered on top of an already-complete
decision — it can recommend further investigation, but it cannot itself
overturn ``InvestigationDecision.accepted``.

This phase does NOT modify ``app.services.hypothesis_investigation`` (19A)
or ``app.services.planner_agent`` (19B) — every type read from those
modules is read-only, and ``InvestigationReport`` is never given a new
field; instead this module introduces a new, composing
``CritiquedInvestigationReport`` (see "Integration workflow") that carries
both the original report and the critique side by side. It makes ZERO LLM
calls — ``HeuristicCriticAgent`` is fully deterministic, by explicit
instruction ("a future phase may replace the heuristic critic with an
LLM-based critic; this phase establishes the architecture only").

# Updated architecture

```
                                  Problem
                                     │
                                     ▼
                  PlannedInvestigationAgent's internals   [19B, unmodified
                     (Planner -> Hypothesis Generation ->   functions/types,
                      Evidence Evaluation -> Decision)      reused as-is]
                                     │
                                     ▼
                InvestigationPlan + InvestigationDecision + evaluations
                                     │
                                     ▼
                       CriticAgent.critique(plan, decision, evaluations)
                              (THIS PHASE — exactly one new agent,
                               zero LLM calls, fully deterministic)
                                     │
                                     ▼
                              CritiqueResult
                    (verdict, confidence, findings, unresolved_questions,
                     missing_evidence, recommended_actions, explanation)
                                     │
                                     ▼
                  build_investigation_report(...)   [19A, unmodified]
                                     │
                                     ▼
                CritiquedInvestigationReport(investigation=, critique=)
                          (THIS PHASE — composes, does not modify,
                           InvestigationReport)
```

# Critique lifecycle

```
CriticAgent.critique(plan, decision, evaluations)
  1. decision.is_uncertain or decision.accepted is None
     -> verdict=INCONCLUSIVE (nothing was accepted; there is nothing for
        the critic to approve or challenge — see "Verdict definitions")
  2. else, evaluate the ACCEPTED hypothesis's own EvidenceEvaluation
     against four independent heuristics, in this fixed priority order
     (most serious finding wins — see "Critique heuristics" for why this
     order):
       a. missing evidence on the accepted hypothesis -> NEED_MORE_EVIDENCE
       b. contradiction ratio over the acceptance threshold -> NEED_MORE_EVIDENCE
       c. score margin over the runner-up under the margin threshold
          -> ALTERNATIVE_HYPOTHESIS_PLAUSIBLE
       d. none of the above triggered -> APPROVED
  3. assemble findings / unresolved_questions / missing_evidence /
     recommended_actions / explanation from the SAME already-computed
     evaluation objects (no new evidence is gathered; no new evidence
     search runs)
  4. -> CritiqueResult (immutable)
```

# Critique heuristics

Each heuristic reuses signals ``HypothesisEvaluator``/``score_hypothesis``
(Phase 19A) already computed for the accepted hypothesis — the critic adds
no new retrieval and invents no new confidence formula:

1. **Evidence completeness** — ``accepted_evaluation.missing_evidence``.
   If the evidence search for the accepted hypothesis's own validation
   keywords found literally nothing, the accepted hypothesis was never
   actually checked against any retrieved incident; this is reported FIRST
   (highest priority) because it represents an absence of grounding, which
   is a stronger objection than weak-but-present grounding.

2. **Contradiction strength** — ``contradicting_count / (supporting_count +
   contradicting_count)`` for the accepted hypothesis (0.0 if there was no
   evidence at all, since heuristic 1 already covers that case).
   ``CONTRADICTION_RATIO_THRESHOLD = 0.5`` — chosen because it is the
   simplest non-arbitrary cut point available: "the majority of the
   evidence this hypothesis's own keywords surfaced was judged
   contradicting" is the natural plain-language reading of "more
   contradicting than supporting," and 0.5 is exactly that majority line
   (not tuned against any dataset, but not picked from thin air either —
   see "Risks discovered").

3. **Score margin between hypotheses** — ``accepted_score.composite_score
   - runner_up_score`` where ``runner_up_score`` is the highest
   ``composite_score`` among ``decision.rejected`` (``0.0`` if nothing was
   rejected — a single-hypothesis investigation has no competitor to be
   plausible). ``MARGIN_THRESHOLD = 0.10`` — reuses the same 0-to-1
   composite-score scale ``ACCEPTANCE_COMPOSITE_FLOOR`` (Phase 19A, 0.60)
   already operates on; a margin under one tenth of that same scale is
   read as "the runner-up was close enough to not be confidently ruled
   out," the least arbitrary fraction of that scale available (10%,
   not 1% or 50%).

4. **Missing validation evidence** — folded into heuristic 1 above (the
   accepted hypothesis's own ``EvidenceEvaluation.missing_evidence``); not
   a separate heuristic, since Phase 19A already represents "no evidence
   was found" as exactly that field.

5. **Uncertainty** — handled upstream, in lifecycle step 1: if the
   decision itself is uncertain, none of heuristics 1-3 are even
   evaluated, because there is no accepted hypothesis to apply them to.

These are checked in the fixed order 1 -> 2 -> 3 -> APPROVED — "most
serious objection wins" — because each is a strictly stronger objection
than the next: no evidence at all (1) is worse than evidence that mostly
contradicts (2), which is worse than evidence that supports but has a
close competitor (3). A hypothesis that triggers none of them is
``APPROVED``.

# Verdict definitions

- **``APPROVED``** — the accepted hypothesis has non-empty evidence, a
  contradiction ratio under the threshold, and a comfortable margin over
  its closest competitor (or no competitor at all). The critic found no
  basis to challenge the decision.
- **``NEED_MORE_EVIDENCE``** — the accepted hypothesis's own evidence
  search found nothing, or found evidence that mostly contradicts it.
  Recommends re-running evidence evaluation with broader validation
  keywords, or gathering more historical incidents, before trusting the
  decision.
- **``ALTERNATIVE_HYPOTHESIS_PLAUSIBLE``** — the accepted hypothesis has
  adequate evidence, but a rejected hypothesis scored close enough that it
  has not actually been ruled out with confidence. Recommends examining
  the runner-up specifically, not generating brand-new hypotheses.
- **``INCONCLUSIVE``** — there was no accepted hypothesis to critique at
  all (``InvestigationDecision.is_uncertain``); the critic has nothing to
  approve or challenge, so it reports this state explicitly rather than
  forcing one of the other three verdicts onto an empty decision.

These four outcomes are never collapsed into a boolean, per the brief's
explicit requirement — each is informationally distinct (compare
``NEED_MORE_EVIDENCE`` recommending more evidence-gathering against
``ALTERNATIVE_HYPOTHESIS_PLAUSIBLE`` recommending re-examining a specific
competitor; collapsing both to "not approved" would lose exactly the
distinction an operator needs to decide what to do next).

# Integration workflow

``CriticAgent`` is an ABC (``critique(plan, decision, evaluations) ->
CritiqueResult``), the same swappable-interface pattern Phase 18A's
``RoutingPolicy`` and Phase 19B's ``PlannerAgent`` both already established
— a future ``LLMCriticAgent`` implements the same method without changing
any caller. ``HeuristicCriticAgent`` is the only implementation this phase
ships, fully deterministic, zero LLM calls.

``CriticReviewedInvestigationAgent`` is the new orchestrator: it reuses
the EXACT same sequence ``PlannedInvestigationAgent.investigate()`` (Phase
19B) already runs — same calls to ``PlannerAgent.plan``,
``plan_then_generate_hypotheses``, ``HypothesisEvaluator.evaluate``,
``score_hypothesis``, ``make_investigation_decision``,
``build_investigation_report``, all imported unmodified from 19A/19B —
but additionally keeps the intermediate ``InvestigationDecision`` and
``evaluations`` mapping in scope (which ``PlannedInvestigationAgent``
discards after building its report) so it can hand them to
``CriticAgent.critique()`` and assemble a ``CritiquedInvestigationReport``.
This duplicates a short call sequence rather than modifying
``PlannedInvestigationAgent`` itself — the brief's "do not modify or
regress any previous phase" leaves no other option once the critic needs
intermediate objects a prior phase's orchestrator does not expose.

``CritiquedInvestigationReport`` is a new, separate frozen dataclass with
exactly two fields, ``investigation: InvestigationReport`` (the unmodified
19A type, untouched) and ``critique: CritiqueResult`` — composition, not
inheritance or field-injection, so ``InvestigationReport`` itself remains
exactly as Phase 19A left it.

# Explainability

Every ``CritiqueResult.explanation`` is a plain-text sentence naming
exactly which heuristic produced the verdict and the numeric signal that
triggered it (e.g. ``"no evidence was found for the accepted hypothesis's
own validation keywords"`` or ``"runner-up h2 scored 0.58, within 0.10 of
the accepted hypothesis's 0.64"``) — the same explainability contract
every rule-based component in this codebase has followed since Phase 18A
(``RoutingDecision.reason``) and Phase 19B (``InvestigationPlan.
strategy_rationale``).

# Risks discovered

- **``CONTRADICTION_RATIO_THRESHOLD`` and ``MARGIN_THRESHOLD`` are
  reasoned defaults, not validated constants.** Like Phase 18A/19B's
  thresholds, neither was tuned against a gold dataset of investigations
  with known-correct verdicts; both are documented analytically (see
  "Critique heuristics") but could both over- and under-trigger on real
  data.
- **The critic only re-reads the accepted hypothesis's OWN evidence
  search.** It does not cross-check the accepted hypothesis against the
  REJECTED hypotheses' evidence (e.g. "did h2's search actually surface
  evidence that argues against h1?") — heuristic 3 (score margin) is a
  proxy for "a competitor remains plausible," not a direct evidentiary
  cross-check, which would require new retrieval calls this phase does
  not make.
- **A single-hypothesis investigation can never trigger
  ``ALTERNATIVE_HYPOTHESIS_PLAUSIBLE``** (there is no rejected hypothesis
  to compare against), which is correct but means the critic's coverage of
  that verdict is entirely dependent on ``n_hypotheses >= 2`` upstream.
- **The critic cannot overturn the decision.** Per the brief, it does not
  make the final investigation decision — even an ``APPROVED``-contradicting
  ``NEED_MORE_EVIDENCE`` verdict leaves ``InvestigationDecision.accepted``
  and ``InvestigationReport.selected_hypothesis`` exactly as Phase 19A
  computed them. A future iterative-loop phase (explicitly out of scope
  here) would be where a ``NEED_MORE_EVIDENCE``/``ALTERNATIVE_HYPOTHESIS_
  PLAUSIBLE`` verdict actually triggers another investigation pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from app.services.hypothesis_investigation import (
    DEFAULT_HYPOTHESIS_COUNT,
    EvidenceEvaluation,
    HypothesisEvaluator,
    HypothesisGenerator,
    InvestigationDecision,
    InvestigationReport,
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

CONTRADICTION_RATIO_THRESHOLD = 0.5
MARGIN_THRESHOLD = 0.10


class CritiqueVerdict(str, Enum):
    APPROVED = "approved"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    ALTERNATIVE_HYPOTHESIS_PLAUSIBLE = "alternative_hypothesis_plausible"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CritiqueResult:
    """The critic's independent, immutable output. See module docstring's
    "Verdict definitions" and "Explainability".
    """

    verdict: CritiqueVerdict
    confidence: float
    findings: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class CritiquedInvestigationReport:
    """Composes the unmodified Phase 19A ``InvestigationReport`` with this
    phase's ``CritiqueResult`` — see module docstring's "Integration
    workflow" for why this is composition, not a new field on
    ``InvestigationReport``.
    """

    investigation: InvestigationReport
    critique: CritiqueResult


# ── Critic interface ─────────────────────────────────────────────────────────────


class CriticAgent(ABC):
    """The swappable extension point — mirrors ``RoutingPolicy`` (18A) and
    ``PlannerAgent`` (19B). A future ``LLMCriticAgent`` implements this
    single method without changing any caller.
    """

    @abstractmethod
    def critique(
        self,
        plan: InvestigationPlan,
        decision: InvestigationDecision,
        evaluations: Mapping[str, EvidenceEvaluation],
    ) -> CritiqueResult:
        """Return an independent ``CritiqueResult`` for an already-made
        ``decision``. Must NOT generate hypotheses, retrieve new evidence,
        or change ``decision`` itself.
        """


# ── Heuristic critic ──────────────────────────────────────────────────────────────


class HeuristicCriticAgent(CriticAgent):
    """Fully deterministic ``CriticAgent`` — see module docstring's
    "Critique heuristics". Makes zero LLM calls and zero new retrieval
    calls.
    """

    def critique(
        self,
        plan: InvestigationPlan,
        decision: InvestigationDecision,
        evaluations: Mapping[str, EvidenceEvaluation],
    ) -> CritiqueResult:
        if decision.is_uncertain or decision.accepted is None:
            return self._inconclusive(plan, decision)

        accepted = decision.accepted
        accepted_score = decision.accepted_score
        accepted_evaluation = evaluations.get(accepted.id)

        if accepted_evaluation is None or accepted_evaluation.missing_evidence:
            return self._need_more_evidence_missing(plan, accepted, accepted_evaluation)

        supporting_count = len(accepted_evaluation.supporting_evidence)
        contradicting_count = len(accepted_evaluation.contradicting_evidence)
        total = supporting_count + contradicting_count
        contradiction_ratio = (contradicting_count / total) if total else 0.0

        if contradiction_ratio >= CONTRADICTION_RATIO_THRESHOLD:
            return self._need_more_evidence_contradicted(
                plan, accepted, contradiction_ratio, supporting_count, contradicting_count
            )

        runner_up_id, runner_up_score = self._runner_up(decision)
        margin = (
            round(accepted_score.composite_score - runner_up_score, 9) if accepted_score else 0.0
        )

        if runner_up_id is not None and margin < MARGIN_THRESHOLD:
            return self._alternative_plausible(
                plan, accepted, accepted_score, runner_up_id, runner_up_score, margin
            )

        return self._approved(plan, accepted, accepted_score, contradiction_ratio, margin)

    # ── Verdict assembly ──────────────────────────────────────────────────────────

    def _inconclusive(
        self, plan: InvestigationPlan, decision: InvestigationDecision
    ) -> CritiqueResult:
        findings = tuple(
            f"hypothesis {hypothesis.id!r} ({hypothesis.root_cause}) scored "
            f"{score.composite_score:.2f}, below the acceptance floor"
            for hypothesis, score in decision.rejected
        )
        explanation = (
            f"no hypothesis was accepted by the decision stage ({decision.rationale}); "
            "the critic has nothing to approve or challenge"
        )
        return CritiqueResult(
            verdict=CritiqueVerdict.INCONCLUSIVE,
            confidence=0.0,
            findings=findings,
            unresolved_questions=(f"why did no hypothesis for {plan.problem!r} clear the floor?",),
            missing_evidence=(),
            recommended_actions=(
                "regenerate hypotheses with a broader or revised investigation plan",
                "gather more historical evidence before re-attempting this investigation",
            ),
            explanation=explanation,
        )

    def _need_more_evidence_missing(
        self,
        plan: InvestigationPlan,
        accepted,
        accepted_evaluation: EvidenceEvaluation | None,
    ) -> CritiqueResult:
        missing = accepted_evaluation.missing_evidence if accepted_evaluation else (
            f"no evidence evaluation was recorded for hypothesis {accepted.id!r}",
        )
        explanation = (
            f"the accepted hypothesis {accepted.id!r} ({accepted.root_cause}) has no "
            "supporting or contradicting evidence at all - its own validation-keyword "
            "search found nothing to check it against"
        )
        return CritiqueResult(
            verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
            confidence=1.0,
            findings=(f"hypothesis {accepted.id!r} was accepted with zero retrieved evidence",),
            unresolved_questions=(
                f"is {accepted.root_cause!r} actually grounded in any historical incident?",
            ),
            missing_evidence=tuple(missing),
            recommended_actions=(
                "broaden or revise the hypothesis's validation keywords and re-search",
                "treat this acceptance as provisional pending further evidence",
            ),
            explanation=explanation,
        )

    def _need_more_evidence_contradicted(
        self,
        plan: InvestigationPlan,
        accepted,
        contradiction_ratio: float,
        supporting_count: int,
        contradicting_count: int,
    ) -> CritiqueResult:
        explanation = (
            f"hypothesis {accepted.id!r} ({accepted.root_cause}) has a contradiction ratio "
            f"of {contradiction_ratio:.2f} (>= {CONTRADICTION_RATIO_THRESHOLD:.2f}): "
            f"{contradicting_count} contradicting vs {supporting_count} supporting result(s)"
        )
        return CritiqueResult(
            verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
            confidence=round(min(1.0, contradiction_ratio), 4),
            findings=(
                f"{contradicting_count} of {contradicting_count + supporting_count} "
                f"retrieved result(s) for hypothesis {accepted.id!r} were judged contradicting",
            ),
            unresolved_questions=(
                f"why does the majority of retrieved evidence for {accepted.id!r} "
                "not support the accepted root cause?",
            ),
            missing_evidence=(),
            recommended_actions=(
                "re-examine the contradicting evidence individually before trusting this "
                "acceptance",
                "consider whether the accepted hypothesis's validation keywords are too broad",
            ),
            explanation=explanation,
        )

    def _alternative_plausible(
        self,
        plan: InvestigationPlan,
        accepted,
        accepted_score,
        runner_up_id: str,
        runner_up_score: float,
        margin: float,
    ) -> CritiqueResult:
        explanation = (
            f"runner-up {runner_up_id!r} scored {runner_up_score:.2f}, within "
            f"{MARGIN_THRESHOLD:.2f} of accepted hypothesis {accepted.id!r}'s "
            f"{accepted_score.composite_score:.2f} (margin={margin:.2f})"
        )
        return CritiqueResult(
            verdict=CritiqueVerdict.ALTERNATIVE_HYPOTHESIS_PLAUSIBLE,
            confidence=round(max(0.0, 1.0 - margin / MARGIN_THRESHOLD), 4),
            findings=(
                f"hypothesis {accepted.id!r} and runner-up {runner_up_id!r} scored "
                f"within {MARGIN_THRESHOLD:.2f} of each other",
            ),
            unresolved_questions=(
                f"has runner-up {runner_up_id!r} actually been ruled out, or only "
                "narrowly outscored?",
            ),
            missing_evidence=(),
            recommended_actions=(
                f"re-examine runner-up {runner_up_id!r}'s evidence specifically",
                "do not treat the accepted hypothesis as conclusively ruled-in over its "
                "competitor",
            ),
            explanation=explanation,
        )

    def _approved(
        self,
        plan: InvestigationPlan,
        accepted,
        accepted_score,
        contradiction_ratio: float,
        margin: float,
    ) -> CritiqueResult:
        confidence = accepted_score.composite_score if accepted_score else 0.0
        explanation = (
            f"hypothesis {accepted.id!r} ({accepted.root_cause}) has non-empty evidence, "
            f"a contradiction ratio of {contradiction_ratio:.2f} "
            f"(< {CONTRADICTION_RATIO_THRESHOLD:.2f}), and "
            + (
                f"a margin of {margin:.2f} over its closest competitor "
                f"(>= {MARGIN_THRESHOLD:.2f})"
                if margin
                else "no remaining competitor"
            )
        )
        return CritiqueResult(
            verdict=CritiqueVerdict.APPROVED,
            confidence=round(confidence, 4),
            findings=(f"hypothesis {accepted.id!r} was not challenged by any heuristic",),
            unresolved_questions=(),
            missing_evidence=(),
            recommended_actions=(),
            explanation=explanation,
        )

    def _runner_up(
        self, decision: InvestigationDecision
    ) -> tuple[str | None, float]:
        if not decision.rejected:
            return None, 0.0
        hypothesis, score = max(decision.rejected, key=lambda pair: pair[1].composite_score)
        return hypothesis.id, score.composite_score


# ── Orchestrator: Planner -> 19A pipeline -> Critic, end to end ─────────────────


class CriticReviewedInvestigationAgent:
    """Wires a ``CriticAgent`` after the unmodified Phase 19A/19B pipeline
    — see module docstring's "Integration workflow" for why this
    duplicates (rather than calls into) ``PlannedInvestigationAgent``'s
    short call sequence.
    """

    def __init__(
        self,
        db: Session,
        *,
        planner: PlannerAgent | None = None,
        critic: CriticAgent | None = None,
        search_service: IncidentSearchService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
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
    ) -> tuple[InvestigationPlan, CritiquedInvestigationReport]:
        initial_results = self.search_service.retrieve(
            problem, limit=10, expand=True, rerank=True,
            call_site="critic_agent.investigate",
        )
        _, retrieval_confidence_level = IncidentSearchService.confidence_for(initial_results)

        plan = self._planner.plan(
            problem, retrieved_incidents=initial_results, routing_observation=routing_observation
        )

        retrieval_context = f"Retrieval confidence: {retrieval_confidence_level}"
        hypotheses = plan_then_generate_hypotheses(
            plan, self._generator, retrieval_context=retrieval_context, n=n_hypotheses
        )
        if not hypotheses:
            decision = make_investigation_decision(())
            report = build_investigation_report(problem, decision, {})
            critique = self._critic.critique(plan, decision, {})
            return plan, CritiquedInvestigationReport(investigation=report, critique=critique)

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
        report = build_investigation_report(problem, decision, evaluations)
        critique = self._critic.critique(plan, decision, evaluations)
        return plan, CritiquedInvestigationReport(investigation=report, critique=critique)
