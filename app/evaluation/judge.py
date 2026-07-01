"""LLM-as-a-Judge Evaluation Framework — interfaces and report model
(Phase 20B).

Phase 20A's reasoning evaluation (``app.evaluation.reasoning_harness``)
judges correctness through string equality and substring matching against
a fixed gold answer (``expected_strategy ==``, ``_root_cause_matches``,
etc.) — useful for regression-testing against KNOWN scenarios, but unable
to judge SEMANTIC quality ("was this a *reasonable* plan", "were these
hypotheses *plausible*") on problems with no single correct answer. This
phase introduces a parallel, semantic evaluation layer: a ``Judge``
abstraction that scores each reasoning stage on a 1-10 rubric with a
written explanation, strengths, weaknesses, and recommendations — never a
free-form dictionary.

This phase does NOT modify ``app.evaluation.reasoning_harness``/
``reasoning_dataset``/``reasoning_regression``/``reasoning_benchmark``
(Phase 20A) or any agent module (19A-19D) — every type imported from those
modules here is read-only. The two evaluation systems (heuristic, Phase
20A; semantic, this phase) COEXIST: nothing here replaces
``InvestigationResult``/``ReasoningMetrics``, and nothing in 20A is
removed or deprecated.

# Updated architecture

```
                    Reasoning Harness (20A, unmodified - reads
                          InvestigationSession after the fact)
                                     │
                                     ▼
                              Judge interface
                       (THIS PHASE - the harness/caller depends
                        ONLY on this abstraction, never on a
                        concrete LLM/rule implementation)
                          ┌──────────┼──────────┐
                          ▼          ▼          ▼
                     RuleJudge   LLMJudge   (HumanJudge -
                     (this        (this      architecture only;
                      phase,        phase,    not implemented this
                      deterministic, behind a  phase - see "Risks
                      no LLM call) JudgeLLMClient discovered")
                                    abstraction,
                                    never a
                                    concrete
                                    SDK)
                          │          │
                          └────┬─────┘
                               ▼
                       JudgeEvaluation
              (score, explanation, strengths, weaknesses,
               recommendations - per reasoning stage)
                               │
                               ▼
              JudgedReasoningBenchmarkRun (composes Phase 20A's
              ReasoningBenchmarkRun, unmodified, with judge output)
```

# Judge lifecycle

```
Judge.evaluate_plan(problem, plan)                      -> JudgeEvaluation
Judge.evaluate_hypotheses(problem, plan, hypotheses)     -> JudgeEvaluation
Judge.evaluate_decision(problem, hypotheses, decision,
                         evaluations)                    -> JudgeEvaluation
Judge.evaluate_critique(problem, decision, critique)     -> JudgeEvaluation
Judge.evaluate_session(problem, session)                 -> JudgeEvaluation
```

Five separate methods, not one generic ``evaluate(stage, **kwargs)`` —
per this phase's explicit instruction ("avoid one giant generic method;
each reasoning stage deserves its own evaluation contract"). Each stage
has its own typed signature (built entirely from already-existing 19A-19D
types) and its own documented rubric criteria (see "Evaluation rubrics"),
so an implementer cannot accidentally apply the wrong criteria to the
wrong stage — the type signature itself prevents it.

# Evaluation rubrics

Every ``JudgeScore`` is a value in ``[SCORE_MIN, SCORE_MAX]`` (``1..10``),
classified into one of five DOCUMENTED, FIXED bands via
``classify_score()`` (never re-derived ad hoc by a Judge implementation):

```
 1-2   Poor        - fundamentally inadequate for this stage
 3-4   Weak         - significant gaps, partially usable
 5-6   Acceptable   - meets the minimum bar, room to improve
 7-8   Good         - solid, only minor gaps
 9-10  Excellent    - no meaningful gaps for this stage
```

Per-stage criteria (every ``Judge`` implementation, rule-based or LLM-
based, must evaluate against these SAME named criteria — they are the
contract, not a suggestion any one implementation may redefine):

- **Planner** (``evaluate_plan``): chosen strategy, investigation
  objective, prioritization, appropriateness (does the plan fit the
  problem text at all).
- **Hypotheses** (``evaluate_hypotheses``): correctness (plausibility
  given available context, NOT compared against any external gold
  answer - a Judge has no gold answer to consult), diversity (are the
  hypotheses genuinely different explanations, not rephrasings of one
  idea), completeness (do they cover the plan's stated priorities),
  plausibility (does each one have a coherent rationale).
- **Decision** (``evaluate_decision``): selected hypothesis (is the
  accepted one defensible given the others), supporting evidence (does
  the evidence actually back the accepted hypothesis), reasoning quality
  (is ``InvestigationDecision.rationale`` itself sound), confidence
  (is the composite score's magnitude justified by the evidence).
- **Critic** (``evaluate_critique``): justification (does
  ``CritiqueResult.explanation`` actually support its own verdict),
  correctness (is the verdict itself the right call given what came
  before it), usefulness (would an operator find the critique
  actionable).
- **Overall investigation** (``evaluate_session``): coherence (do the
  five stages tell one consistent story), efficiency (was the iteration
  count proportionate to the problem's difficulty), reasoning quality
  (the investigation's reasoning as a whole, not stage-by-stage),
  final usefulness (would the final report actually help someone resolve
  the problem).

``CRITERIA`` (below) names these per-stage tuples as a single source of
truth that both ``RuleJudge`` and ``LLMJudge`` read from, rather than each
hand-listing its own criteria strings.

# Explainability

Every ``JudgeEvaluation.explanation`` is REQUIRED to be non-empty
(enforced in ``JudgeEvaluation.__post_init__``) — "every score must
include why the score was assigned, not only the number" is a hard
invariant of this phase's data model, not a documentation convention a
Judge implementation could skip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.services.critic_agent import CritiqueResult
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    InvestigationDecision,
    InvestigationHypothesis,
)
from app.services.investigation_orchestrator import InvestigationSession
from app.services.planner_agent import InvestigationPlan

SCORE_MIN = 1.0
SCORE_MAX = 10.0

# Fixed, documented rubric bands - see module docstring's "Evaluation
# rubrics". (lower_bound_inclusive, upper_bound_inclusive, band_name).
RUBRIC_BANDS: tuple[tuple[float, float, str], ...] = (
    (1.0, 2.0, "Poor"),
    (2.0, 4.0, "Weak"),
    (4.0, 6.0, "Acceptable"),
    (6.0, 8.0, "Good"),
    (8.0, 10.0, "Excellent"),
)

STAGE_PLAN = "plan"
STAGE_HYPOTHESES = "hypotheses"
STAGE_DECISION = "decision"
STAGE_CRITIQUE = "critique"
STAGE_SESSION = "session"

# Single source of truth for per-stage rubric criteria - see module
# docstring's "Evaluation rubrics". Every Judge implementation evaluates
# against these same named criteria.
CRITERIA: dict[str, tuple[str, ...]] = {
    STAGE_PLAN: ("chosen_strategy", "investigation_objective", "prioritization",
                 "appropriateness"),
    STAGE_HYPOTHESES: ("correctness", "diversity", "completeness", "plausibility"),
    STAGE_DECISION: ("selected_hypothesis", "supporting_evidence", "reasoning_quality",
                      "confidence"),
    STAGE_CRITIQUE: ("justification", "correctness", "usefulness"),
    STAGE_SESSION: ("coherence", "efficiency", "reasoning_quality", "final_usefulness"),
}


def classify_score(value: float) -> str:
    """Map a numeric score to its fixed rubric band — see module
    docstring's "Evaluation rubrics". Clamped to ``[SCORE_MIN, SCORE_MAX]``
    before classification so a Judge implementation cannot produce an
    out-of-band label by passing an out-of-range value.
    """
    clamped = max(SCORE_MIN, min(SCORE_MAX, value))
    for low, high, band in RUBRIC_BANDS:
        if low <= clamped <= high:
            return band
    return RUBRIC_BANDS[-1][2]  # unreachable given the bands above; defensive only


# ── Report data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeScore:
    """A single numeric rubric score plus its derived band. Never
    constructed with ``band`` independently of ``value`` - use
    ``make_judge_score`` so the two can never disagree.
    """

    value: float
    band: str


def make_judge_score(value: float) -> JudgeScore:
    clamped = max(SCORE_MIN, min(SCORE_MAX, value))
    return JudgeScore(value=clamped, band=classify_score(clamped))


@dataclass(frozen=True)
class JudgeFinding:
    """One typed finding (a strength, weakness, or recommendation),
    tagged with which rubric criterion it concerns — never a bare string
    in a free-form list, per this phase's "avoid free-form dictionaries"
    instruction (extended here to apply to untyped string lists too).
    """

    criterion: str
    detail: str


@dataclass(frozen=True)
class JudgeEvaluation:
    """The complete, immutable judgment for ONE reasoning stage of ONE
    investigation. ``explanation`` must be non-empty - see module
    docstring's "Explainability".
    """

    stage: str
    score: JudgeScore
    explanation: str
    strengths: tuple[JudgeFinding, ...] = field(default_factory=tuple)
    weaknesses: tuple[JudgeFinding, ...] = field(default_factory=tuple)
    recommendations: tuple[JudgeFinding, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.explanation.strip():
            raise ValueError(
                f"JudgeEvaluation for stage {self.stage!r} must include a non-empty "
                "explanation - every score must include why it was assigned"
            )


# ── Judge interface ────────────────────────────────────────────────────────────


class Judge(ABC):
    """The swappable extension point this whole phase exists to introduce.
    A caller (the reasoning harness, a future evaluation script) depends
    ONLY on this interface, never on ``RuleJudge``/``LLMJudge`` directly —
    the same pattern ``PlannerAgent``/``CriticAgent``/``RoutingPolicy``
    already establish in this codebase.
    """

    @abstractmethod
    def evaluate_plan(self, problem: str, plan: InvestigationPlan) -> JudgeEvaluation:
        """Judge the planner's output against ``CRITERIA[STAGE_PLAN]``."""

    @abstractmethod
    def evaluate_hypotheses(
        self,
        problem: str,
        plan: InvestigationPlan,
        hypotheses: Sequence[InvestigationHypothesis],
    ) -> JudgeEvaluation:
        """Judge the generated hypotheses against
        ``CRITERIA[STAGE_HYPOTHESES]``.
        """

    @abstractmethod
    def evaluate_decision(
        self,
        problem: str,
        hypotheses: Sequence[InvestigationHypothesis],
        decision: InvestigationDecision,
        evaluations: Mapping[str, EvidenceEvaluation],
    ) -> JudgeEvaluation:
        """Judge the decision stage against ``CRITERIA[STAGE_DECISION]``."""

    @abstractmethod
    def evaluate_critique(
        self, problem: str, decision: InvestigationDecision, critique: CritiqueResult
    ) -> JudgeEvaluation:
        """Judge the critic's output against ``CRITERIA[STAGE_CRITIQUE]``."""

    @abstractmethod
    def evaluate_session(
        self, problem: str, session: InvestigationSession
    ) -> JudgeEvaluation:
        """Judge the whole investigation against ``CRITERIA[STAGE_SESSION]``."""
