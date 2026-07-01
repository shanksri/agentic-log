"""RuleJudge — deterministic, no-LLM Judge implementation (Phase 20B).

Exists so unit tests (and any caller wanting a fast, free, reproducible
judge) never require an LLM call — per this phase's explicit instruction
("unit tests must never require OpenAI"). ``RuleJudge`` is NOT a substitute
for semantic judgment: it scores each stage using only shallow, documented,
deterministic proxies (counts, presence/absence, already-computed
confidence numbers) because a rule-based judge has no way to assess true
semantic correctness/plausibility — see "Risks discovered" below for
exactly what this gives up relative to ``LLMJudge``.

# Per-stage scoring heuristics

Every heuristic below maps directly onto the criteria named in
``app.evaluation.judge.CRITERIA`` for that stage - no heuristic invents a
criterion the interface doesn't already name.

- **Plan** (``chosen_strategy``/``investigation_objective``/
  ``prioritization``/``appropriateness``): starts at a baseline of 6.0
  ("Acceptable" - a plan that merely exists and is well-formed earns the
  middle of the scale, since a rule-based judge cannot tell whether the
  *content* is actually appropriate for the problem). +2.0 if
  ``strategy != PlanningStrategy.UNKNOWN`` (a concrete strategy was
  identified at all - addresses ``appropriateness``/``chosen_strategy``);
  +1.0 if ``objective`` is non-empty (``investigation_objective``); +1.0 if
  ``priority_list`` has more than one entry (``prioritization`` - a single
  vague priority is weaker than several ordered ones).
- **Hypotheses** (``correctness``/``diversity``/``completeness``/
  ``plausibility``): baseline 5.0 (a rule-based judge cannot assess
  ``correctness``/``plausibility`` without a gold answer it does not have
  - see "Risks discovered"). +2.0 if more than one DISTINCT root cause was
  generated (``diversity``); +2.0 if every hypothesis has at least one
  non-empty ``validation_keywords`` entry (``completeness`` - a hypothesis
  with no way to be evidence-checked is incomplete); 0 hypotheses at all
  is scored 1.0 (``Poor`` - nothing to evaluate).
- **Decision** (``selected_hypothesis``/``supporting_evidence``/
  ``reasoning_quality``/``confidence``): if uncertain (no accepted
  hypothesis), scored at 4.0 ("Weak" - a defensible, sometimes-correct
  outcome, but the investigation produced no answer); if accepted, the
  score is ``accepted_score.composite_score`` (already a 0-1 measure
  combining the hypothesis's own confidence with retrieval/evidence
  signals - Phase 19A, unmodified) rescaled linearly onto ``[SCORE_MIN,
  SCORE_MAX]`` - reusing an already-computed, already-justified number
  rather than inventing a new one.
- **Critique** (``justification``/``correctness``/``usefulness``):
  derived directly from ``CritiqueVerdict`` via a fixed mapping
  (``APPROVED``->9.0, ``ALTERNATIVE_HYPOTHESIS_PLAUSIBLE``->6.0,
  ``NEED_MORE_EVIDENCE``->5.0, ``INCONCLUSIVE``->4.0) - this phase already
  established the same logic Phase 19C's own verdict definitions document
  (closer to APPROVED = closer to a complete, justified, useful critique).
- **Session** (``coherence``/``efficiency``/``reasoning_quality``/
  ``final_usefulness``): the mean of the four stage scores above
  (``coherence``/``reasoning_quality`` are properties of the WHOLE chain,
  which a simple mean approximates), MINUS a small efficiency penalty of
  ``0.5 * max(0, total_iterations - 1)`` (``efficiency`` - more iterations
  than the unavoidable minimum of one is a cost, not free), floored at
  ``SCORE_MIN``.

These numbers (the +2.0/+1.0 increments, the 4.0/5.0/6.0 baselines, the
0.5-per-extra-iteration penalty) are explicit, fixed, and documented here
exactly because the brief requires every rubric to be documented - they
are NOT tuned against any dataset (this phase explicitly forbids tuning
scores) and should be read as "a reasonable, explainable default," the
same status Phase 18A/19B/19C's own thresholds already have.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.evaluation.judge import (
    CRITERIA,
    SCORE_MAX,
    SCORE_MIN,
    STAGE_CRITIQUE,
    STAGE_DECISION,
    STAGE_HYPOTHESES,
    STAGE_PLAN,
    STAGE_SESSION,
    Judge,
    JudgeEvaluation,
    JudgeFinding,
    make_judge_score,
)
from app.services.critic_agent import CritiqueResult, CritiqueVerdict
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    InvestigationDecision,
    InvestigationHypothesis,
)
from app.services.investigation_orchestrator import InvestigationSession
from app.services.planner_agent import InvestigationPlan, PlanningStrategy

_CRITIQUE_VERDICT_SCORES: dict[CritiqueVerdict, float] = {
    CritiqueVerdict.APPROVED: 9.0,
    CritiqueVerdict.ALTERNATIVE_HYPOTHESIS_PLAUSIBLE: 6.0,
    CritiqueVerdict.NEED_MORE_EVIDENCE: 5.0,
    CritiqueVerdict.INCONCLUSIVE: 4.0,
}


class RuleJudge(Judge):
    """Deterministic ``Judge`` — see module docstring. Makes zero LLM
    calls; identical inputs always produce an identical
    ``JudgeEvaluation``.
    """

    def evaluate_plan(self, problem: str, plan: InvestigationPlan) -> JudgeEvaluation:
        score = 6.0
        strengths: list[JudgeFinding] = []
        weaknesses: list[JudgeFinding] = []

        if plan.strategy != PlanningStrategy.UNKNOWN:
            score += 2.0
            strengths.append(JudgeFinding("chosen_strategy", f"identified {plan.strategy.value}"))
        else:
            weaknesses.append(JudgeFinding("chosen_strategy", "no concrete strategy identified"))

        if plan.objective.strip():
            score += 1.0
            strengths.append(JudgeFinding("investigation_objective", "objective is stated"))
        else:
            weaknesses.append(JudgeFinding("investigation_objective", "objective is empty"))

        if len(plan.priority_list) > 1:
            score += 1.0
            strengths.append(
                JudgeFinding("prioritization", f"{len(plan.priority_list)} ordered priorities")
            )
        else:
            weaknesses.append(JudgeFinding("prioritization", "fewer than two priorities listed"))

        return JudgeEvaluation(
            stage=STAGE_PLAN, score=make_judge_score(score),
            explanation=(
                f"plan for strategy {plan.strategy.value!r} scored {score:.1f}/10 based on "
                f"{CRITERIA[STAGE_PLAN]}"
            ),
            strengths=tuple(strengths), weaknesses=tuple(weaknesses),
            recommendations=(
                () if plan.strategy != PlanningStrategy.UNKNOWN
                else (JudgeFinding("appropriateness", "investigate why no strategy matched"),)
            ),
        )

    def evaluate_hypotheses(
        self,
        problem: str,
        plan: InvestigationPlan,
        hypotheses: Sequence[InvestigationHypothesis],
    ) -> JudgeEvaluation:
        if not hypotheses:
            return JudgeEvaluation(
                stage=STAGE_HYPOTHESES, score=make_judge_score(SCORE_MIN),
                explanation="no hypotheses were generated; nothing to evaluate",
                weaknesses=(JudgeFinding("completeness", "zero hypotheses generated"),),
                recommendations=(JudgeFinding("completeness", "regenerate hypotheses"),),
            )

        score = 5.0
        strengths: list[JudgeFinding] = []
        weaknesses: list[JudgeFinding] = []

        distinct = {hypothesis.root_cause for hypothesis in hypotheses}
        if len(distinct) > 1:
            score += 2.0
            strengths.append(JudgeFinding("diversity", f"{len(distinct)} distinct root causes"))
        else:
            weaknesses.append(JudgeFinding("diversity", "only one distinct root cause"))

        if all(hypothesis.validation_keywords for hypothesis in hypotheses):
            score += 2.0
            strengths.append(JudgeFinding("completeness", "every hypothesis has keywords"))
        else:
            weaknesses.append(
                JudgeFinding("completeness", "at least one hypothesis has no validation keywords")
            )

        return JudgeEvaluation(
            stage=STAGE_HYPOTHESES, score=make_judge_score(score),
            explanation=(
                f"{len(hypotheses)} hypothesis(es) scored {score:.1f}/10 based on "
                f"{CRITERIA[STAGE_HYPOTHESES]}"
            ),
            strengths=tuple(strengths), weaknesses=tuple(weaknesses),
        )

    def evaluate_decision(
        self,
        problem: str,
        hypotheses: Sequence[InvestigationHypothesis],
        decision: InvestigationDecision,
        evaluations: Mapping[str, EvidenceEvaluation],
    ) -> JudgeEvaluation:
        if decision.is_uncertain or decision.accepted is None or decision.accepted_score is None:
            return JudgeEvaluation(
                stage=STAGE_DECISION, score=make_judge_score(4.0),
                explanation=f"no hypothesis was accepted: {decision.rationale}",
                weaknesses=(JudgeFinding("selected_hypothesis", "no hypothesis accepted"),),
            )

        composite = decision.accepted_score.composite_score
        rescaled = SCORE_MIN + composite * (SCORE_MAX - SCORE_MIN)
        evaluation = evaluations.get(decision.accepted.id)
        strengths: list[JudgeFinding] = []
        weaknesses: list[JudgeFinding] = []
        if evaluation and evaluation.supporting_evidence:
            count = len(evaluation.supporting_evidence)
            strengths.append(JudgeFinding("supporting_evidence", f"{count} item(s)"))
        else:
            weaknesses.append(JudgeFinding("supporting_evidence", "no supporting evidence"))

        return JudgeEvaluation(
            stage=STAGE_DECISION, score=make_judge_score(rescaled),
            explanation=(
                f"accepted {decision.accepted.id!r} with composite_score={composite:.2f}, "
                f"rescaled to {rescaled:.1f}/10 based on {CRITERIA[STAGE_DECISION]}"
            ),
            strengths=tuple(strengths), weaknesses=tuple(weaknesses),
        )

    def evaluate_critique(
        self, problem: str, decision: InvestigationDecision, critique: CritiqueResult
    ) -> JudgeEvaluation:
        score = _CRITIQUE_VERDICT_SCORES[critique.verdict]
        strengths = (
            (JudgeFinding("justification", "explanation provided"),)
            if critique.explanation.strip() else ()
        )
        weaknesses = (
            () if critique.explanation.strip()
            else (JudgeFinding("justification", "no explanation provided"),)
        )
        return JudgeEvaluation(
            stage=STAGE_CRITIQUE, score=make_judge_score(score),
            explanation=(
                f"verdict {critique.verdict.value!r} scored {score:.1f}/10 based on "
                f"{CRITERIA[STAGE_CRITIQUE]}: {critique.explanation}"
            ),
            strengths=strengths, weaknesses=weaknesses,
        )

    def evaluate_session(
        self, problem: str, session: InvestigationSession
    ) -> JudgeEvaluation:
        final = session.iterations[-1]
        plan_score = self.evaluate_plan(problem, final.plan).score.value
        hypotheses_score = self.evaluate_hypotheses(
            problem, final.plan, final.hypotheses
        ).score.value
        decision_score = self.evaluate_decision(
            problem, final.hypotheses, final.decision, final.evaluations
        ).score.value
        critique_score = self.evaluate_critique(
            problem, final.decision, final.critique
        ).score.value

        mean_score = (plan_score + hypotheses_score + decision_score + critique_score) / 4.0
        efficiency_penalty = 0.5 * max(0, session.total_iterations - 1)
        final_score = max(SCORE_MIN, mean_score - efficiency_penalty)

        weaknesses = (
            (JudgeFinding("efficiency", f"{session.total_iterations} iterations were needed"),)
            if efficiency_penalty > 0 else ()
        )

        return JudgeEvaluation(
            stage=STAGE_SESSION, score=make_judge_score(final_score),
            explanation=(
                f"mean stage score {mean_score:.1f} over {session.total_iterations} "
                f"iteration(s) (efficiency penalty {efficiency_penalty:.1f}) -> "
                f"{final_score:.1f}/10 based on {CRITERIA[STAGE_SESSION]}; "
                f"stopped because: {session.stop_explanation}"
            ),
            weaknesses=weaknesses,
        )
