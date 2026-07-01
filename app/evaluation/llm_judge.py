"""LLMJudge — semantic, LLM-backed Judge implementation (Phase 20B).

Per this phase's explicit design philosophy ("the evaluation framework
must not depend directly on any specific LLM"), ``LLMJudge`` depends only
on ``JudgeLLMClient`` — a minimal, two-method Protocol this module defines
itself, NOT on ``app.services.llm_service.LLMService`` or any OpenAI SDK
type. ``LLMJudge`` never imports ``openai``; it never constructs a prompt
that is provider-specific (no OpenAI-specific function-calling schema,
no Anthropic-specific tool-use block) — it sends one plain-text prompt and
expects one plain-text response containing a JSON object, the lowest
common denominator any text-completion client can satisfy.

This phase does NOT wire a concrete ``JudgeLLMClient`` implementation to a
real provider (no ``OpenAIJudgeLLMClient`` is built here) — "do not
optimize prompts" / "this phase builds the judging framework" together
mean the framework, the prompts, and the parsing contract are what this
phase delivers; a concrete client adapter is left for a future phase to
wire up against whichever provider is in use at that time, exactly the
same deferral Phase 19C made for "a future phase may replace the
heuristic critic with an LLM-based critic."

# JudgeLLMClient contract

```python
class JudgeLLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...
```

One method, one string in, one string out — text completion, nothing
more. Any object satisfying this (a thin wrapper around any SDK, or a test
double) works as ``LLMJudge``'s backing client.

# Prompt construction

Each ``evaluate_*`` method builds ONE plain-text prompt naming: the
problem, the artifact being judged (rendered as plain text from already-
existing dataclass fields — no new serialization format), the stage's
fixed rubric criteria (``app.evaluation.judge.CRITERIA``), and the fixed
rubric bands (``app.evaluation.judge.RUBRIC_BANDS``), then asks for a
JSON object with keys ``score``, ``explanation``, ``strengths``,
``weaknesses``, ``recommendations`` (the last three as JSON arrays of
``{"criterion": ..., "detail": ...}`` objects). This is the SAME shape
every stage's prompt asks for — one shared parsing function
(``_parse_response``) handles all five.

# Response parsing

``_parse_response(stage, raw_text)`` parses the JSON object and builds a
``JudgeEvaluation``. It is deliberately STRICT, not defensive: a malformed
or incomplete response raises ``JudgeResponseError`` rather than silently
substituting a default score - per this phase's stop condition ("do not
implement automatic self-improvement"), there is no retry/repair loop
here; a caller that wants resilience against occasional malformed LLM
output must add that on top (see "Risks discovered").
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.evaluation.judge import (
    CRITERIA,
    RUBRIC_BANDS,
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
from app.services.critic_agent import CritiqueResult
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    InvestigationDecision,
    InvestigationHypothesis,
)
from app.services.investigation_orchestrator import InvestigationSession
from app.services.planner_agent import InvestigationPlan


class JudgeLLMClient(Protocol):
    """The minimal abstraction ``LLMJudge`` depends on - see module
    docstring's "JudgeLLMClient contract". Not bound to any SDK.
    """

    def complete(self, prompt: str) -> str: ...


class JudgeResponseError(ValueError):
    """Raised when an LLM response cannot be parsed into a
    ``JudgeEvaluation`` - see module docstring's "Response parsing".
    """


def _rubric_text() -> str:
    return "; ".join(f"{low:.0f}-{high:.0f}={band}" for low, high, band in RUBRIC_BANDS)


def _findings_text(label: str, criterion: str, detail: str) -> str:
    return f"{label}[{criterion}]: {detail}"


def _parse_findings(raw: Any, stage: str, kind: str) -> tuple[JudgeFinding, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise JudgeResponseError(f"stage {stage!r}: {kind!r} must be a list, got {type(raw)!r}")
    findings = []
    for item in raw:
        if not isinstance(item, dict) or "criterion" not in item or "detail" not in item:
            raise JudgeResponseError(
                f"stage {stage!r}: each {kind!r} entry must have 'criterion' and 'detail'"
            )
        findings.append(JudgeFinding(criterion=str(item["criterion"]), detail=str(item["detail"])))
    return tuple(findings)


def _parse_response(stage: str, raw_text: str) -> JudgeEvaluation:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"stage {stage!r}: response is not valid JSON: {exc}") from exc

    if "score" not in data or "explanation" not in data:
        raise JudgeResponseError(f"stage {stage!r}: response missing 'score' or 'explanation'")
    try:
        score_value = float(data["score"])
    except (TypeError, ValueError) as exc:
        raise JudgeResponseError(f"stage {stage!r}: 'score' is not numeric: {exc}") from exc

    explanation = str(data["explanation"])
    return JudgeEvaluation(
        stage=stage,
        score=make_judge_score(score_value),
        explanation=explanation,
        strengths=_parse_findings(data.get("strengths"), stage, "strengths"),
        weaknesses=_parse_findings(data.get("weaknesses"), stage, "weaknesses"),
        recommendations=_parse_findings(data.get("recommendations"), stage, "recommendations"),
    )


def _response_format_instructions(stage: str) -> str:
    return (
        f"Rubric bands (1-10): {_rubric_text()}. Criteria for this stage: "
        f"{CRITERIA[stage]}. Respond with ONLY a JSON object with keys "
        '"score" (number 1-10), "explanation" (string, required, non-empty), '
        '"strengths", "weaknesses", "recommendations" (each a JSON array of '
        '{"criterion": "...", "detail": "..."} objects, may be empty).'
    )


class LLMJudge(Judge):
    """Semantic ``Judge`` backed by any ``JudgeLLMClient`` - see module
    docstring. Makes exactly one ``client.complete()`` call per
    ``evaluate_*`` call; never calls an agent, never retrieves evidence,
    never makes more than one LLM call per evaluation.
    """

    def __init__(self, client: JudgeLLMClient) -> None:
        self._client = client

    def evaluate_plan(self, problem: str, plan: InvestigationPlan) -> JudgeEvaluation:
        prompt = (
            f"Problem: {problem}\n\n"
            f"Investigation plan:\nStrategy: {plan.strategy.value}\n"
            f"Objective: {plan.objective}\nPriorities: {list(plan.priority_list)}\n"
            f"Evidence priorities: {list(plan.evidence_priorities)}\n"
            f"Assumptions: {list(plan.assumptions)}\n"
            f"Expected difficulty: {plan.expected_difficulty}\n"
            f"Strategy rationale: {plan.strategy_rationale}\n\n"
            f"Judge this plan. {_response_format_instructions(STAGE_PLAN)}"
        )
        return _parse_response(STAGE_PLAN, self._client.complete(prompt))

    def evaluate_hypotheses(
        self,
        problem: str,
        plan: InvestigationPlan,
        hypotheses: Sequence[InvestigationHypothesis],
    ) -> JudgeEvaluation:
        rendered = "\n".join(
            f"- {hypothesis.id}: {hypothesis.root_cause} "
            f"(confidence={hypothesis.raw_confidence:.2f}, rationale={hypothesis.rationale}, "
            f"keywords={list(hypothesis.validation_keywords)})"
            for hypothesis in hypotheses
        ) or "(no hypotheses were generated)"
        prompt = (
            f"Problem: {problem}\nPlan objective: {plan.objective}\n\n"
            f"Generated hypotheses:\n{rendered}\n\n"
            f"Judge these hypotheses. {_response_format_instructions(STAGE_HYPOTHESES)}"
        )
        return _parse_response(STAGE_HYPOTHESES, self._client.complete(prompt))

    def evaluate_decision(
        self,
        problem: str,
        hypotheses: Sequence[InvestigationHypothesis],
        decision: InvestigationDecision,
        evaluations: Mapping[str, EvidenceEvaluation],
    ) -> JudgeEvaluation:
        if decision.accepted is None or decision.accepted_score is None:
            accepted_text = "(no hypothesis was accepted)"
        else:
            evaluation = evaluations.get(decision.accepted.id)
            accepted_text = (
                f"{decision.accepted.id}: {decision.accepted.root_cause} "
                f"(composite_score={decision.accepted_score.composite_score:.2f}); "
                f"supporting evidence: "
                f"{list(evaluation.supporting_evidence) if evaluation else []}"
            )
        prompt = (
            f"Problem: {problem}\nCandidates considered: {[h.id for h in hypotheses]}\n"
            f"Decision rationale: {decision.rationale}\nAccepted: {accepted_text}\n\n"
            f"Judge this decision. {_response_format_instructions(STAGE_DECISION)}"
        )
        return _parse_response(STAGE_DECISION, self._client.complete(prompt))

    def evaluate_critique(
        self, problem: str, decision: InvestigationDecision, critique: CritiqueResult
    ) -> JudgeEvaluation:
        prompt = (
            f"Problem: {problem}\nDecision rationale: {decision.rationale}\n"
            f"Critique verdict: {critique.verdict.value}\n"
            f"Critique explanation: {critique.explanation}\n"
            f"Findings: {[f.detail for f in critique.findings]}\n"
            f"Unresolved questions: {list(critique.unresolved_questions)}\n\n"
            f"Judge this critique. {_response_format_instructions(STAGE_CRITIQUE)}"
        )
        return _parse_response(STAGE_CRITIQUE, self._client.complete(prompt))

    def evaluate_session(
        self, problem: str, session: InvestigationSession
    ) -> JudgeEvaluation:
        timeline = "\n".join(
            f"Iteration {iteration.iteration_number}: strategy="
            f"{iteration.plan.strategy.value}, hypotheses="
            f"{[h.root_cause for h in iteration.hypotheses]}, "
            f"critique={iteration.critique.verdict.value}"
            for iteration in session.iterations
        )
        prompt = (
            f"Problem: {problem}\nInvestigation timeline:\n{timeline}\n\n"
            f"Stopping reason: {session.stopping_reason.value} "
            f"({session.stop_explanation})\n"
            f"Total iterations: {session.total_iterations}\n\n"
            f"Judge the whole investigation. {_response_format_instructions(STAGE_SESSION)}"
        )
        return _parse_response(STAGE_SESSION, self._client.complete(prompt))
