"""LLM-backed ``CriticAgent`` — the second reasoning model in the pipeline.

# Why this exists

``HeuristicCriticAgent`` decides by arithmetic: a contradiction ratio at or
above 0.5 means "need more evidence", otherwise approve. Measured across a
34-investigation probe, that path produced **78 supporting evidence items
against 2**, only 1 of 34 investigations found any contradicting evidence at
all, and the critic's ``REJECTED``-equivalent verdict **never fired once**.
The reason is structural: evidence is retrieved using the hypothesis's own
keywords, so a hypothesis always finds agreement with itself, and the
resulting ratio is pinned near zero.

The consequence was visible: a query whose exact incident was retrieved at
rank 1 (top-1 0.9301, HIGH) returned an unrelated network root cause at
confidence 0.8, and the critic APPROVED it. Nothing in the pipeline asked the
only question that matters — *does this root cause actually address the
problem that was asked?* That is a reasoning question, not an arithmetic one.

# Why a different model family

``HypothesisGenerator`` proposes root causes using ``OPENAI_MODEL``. If the
same model then critiques them, it is grading its own work, and agreement
tells you nothing. This critic runs on ``GEMINI_MODEL`` precisely so the
proposer and the challenger are independent. That independence is the point
of the design, not an implementation detail.

# What it is allowed to do

The ``CriticAgent`` contract is unchanged: return a ``CritiqueResult`` for an
already-made decision, and do NOT generate hypotheses, retrieve new evidence,
or alter the decision. This makes exactly one LLM call per critique and reads
only what it is handed.

# Fallback, and why it is not optional

Gemini's free tier allows 20 requests per day per model, and a critique runs
once per iteration (up to ``DEFAULT_MAX_ITERATIONS``). Quota exhaustion is
therefore expected, not exceptional. Every failure path — missing key, quota,
overload, unparseable response — falls back to ``HeuristicCriticAgent`` and
records that it did so in the returned ``findings``. An investigation must
never fail because the critic was unavailable, and a reader must always be
able to tell which critic produced a verdict.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping

from app.services.critic_agent import (
    CriticAgent,
    CritiqueResult,
    CritiqueVerdict,
    HeuristicCriticAgent,
)
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    InvestigationDecision,
)
from app.services.planner_agent import InvestigationPlan

logger = logging.getLogger(__name__)

FALLBACK_NOTE = "llm critic unavailable; fell back to the heuristic critic"

_VERDICT_BY_NAME = {v.value: v for v in CritiqueVerdict}

_PROMPT = """You are reviewing an automated incident investigation. Another \
system proposed a root cause; your job is to challenge it, not to agree with it.

PROBLEM AS STATED:
{problem}

PROPOSED ROOT CAUSE:
{root_cause}

ITS STATED REASONING:
{rationale}

EVIDENCE THE SYSTEM COUNTED AS SUPPORTING:
{supporting}

EVIDENCE THE SYSTEM COUNTED AS CONTRADICTING OR MISSING:
{contradicting}

The single most important question: does the proposed root cause actually \
explain THE PROBLEM AS STATED? A root cause can be fluent, well-evidenced and \
entirely about a different problem. Judge that first. Note that the supporting \
evidence was retrieved using the hypothesis's own keywords, so its apparent \
agreement is weak evidence by construction.

Return JSON with exactly these keys:
  "verdict": one of "approved", "need_more_evidence", \
"alternative_hypothesis_plausible", "inconclusive"
  "confidence": a number from 0.0 to 1.0
  "findings": array of short strings, what you observed
  "unresolved_questions": array of short strings
  "missing_evidence": array of short strings, what would be needed to confirm
  "recommended_actions": array of short strings
  "explanation": one or two sentences stating the verdict's reason

Use "approved" only if the root cause plausibly explains the stated problem. \
If it addresses a different problem, do not approve it regardless of how much \
evidence supports it."""


def _bullets(items, empty: str) -> str:
    listed = [str(item) for item in items if str(item).strip()]
    return "\n".join(f"  - {item}" for item in listed) if listed else f"  ({empty})"


class LLMCriticAgent(CriticAgent):
    """``CriticAgent`` backed by a second model family. Falls back to
    ``HeuristicCriticAgent`` on any failure, never raising into the caller.
    """

    def __init__(self, client=None, *, fallback: CriticAgent | None = None) -> None:
        self._client = client
        self._fallback = fallback or HeuristicCriticAgent()

    def critique(
        self,
        plan: InvestigationPlan,
        decision: InvestigationDecision,
        evaluations: Mapping[str, EvidenceEvaluation],
    ) -> CritiqueResult:
        # An abstained decision has no root cause to challenge; the heuristic
        # critic already words this correctly, so there is nothing to add and
        # no reason to spend a call.
        if decision.is_uncertain or decision.accepted is None:
            return self._fallback.critique(plan, decision, evaluations)

        try:
            raw = self._client_or_default().complete(
                self._build_prompt(plan, decision, evaluations)
            )
            return self._parse(raw)
        except Exception as exc:  # noqa: BLE001 — any critic failure must degrade, not propagate
            logger.warning(
                "llm_critic.fell_back",
                exc_info=True,
                extra={"error": type(exc).__name__},
            )
            return self._with_fallback_note(
                self._fallback.critique(plan, decision, evaluations), exc
            )

    def _client_or_default(self):
        if self._client is not None:
            return self._client
        from app.evaluation.gemini_judge_client import GeminiJudgeClient

        self._client = GeminiJudgeClient()
        return self._client

    def _build_prompt(
        self,
        plan: InvestigationPlan,
        decision: InvestigationDecision,
        evaluations: Mapping[str, EvidenceEvaluation],
    ) -> str:
        accepted = decision.accepted
        evaluation = evaluations.get(accepted.id)
        supporting = evaluation.supporting_evidence if evaluation else ()
        contradicting = list(evaluation.contradicting_evidence) if evaluation else []
        if evaluation and evaluation.missing_evidence:
            contradicting += list(evaluation.missing_evidence)
        return _PROMPT.format(
            problem=plan.problem,
            root_cause=accepted.root_cause,
            rationale=accepted.rationale or "(none given)",
            supporting=_bullets(supporting, "none"),
            contradicting=_bullets(contradicting, "none reported"),
        )

    def _parse(self, raw: str) -> CritiqueResult:
        data = json.loads(raw)
        verdict = _VERDICT_BY_NAME.get(str(data.get("verdict", "")).strip().lower())
        if verdict is None:
            raise ValueError(f"unrecognised verdict {data.get('verdict')!r}")
        confidence = float(data.get("confidence", 0.0))
        return CritiqueResult(
            verdict=verdict,
            confidence=round(min(1.0, max(0.0, confidence)), 4),
            findings=_strings(data.get("findings")),
            unresolved_questions=_strings(data.get("unresolved_questions")),
            missing_evidence=_strings(data.get("missing_evidence")),
            recommended_actions=_strings(data.get("recommended_actions")),
            explanation=str(data.get("explanation") or "").strip()
            or f"llm critic returned {verdict.value}",
        )

    @staticmethod
    def _with_fallback_note(result: CritiqueResult, exc: Exception) -> CritiqueResult:
        """Mark a verdict as heuristic-produced. Without this a reader cannot
        tell whether a verdict came from the reasoning critic or from the
        arithmetic one, which are very different claims.
        """
        note = f"{FALLBACK_NOTE} ({type(exc).__name__})"
        return CritiqueResult(
            verdict=result.verdict,
            confidence=result.confidence,
            findings=(*result.findings, note),
            unresolved_questions=result.unresolved_questions,
            missing_evidence=result.missing_evidence,
            recommended_actions=result.recommended_actions,
            explanation=result.explanation,
        )


def _strings(value) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if str(item).strip())
