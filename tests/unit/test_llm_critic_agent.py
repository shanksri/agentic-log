"""Tests for the LLM-backed CriticAgent.

No network: the client is injected as a fake. The behaviour that matters most
is the fallback, because Gemini's free tier allows 20 requests per day per
model and a critique runs once per iteration, so exhaustion is routine rather
than exceptional. An investigation must never fail because the critic was
unavailable, and a reader must always be able to tell which critic produced a
verdict.
"""
from __future__ import annotations

import json

import pytest

from app.services.critic_agent import CritiqueResult, CritiqueVerdict
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    HypothesisScore,
    InvestigationDecision,
    InvestigationHypothesis,
)
from app.services.llm_critic_agent import FALLBACK_NOTE, LLMCriticAgent
from app.services.planner_agent import InvestigationPlan, PlanningStrategy

PROBLEM = "ZookeeperConsumerConnectorMBean needs to close SimpleConsumer when done"


class FakeClient:
    def __init__(self, response=None, *, raises=None):
        self._response = response
        self._raises = raises
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._raises is not None:
            raise self._raises
        return self._response


class CountingFallback:
    def __init__(self):
        self.calls = 0

    def critique(self, plan, decision, evaluations) -> CritiqueResult:
        self.calls += 1
        return CritiqueResult(
            verdict=CritiqueVerdict.APPROVED, confidence=0.8, findings=("heuristic finding",),
            unresolved_questions=(), missing_evidence=(), recommended_actions=(),
            explanation="heuristic explanation",
        )


def _payload(**overrides) -> str:
    data = {
        "verdict": "alternative_hypothesis_plausible",
        "confidence": 0.8,
        "findings": ["root cause addresses network latency, problem is a resource leak"],
        "unresolved_questions": ["why is the consumer not closed?"],
        "missing_evidence": ["an incident about resource lifecycle"],
        "recommended_actions": ["re-run with a lifecycle-focused hypothesis"],
        "explanation": "addresses a different problem than the one asked",
    }
    data.update(overrides)
    return json.dumps(data)


def _fixture(*, is_uncertain: bool = False):
    hyp = InvestigationHypothesis(
        id="h1", root_cause="High network latency between Zookeeper and SimpleConsumer",
        rationale="retrieved incidents mention timeouts",
        validation_keywords=("network", "latency"), raw_confidence=0.8,
    )
    ev = EvidenceEvaluation(
        hypothesis_id="h1", query="network latency",
        supporting_evidence=("ConsoleConsumer throws SocketTimeoutException (similarity=0.550)",),
        contradicting_evidence=(), missing_evidence=(),
        evidence_confidence_level="HIGH", evidence_top1_score=0.626,
    )
    score = HypothesisScore(
        hypothesis_id="h1", raw_confidence=0.8, retrieval_confidence_level="HIGH",
        evidence_confidence_level="HIGH", supporting_count=1, contradicting_count=0,
        missing_count=0, composite_score=0.8,
    )
    decision = InvestigationDecision(
        accepted=None if is_uncertain else hyp,
        accepted_score=None if is_uncertain else score,
        rejected=(), is_uncertain=is_uncertain, rationale="r",
    )
    plan = InvestigationPlan(
        problem=PROBLEM, strategy=PlanningStrategy.NETWORK, objective="o",
        priority_list=("p",), evidence_priorities=("e",), assumptions=("a",),
        expected_difficulty="medium", strategy_rationale="matched network keyword",
    )
    return plan, decision, {"h1": ev}


# ── Happy path ─────────────────────────────────────────────────────────────────


def test_parses_the_verdict_and_fields() -> None:
    plan, decision, evals = _fixture()
    result = LLMCriticAgent(FakeClient(_payload())).critique(plan, decision, evals)
    assert result.verdict == CritiqueVerdict.ALTERNATIVE_HYPOTHESIS_PLAUSIBLE
    assert result.confidence == 0.8
    assert result.findings and result.unresolved_questions
    assert "different problem" in result.explanation


@pytest.mark.parametrize(
    "name,expected",
    [
        ("approved", CritiqueVerdict.APPROVED),
        ("need_more_evidence", CritiqueVerdict.NEED_MORE_EVIDENCE),
        ("inconclusive", CritiqueVerdict.INCONCLUSIVE),
        ("APPROVED", CritiqueVerdict.APPROVED),
        ("  approved  ", CritiqueVerdict.APPROVED),
    ],
)
def test_every_verdict_name_maps(name, expected) -> None:
    plan, decision, evals = _fixture()
    result = LLMCriticAgent(FakeClient(_payload(verdict=name))).critique(plan, decision, evals)
    assert result.verdict == expected


def test_confidence_is_clamped_to_the_unit_interval() -> None:
    plan, decision, evals = _fixture()
    for given, expected in [(5.0, 1.0), (-2.0, 0.0)]:
        result = LLMCriticAgent(FakeClient(_payload(confidence=given))).critique(
            plan, decision, evals
        )
        assert result.confidence == expected


# ── The prompt ─────────────────────────────────────────────────────────────────


def test_prompt_contains_the_problem_and_the_proposed_root_cause() -> None:
    plan, decision, evals = _fixture()
    client = FakeClient(_payload())
    LLMCriticAgent(client).critique(plan, decision, evals)
    prompt = client.prompts[0]
    assert PROBLEM in prompt
    assert "High network latency" in prompt


def test_prompt_warns_that_supporting_evidence_is_weak_by_construction() -> None:
    """Evidence is retrieved with the hypothesis's own keywords, so the critic
    must be told not to read that agreement as confirmation."""
    plan, decision, evals = _fixture()
    client = FakeClient(_payload())
    LLMCriticAgent(client).critique(plan, decision, evals)
    assert "hypothesis's own keywords" in client.prompts[0]


# ── Fallback ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("quota exhausted"), ValueError("bad"), TimeoutError("slow")],
)
def test_any_client_failure_falls_back_instead_of_raising(failure) -> None:
    plan, decision, evals = _fixture()
    fallback = CountingFallback()
    result = LLMCriticAgent(FakeClient(raises=failure), fallback=fallback).critique(
        plan, decision, evals
    )
    assert fallback.calls == 1
    assert result.verdict == CritiqueVerdict.APPROVED


def test_unparseable_response_falls_back() -> None:
    plan, decision, evals = _fixture()
    fallback = CountingFallback()
    LLMCriticAgent(FakeClient("not json at all"), fallback=fallback).critique(
        plan, decision, evals
    )
    assert fallback.calls == 1


def test_unrecognised_verdict_falls_back() -> None:
    plan, decision, evals = _fixture()
    fallback = CountingFallback()
    LLMCriticAgent(FakeClient(_payload(verdict="looks_fine_to_me")), fallback=fallback).critique(
        plan, decision, evals
    )
    assert fallback.calls == 1


def test_fallback_verdict_is_labelled_so_readers_know_which_critic_ran() -> None:
    plan, decision, evals = _fixture()
    result = LLMCriticAgent(
        FakeClient(raises=RuntimeError("quota")), fallback=CountingFallback()
    ).critique(plan, decision, evals)
    assert any(FALLBACK_NOTE in f for f in result.findings)
    assert any("RuntimeError" in f for f in result.findings)


def test_successful_critique_carries_no_fallback_note() -> None:
    plan, decision, evals = _fixture()
    result = LLMCriticAgent(FakeClient(_payload())).critique(plan, decision, evals)
    assert not any(FALLBACK_NOTE in f for f in result.findings)


# ── Contract ───────────────────────────────────────────────────────────────────


def test_abstained_decision_does_not_spend_a_call() -> None:
    """There is no root cause to challenge, and the free tier is 20/day."""
    plan, decision, evals = _fixture(is_uncertain=True)
    client = FakeClient(_payload())
    fallback = CountingFallback()
    LLMCriticAgent(client, fallback=fallback).critique(plan, decision, evals)
    assert client.prompts == []
    assert fallback.calls == 1


def test_exactly_one_call_per_critique() -> None:
    plan, decision, evals = _fixture()
    client = FakeClient(_payload())
    LLMCriticAgent(client).critique(plan, decision, evals)
    assert len(client.prompts) == 1
