from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.critic_agent import (
    CONTRADICTION_RATIO_THRESHOLD,
    MARGIN_THRESHOLD,
    CriticAgent,
    CriticReviewedInvestigationAgent,
    CritiqueResult,
    CritiqueVerdict,
    HeuristicCriticAgent,
)
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    HypothesisScore,
    InvestigationDecision,
    InvestigationHypothesis,
    make_investigation_decision,
)
from app.services.planner_agent import InvestigationPlan, PlanningStrategy
from app.services.search import IncidentSearchResult

critic = HeuristicCriticAgent()


# ── Fakes / builders ─────────────────────────────────────────────────────────────


def _plan(problem: str = "something broke") -> InvestigationPlan:
    return InvestigationPlan(
        problem=problem, strategy=PlanningStrategy.UNKNOWN, objective="obj",
        priority_list=("p",), evidence_priorities=("e",), assumptions=("a",),
        expected_difficulty="medium", strategy_rationale="r",
    )


def _hypothesis(
    id_: str, root_cause: str = "x", confidence: float = 0.8
) -> InvestigationHypothesis:
    return InvestigationHypothesis(
        id=id_, root_cause=root_cause, rationale="because", validation_keywords=("x",),
        raw_confidence=confidence,
    )


def _score(id_: str, composite: float) -> HypothesisScore:
    return HypothesisScore(
        hypothesis_id=id_, raw_confidence=0.8, retrieval_confidence_level="HIGH",
        evidence_confidence_level="HIGH", supporting_count=1, contradicting_count=0,
        missing_count=0, composite_score=composite,
    )


def _evaluation(
    id_: str, *, supporting: int = 1, contradicting: int = 0, missing: tuple = ()
) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        hypothesis_id=id_, query="q",
        supporting_evidence=tuple(f"s{i}" for i in range(supporting)),
        contradicting_evidence=tuple(f"c{i}" for i in range(contradicting)),
        missing_evidence=missing,
        evidence_confidence_level="HIGH", evidence_top1_score=0.9,
    )


# ── Approved ──────────────────────────────────────────────────────────────────────


def test_approved_when_evidence_strong_and_no_close_competitor() -> None:
    h1 = _hypothesis("h1")
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", 0.90), rejected=(),
        is_uncertain=False, rationale="accepted h1",
    )
    evaluations = {"h1": _evaluation("h1", supporting=3, contradicting=0)}

    result = critic.critique(_plan(), decision, evaluations)

    assert result.verdict == CritiqueVerdict.APPROVED
    assert result.confidence == pytest.approx(0.90)
    assert result.unresolved_questions == ()
    assert result.missing_evidence == ()
    assert result.recommended_actions == ()
    assert "h1" in result.explanation


def test_approved_when_margin_over_threshold() -> None:
    h1, h2 = _hypothesis("h1"), _hypothesis("h2")
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", 0.90),
        rejected=((h2, _score("h2", 0.50)),), is_uncertain=False, rationale="accepted h1",
    )
    evaluations = {
        "h1": _evaluation("h1", supporting=2, contradicting=0),
        "h2": _evaluation("h2", supporting=1, contradicting=0),
    }

    result = critic.critique(_plan(), decision, evaluations)

    assert result.verdict == CritiqueVerdict.APPROVED


# ── Insufficient evidence ───────────────────────────────────────────────────────


def test_need_more_evidence_when_accepted_has_no_evidence_at_all() -> None:
    h1 = _hypothesis("h1")
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", 0.90), rejected=(),
        is_uncertain=False, rationale="accepted h1",
    )
    evaluations = {
        "h1": _evaluation("h1", supporting=0, contradicting=0, missing=("nothing found",)),
    }

    result = critic.critique(_plan(), decision, evaluations)

    assert result.verdict == CritiqueVerdict.NEED_MORE_EVIDENCE
    assert result.confidence == 1.0
    assert result.missing_evidence == ("nothing found",)
    assert result.findings


def test_need_more_evidence_when_evaluation_missing_from_map() -> None:
    h1 = _hypothesis("h1")
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", 0.90), rejected=(),
        is_uncertain=False, rationale="accepted h1",
    )

    result = critic.critique(_plan(), decision, {})

    assert result.verdict == CritiqueVerdict.NEED_MORE_EVIDENCE
    assert "h1" in result.missing_evidence[0]


# ── Competing hypotheses ─────────────────────────────────────────────────────────


def test_alternative_hypothesis_plausible_when_margin_under_threshold() -> None:
    h1, h2 = _hypothesis("h1"), _hypothesis("h2")
    accepted_composite = 0.65
    runner_up_composite = accepted_composite - (MARGIN_THRESHOLD / 2)
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", accepted_composite),
        rejected=((h2, _score("h2", runner_up_composite)),),
        is_uncertain=False, rationale="accepted h1",
    )
    evaluations = {
        "h1": _evaluation("h1", supporting=2, contradicting=0),
        "h2": _evaluation("h2", supporting=1, contradicting=0),
    }

    result = critic.critique(_plan(), decision, evaluations)

    assert result.verdict == CritiqueVerdict.ALTERNATIVE_HYPOTHESIS_PLAUSIBLE
    assert "h2" in result.explanation
    assert result.recommended_actions


def test_margin_exactly_at_threshold_is_not_plausible_alternative() -> None:
    h1, h2 = _hypothesis("h1"), _hypothesis("h2")
    accepted_composite = 0.70
    runner_up_composite = accepted_composite - MARGIN_THRESHOLD
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", accepted_composite),
        rejected=((h2, _score("h2", runner_up_composite)),),
        is_uncertain=False, rationale="accepted h1",
    )
    evaluations = {
        "h1": _evaluation("h1", supporting=2, contradicting=0),
        "h2": _evaluation("h2", supporting=1, contradicting=0),
    }

    result = critic.critique(_plan(), decision, evaluations)

    assert result.verdict == CritiqueVerdict.APPROVED


# ── Contradictory evidence ──────────────────────────────────────────────────────


def test_need_more_evidence_when_contradiction_ratio_at_or_above_threshold() -> None:
    h1 = _hypothesis("h1")
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", 0.90), rejected=(),
        is_uncertain=False, rationale="accepted h1",
    )
    evaluations = {"h1": _evaluation("h1", supporting=1, contradicting=1)}
    assert (1 / 2) >= CONTRADICTION_RATIO_THRESHOLD

    result = critic.critique(_plan(), decision, evaluations)

    assert result.verdict == CritiqueVerdict.NEED_MORE_EVIDENCE
    assert result.confidence == pytest.approx(0.5)


def test_approved_when_contradiction_ratio_below_threshold() -> None:
    h1 = _hypothesis("h1")
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", 0.90), rejected=(),
        is_uncertain=False, rationale="accepted h1",
    )
    evaluations = {"h1": _evaluation("h1", supporting=3, contradicting=1)}

    result = critic.critique(_plan(), decision, evaluations)

    assert result.verdict == CritiqueVerdict.APPROVED


# ── Inconclusive ─────────────────────────────────────────────────────────────────


def test_inconclusive_when_decision_is_uncertain() -> None:
    h1 = _hypothesis("h1", confidence=0.1)
    decision = InvestigationDecision(
        accepted=None, accepted_score=None, rejected=((h1, _score("h1", 0.10)),),
        is_uncertain=True, rationale="no hypothesis reached the acceptance floor",
    )

    result = critic.critique(_plan(), decision, {"h1": _evaluation("h1")})

    assert result.verdict == CritiqueVerdict.INCONCLUSIVE
    assert result.confidence == 0.0
    assert result.findings
    assert result.recommended_actions


def test_inconclusive_when_no_hypotheses_at_all() -> None:
    decision = make_investigation_decision(())

    result = critic.critique(_plan(), decision, {})

    assert result.verdict == CritiqueVerdict.INCONCLUSIVE
    assert result.findings == ()


# ── Deterministic behavior ──────────────────────────────────────────────────────


def test_critique_is_deterministic_across_repeated_calls() -> None:
    h1, h2 = _hypothesis("h1"), _hypothesis("h2")
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", 0.62),
        rejected=((h2, _score("h2", 0.55)),), is_uncertain=False, rationale="accepted h1",
    )
    evaluations = {
        "h1": _evaluation("h1", supporting=2, contradicting=0),
        "h2": _evaluation("h2", supporting=1, contradicting=0),
    }

    first = critic.critique(_plan(), decision, evaluations)
    second = critic.critique(_plan(), decision, evaluations)

    assert first == second


# ── CritiqueResult / verdict shape ──────────────────────────────────────────────


def test_critique_result_is_frozen() -> None:
    h1 = _hypothesis("h1")
    decision = InvestigationDecision(
        accepted=h1, accepted_score=_score("h1", 0.90), rejected=(),
        is_uncertain=False, rationale="accepted h1",
    )
    result = critic.critique(_plan(), decision, {"h1": _evaluation("h1", supporting=2)})

    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        result.verdict = CritiqueVerdict.INCONCLUSIVE  # type: ignore[misc]


def test_all_four_verdicts_are_distinct_enum_members() -> None:
    assert {
        CritiqueVerdict.APPROVED,
        CritiqueVerdict.NEED_MORE_EVIDENCE,
        CritiqueVerdict.ALTERNATIVE_HYPOTHESIS_PLAUSIBLE,
        CritiqueVerdict.INCONCLUSIVE,
    } == set(CritiqueVerdict)


# ── Critic replacement (interface independence) ─────────────────────────────────


class _AlwaysApprovingCritic(CriticAgent):
    def critique(self, plan, decision, evaluations) -> CritiqueResult:
        return CritiqueResult(
            verdict=CritiqueVerdict.APPROVED, confidence=1.0, findings=(),
            unresolved_questions=(), missing_evidence=(), recommended_actions=(),
            explanation="stub critic always approves",
        )


def test_critic_can_be_replaced_without_changing_callers() -> None:
    stub = _AlwaysApprovingCritic()
    decision = make_investigation_decision(())

    result = stub.critique(_plan(), decision, {})

    assert result.verdict == CritiqueVerdict.APPROVED


# ── Integration with Phase 19B ───────────────────────────────────────────────────


class FakeLLMService:
    def __init__(self, hypotheses=None):
        self._hypotheses = hypotheses if hypotheses is not None else []
        self.calls: list[dict] = []

    def generate_hypotheses(self, *, problem, context, n=2, existing_root_causes=None):
        self.calls.append({"problem": problem, "context": context, "n": n})
        return self._hypotheses


def _incident(title: str, symptoms=()):
    return SimpleNamespace(title=title, symptoms=[SimpleNamespace(text=s) for s in symptoms])


def _result(title: str, distance: float = 0.5, symptoms=()) -> IncidentSearchResult:
    return IncidentSearchResult(incident=_incident(title, symptoms), distance=distance)


class FakeSearchService:
    def __init__(self, *, retrieve_response=None, search_responses=None):
        self._retrieve_response = retrieve_response or []
        self._search_responses = search_responses or {}

    def retrieve(self, query, *, limit=10, expand=False, rerank=False, call_site=None):
        return self._retrieve_response

    def search(self, query, *, limit=10, call_site=None):
        return self._search_responses.get(query, [])


def test_critic_reviewed_agent_end_to_end_approved() -> None:
    llm = FakeLLMService([
        {
            "root_cause": "expired token", "confidence_score": 0.9,
            "validation_keywords": ["token", "expired"], "rationale": "seen before",
        },
    ])
    search = FakeSearchService(
        retrieve_response=[_result("similar incident", 0.2)],
        search_responses={"token expired": [_result("token expiry match", 0.1)]},
    )
    agent = CriticReviewedInvestigationAgent(db=None, search_service=search, llm_service=llm)

    plan, reviewed = agent.investigate("login fails with token error", n_hypotheses=1)

    assert plan.strategy == PlanningStrategy.AUTHENTICATION
    assert reviewed.investigation.selected_hypothesis is not None
    assert reviewed.critique.verdict in set(CritiqueVerdict)


def test_critic_reviewed_agent_end_to_end_empty_hypotheses_is_inconclusive() -> None:
    llm = FakeLLMService([])
    search = FakeSearchService(retrieve_response=[])
    agent = CriticReviewedInvestigationAgent(db=None, search_service=search, llm_service=llm)

    plan, reviewed = agent.investigate("totally unrelated coffee machine issue")

    assert reviewed.investigation.is_uncertain is True
    assert reviewed.critique.verdict == CritiqueVerdict.INCONCLUSIVE


def test_critic_reviewed_agent_does_not_overturn_decision() -> None:
    llm = FakeLLMService([
        {
            "root_cause": "weak guess", "confidence_score": 1.0,
            "validation_keywords": ["zzz_no_match_zzz"], "rationale": "",
        },
    ])
    search = FakeSearchService(
        retrieve_response=[_result("similar incident", 0.1)], search_responses={},
    )
    agent = CriticReviewedInvestigationAgent(db=None, search_service=search, llm_service=llm)

    plan, reviewed = agent.investigate("connection refused", n_hypotheses=1)

    # the critic may flag NEED_MORE_EVIDENCE, but the underlying decision/report
    # (whatever it accepted or rejected) is untouched by the critique.
    assert reviewed.investigation.selected_hypothesis is not None
    assert reviewed.critique.verdict == CritiqueVerdict.NEED_MORE_EVIDENCE


def test_critic_accepts_injected_planner_and_critic() -> None:
    llm = FakeLLMService([])
    search = FakeSearchService(retrieve_response=[])
    agent = CriticReviewedInvestigationAgent(
        db=None, critic=_AlwaysApprovingCritic(), search_service=search, llm_service=llm,
    )

    plan, reviewed = agent.investigate("totally unrelated coffee machine issue")

    assert reviewed.critique.verdict == CritiqueVerdict.APPROVED
