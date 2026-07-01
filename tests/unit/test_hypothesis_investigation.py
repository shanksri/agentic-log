from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    composite_hypothesis_confidence,
)
from app.services.hypothesis_investigation import (
    ACCEPTANCE_COMPOSITE_FLOOR,
    EvidenceEvaluation,
    HypothesisDrivenInvestigationAgent,
    HypothesisEvaluator,
    HypothesisGenerator,
    HypothesisScore,
    InvestigationHypothesis,
    build_investigation_report,
    make_investigation_decision,
    score_hypothesis,
)
from app.services.search import IncidentSearchResult

# ── Fakes ──────────────────────────────────────────────────────────────────────


def _incident(title: str, symptoms=()):
    return SimpleNamespace(title=title, symptoms=[SimpleNamespace(text=s) for s in symptoms])


def _result(title: str, distance: float, symptoms=()) -> IncidentSearchResult:
    return IncidentSearchResult(incident=_incident(title, symptoms), distance=distance)


class FakeLLMService:
    def __init__(self, hypotheses_by_call=None):
        self._hypotheses_by_call = hypotheses_by_call or []
        self._call_index = 0
        self.calls: list[dict] = []

    def generate_hypotheses(self, *, problem, context, n=2, existing_root_causes=None):
        self.calls.append(
            {
                "problem": problem, "context": context, "n": n,
                "existing_root_causes": existing_root_causes,
            }
        )
        if self._call_index < len(self._hypotheses_by_call):
            result = self._hypotheses_by_call[self._call_index]
        else:
            result = []
        self._call_index += 1
        return result


class FakeSearchService:
    def __init__(self, *, retrieve_response=None, search_responses=None):
        self._retrieve_response = retrieve_response or []
        self._search_responses = search_responses or {}
        self.search_calls: list[dict] = []
        self.retrieve_calls: list[dict] = []

    def retrieve(self, query, *, limit=10, expand=False, rerank=False, call_site=None):
        self.retrieve_calls.append({"query": query, "limit": limit})
        return self._retrieve_response

    def search(self, query, *, limit=10, call_site=None):
        self.search_calls.append({"query": query, "limit": limit, "call_site": call_site})
        return self._search_responses.get(query, [])


def _hypothesis(id_="h1", root_cause="cause", raw_confidence=0.8, keywords=("kw1",)):
    return InvestigationHypothesis(
        id=id_, root_cause=root_cause, rationale="because", validation_keywords=tuple(keywords),
        raw_confidence=raw_confidence,
    )


def _evaluation(hypothesis_id="h1", *, supporting=(), contradicting=(), missing=(),
                 evidence_confidence_level=CONFIDENCE_HIGH, top1=0.9):
    return EvidenceEvaluation(
        hypothesis_id=hypothesis_id, query="q", supporting_evidence=supporting,
        contradicting_evidence=contradicting, missing_evidence=missing,
        evidence_confidence_level=evidence_confidence_level, evidence_top1_score=top1,
    )


# ── HypothesisGenerator ──────────────────────────────────────────────────────────


def test_generate_converts_raw_dicts_to_hypotheses() -> None:
    llm = FakeLLMService([[{
        "root_cause": "memory leak", "confidence_score": 0.7,
        "validation_keywords": ["memory", "leak"], "rationale": "saw it before",
    }]])
    generator = HypothesisGenerator(llm)

    [hypothesis] = generator.generate(problem="p", context="c", n=1)

    assert hypothesis.id == "h1"
    assert hypothesis.root_cause == "memory leak"
    assert hypothesis.raw_confidence == pytest.approx(0.7)
    assert hypothesis.validation_keywords == ("memory", "leak")
    assert hypothesis.rationale == "saw it before"


def test_generate_assigns_sequential_ids() -> None:
    llm = FakeLLMService([[
        {"root_cause": "a", "confidence_score": 0.5, "validation_keywords": [], "rationale": ""},
        {"root_cause": "b", "confidence_score": 0.5, "validation_keywords": [], "rationale": ""},
    ]])
    generator = HypothesisGenerator(llm)

    hypotheses = generator.generate(problem="p", context="c", n=2)

    assert [h.id for h in hypotheses] == ["h1", "h2"]


def test_generate_coerces_non_list_keywords_to_single_item_tuple() -> None:
    llm = FakeLLMService([[
        {
            "root_cause": "a", "confidence_score": 0.5,
            "validation_keywords": "single", "rationale": "",
        },
    ]])
    generator = HypothesisGenerator(llm)

    [hypothesis] = generator.generate(problem="p", context="c", n=1)

    assert hypothesis.validation_keywords == ("single",)


def test_generate_clamps_out_of_range_confidence() -> None:
    llm = FakeLLMService([[
        {"root_cause": "a", "confidence_score": 5.0, "validation_keywords": [], "rationale": ""},
    ]])
    generator = HypothesisGenerator(llm)

    [hypothesis] = generator.generate(problem="p", context="c", n=1)

    assert hypothesis.raw_confidence == 1.0


def test_generate_invalid_confidence_defaults_to_zero() -> None:
    llm = FakeLLMService([[
        {
            "root_cause": "a", "confidence_score": "not-a-number",
            "validation_keywords": [], "rationale": "",
        },
    ]])
    generator = HypothesisGenerator(llm)

    [hypothesis] = generator.generate(problem="p", context="c", n=1)

    assert hypothesis.raw_confidence == 0.0


def test_generate_empty_response_returns_empty_tuple() -> None:
    llm = FakeLLMService([[]])
    generator = HypothesisGenerator(llm)

    assert generator.generate(problem="p", context="c", n=3) == ()


def test_generate_passes_existing_root_causes_through() -> None:
    llm = FakeLLMService([[]])
    generator = HypothesisGenerator(llm)

    generator.generate(problem="p", context="c", n=2, existing_root_causes=["x", "y"])

    assert llm.calls[0]["existing_root_causes"] == ["x", "y"]


# ── HypothesisEvaluator ──────────────────────────────────────────────────────────


def test_evaluate_classifies_high_similarity_as_supporting() -> None:
    search = FakeSearchService(search_responses={"kw1": [_result("match", 0.1)]})
    evaluator = HypothesisEvaluator(search)

    evaluation = evaluator.evaluate(_hypothesis(keywords=("kw1",)))

    assert evaluation.supporting_evidence == ("match (similarity=0.900)",)
    assert evaluation.contradicting_evidence == ()
    assert evaluation.missing_evidence == ()


def test_evaluate_classifies_low_similarity_as_contradicting() -> None:
    search = FakeSearchService(search_responses={"kw1": [_result("weak", 0.9)]})  # similarity=0.1
    evaluator = HypothesisEvaluator(search)

    evaluation = evaluator.evaluate(_hypothesis(keywords=("kw1",)))

    assert evaluation.contradicting_evidence == ("weak (similarity=0.100)",)
    assert evaluation.supporting_evidence == ()


def test_evaluate_boundary_similarity_exactly_at_threshold_is_supporting() -> None:
    # similarity_score = 1 - distance = 0.40 exactly == LOW_CONFIDENCE_THRESHOLD
    search = FakeSearchService(search_responses={"kw1": [_result("boundary", 0.6)]})
    evaluator = HypothesisEvaluator(search)

    evaluation = evaluator.evaluate(_hypothesis(keywords=("kw1",)))

    assert evaluation.supporting_evidence == ("boundary (similarity=0.400)",)
    assert evaluation.contradicting_evidence == ()


def test_evaluate_no_results_is_missing_evidence() -> None:
    search = FakeSearchService(search_responses={})
    evaluator = HypothesisEvaluator(search)

    evaluation = evaluator.evaluate(_hypothesis(keywords=("kw1",)))

    assert evaluation.missing_evidence == ("no incidents found for validation query 'kw1'",)
    assert evaluation.supporting_evidence == ()
    assert evaluation.contradicting_evidence == ()
    assert evaluation.evidence_confidence_level == CONFIDENCE_LOW


def test_evaluate_falls_back_to_root_cause_when_no_keywords() -> None:
    search = FakeSearchService(search_responses={"the root cause": [_result("x", 0.1)]})
    evaluator = HypothesisEvaluator(search)

    evaluator.evaluate(_hypothesis(root_cause="the root cause", keywords=()))

    assert search.search_calls[0]["query"] == "the root cause"


def test_evaluate_joins_multiple_keywords_into_one_query() -> None:
    search = FakeSearchService(search_responses={"alpha beta": []})
    evaluator = HypothesisEvaluator(search)

    evaluator.evaluate(_hypothesis(keywords=("alpha", "beta")))

    assert search.search_calls[0]["query"] == "alpha beta"


def test_evaluate_mixed_results_split_correctly() -> None:
    search = FakeSearchService(
        search_responses={"kw1": [_result("strong", 0.1), _result("weak", 0.9)]}
    )
    evaluator = HypothesisEvaluator(search)

    evaluation = evaluator.evaluate(_hypothesis(keywords=("kw1",)))

    assert evaluation.supporting_evidence == ("strong (similarity=0.900)",)
    assert evaluation.contradicting_evidence == ("weak (similarity=0.100)",)


# ── score_hypothesis ─────────────────────────────────────────────────────────────


def test_score_hypothesis_matches_composite_hypothesis_confidence_directly() -> None:
    hypothesis = _hypothesis(raw_confidence=0.8)
    evaluation = _evaluation(evidence_confidence_level=CONFIDENCE_HIGH)

    score = score_hypothesis(hypothesis, evaluation, retrieval_confidence_level=CONFIDENCE_MEDIUM)

    expected = composite_hypothesis_confidence(
        raw_confidence=0.8, retrieval_confidence_level=CONFIDENCE_MEDIUM,
        validation_keyword_recall_ok=True,
    )
    assert score.composite_score == pytest.approx(expected)


def test_score_hypothesis_low_evidence_confidence_means_keyword_recall_not_ok() -> None:
    hypothesis = _hypothesis(raw_confidence=0.8)
    evaluation = _evaluation(evidence_confidence_level=CONFIDENCE_LOW)

    score = score_hypothesis(hypothesis, evaluation, retrieval_confidence_level=CONFIDENCE_HIGH)

    expected = composite_hypothesis_confidence(
        raw_confidence=0.8, retrieval_confidence_level=CONFIDENCE_HIGH,
        validation_keyword_recall_ok=False,
    )
    assert score.composite_score == pytest.approx(expected)


def test_score_hypothesis_records_evidence_counts() -> None:
    hypothesis = _hypothesis()
    evaluation = _evaluation(supporting=("a", "b"), contradicting=("c",), missing=())

    score = score_hypothesis(hypothesis, evaluation, retrieval_confidence_level=CONFIDENCE_HIGH)

    assert score.supporting_count == 2
    assert score.contradicting_count == 1
    assert score.missing_count == 0


# ── make_investigation_decision ──────────────────────────────────────────────────


def test_decision_no_hypotheses_is_uncertain() -> None:
    decision = make_investigation_decision(())
    assert decision.is_uncertain is True
    assert decision.accepted is None
    assert decision.rejected == ()


def test_decision_single_eligible_hypothesis_is_accepted() -> None:
    h = _hypothesis()
    score = HypothesisScore(
        hypothesis_id=h.id, raw_confidence=0.9, retrieval_confidence_level=CONFIDENCE_HIGH,
        evidence_confidence_level=CONFIDENCE_HIGH, supporting_count=1, contradicting_count=0,
        missing_count=0, composite_score=ACCEPTANCE_COMPOSITE_FLOOR + 0.1,
    )

    decision = make_investigation_decision([(h, score)])

    assert decision.is_uncertain is False
    assert decision.accepted is h
    assert decision.accepted_score is score
    assert decision.rejected == ()


def test_decision_none_eligible_is_uncertain_with_all_rejected() -> None:
    h1, h2 = _hypothesis("h1"), _hypothesis("h2")
    s1 = HypothesisScore("h1", 0.5, CONFIDENCE_LOW, CONFIDENCE_LOW, 0, 1, 0, 0.2)
    s2 = HypothesisScore("h2", 0.5, CONFIDENCE_LOW, CONFIDENCE_LOW, 0, 1, 0, 0.3)

    decision = make_investigation_decision([(h1, s1), (h2, s2)])

    assert decision.is_uncertain is True
    assert decision.accepted is None
    assert {h for h, _ in decision.rejected} == {h1, h2}


def test_decision_picks_highest_composite_score_among_eligible() -> None:
    h1, h2, h3 = _hypothesis("h1"), _hypothesis("h2"), _hypothesis("h3")
    s1 = HypothesisScore("h1", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.65)
    s2 = HypothesisScore("h2", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.90)
    s3 = HypothesisScore("h3", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.10)

    decision = make_investigation_decision([(h1, s1), (h2, s2), (h3, s3)])

    assert decision.accepted is h2
    assert {h for h, _ in decision.rejected} == {h1, h3}


def test_decision_only_one_hypothesis_accepted_even_if_multiple_eligible() -> None:
    h1, h2 = _hypothesis("h1"), _hypothesis("h2")
    s1 = HypothesisScore("h1", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.70)
    s2 = HypothesisScore("h2", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.80)

    decision = make_investigation_decision([(h1, s1), (h2, s2)])

    accepted_ids = [decision.accepted.id]
    rejected_ids = [h.id for h, _ in decision.rejected]
    assert accepted_ids == ["h2"]
    assert rejected_ids == ["h1"]


def test_decision_tie_broken_by_generation_order() -> None:
    h1, h2 = _hypothesis("h1"), _hypothesis("h2")
    s1 = HypothesisScore("h1", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.70)
    s2 = HypothesisScore("h2", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.70)

    decision = make_investigation_decision([(h1, s1), (h2, s2)])

    assert decision.accepted.id == "h1"  # first of the tied pair wins


# ── build_investigation_report ───────────────────────────────────────────────────


def test_report_accepted_case_copies_evidence_from_accepted_hypothesis() -> None:
    h = _hypothesis("h1")
    score = HypothesisScore("h1", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.75)
    decision = make_investigation_decision([(h, score)])
    evaluation = _evaluation("h1", supporting=("inc-a",), contradicting=("inc-b",), missing=())

    report = build_investigation_report("the problem", decision, {"h1": evaluation})

    assert report.is_uncertain is False
    assert report.selected_hypothesis is h
    assert report.confidence == pytest.approx(0.75)
    assert report.supporting_evidence == ("inc-a",)
    assert report.contradicting_evidence == ("inc-b",)


def test_report_uncertain_case_has_no_selected_hypothesis() -> None:
    h1 = _hypothesis("h1")
    score = HypothesisScore("h1", 0.5, CONFIDENCE_LOW, CONFIDENCE_LOW, 0, 0, 1, 0.1)
    decision = make_investigation_decision([(h1, score)])

    report = build_investigation_report("p", decision, {"h1": _evaluation("h1")})

    assert report.is_uncertain is True
    assert report.selected_hypothesis is None
    assert report.confidence == 0.0
    assert report.supporting_evidence == ()
    assert report.contradicting_evidence == ()
    assert len(report.remaining_uncertainty) >= 1


def test_report_remaining_uncertainty_mentions_rejected_alternatives() -> None:
    h1, h2 = _hypothesis("h1", root_cause="cause-a"), _hypothesis("h2", root_cause="cause-b")
    s1 = HypothesisScore("h1", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.80)
    s2 = HypothesisScore("h2", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 1, 0, 0, 0.65)
    decision = make_investigation_decision([(h1, s1), (h2, s2)])
    evaluations = {"h1": _evaluation("h1"), "h2": _evaluation("h2")}

    report = build_investigation_report("p", decision, evaluations)

    assert any("cause-b" in entry for entry in report.remaining_uncertainty)
    assert report.rejected_hypotheses == (h2,)


def test_report_includes_missing_evidence_in_remaining_uncertainty() -> None:
    h = _hypothesis("h1")
    score = HypothesisScore("h1", 0.9, CONFIDENCE_HIGH, CONFIDENCE_HIGH, 0, 0, 1, 0.75)
    decision = make_investigation_decision([(h, score)])
    evaluation = _evaluation("h1", missing=("no incidents found",))

    report = build_investigation_report("p", decision, {"h1": evaluation})

    assert "no incidents found" in report.remaining_uncertainty


# ── HypothesisDrivenInvestigationAgent (end-to-end with fakes) ──────────────────


def test_agent_end_to_end_produces_accepted_report() -> None:
    initial_results = [_result("seen before", 0.2)]
    llm = FakeLLMService([[
        {
            "root_cause": "leak", "confidence_score": 0.95,
            "validation_keywords": ["leak"], "rationale": "r",
        },
        {
            "root_cause": "timeout", "confidence_score": 0.3,
            "validation_keywords": ["timeout"], "rationale": "r",
        },
    ]])
    search = FakeSearchService(
        retrieve_response=initial_results,
        search_responses={"leak": [_result("leak match", 0.1)], "timeout": []},
    )
    agent = HypothesisDrivenInvestigationAgent(db=None, search_service=search, llm_service=llm)

    report = agent.investigate("things are broken", n_hypotheses=2)

    assert isinstance(report.confidence, float)
    assert report.selected_hypothesis is not None
    assert report.selected_hypothesis.root_cause == "leak"  # higher confidence + strong evidence


def test_agent_no_hypotheses_generated_is_uncertain() -> None:
    llm = FakeLLMService([[]])
    search = FakeSearchService(retrieve_response=[])
    agent = HypothesisDrivenInvestigationAgent(db=None, search_service=search, llm_service=llm)

    report = agent.investigate("things are broken")

    assert report.is_uncertain is True
    assert report.selected_hypothesis is None


def test_agent_calls_retrieve_with_expand_and_rerank() -> None:
    llm = FakeLLMService([[]])
    search = FakeSearchService(retrieve_response=[])
    agent = HypothesisDrivenInvestigationAgent(db=None, search_service=search, llm_service=llm)

    agent.investigate("problem text")

    assert search.retrieve_calls == [{"query": "problem text", "limit": 10}]


def test_agent_evaluates_every_hypothesis_independently() -> None:
    llm = FakeLLMService([[
        {
            "root_cause": "a", "confidence_score": 0.5,
            "validation_keywords": ["alpha"], "rationale": "",
        },
        {
            "root_cause": "b", "confidence_score": 0.5,
            "validation_keywords": ["beta"], "rationale": "",
        },
    ]])
    search = FakeSearchService(retrieve_response=[], search_responses={"alpha": [], "beta": []})
    agent = HypothesisDrivenInvestigationAgent(db=None, search_service=search, llm_service=llm)

    agent.investigate("p", n_hypotheses=2)

    queried = {call["query"] for call in search.search_calls}
    assert queried == {"alpha", "beta"}


# ── Immutability ──────────────────────────────────────────────────────────────────


def test_hypothesis_is_frozen() -> None:
    h = _hypothesis()
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        h.root_cause = "changed"  # type: ignore[misc]
