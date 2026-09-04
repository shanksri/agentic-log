"""Tests for cross-encoder relevance scoring in evidence evaluation.

Covers the ``HypothesisEvaluator`` support/contradiction split and the
``CrossEncoderRelevanceScorer`` wrapper. No model is ever loaded here -- the
scorer is injected as a fake, and the cache tests assert on construction
rather than on encoding.

Background: the legacy split thresholded cosine similarity to the
*hypothesis's own keywords*, so a hypothesis always found agreement with
itself. Across a 34-investigation probe that produced 78 supporting items
against 2 contradicting, and the critic's ``rejected`` verdict never fired.
These tests pin the replacement and, just as importantly, pin that the old
behavior is still what you get until the flag is switched on.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.hypothesis_investigation import (
    HypothesisEvaluator,
    InvestigationHypothesis,
    _incident_passage,
)
from app.services.relevance_scorer import (
    CrossEncoderRelevanceScorer,
    RelevanceScorerError,
    get_relevance_scorer,
    reset_relevance_scorer_cache,
)
from app.services.search import IncidentSearchResult

PROBLEM = "ZookeeperConsumerConnectorMBean needs to close SimpleConsumer when done"


def _result(title: str, distance: float, symptoms=()) -> IncidentSearchResult:
    incident = SimpleNamespace(
        title=title, symptoms=[SimpleNamespace(text=s) for s in symptoms]
    )
    return IncidentSearchResult(incident=incident, distance=distance)


class FakeSearchService:
    def __init__(self, results):
        self._results = results

    def search(self, query, *, limit=10, call_site=None):
        return self._results


class FakeScorer:
    """Returns canned logits in order, and records what it was asked."""

    def __init__(self, scores, *, raises=False):
        self._scores = scores
        self.raises = raises
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        if self.raises:
            raise RelevanceScorerError("boom")
        return list(self._scores)


def _hypothesis(keywords=("network", "latency")):
    return InvestigationHypothesis(
        id="h1", root_cause="High network latency", rationale="because",
        validation_keywords=tuple(keywords), raw_confidence=0.8,
    )


# The real measured case: five incidents, cosine 0.525-0.626 so all five clear
# the 0.40 cosine floor, but only the first genuinely answers the problem.
_REAL_CASE = [
    ("ZookeeperConsumerConnectorMBean needs to close SimpleConsumer", 0.626, 9.81),
    ("avoid reimplementing ZookeeperConsumerConnectorMBean in javaapi", 0.560, 0.73),
    ("Provide aggregate stats at high level Producer and ZookeeperCo", 0.557, -2.95),
    ("ConsoleConsumer throws SocketTimeoutException when fetching", 0.550, -8.11),
    ("Producer perf test fails against localhost with > 10 threads", 0.525, -11.44),
]


@pytest.fixture
def real_case_evaluator():
    results = [_result(t, 1.0 - cos) for t, cos, _ in _REAL_CASE]
    scorer = FakeScorer([rel for _, _, rel in _REAL_CASE])
    return HypothesisEvaluator(FakeSearchService(results), relevance_scorer=scorer), scorer


# ── The flag gates the behavior ────────────────────────────────────────────────


def test_disabled_by_default_keeps_the_legacy_cosine_split(real_case_evaluator) -> None:
    evaluator, scorer = real_case_evaluator
    out = evaluator.evaluate(_hypothesis(), problem=PROBLEM)
    assert len(out.supporting_evidence) == 5
    assert out.contradicting_evidence == ()
    assert scorer.calls == [], "scorer must not be consulted while the flag is off"


def test_enabled_splits_on_relevance_not_cosine(real_case_evaluator, monkeypatch) -> None:
    monkeypatch.setattr("app.services.hypothesis_investigation.settings",
                        SimpleNamespace(evidence_relevance_enabled=True, relevance_threshold=0.0))
    evaluator, _ = real_case_evaluator
    out = evaluator.evaluate(_hypothesis(), problem=PROBLEM)
    # Only the two above the 0.0 boundary survive; the SocketTimeoutException
    # incident that fed the bogus network story is now contradicting.
    assert len(out.supporting_evidence) == 2
    assert len(out.contradicting_evidence) == 3
    assert "SocketTimeoutException" in " ".join(out.contradicting_evidence)


def test_relevance_is_scored_against_the_problem_not_the_hypothesis(
    real_case_evaluator, monkeypatch
) -> None:
    """The whole point of the change: the query must be the original problem."""
    monkeypatch.setattr("app.services.hypothesis_investigation.settings",
                        SimpleNamespace(evidence_relevance_enabled=True, relevance_threshold=0.0))
    evaluator, scorer = real_case_evaluator
    evaluator.evaluate(_hypothesis(), problem=PROBLEM)
    query, passages = scorer.calls[0]
    assert query == PROBLEM
    assert "High network latency" not in query
    assert len(passages) == 5


# ── Fallbacks ──────────────────────────────────────────────────────────────────


def test_no_problem_supplied_falls_back_to_cosine(real_case_evaluator, monkeypatch) -> None:
    monkeypatch.setattr("app.services.hypothesis_investigation.settings",
                        SimpleNamespace(evidence_relevance_enabled=True, relevance_threshold=0.0))
    evaluator, scorer = real_case_evaluator
    out = evaluator.evaluate(_hypothesis())
    assert len(out.supporting_evidence) == 5
    assert scorer.calls == []


def test_scorer_failure_degrades_to_cosine_instead_of_raising(monkeypatch) -> None:
    monkeypatch.setattr("app.services.hypothesis_investigation.settings",
                        SimpleNamespace(evidence_relevance_enabled=True, relevance_threshold=0.0))
    results = [_result(t, 1.0 - cos) for t, cos, _ in _REAL_CASE]
    scorer = FakeScorer([], raises=True)
    evaluator = HypothesisEvaluator(FakeSearchService(results), relevance_scorer=scorer)
    out = evaluator.evaluate(_hypothesis(), problem=PROBLEM)
    assert len(out.supporting_evidence) == 5, "must fall back, not fail the investigation"


def test_no_results_reports_missing_evidence_without_scoring(monkeypatch) -> None:
    monkeypatch.setattr("app.services.hypothesis_investigation.settings",
                        SimpleNamespace(evidence_relevance_enabled=True, relevance_threshold=0.0))
    scorer = FakeScorer([])
    evaluator = HypothesisEvaluator(FakeSearchService([]), relevance_scorer=scorer)
    out = evaluator.evaluate(_hypothesis(), problem=PROBLEM)
    assert out.supporting_evidence == () and out.contradicting_evidence == ()
    assert len(out.missing_evidence) == 1
    assert scorer.calls == []


# ── Threshold semantics ────────────────────────────────────────────────────────


def test_threshold_is_inclusive_at_the_boundary(monkeypatch) -> None:
    monkeypatch.setattr("app.services.hypothesis_investigation.settings",
                        SimpleNamespace(evidence_relevance_enabled=True, relevance_threshold=0.0))
    results = [_result("exactly at the boundary", 0.5)]
    evaluator = HypothesisEvaluator(
        FakeSearchService(results), relevance_scorer=FakeScorer([0.0])
    )
    out = evaluator.evaluate(_hypothesis(), problem=PROBLEM)
    assert len(out.supporting_evidence) == 1


def test_raising_the_threshold_shrinks_supporting_evidence(monkeypatch) -> None:
    monkeypatch.setattr("app.services.hypothesis_investigation.settings",
                        SimpleNamespace(evidence_relevance_enabled=True, relevance_threshold=5.0))
    results = [_result(t, 1.0 - cos) for t, cos, _ in _REAL_CASE]
    evaluator = HypothesisEvaluator(
        FakeSearchService(results),
        relevance_scorer=FakeScorer([rel for _, _, rel in _REAL_CASE]),
    )
    out = evaluator.evaluate(_hypothesis(), problem=PROBLEM)
    assert len(out.supporting_evidence) == 1


def test_relevance_score_is_recorded_in_the_evidence_text(monkeypatch) -> None:
    """Explainability: a reader must be able to see why an item was counted."""
    monkeypatch.setattr("app.services.hypothesis_investigation.settings",
                        SimpleNamespace(evidence_relevance_enabled=True, relevance_threshold=0.0))
    results = [_result("some incident", 0.4)]
    evaluator = HypothesisEvaluator(
        FakeSearchService(results), relevance_scorer=FakeScorer([7.25])
    )
    out = evaluator.evaluate(_hypothesis(), problem=PROBLEM)
    assert "relevance=7.25" in out.supporting_evidence[0]
    assert "similarity=" in out.supporting_evidence[0]


# ── Passage construction ───────────────────────────────────────────────────────


def test_passage_includes_title_and_symptoms() -> None:
    passage = _incident_passage(_result("Pod evicted", 0.5, symptoms=("kubelet restarted",)))
    assert passage == "Pod evicted kubelet restarted"


def test_passage_without_symptoms_has_no_trailing_space() -> None:
    assert _incident_passage(_result("Pod evicted", 0.5)) == "Pod evicted"


# ── Scorer wrapper ─────────────────────────────────────────────────────────────


def test_empty_passages_short_circuit_without_loading_a_model() -> None:
    scorer = CrossEncoderRelevanceScorer("this-model-does-not-exist")
    assert scorer.score("q", []) == []


def test_model_load_failure_raises_the_typed_error() -> None:
    scorer = CrossEncoderRelevanceScorer("definitely/not-a-real-model-xyz")
    with pytest.raises(RelevanceScorerError, match="Failed to load relevance model"):
        _ = scorer.model


def test_cache_returns_the_same_instance_and_reset_clears_it() -> None:
    reset_relevance_scorer_cache()
    try:
        first = get_relevance_scorer()
        assert get_relevance_scorer() is first
        reset_relevance_scorer_cache()
        assert get_relevance_scorer() is not first
    finally:
        reset_relevance_scorer_cache()
