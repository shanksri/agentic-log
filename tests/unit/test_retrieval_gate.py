"""Tests for the cross-encoder retrieval confidence gate.

The gate answers "did retrieval surface anything genuinely relevant to this
problem" before any LLM call is made. Its whole reason to exist is the
measured finding that top-1 cosine cannot answer that: on 56 gold queries,
cosine at 0.40 admitted 18 of 20 hard negatives, a 0.900 false-accept rate,
while the cross-encoder at 2.0 admitted 3 of 20 at higher recall.

Two properties matter more than the threshold itself and are pinned here:
the gate is OFF unless explicitly enabled, and it fails OPEN so a broken
relevance backend degrades to today's behavior rather than refusing every
query.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.relevance_scorer import (
    RelevanceScorerError,
    RetrievalGateDecision,
    evaluate_retrieval_gate,
)

PROBLEM = "MemoryQoS does not set memory.high for BestEffort pods on cgroup v2"
PASSAGES = ["some incident", "another incident"]


class FakeScorer:
    def __init__(self, scores, *, raises=False):
        self._scores = scores
        self.raises = raises
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        if self.raises:
            raise RelevanceScorerError("backend down")
        return list(self._scores)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(
        "app.services.relevance_scorer.settings",
        SimpleNamespace(retrieval_gate_enabled=True, retrieval_gate_threshold=2.0),
    )


# ── Off by default ─────────────────────────────────────────────────────────────


def test_disabled_accepts_without_consulting_the_scorer() -> None:
    scorer = FakeScorer([-99.0])
    decision = evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=scorer)
    assert decision.accepted is True
    assert decision.score is None
    assert "disabled" in decision.reason
    assert scorer.calls == [], "a disabled gate must not load or call a model"


# ── Accept / reject ────────────────────────────────────────────────────────────


def test_accepts_when_best_score_clears_the_threshold(enabled) -> None:
    decision = evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=FakeScorer([-8.0, 9.81]))
    assert decision.accepted is True
    assert decision.score == 9.81
    assert "9.81" in decision.reason and "at or above" in decision.reason


def test_rejects_when_every_passage_is_below_the_threshold(enabled) -> None:
    decision = evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=FakeScorer([-8.0, 0.73]))
    assert decision.accepted is False
    assert decision.score == 0.73
    assert "below" in decision.reason


def test_uses_the_best_score_not_the_first_or_the_mean(enabled) -> None:
    decision = evaluate_retrieval_gate(
        PROBLEM, ["a", "b", "c"], scorer=FakeScorer([-11.0, 5.0, -9.0])
    )
    assert decision.accepted is True and decision.score == 5.0


def test_threshold_is_inclusive_at_the_boundary(enabled) -> None:
    assert evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=FakeScorer([2.0])).accepted


def test_explicit_threshold_argument_overrides_settings(enabled) -> None:
    scorer = FakeScorer([3.0])
    assert evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=scorer).accepted
    tightened = evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=scorer, threshold=5.0)
    assert tightened.accepted is False
    assert tightened.threshold == 5.0


def test_the_query_scored_is_the_problem_statement(enabled) -> None:
    scorer = FakeScorer([5.0])
    evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=scorer)
    assert scorer.calls[0][0] == PROBLEM


# ── Fails open ─────────────────────────────────────────────────────────────────


def test_scorer_failure_accepts_rather_than_blocking_every_query(enabled) -> None:
    decision = evaluate_retrieval_gate(
        PROBLEM, PASSAGES, scorer=FakeScorer([], raises=True)
    )
    assert decision.accepted is True
    assert decision.score is None
    assert "failed open" in decision.reason


def test_no_passages_accepts_without_scoring(enabled) -> None:
    scorer = FakeScorer([5.0])
    decision = evaluate_retrieval_gate(PROBLEM, [], scorer=scorer)
    assert decision.accepted is True
    assert scorer.calls == []


def test_scorer_returning_nothing_accepts(enabled) -> None:
    decision = evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=FakeScorer([]))
    assert decision.accepted is True
    assert decision.score is None


# ── Decision record ────────────────────────────────────────────────────────────


def test_decision_is_immutable(enabled) -> None:
    decision = evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=FakeScorer([5.0]))
    assert isinstance(decision, RetrievalGateDecision)
    with pytest.raises(Exception):
        decision.accepted = False  # type: ignore[misc]


def test_reason_always_explains_the_outcome(enabled) -> None:
    for scores in ([9.0], [0.1], []):
        decision = evaluate_retrieval_gate(PROBLEM, PASSAGES, scorer=FakeScorer(scores))
        assert decision.reason.strip(), "every decision must carry a human-readable reason"
