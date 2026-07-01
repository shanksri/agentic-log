from __future__ import annotations

from app.evaluation.failure_analysis import (
    CauseLevel,
    CauseStep,
    Component,
    FailureCategory,
    FailureCluster,
    FailureRecord,
    Severity,
)
from app.evaluation.recommendation_engine import Priority, generate_recommendations


def _failure(component, category, severity, subject_id) -> FailureRecord:
    return FailureRecord(
        component=component, stage="x", category=category, severity=severity,
        subject_id=subject_id, description="x", evidence=("x",), metrics_involved=(),
        cause_chain=(CauseStep(CauseLevel.SYSTEMIC, "x"),),
    )


def _cluster(component, category, severity, n: int, common_cause="x") -> FailureCluster:
    failures = tuple(
        _failure(component, category, severity, f"s{i}") for i in range(n)
    )
    return FailureCluster(
        component=component, category=category, failures=failures, severity=severity,
        common_cause=common_cause,
    )


def test_empty_clusters_produce_no_recommendations() -> None:
    assert generate_recommendations(()) == ()


def test_recommendation_fields_trace_back_to_the_cluster() -> None:
    cluster = _cluster(
        Component.PLANNER, FailureCategory.STRATEGY_MISMATCH, Severity.HIGH, 3,
        common_cause="planner rule priority ordering",
    )
    recommendations = generate_recommendations((cluster,))
    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec.root_cause == "planner rule priority ordering"
    assert rec.estimated_impact == 3
    assert rec.priority == Priority.HIGH
    assert "planner" in rec.problem
    assert "keyword priority" in rec.recommended_action


def test_confidence_increases_with_cluster_size() -> None:
    small = _cluster(Component.PLANNER, FailureCategory.STRATEGY_MISMATCH, Severity.LOW, 1)
    large = _cluster(Component.PLANNER, FailureCategory.STRATEGY_MISMATCH, Severity.LOW, 10)

    small_rec = generate_recommendations((small,))[0]
    large_rec = generate_recommendations((large,))[0]

    assert large_rec.confidence > small_rec.confidence
    assert large_rec.confidence <= 1.0


def test_priority_matches_severity() -> None:
    cluster = _cluster(Component.CRITIC, FailureCategory.INCORRECT_CRITIQUE, Severity.CRITICAL, 1)
    rec = generate_recommendations((cluster,))[0]
    assert rec.priority == Priority.CRITICAL


def test_every_failure_category_has_a_defined_action() -> None:
    for category in FailureCategory:
        cluster = _cluster(Component.JUDGE, category, Severity.LOW, 1)
        rec = generate_recommendations((cluster,))[0]
        assert rec.recommended_action
        assert "investigate judge failures of category" not in rec.recommended_action


def test_recommendations_are_sorted_by_priority_then_impact() -> None:
    low = _cluster(Component.RETRIEVAL, FailureCategory.INCOMPLETE_RECALL, Severity.LOW, 1)
    critical_small = _cluster(
        Component.PLANNER, FailureCategory.STRATEGY_MISMATCH, Severity.CRITICAL, 1
    )
    critical_large = _cluster(
        Component.CRITIC, FailureCategory.INCORRECT_CRITIQUE, Severity.CRITICAL, 5
    )

    recommendations = generate_recommendations((low, critical_small, critical_large))

    assert recommendations[0].priority == Priority.CRITICAL
    assert recommendations[0].estimated_impact == 5
    assert recommendations[1].priority == Priority.CRITICAL
    assert recommendations[1].estimated_impact == 1
    assert recommendations[2].priority == Priority.LOW


def test_recommendation_ordering_is_deterministic() -> None:
    clusters = (
        _cluster(Component.RETRIEVAL, FailureCategory.INCOMPLETE_RECALL, Severity.MEDIUM, 2),
        _cluster(Component.PLANNER, FailureCategory.STRATEGY_MISMATCH, Severity.MEDIUM, 2),
    )
    first = generate_recommendations(clusters)
    second = generate_recommendations(clusters)
    assert first == second


def test_recommendation_is_frozen() -> None:
    cluster = _cluster(Component.PLANNER, FailureCategory.STRATEGY_MISMATCH, Severity.LOW, 1)
    rec = generate_recommendations((cluster,))[0]
    import pytest

    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        rec.priority = Priority.CRITICAL  # type: ignore[misc]
