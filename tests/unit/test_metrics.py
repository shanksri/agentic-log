from __future__ import annotations

import math
import uuid

import pytest

from app.evaluation.gold_dataset import ExpectedIncident, GoldQuery
from app.evaluation.gold_loader import ResolvedExpectedIncident, ResolvedGoldQuery
from app.evaluation.metrics import (
    QueryMetricResult,
    dcg_at_k,
    ideal_dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_grades_from_resolved_gold,
    score_query,
)
from app.services.identity import ResolvedIdentity

A, B, C, D = (uuid.uuid4() for _ in range(4))


def _gain(relevance: int) -> float:
    """Independent re-derivation of the documented gain formula, used to
    compute expected values in tests without relying on the implementation
    under test.
    """
    return 2**relevance - 1


def _resolved_gold_query(
    *,
    query_id: str = "q-1",
    category: str = "lexical-overlap",
    expected: list[tuple[uuid.UUID | None, str, str, int]],
) -> ResolvedGoldQuery:
    """Build a ResolvedGoldQuery from (incident_id_or_None, source_type,
    source_external_id, relevance) tuples. incident_id=None means the
    expected incident is unresolved.
    """
    resolved_incidents = []
    expected_incidents = []
    for incident_id, source_type, source_external_id, relevance in expected:
        exp = ExpectedIncident(
            source_type=source_type, source_external_id=source_external_id, relevance=relevance
        )
        expected_incidents.append(exp)
        resolved = (
            ResolvedIdentity(
                source_type=source_type,
                source_external_id=source_external_id,
                incident_id=incident_id,
            )
            if incident_id is not None
            else None
        )
        resolved_incidents.append(ResolvedExpectedIncident(expected=exp, resolved=resolved))

    query = GoldQuery(
        id=query_id,
        query="some query",
        category=category,
        difficulty="medium",
        expected_incidents=tuple(expected_incidents),
    )
    return ResolvedGoldQuery(query=query, resolved_incidents=tuple(resolved_incidents))


def _no_match_query(query_id: str = "q-neg") -> ResolvedGoldQuery:
    query = GoldQuery(
        id=query_id,
        query="irrelevant query",
        category="no-match-expected",
        difficulty="easy",
        expected_incidents=(),
    )
    return ResolvedGoldQuery(query=query, resolved_incidents=())


# ── relevant_grades_from_resolved_gold ───────────────────────────────────────


def test_relevant_grades_excludes_unresolved_entries() -> None:
    rgq = _resolved_gold_query(
        expected=[
            (A, "github", "a", 3),
            (None, "github", "b", 2),  # unresolved
        ]
    )
    assert relevant_grades_from_resolved_gold(rgq) == {A: 3}


def test_relevant_grades_empty_for_no_match_expected() -> None:
    assert relevant_grades_from_resolved_gold(_no_match_query()) == {}


def test_relevant_grades_keeps_max_when_two_identities_collide_on_one_incident() -> None:
    rgq = _resolved_gold_query(
        expected=[
            (A, "github", "a", 1),
            (A, "jira", "a-dup", 3),
        ]
    )
    assert relevant_grades_from_resolved_gold(rgq) == {A: 3}


# ── recall_at_k ───────────────────────────────────────────────────────────────


def test_recall_at_k_perfect() -> None:
    assert recall_at_k([A, B], {A, B}, k=2) == 1.0


def test_recall_at_k_partial() -> None:
    assert recall_at_k([A, C], {A, B}, k=2) == 0.5


def test_recall_at_k_zero_hits() -> None:
    assert recall_at_k([C, D], {A, B}, k=2) == 0.0


def test_recall_at_k_none_when_no_relevant_items() -> None:
    assert recall_at_k([A, B], set(), k=5) is None


def test_recall_at_k_empty_retrieved_with_relevant_items_is_zero() -> None:
    assert recall_at_k([], {A}, k=5) == 0.0


def test_recall_at_k_k_larger_than_retrieved_count() -> None:
    assert recall_at_k([A], {A, B}, k=100) == 0.5


def test_recall_at_k_k_larger_than_relevant_count_uses_relevant_count_as_denominator() -> None:
    # k=100 but only 1 relevant item exists; denominator must stay 1, not 100.
    assert recall_at_k([A], {A}, k=100) == 1.0


def test_recall_at_k_duplicate_retrieved_ids_counted_once() -> None:
    assert recall_at_k([A, A, A], {A, B}, k=3) == 0.5


def test_recall_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k([A], {A}, k=0)


# ── reciprocal_rank ───────────────────────────────────────────────────────────


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank([A, B, C], {A}) == 1.0


def test_reciprocal_rank_later_position() -> None:
    assert reciprocal_rank([C, B, A], {A}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit_is_zero() -> None:
    assert reciprocal_rank([C, D], {A, B}) == 0.0


def test_reciprocal_rank_empty_retrieved_with_relevant_items_is_zero() -> None:
    assert reciprocal_rank([], {A}) == 0.0


def test_reciprocal_rank_none_when_no_relevant_items() -> None:
    assert reciprocal_rank([A, B], set()) is None


def test_reciprocal_rank_uses_first_occurrence_of_duplicate() -> None:
    # A appears at rank 1 and rank 3; first occurrence (rank 1) governs.
    assert reciprocal_rank([A, B, A], {A}) == 1.0


def test_reciprocal_rank_not_bounded_by_any_k() -> None:
    fillers = [uuid.uuid4() for _ in range(50)]
    long_list = [*fillers, A]
    assert reciprocal_rank(long_list, {A}) == pytest.approx(1 / 51)


# ── dcg_at_k / ideal_dcg_at_k / ndcg_at_k ─────────────────────────────────────


def test_dcg_at_k_empty_retrieved_is_zero() -> None:
    assert dcg_at_k([], {A: 3}, k=5) == 0.0


def test_dcg_at_k_no_relevant_hits_is_zero() -> None:
    assert dcg_at_k([C, D], {A: 3}, k=5) == 0.0


def test_dcg_at_k_single_hit_matches_hand_computed_formula() -> None:
    expected = _gain(3) / math.log2(2 + 1)  # rank 2
    assert dcg_at_k([C, A], {A: 3}, k=2) == pytest.approx(expected)


def test_dcg_at_k_ignores_ranks_beyond_k() -> None:
    within_k = dcg_at_k([A, C], {A: 3, C: 0}, k=1)
    assert within_k == pytest.approx(_gain(3) / math.log2(2))


def test_dcg_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        dcg_at_k([A], {A: 1}, k=0)


def test_ideal_dcg_at_k_sorts_grades_descending() -> None:
    expected = _gain(3) / math.log2(2) + _gain(1) / math.log2(3)
    # Order of input grades must not matter — ideal_dcg_at_k sorts internally.
    assert ideal_dcg_at_k([1, 3], k=2) == pytest.approx(expected)
    assert ideal_dcg_at_k([3, 1], k=2) == pytest.approx(expected)


def test_ideal_dcg_at_k_truncates_to_k() -> None:
    expected = _gain(3) / math.log2(2)  # only the top grade within k=1
    assert ideal_dcg_at_k([3, 2, 1], k=1) == pytest.approx(expected)


def test_ideal_dcg_at_k_k_larger_than_grade_count_uses_all_grades() -> None:
    expected = _gain(2) / math.log2(2) + _gain(1) / math.log2(3)
    assert ideal_dcg_at_k([2, 1], k=100) == pytest.approx(expected)


def test_ideal_dcg_at_k_empty_grades_is_zero() -> None:
    assert ideal_dcg_at_k([], k=5) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    # Relevant items occupy the very top ranks in descending-grade order —
    # this is what "perfect ranking" means for graded NDCG (an irrelevant
    # item ahead of a relevant one, even if relative order among relevant
    # items is preserved, would already reduce NDCG below 1.0).
    relevance_by_id = {A: 3, B: 1}
    retrieved = [A, B, C]
    assert ndcg_at_k(retrieved, relevance_by_id, k=3) == pytest.approx(1.0)


def test_ndcg_completely_incorrect_ranking_is_zero() -> None:
    relevance_by_id = {A: 3}
    retrieved = [C, D]
    assert ndcg_at_k(retrieved, relevance_by_id, k=2) == 0.0


def test_ndcg_partially_correct_ranking_matches_hand_computed_formula() -> None:
    relevance_by_id = {A: 3, B: 2}
    retrieved = [C, B, A]  # irrelevant first, then both relevant out of ideal order

    dcg = (
        0  # C, irrelevant
        + _gain(2) / math.log2(3)  # B at rank 2
        + _gain(3) / math.log2(4)  # A at rank 3
    )
    idcg = _gain(3) / math.log2(2) + _gain(2) / math.log2(3)  # ideal: A then B
    expected = dcg / idcg

    assert ndcg_at_k(retrieved, relevance_by_id, k=3) == pytest.approx(expected)


def test_ndcg_irrelevant_item_pushing_relevant_item_back_reduces_score() -> None:
    relevance_by_id = {A: 3, B: 1}
    perfect = ndcg_at_k([A, B], relevance_by_id, k=2)
    pushed_back = ndcg_at_k([C, A, B], relevance_by_id, k=3)
    assert pushed_back < perfect


def test_ndcg_none_when_no_relevant_items() -> None:
    assert ndcg_at_k([A, B], {}, k=5) is None


def test_ndcg_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([A], {A: 1}, k=0)


# ── score_query: end-to-end via ResolvedGoldQuery ────────────────────────────


def test_score_query_perfect_ranking() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3), (B, "github", "b", 1)])
    result = score_query([A, B, C], rgq, k=3)

    assert isinstance(result, QueryMetricResult)
    assert result.query_id == "q-1"
    assert result.k == 3
    assert result.num_relevant == 2
    assert result.num_unresolved_expected == 0
    assert result.num_retrieved == 3
    assert result.num_duplicate_retrieved == 0
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.ndcg_at_k == pytest.approx(1.0)


def test_score_query_completely_incorrect_ranking() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3)])
    result = score_query([C, D], rgq, k=2)

    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.dcg_at_k == 0.0
    assert result.idcg_at_k > 0.0
    assert result.ndcg_at_k == 0.0


def test_score_query_partially_correct_ranking() -> None:
    rgq = _resolved_gold_query(
        expected=[(A, "github", "a", 3), (B, "github", "b", 2)]
    )
    result = score_query([C, B, A], rgq, k=3)

    assert result.recall_at_k == 1.0  # both eventually within top 3
    assert result.reciprocal_rank == pytest.approx(0.5)  # first hit (B) at rank 2
    assert 0.0 < result.ndcg_at_k < 1.0


def test_score_query_multiple_relevant_incidents_with_graded_relevance() -> None:
    rgq = _resolved_gold_query(
        expected=[
            (A, "github", "a", 3),
            (B, "github", "b", 2),
            (C, "github", "c", 1),
        ]
    )
    result = score_query([A, B, C], rgq, k=3)

    assert result.num_relevant == 3
    assert result.recall_at_k == 1.0
    assert result.ndcg_at_k == pytest.approx(1.0)


def test_score_query_duplicate_retrieved_ids() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3), (B, "github", "b", 1)])
    result = score_query([A, A, A, B], rgq, k=4)

    assert result.num_retrieved == 4
    assert result.num_duplicate_retrieved == 2  # [A, A, A, B] -> [A, B], 2 dropped
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0


def test_score_query_no_match_expected_returns_none_for_undefined_metrics() -> None:
    result = score_query([A, B, C], _no_match_query(), k=5)

    assert result.num_relevant == 0
    assert result.num_unresolved_expected == 0
    assert result.recall_at_k is None
    assert result.reciprocal_rank is None
    assert result.dcg_at_k == 0.0
    assert result.idcg_at_k == 0.0
    assert result.ndcg_at_k is None


def test_score_query_all_expected_incidents_unresolved_behaves_like_no_match() -> None:
    rgq = _resolved_gold_query(
        expected=[
            (None, "github", "a", 3),
            (None, "github", "b", 2),
        ]
    )
    result = score_query([A, B], rgq, k=5)

    assert result.num_relevant == 0
    assert result.num_unresolved_expected == 2
    assert result.recall_at_k is None
    assert result.ndcg_at_k is None


def test_score_query_partially_resolved_gold_excludes_unresolved_from_scoring() -> None:
    rgq = _resolved_gold_query(
        expected=[
            (A, "github", "a", 3),  # resolved
            (None, "github", "b", 2),  # unresolved — must not count against recall
        ]
    )
    result = score_query([A], rgq, k=5)

    assert result.num_relevant == 1
    assert result.num_unresolved_expected == 1
    assert result.recall_at_k == 1.0  # perfect recall over the 1 resolvable item
    assert result.reciprocal_rank == 1.0


def test_score_query_empty_retrieval_with_relevant_items() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3)])
    result = score_query([], rgq, k=5)

    assert result.num_retrieved == 0
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.dcg_at_k == 0.0
    assert result.ndcg_at_k == 0.0


def test_score_query_retrieved_incidents_not_in_gold_set_do_not_error() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3)])
    result = score_query([C, D, A], rgq, k=3)

    assert result.num_retrieved == 3
    assert result.recall_at_k == 1.0


def test_score_query_k_larger_than_retrieved_count() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3)])
    result = score_query([A], rgq, k=100)

    assert result.k == 100
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0


def test_score_query_k_larger_than_gold_count() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3)])
    result = score_query([A, B, C], rgq, k=100)

    assert result.recall_at_k == 1.0  # denominator is 1 (num_relevant), not 100


def test_score_query_rejects_non_positive_k() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3)])
    with pytest.raises(ValueError):
        score_query([A], rgq, k=0)


def test_score_query_result_is_frozen_and_primitive_only() -> None:
    rgq = _resolved_gold_query(expected=[(A, "github", "a", 3)])
    result = score_query([A], rgq, k=5)

    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        result.recall_at_k = 0.0  # type: ignore[misc]

    for value in (
        result.query_id,
        result.k,
        result.num_relevant,
        result.num_unresolved_expected,
        result.num_retrieved,
        result.num_duplicate_retrieved,
        result.dcg_at_k,
        result.idcg_at_k,
    ):
        assert isinstance(value, (str, int, float))
