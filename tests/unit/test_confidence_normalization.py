from __future__ import annotations

import pytest

from app.services.confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    classify_confidence,
)
from app.services.confidence_normalization import (
    BM25_MIDPOINT,
    HYBRID_MIDPOINT,
    BM25ConfidenceNormalizer,
    ConfidenceNormalizer,
    DenseConfidenceNormalizer,
    HybridConfidenceNormalizer,
    NormalizedConfidence,
    get_confidence_normalizer,
    normalize_confidence,
    register_confidence_normalizer,
)
from app.services.routing import RoutingStrategy

# ── Dense normalization ──────────────────────────────────────────────────────────


def test_dense_normalize_zero_distance_is_full_confidence() -> None:
    result = DenseConfidenceNormalizer().normalize(0.0)
    assert result.value == pytest.approx(1.0)
    assert result.level == CONFIDENCE_HIGH


def test_dense_normalize_distance_one_is_zero_confidence() -> None:
    result = DenseConfidenceNormalizer().normalize(1.0)
    assert result.value == pytest.approx(0.0)
    assert result.level == CONFIDENCE_LOW


def test_dense_normalize_matches_similarity_score_formula() -> None:
    result = DenseConfidenceNormalizer().normalize(0.3)
    assert result.value == pytest.approx(0.7)


def test_dense_normalize_none_raw_score_is_zero() -> None:
    result = DenseConfidenceNormalizer().normalize(None)
    assert result.value == 0.0
    assert result.level == CONFIDENCE_LOW


def test_dense_normalize_clamps_distance_greater_than_one() -> None:
    result = DenseConfidenceNormalizer().normalize(1.5)
    assert result.value == 0.0


def test_dense_normalize_clamps_negative_distance() -> None:
    result = DenseConfidenceNormalizer().normalize(-0.2)
    assert result.value == 1.0


def test_dense_normalizer_strategy_label() -> None:
    assert DenseConfidenceNormalizer().strategy == RoutingStrategy.DENSE


# ── BM25 normalization ───────────────────────────────────────────────────────────


def test_bm25_normalize_zero_score_is_zero_confidence() -> None:
    result = BM25ConfidenceNormalizer().normalize(0.0)
    assert result.value == 0.0
    assert result.level == CONFIDENCE_LOW


def test_bm25_normalize_at_midpoint_is_half() -> None:
    result = BM25ConfidenceNormalizer().normalize(BM25_MIDPOINT)
    assert result.value == pytest.approx(0.5)


def test_bm25_normalize_large_score_approaches_but_never_reaches_one() -> None:
    result = BM25ConfidenceNormalizer().normalize(10_000.0)
    assert result.value < 1.0
    assert result.value > 0.99


def test_bm25_normalize_negative_score_treated_as_zero() -> None:
    result = BM25ConfidenceNormalizer().normalize(-5.0)
    assert result.value == 0.0


def test_bm25_normalize_none_raw_score_is_zero() -> None:
    result = BM25ConfidenceNormalizer().normalize(None)
    assert result.value == 0.0
    assert result.level == CONFIDENCE_LOW


def test_bm25_normalize_is_monotonic() -> None:
    low = BM25ConfidenceNormalizer().normalize(1.0).value
    high = BM25ConfidenceNormalizer().normalize(8.0).value
    assert high > low


def test_bm25_normalizer_strategy_label() -> None:
    assert BM25ConfidenceNormalizer().strategy == RoutingStrategy.BM25


# ── Hybrid normalization ──────────────────────────────────────────────────────────


def test_hybrid_normalize_zero_score_is_zero_confidence() -> None:
    result = HybridConfidenceNormalizer().normalize(0.0)
    assert result.value == 0.0


def test_hybrid_normalize_at_midpoint_is_half() -> None:
    result = HybridConfidenceNormalizer().normalize(HYBRID_MIDPOINT)
    assert result.value == pytest.approx(0.5)


def test_hybrid_normalize_typical_rrf_single_source_score() -> None:
    # rank-1, single retriever, default rrf_k=60 (Phase 17B): 1/61
    result = HybridConfidenceNormalizer().normalize(1.0 / 61.0)
    assert 0.0 < result.value < 1.0
    assert result.level in {CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH}


def test_hybrid_normalize_none_raw_score_is_zero() -> None:
    result = HybridConfidenceNormalizer().normalize(None)
    assert result.value == 0.0
    assert result.level == CONFIDENCE_LOW


def test_hybrid_normalizer_strategy_label() -> None:
    assert HybridConfidenceNormalizer().strategy == RoutingStrategy.HYBRID


# ── Output always within [0, 1] ──────────────────────────────────────────────────


@pytest.mark.parametrize("raw_score", [None, 0.0, -100.0, 0.5, 1.0, 2.0, 1e6])
def test_dense_output_always_in_unit_interval(raw_score) -> None:
    result = DenseConfidenceNormalizer().normalize(raw_score)
    assert 0.0 <= result.value <= 1.0


@pytest.mark.parametrize("raw_score", [None, -10.0, 0.0, 0.5, 4.0, 1000.0, 1e9])
def test_bm25_output_always_in_unit_interval(raw_score) -> None:
    result = BM25ConfidenceNormalizer().normalize(raw_score)
    assert 0.0 <= result.value <= 1.0


@pytest.mark.parametrize("raw_score", [None, -1.0, 0.0, 0.001, 0.016, 0.033, 1000.0])
def test_hybrid_output_always_in_unit_interval(raw_score) -> None:
    result = HybridConfidenceNormalizer().normalize(raw_score)
    assert 0.0 <= result.value <= 1.0


# ── Threshold classification (shared, unmodified classify_confidence) ───────────


def test_classification_uses_same_thresholds_as_existing_classify_confidence() -> None:
    for value in (0.1, 0.40, 0.45, 0.55, 0.9):
        normalized = DenseConfidenceNormalizer().normalize(1.0 - value)
        assert normalized.level == classify_confidence(value)


def test_low_confidence_below_threshold() -> None:
    result = DenseConfidenceNormalizer().normalize(1.0 - (LOW_CONFIDENCE_THRESHOLD - 0.01))
    assert result.level == CONFIDENCE_LOW


def test_medium_confidence_between_thresholds() -> None:
    midpoint_value = (LOW_CONFIDENCE_THRESHOLD + HIGH_CONFIDENCE_THRESHOLD) / 2
    result = DenseConfidenceNormalizer().normalize(1.0 - midpoint_value)
    assert result.level == CONFIDENCE_MEDIUM


def test_high_confidence_at_or_above_threshold() -> None:
    result = DenseConfidenceNormalizer().normalize(1.0 - HIGH_CONFIDENCE_THRESHOLD)
    assert result.level == CONFIDENCE_HIGH


# ── Boundary conditions ───────────────────────────────────────────────────────────


def test_dense_boundary_value_exactly_at_low_threshold_is_medium_not_low() -> None:
    # value == LOW_CONFIDENCE_THRESHOLD is NOT < threshold, so it is MEDIUM.
    result = DenseConfidenceNormalizer().normalize(1.0 - LOW_CONFIDENCE_THRESHOLD)
    assert result.value == pytest.approx(LOW_CONFIDENCE_THRESHOLD)
    assert result.level == CONFIDENCE_MEDIUM


def test_dense_boundary_value_exactly_at_high_threshold_is_high() -> None:
    result = DenseConfidenceNormalizer().normalize(1.0 - HIGH_CONFIDENCE_THRESHOLD)
    assert result.value == pytest.approx(HIGH_CONFIDENCE_THRESHOLD)
    assert result.level == CONFIDENCE_HIGH


def test_bm25_boundary_score_for_exact_low_threshold_value() -> None:
    # score / (score + 4) == 0.40  =>  score == 4 * 0.40 / 0.60
    score = BM25_MIDPOINT * LOW_CONFIDENCE_THRESHOLD / (1 - LOW_CONFIDENCE_THRESHOLD)
    result = BM25ConfidenceNormalizer().normalize(score)
    assert result.value == pytest.approx(LOW_CONFIDENCE_THRESHOLD)
    assert result.level == CONFIDENCE_MEDIUM  # exactly-at-threshold is MEDIUM, not LOW


# ── Strategy independence ─────────────────────────────────────────────────────────


def _describe(confidence: NormalizedConfidence) -> tuple[float, str]:
    """A stand-in for a downstream consumer (e.g. the Investigation Agent)
    that only ever reads .value/.level, never .strategy/.raw_score.
    """
    return (round(confidence.value, 6), confidence.level)


def test_strategy_independent_consumer_sees_identical_output_for_equivalent_confidence() -> None:
    dense = DenseConfidenceNormalizer().normalize(0.5)  # value = 0.5
    bm25 = BM25ConfidenceNormalizer().normalize(BM25_MIDPOINT)  # value = 0.5
    hybrid = HybridConfidenceNormalizer().normalize(HYBRID_MIDPOINT)  # value = 0.5

    assert _describe(dense) == _describe(bm25) == _describe(hybrid)


def test_normalized_confidence_does_not_require_strategy_to_interpret() -> None:
    result = normalize_confidence(RoutingStrategy.BM25, 4.0)
    # A consumer can use .value/.level without ever inspecting .strategy.
    value, level = result.value, result.level
    assert value == pytest.approx(0.5)
    assert level == CONFIDENCE_MEDIUM


# ── Backward compatibility for Dense ─────────────────────────────────────────────


def test_dense_normalized_value_matches_today_similarity_score_for_typical_distances() -> None:
    from app.services.search import IncidentSearchResult

    for distance in (0.0, 0.1, 0.3, 0.42, 0.6, 0.9, 1.0):
        legacy = IncidentSearchResult(incident=None, distance=distance).similarity_score
        normalized = DenseConfidenceNormalizer().normalize(distance).value
        assert normalized == pytest.approx(legacy)


def test_dense_classification_matches_classify_confidence_directly_on_similarity_score() -> None:
    from app.services.search import IncidentSearchResult

    for distance in (0.0, 0.3, 0.45, 0.6, 1.0):
        similarity = IncidentSearchResult(incident=None, distance=distance).similarity_score
        legacy_level = classify_confidence(similarity)
        normalized_level = DenseConfidenceNormalizer().normalize(distance).level
        assert normalized_level == legacy_level


# ── Registry / factory ────────────────────────────────────────────────────────────


def test_get_confidence_normalizer_returns_correct_type_per_strategy() -> None:
    dense = get_confidence_normalizer(RoutingStrategy.DENSE)
    bm25 = get_confidence_normalizer(RoutingStrategy.BM25)
    hybrid = get_confidence_normalizer(RoutingStrategy.HYBRID)
    assert isinstance(dense, DenseConfidenceNormalizer)
    assert isinstance(bm25, BM25ConfidenceNormalizer)
    assert isinstance(hybrid, HybridConfidenceNormalizer)


def test_normalize_confidence_dispatches_via_registry() -> None:
    result = normalize_confidence(RoutingStrategy.DENSE, 0.2)
    assert result.strategy == RoutingStrategy.DENSE
    assert result.value == pytest.approx(0.8)


def test_register_confidence_normalizer_swaps_without_changing_call_sites() -> None:
    class _AlwaysOneNormalizer(ConfidenceNormalizer):
        strategy = RoutingStrategy.DENSE

        def normalize(self, raw_score):
            return NormalizedConfidence(
                value=1.0, level=CONFIDENCE_HIGH, strategy=self.strategy, raw_score=raw_score
            )

    original = get_confidence_normalizer(RoutingStrategy.DENSE)
    try:
        register_confidence_normalizer(RoutingStrategy.DENSE, _AlwaysOneNormalizer())
        result = normalize_confidence(RoutingStrategy.DENSE, 0.99)  # would normally be LOW
        assert result.value == 1.0
        assert result.level == CONFIDENCE_HIGH
    finally:
        register_confidence_normalizer(RoutingStrategy.DENSE, original)


def test_normalized_confidence_is_frozen() -> None:
    result = normalize_confidence(RoutingStrategy.DENSE, 0.5)
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        result.value = 0.0  # type: ignore[misc]
