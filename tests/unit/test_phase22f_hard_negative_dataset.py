"""Tests for Phase 22F's hard-negative dataset
(tests/eval/gold_queries_hard_negatives_v1.json) and the existing easy
negatives it's designed to sit alongside (tests/eval/gold_queries.json) --
shape/metadata checks only, no retrieval, no DB. Also pins the existing
confidence-classification behavior (thresholds unchanged) per Phase 22F's
strict scope."""
from __future__ import annotations

import json
from pathlib import Path

from app.services.confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    classify_confidence,
)

HARD_NEGATIVES_PATH = (
    Path(__file__).parent.parent / "eval" / "gold_queries_hard_negatives_v1.json"
)
EASY_NEGATIVES_PATH = Path(__file__).parent.parent / "eval" / "gold_queries.json"

REQUIRED_FIELDS = {
    "id", "query", "query_type", "category", "expected_incident_ids",
    "nearest_real_incident", "why_not_represented", "adjacent_topic_family",
}

EXPECTED_CATEGORIES = {
    "payment/billing", "authentication/authorization", "monitoring/alerts",
    "kubernetes/containers", "networking", "integrations", "databases",
    "apis", "configuration", "deployment/upgrades",
}


def _load_hard_negatives() -> list[dict]:
    return json.loads(HARD_NEGATIVES_PATH.read_text(encoding="utf-8"))["queries"]


def test_hard_negative_count_in_target_range() -> None:
    queries = _load_hard_negatives()
    assert 15 <= len(queries) <= 25


def test_hard_negative_ids_are_unique() -> None:
    queries = _load_hard_negatives()
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))


def test_every_hard_negative_has_required_fields() -> None:
    queries = _load_hard_negatives()
    for q in queries:
        missing = REQUIRED_FIELDS - set(q.keys())
        assert not missing, f"{q.get('id')} missing fields: {missing}"


def test_every_hard_negative_has_empty_expected_incidents() -> None:
    # A hard negative is a negative -- it must not carry a real expected match.
    queries = _load_hard_negatives()
    for q in queries:
        assert q["expected_incident_ids"] == [], f"{q['id']} should have no expected match"


def test_hard_negative_categories_cover_the_requested_spread() -> None:
    queries = _load_hard_negatives()
    seen_categories = {q["category"] for q in queries}
    # Every category actually used must be one of the requested ones, and at
    # least 8 of the 10 requested categories should be represented (spread
    # requirement, not a rigid one-of-each rule).
    assert seen_categories <= EXPECTED_CATEGORIES
    assert len(seen_categories) >= 8


def test_hard_negative_justifications_are_non_trivial() -> None:
    # Every "why_not_represented" must be a real sentence, not a placeholder.
    queries = _load_hard_negatives()
    for q in queries:
        assert len(q["why_not_represented"]) > 30, q["id"]
        assert len(q["nearest_real_incident"]) > 10, q["id"]


def test_positive_and_hard_negative_queries_are_distinguishable_by_id_prefix() -> None:
    # Sanity check that the new dataset can never collide with, or be
    # confused for, the existing easy-negative/positive gold set's ids.
    hard_ids = {q["id"] for q in _load_hard_negatives()}
    easy_and_positive_ids = {
        q["id"] for q in json.loads(EASY_NEGATIVES_PATH.read_text(encoding="utf-8"))["queries"]
    }
    assert hard_ids.isdisjoint(easy_and_positive_ids)
    assert all(hard_id.startswith("hard-neg-") for hard_id in hard_ids)


def test_easy_negatives_still_intact() -> None:
    # Phase 22F must not touch the existing easy-negative gold data.
    data = json.loads(EASY_NEGATIVES_PATH.read_text(encoding="utf-8"))
    negatives = [q for q in data["queries"] if len(q["expected_incident_ids"]) == 0]
    assert len(negatives) == 4
    assert {q["id"] for q in negatives} == {"neg-01", "neg-02", "neg-03", "neg-04"}


# ── Existing confidence classification behavior must be unchanged ──────────


def test_confidence_thresholds_unchanged() -> None:
    assert LOW_CONFIDENCE_THRESHOLD == 0.40
    assert HIGH_CONFIDENCE_THRESHOLD == 0.55


def test_classify_confidence_behavior_unchanged() -> None:
    assert classify_confidence(None) == CONFIDENCE_LOW
    assert classify_confidence(0.10) == CONFIDENCE_LOW
    assert classify_confidence(0.399999) == CONFIDENCE_LOW
    assert classify_confidence(0.40) == CONFIDENCE_MEDIUM
    assert classify_confidence(0.549999) == CONFIDENCE_MEDIUM
    assert classify_confidence(0.55) == CONFIDENCE_HIGH
    assert classify_confidence(0.90) == CONFIDENCE_HIGH
