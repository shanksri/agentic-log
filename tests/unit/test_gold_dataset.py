from __future__ import annotations

from app.evaluation.gold_dataset import (
    CorpusFingerprintPlaceholder,
    ExpectedIncident,
    GoldDataset,
    GoldQuery,
)


def _expected(
    *, source_type: str = "github", source_external_id: str = "acme/api#1", relevance: int = 3
) -> ExpectedIncident:
    return ExpectedIncident(
        source_type=source_type, source_external_id=source_external_id, relevance=relevance
    )


def _query(
    *,
    id: str = "q-1",
    query: str = "scheduler crashes",
    category: str = "lexical-overlap",
    difficulty: str = "easy",
    expected_incidents: tuple[ExpectedIncident, ...] = (),
) -> GoldQuery:
    return GoldQuery(
        id=id,
        query=query,
        category=category,
        difficulty=difficulty,
        expected_incidents=expected_incidents or (_expected(),),
    )


def _dataset(*, queries: tuple[GoldQuery, ...] | None = None) -> GoldDataset:
    return GoldDataset(
        version="2.0.0",
        description="test dataset",
        created_at="2026-06-26T00:00:00Z",
        queries=queries if queries is not None else (_query(),),
    )


# ── ExpectedIncident ──────────────────────────────────────────────────────────


def test_expected_incident_valid_has_no_issues() -> None:
    assert _expected().issues() == []


def test_expected_incident_rejects_empty_source_type() -> None:
    issues = _expected(source_type="").issues()
    assert any("source_type" in issue for issue in issues)


def test_expected_incident_rejects_empty_source_external_id() -> None:
    issues = _expected(source_external_id="").issues()
    assert any("source_external_id" in issue for issue in issues)


def test_expected_incident_rejects_relevance_out_of_range() -> None:
    assert any("relevance" in issue for issue in _expected(relevance=0).issues())
    assert any("relevance" in issue for issue in _expected(relevance=4).issues())


def test_expected_incident_accepts_relevance_bounds() -> None:
    assert _expected(relevance=1).issues() == []
    assert _expected(relevance=3).issues() == []


# ── GoldQuery ──────────────────────────────────────────────────────────────────


def test_gold_query_valid_has_no_issues() -> None:
    assert _query().issues() == []


def test_gold_query_rejects_unknown_category() -> None:
    issues = _query(category="not-a-real-category").issues()
    assert any("category" in issue for issue in issues)


def test_gold_query_rejects_unknown_difficulty() -> None:
    issues = _query(difficulty="extreme").issues()
    assert any("difficulty" in issue for issue in issues)


def test_gold_query_non_negative_category_requires_expected_incidents() -> None:
    query = GoldQuery(
        id="q-empty",
        query="some query",
        category="paraphrase",
        difficulty="easy",
        expected_incidents=(),
    )
    issues = query.issues()
    assert any("requires at least one" in issue for issue in issues)


def test_gold_query_no_match_expected_must_have_zero_expected_incidents() -> None:
    query = GoldQuery(
        id="q-neg",
        query="some query",
        category="no-match-expected",
        difficulty="easy",
        expected_incidents=(_expected(),),
    )
    issues = query.issues()
    assert any("must have zero" in issue for issue in issues)


def test_gold_query_no_match_expected_with_zero_expected_is_valid() -> None:
    query = GoldQuery(
        id="q-neg",
        query="some query",
        category="no-match-expected",
        difficulty="easy",
        expected_incidents=(),
    )
    assert query.issues() == []


def test_gold_query_rejects_duplicate_expected_identities() -> None:
    duplicate = _expected()
    query = _query(expected_incidents=(duplicate, duplicate))
    issues = query.issues()
    assert any("duplicate expected_incident identity" in issue for issue in issues)


def test_gold_query_allows_multiple_distinct_expected_incidents() -> None:
    query = _query(
        expected_incidents=(
            _expected(source_external_id="acme/api#1", relevance=3),
            _expected(source_external_id="acme/api#2", relevance=1),
        )
    )
    assert query.issues() == []


# ── GoldDataset ────────────────────────────────────────────────────────────────


def test_dataset_valid_has_no_issues_and_is_valid() -> None:
    dataset = _dataset()
    assert dataset.issues() == []
    assert dataset.is_valid() is True


def test_dataset_rejects_missing_version_description_created_at() -> None:
    dataset = GoldDataset(version="", description="", created_at="", queries=(_query(),))
    issues = dataset.issues()
    assert any("version" in issue for issue in issues)
    assert any("description" in issue for issue in issues)
    assert any("created_at" in issue for issue in issues)


def test_dataset_rejects_empty_queries() -> None:
    dataset = _dataset(queries=())
    issues = dataset.issues()
    assert any("queries must be non-empty" in issue for issue in issues)
    assert dataset.is_valid() is False


def test_dataset_rejects_duplicate_query_ids() -> None:
    dataset = _dataset(queries=(_query(id="dup"), _query(id="dup")))
    issues = dataset.issues()
    assert any("duplicate query id" in issue for issue in issues)


def test_dataset_aggregates_issues_from_all_queries() -> None:
    dataset = _dataset(
        queries=(
            _query(id="q-1", category="bad-category"),
            _query(id="q-2", difficulty="bad-difficulty"),
        )
    )
    issues = dataset.issues()
    assert any("q-1" in issue and "category" in issue for issue in issues)
    assert any("q-2" in issue and "difficulty" in issue for issue in issues)


def test_dataset_default_corpus_fingerprint_is_uncomputed_placeholder() -> None:
    dataset = _dataset()
    assert dataset.corpus_fingerprint == CorpusFingerprintPlaceholder(computed=False, value=None)


def test_dataset_author_is_optional() -> None:
    dataset = _dataset()
    assert dataset.author is None
