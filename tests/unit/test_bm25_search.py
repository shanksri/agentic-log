from __future__ import annotations

import math

import pytest

from app.services.bm25_search import (
    BM25Config,
    BM25Document,
    BM25Index,
    BM25Retriever,
    BM25SearchResult,
    default_tokenizer,
)


def _doc(document_id: str, text: str) -> BM25Document:
    return BM25Document(document_id=document_id, text=text)


def _bm25_score(
    *, term_frequency: int, document_length: int, n: int, df: int, avgdl: float,
    k1: float = 1.5, b: float = 0.75,
) -> float:
    """Independent re-derivation of the documented BM25 formula, used to
    compute expected values without relying on the implementation under
    test (same approach as tests/unit/test_metrics.py).
    """
    idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
    length_norm = 1 - b + b * (document_length / avgdl)
    denominator = term_frequency + k1 * length_norm
    return idf * (term_frequency * (k1 + 1)) / denominator


# ── Tokenization ───────────────────────────────────────────────────────────────


def test_default_tokenizer_lowercases() -> None:
    assert default_tokenizer("Scheduler CRASHLOOP") == ["scheduler", "crashloop"]


def test_default_tokenizer_strips_punctuation() -> None:
    assert default_tokenizer("triggerer not starting!") == ["triggerer", "not", "starting"]


def test_default_tokenizer_keeps_digits_and_underscores() -> None:
    assert default_tokenizer("dag_version_id is NULL v2") == [
        "dag_version_id", "is", "null", "v2",
    ]


def test_default_tokenizer_empty_string_is_empty_list() -> None:
    assert default_tokenizer("") == []


def test_default_tokenizer_punctuation_only_is_empty_list() -> None:
    assert default_tokenizer("!!! --- ???") == []


# ── Indexing ───────────────────────────────────────────────────────────────────


def test_index_build_populates_size_and_document_length() -> None:
    index = BM25Index()
    index.build([_doc("a", "scheduler crashloop"), _doc("b", "triggerer not starting")])

    assert index.size == 2
    assert index.document_length("a") == 2
    assert index.document_length("b") == 3


def test_index_average_document_length() -> None:
    index = BM25Index()
    index.build([_doc("a", "one two"), _doc("b", "one two three four")])

    assert index.average_document_length == pytest.approx(3.0)


def test_index_average_document_length_empty_corpus_is_zero() -> None:
    index = BM25Index()
    index.build([])
    assert index.average_document_length == 0.0


def test_index_document_frequency_counts_distinct_documents_only() -> None:
    index = BM25Index()
    index.build(
        [
            _doc("a", "memory leak memory leak"),  # "memory" appears twice in one doc
            _doc("b", "memory pressure"),
            _doc("c", "unrelated text"),
        ]
    )
    # "memory" appears in 2 distinct documents, regardless of in-document repetition.
    assert index.document_frequency("memory") == 2
    assert index.document_frequency("nonexistent-term") == 0


def test_index_postings_returns_term_frequency_per_document() -> None:
    index = BM25Index()
    index.build([_doc("a", "memory leak memory leak"), _doc("b", "memory pressure")])

    postings = index.postings("memory")
    assert postings == {"a": 2, "b": 1}


def test_index_postings_unknown_term_is_empty_dict() -> None:
    index = BM25Index()
    index.build([_doc("a", "hello world")])
    assert index.postings("nonexistent") == {}


def test_index_postings_returns_a_copy_not_internal_state() -> None:
    index = BM25Index()
    index.build([_doc("a", "hello world")])
    postings = index.postings("hello")
    postings["b"] = 99  # mutate the returned dict
    assert index.postings("hello") == {"a": 1}  # internal state unaffected


def test_index_build_rejects_duplicate_document_id() -> None:
    index = BM25Index()
    with pytest.raises(ValueError, match="dup"):
        index.build([_doc("dup", "first"), _doc("dup", "second")])


def test_index_build_called_twice_raises() -> None:
    index = BM25Index()
    index.build([_doc("a", "hello")])
    with pytest.raises(RuntimeError):
        index.build([_doc("b", "world")])


def test_index_build_with_empty_text_document_does_not_break_avgdl() -> None:
    index = BM25Index()
    index.build([_doc("a", ""), _doc("b", "hello world")])

    assert index.document_length("a") == 0
    assert index.average_document_length == pytest.approx(1.0)  # (0 + 2) / 2


def test_index_accepts_custom_tokenizer() -> None:
    def whitespace_tokenizer(text: str) -> list[str]:
        return text.split()

    index = BM25Index(tokenizer=whitespace_tokenizer)
    index.build([_doc("a", "hello, world!")])

    # The custom tokenizer keeps punctuation attached; default_tokenizer would not.
    assert index.postings("hello,") == {"a": 1}
    assert index.postings("hello") == {}


# ── Ranking correctness ────────────────────────────────────────────────────────


def test_retrieve_single_term_matches_hand_computed_score() -> None:
    index = BM25Index()
    index.build([_doc("a", "scheduler crashloop error"), _doc("b", "unrelated text here")])
    retriever = BM25Retriever(index)

    [result] = retriever.retrieve("scheduler", limit=10)

    expected = _bm25_score(
        term_frequency=1, document_length=3, n=2, df=1, avgdl=index.average_document_length,
    )
    assert result.document_id == "a"
    assert result.score == pytest.approx(expected)


def test_retrieve_exact_keyword_ranks_above_non_matching_document() -> None:
    index = BM25Index()
    index.build(
        [
            _doc("match", "scheduler crashloop ValidationError dag_version_id NULL"),
            _doc("nomatch", "completely unrelated incident about a different system"),
        ]
    )
    retriever = BM25Retriever(index)

    results = retriever.retrieve("dag_version_id", limit=10)

    assert [r.document_id for r in results] == ["match"]


def test_retrieve_multiple_matching_documents_ranked_by_relevance() -> None:
    index = BM25Index()
    index.build(
        [
            _doc("low", "memory leak observed once"),
            _doc("high", "memory leak memory leak memory leak in compiler"),
            _doc("none", "scheduler crashloop unrelated"),
        ]
    )
    retriever = BM25Retriever(index)

    results = retriever.retrieve("memory leak", limit=10)

    assert [r.document_id for r in results] == ["high", "low"]
    assert results[0].score > results[1].score


def test_retrieve_unique_query_terms_not_weighted_by_query_term_frequency() -> None:
    index = BM25Index()
    index.build([_doc("a", "memory leak issue")])
    retriever = BM25Retriever(index)

    once = retriever.retrieve("memory", limit=10)
    repeated = retriever.retrieve("memory memory memory", limit=10)

    # Classic BM25: repeating a term in the query does not change its score
    # contribution, since scoring iterates unique query terms.
    assert once[0].score == pytest.approx(repeated[0].score)


def test_retrieve_term_appearing_in_every_document_has_nonnegative_idf() -> None:
    index = BM25Index()
    index.build([_doc("a", "common term unique-a"), _doc("b", "common term unique-b")])
    retriever = BM25Retriever(index)

    results = retriever.retrieve("common", limit=10)

    # The smoothed idf must never go negative, even for a term in 100% of docs.
    assert all(r.score >= 0 for r in results)


# ── Empty corpus / unknown query ───────────────────────────────────────────────


def test_retrieve_empty_corpus_returns_empty_list() -> None:
    index = BM25Index()
    index.build([])
    retriever = BM25Retriever(index)

    assert retriever.retrieve("anything", limit=10) == []


def test_retrieve_unknown_query_returns_empty_list() -> None:
    index = BM25Index()
    index.build([_doc("a", "scheduler crashloop")])
    retriever = BM25Retriever(index)

    assert retriever.retrieve("completely unrelated vocabulary", limit=10) == []


# ── Deterministic ordering ────────────────────────────────────────────────────


def test_retrieve_ties_broken_deterministically_by_document_id() -> None:
    index = BM25Index()
    # Identical text -> identical scores for both documents.
    index.build([_doc("z-doc", "shared term"), _doc("a-doc", "shared term")])
    retriever = BM25Retriever(index)

    results = retriever.retrieve("shared term", limit=10)

    assert results[0].score == pytest.approx(results[1].score)
    assert [r.document_id for r in results] == ["a-doc", "z-doc"]  # ascending id wins tie


def test_retrieve_ordering_is_stable_across_repeated_calls() -> None:
    index = BM25Index()
    index.build([_doc("a", "x y z"), _doc("b", "x y"), _doc("c", "x")])
    retriever = BM25Retriever(index)

    first = retriever.retrieve("x y z", limit=10)
    second = retriever.retrieve("x y z", limit=10)

    assert first == second


# ── Configurable K ─────────────────────────────────────────────────────────────


def test_retrieve_respects_limit() -> None:
    index = BM25Index()
    index.build([_doc(str(i), f"term-{i} shared") for i in range(5)])
    retriever = BM25Retriever(index)

    results = retriever.retrieve("shared", limit=2)

    assert len(results) == 2


def test_retrieve_limit_larger_than_corpus_returns_all_matches() -> None:
    index = BM25Index()
    index.build([_doc("a", "shared"), _doc("b", "shared")])
    retriever = BM25Retriever(index)

    results = retriever.retrieve("shared", limit=100)

    assert len(results) == 2


def test_retrieve_rejects_non_positive_limit() -> None:
    index = BM25Index()
    index.build([_doc("a", "hello")])
    retriever = BM25Retriever(index)

    with pytest.raises(ValueError):
        retriever.retrieve("hello", limit=0)


# ── BM25Config / k1, b ─────────────────────────────────────────────────────────


def test_different_k1_b_on_same_index_produce_different_scores() -> None:
    index = BM25Index()
    index.build([_doc("a", "memory leak memory leak"), _doc("b", "memory")])

    default_retriever = BM25Retriever(index)
    alternate_retriever = BM25Retriever(index, config=BM25Config(k1=3.0, b=0.0))

    default_results = default_retriever.retrieve("memory", limit=10)
    alternate_results = alternate_retriever.retrieve("memory", limit=10)
    default_score = next(r.score for r in default_results if r.document_id == "a")
    alternate_score = next(r.score for r in alternate_results if r.document_id == "a")

    assert default_score != pytest.approx(alternate_score)


def test_config_defaults_match_standard_okapi_bm25() -> None:
    config = BM25Config()
    assert config.k1 == 1.5
    assert config.b == 0.75


# ── BM25Retriever.from_documents convenience constructor ─────────────────────


def test_from_documents_builds_and_wraps_in_one_call() -> None:
    retriever = BM25Retriever.from_documents(
        [_doc("a", "scheduler crashloop"), _doc("b", "triggerer not starting")]
    )

    assert isinstance(retriever, BM25Retriever)
    assert retriever.index.size == 2
    results = retriever.retrieve("scheduler", limit=10)
    assert [r.document_id for r in results] == ["a"]


def test_from_documents_accepts_config_and_tokenizer() -> None:
    def whitespace_tokenizer(text: str) -> list[str]:
        return text.split()

    config = BM25Config(k1=2.0, b=0.5)
    retriever = BM25Retriever.from_documents(
        [_doc("a", "hello world")], config=config, tokenizer=whitespace_tokenizer
    )

    assert retriever.config == config
    assert retriever.index.tokenizer is whitespace_tokenizer


# ── Result type ────────────────────────────────────────────────────────────────


def test_bm25_search_result_is_frozen_and_primitive() -> None:
    result = BM25SearchResult(document_id="a", score=1.0)
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        result.score = 2.0  # type: ignore[misc]
