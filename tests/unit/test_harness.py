from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

import app.services.identity as identity_module
from app.evaluation.gold_dataset import ExpectedIncident, GoldDataset, GoldQuery
from app.evaluation.harness import EvaluationReport, evaluate
from app.services.search import IncidentSearchResult

# ── Shared fixtures: SQLite-backed identity resolution + fake search ─────────


class _TestBase(DeclarativeBase):
    pass


class _FakeIncident(_TestBase):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_external_id: Mapped[str] = mapped_column(String, nullable=False)


@pytest.fixture(autouse=True)
def _patch_incident_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity_module, "Incident", _FakeIncident)


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    _TestBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        yield db


def _add_incident(db: Session, *, source_type: str, source_external_id: str) -> str:
    """Returns the incident id as a plain string.

    The fake SQLite-backed Incident model below maps ``id`` as a String
    column (unlike production's Postgres ``UUID(as_uuid=True)``), so
    IdentityResolver returns this id as a ``str``, not a ``uuid.UUID``.
    Tests use ``str`` ids consistently on both the resolution side and the
    fake retrieval side so equality comparisons inside the metric engine
    behave the same way they would against real ``uuid.UUID`` values in
    production — only the Python type differs, not the comparison
    semantics.
    """
    incident_id = str(uuid.uuid4())
    db.add(
        _FakeIncident(
            id=incident_id, source_type=source_type, source_external_id=source_external_id
        )
    )
    db.commit()
    return incident_id


def _result(incident_id: str, *, distance: float = 0.1) -> IncidentSearchResult:
    return IncidentSearchResult(incident=SimpleNamespace(id=incident_id), distance=distance)


class FakeSearchService:
    """Duck-typed stand-in for IncidentSearchService: harness only needs
    ``.db`` and ``.retrieve(query, *, limit, expand, rerank, call_site)``.
    """

    def __init__(self, db: Session, responses: dict[str, list[IncidentSearchResult] | Exception]):
        self.db = db
        self._responses = responses
        self.calls: list[dict] = []

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        expand: bool = False,
        rerank: bool = False,
        call_site: str | None = None,
    ) -> list[IncidentSearchResult]:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "expand": expand,
                "rerank": rerank,
                "call_site": call_site,
            }
        )
        outcome = self._responses.get(query, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _query(
    *,
    id: str,
    query: str,
    category: str = "lexical-overlap",
    difficulty: str = "easy",
    expected: list[tuple[str, str, int]] | None = None,
) -> GoldQuery:
    expected_incidents = tuple(
        ExpectedIncident(source_type=st, source_external_id=ext, relevance=rel)
        for st, ext, rel in (expected or [])
    )
    return GoldQuery(
        id=id, query=query, category=category, difficulty=difficulty,
        expected_incidents=expected_incidents,
    )


def _dataset(queries: tuple[GoldQuery, ...]) -> GoldDataset:
    return GoldDataset(
        version="2.0.0", description="test", created_at="2026-06-26T00:00:00Z", queries=queries
    )


# ── Successful evaluation ─────────────────────────────────────────────────────


def test_evaluate_successful_run_produces_full_report(session: Session) -> None:
    incident_a = _add_incident(session, source_type="github", source_external_id="a")
    gq = _query(id="q-1", query="scheduler crashes", expected=[("github", "a", 3)])
    dataset = _dataset((gq,))
    search_service = FakeSearchService(session, {"scheduler crashes": [_result(incident_a)]})

    report = evaluate(dataset, search_service, k=5)

    assert isinstance(report, EvaluationReport)
    assert report.num_evaluated == 1
    assert report.num_skipped == 0
    assert report.aggregate_metrics.mean_recall_at_k == 1.0
    assert report.aggregate_metrics.mean_reciprocal_rank == 1.0
    assert report.aggregate_metrics.mean_ndcg_at_k == pytest.approx(1.0)
    assert len(report.per_query) == 1
    assert report.per_query[0].query_id == "q-1"
    assert report.per_query[0].skipped is False


def test_evaluate_passes_k_expand_rerank_through_to_search_service(session: Session) -> None:
    incident_a = _add_incident(session, source_type="github", source_external_id="a")
    gq = _query(id="q-1", query="q", expected=[("github", "a", 1)])
    dataset = _dataset((gq,))
    search_service = FakeSearchService(session, {"q": [_result(incident_a)]})

    evaluate(dataset, search_service, k=7, expand=True, rerank=True)

    [call] = search_service.calls
    assert call == {
        "query": "q",
        "limit": 7,
        "expand": True,
        "rerank": True,
        "call_site": "evaluation_harness",
    }
    report = evaluate(dataset, search_service, k=7, expand=True, rerank=True)
    assert report.config.k == 7
    assert report.config.expand is True
    assert report.config.rerank is True


# ── Search exceptions / partial failures ──────────────────────────────────────


def test_evaluate_search_failure_skips_only_that_query(session: Session) -> None:
    incident_a = _add_incident(session, source_type="github", source_external_id="a")
    incident_b = _add_incident(session, source_type="github", source_external_id="b")
    q1 = _query(id="q-1", query="fails", expected=[("github", "a", 3)])
    q2 = _query(id="q-2", query="works", expected=[("github", "b", 3)])
    dataset = _dataset((q1, q2))
    search_service = FakeSearchService(
        session,
        {"fails": RuntimeError("embedding backend down"), "works": [_result(incident_b)]},
    )

    report = evaluate(dataset, search_service, k=5)

    assert report.num_evaluated == 1
    assert report.num_skipped == 1
    failed = next(o for o in report.per_query if o.query_id == "q-1")
    succeeded = next(o for o in report.per_query if o.query_id == "q-2")
    assert failed.skipped is True
    assert failed.metric is None
    assert "search_failed" in failed.skip_reason
    assert "embedding backend down" in failed.skip_reason
    assert succeeded.skipped is False
    assert succeeded.metric.recall_at_k == 1.0


def test_evaluate_skipped_query_still_reports_resolution_fields(session: Session) -> None:
    # No incident added for "a" -> unresolved, AND the search call fails too.
    q1 = _query(id="q-1", query="fails", expected=[("github", "a", 3)])
    dataset = _dataset((q1,))
    search_service = FakeSearchService(session, {"fails": RuntimeError("boom")})

    report = evaluate(dataset, search_service, k=5)

    [outcome] = report.per_query
    assert outcome.skipped is True
    assert outcome.num_relevant == 0
    assert outcome.num_unresolved_expected == 1
    # Resolution-derived aggregate must count this query even though it was skipped.
    assert report.aggregate_metrics.queries_with_unresolved_incidents == 1


def test_evaluate_all_queries_failing_search_yields_zero_evaluated(session: Session) -> None:
    incident_a = _add_incident(session, source_type="github", source_external_id="a")
    q1 = _query(id="q-1", query="fails-1", expected=[("github", "a", 3)])
    dataset = _dataset((q1,))
    search_service = FakeSearchService(session, {"fails-1": RuntimeError("down")})

    report = evaluate(dataset, search_service, k=5)

    assert report.num_evaluated == 0
    assert report.num_skipped == 1
    assert report.aggregate_metrics.mean_recall_at_k is None
    assert report.aggregate_metrics.mean_reciprocal_rank is None
    assert report.aggregate_metrics.mean_ndcg_at_k is None


# ── Fully unresolved queries ───────────────────────────────────────────────────


def test_evaluate_fully_unresolved_query_scores_like_no_match(session: Session) -> None:
    # Gold references an incident that does not (or no longer) exists.
    q1 = _query(id="q-1", query="stale gold", expected=[("github", "ghost", 3)])
    dataset = _dataset((q1,))
    search_service = FakeSearchService(session, {"stale gold": []})

    report = evaluate(dataset, search_service, k=5)

    [outcome] = report.per_query
    assert outcome.skipped is False
    assert outcome.num_relevant == 0
    assert outcome.num_unresolved_expected == 1
    assert outcome.metric.recall_at_k is None
    assert outcome.metric.ndcg_at_k is None
    assert report.coverage.fully_unresolved_queries == 1
    assert report.coverage.fully_resolved_queries == 0


def test_evaluate_no_match_expected_query_handled_correctly(session: Session) -> None:
    q1 = _query(id="q-neg", query="irrelevant", category="no-match-expected", expected=[])
    dataset = _dataset((q1,))
    search_service = FakeSearchService(session, {"irrelevant": [_result(uuid.uuid4())]})

    report = evaluate(dataset, search_service, k=5)

    [outcome] = report.per_query
    assert outcome.skipped is False
    assert outcome.metric.recall_at_k is None
    assert outcome.metric.ndcg_at_k is None
    assert report.coverage.no_match_expected_queries == 1
    assert report.coverage.fully_unresolved_queries == 0  # no-match-expected is its own bucket


# ── Mixed categories / difficulties / aggregation correctness ────────────────


def test_evaluate_mixed_categories_and_difficulties_breakdowns(session: Session) -> None:
    incident_a = _add_incident(session, source_type="github", source_external_id="a")
    incident_b = _add_incident(session, source_type="github", source_external_id="b")

    q1 = _query(
        id="q-1", query="lex query", category="lexical-overlap", difficulty="easy",
        expected=[("github", "a", 3)],
    )
    q2 = _query(
        id="q-2", query="para query", category="paraphrase", difficulty="hard",
        expected=[("github", "b", 2)],
    )
    q3 = _query(
        id="q-3", query="neg query", category="no-match-expected", difficulty="medium",
        expected=[],
    )
    dataset = _dataset((q1, q2, q3))
    search_service = FakeSearchService(
        session,
        {
            "lex query": [_result(incident_a)],      # perfect hit
            "para query": [_result(uuid.uuid4())],   # miss
            "neg query": [],                          # correctly empty
        },
    )

    report = evaluate(dataset, search_service, k=5)

    assert set(report.category_breakdown) == {"lexical-overlap", "paraphrase", "no-match-expected"}
    assert set(report.difficulty_breakdown) == {"easy", "hard", "medium"}

    lex_bucket = report.category_breakdown["lexical-overlap"]
    para_bucket = report.category_breakdown["paraphrase"]
    neg_bucket = report.category_breakdown["no-match-expected"]

    assert lex_bucket.num_queries == 1
    assert lex_bucket.mean_recall_at_k == 1.0
    assert para_bucket.mean_recall_at_k == 0.0
    assert neg_bucket.mean_recall_at_k is None  # excluded, not averaged in as 0


def test_evaluate_overall_aggregate_excludes_none_metrics_from_means(session: Session) -> None:
    incident_a = _add_incident(session, source_type="github", source_external_id="a")
    q1 = _query(id="q-1", query="hit", expected=[("github", "a", 3)])
    q2 = _query(id="q-2", query="neg", category="no-match-expected", expected=[])
    dataset = _dataset((q1, q2))
    search_service = FakeSearchService(session, {"hit": [_result(incident_a)], "neg": []})

    report = evaluate(dataset, search_service, k=5)

    # Only q-1 contributes a defined recall/ndcg; q-2's None must not be
    # averaged in as 0, and must not reduce the mean below 1.0.
    assert report.aggregate_metrics.mean_recall_at_k == 1.0
    assert report.aggregate_metrics.mean_ndcg_at_k == pytest.approx(1.0)
    assert report.aggregate_metrics.num_queries == 2  # bucket size includes both


def test_evaluate_resolution_coverage_computed_across_relevant_and_unresolved(
    session: Session,
) -> None:
    incident_a = _add_incident(session, source_type="github", source_external_id="a")
    # q-1: 1 resolved expected incident. q-2: 1 unresolved expected incident.
    q1 = _query(id="q-1", query="q1", expected=[("github", "a", 3)])
    q2 = _query(id="q-2", query="q2", expected=[("github", "ghost", 1)])
    dataset = _dataset((q1, q2))
    search_service = FakeSearchService(session, {"q1": [_result(incident_a)], "q2": []})

    report = evaluate(dataset, search_service, k=5)

    # 1 resolved out of 2 total expected incidents.
    assert report.aggregate_metrics.resolution_coverage == pytest.approx(0.5)
    assert report.aggregate_metrics.queries_with_unresolved_incidents == 1
    assert report.resolution_summary.total_expected_incidents == 2
    assert report.resolution_summary.resolved_count == 1


# ── Empty dataset ──────────────────────────────────────────────────────────────


def test_evaluate_empty_dataset_does_not_raise_and_zeros_everything(session: Session) -> None:
    dataset = _dataset(())
    search_service = FakeSearchService(session, {})

    report = evaluate(dataset, search_service, k=5)

    assert report.num_evaluated == 0
    assert report.num_skipped == 0
    assert report.per_query == ()
    assert report.aggregate_metrics.num_queries == 0
    assert report.aggregate_metrics.mean_recall_at_k is None
    assert report.category_breakdown == {}
    assert report.difficulty_breakdown == {}
    assert report.coverage.total_queries == 0
    assert report.resolution_summary.total_expected_incidents == 0


# ── Report correctness: dataset metadata, config, corpus stats, execution ───


def test_evaluate_report_carries_dataset_metadata_verbatim(session: Session) -> None:
    dataset = GoldDataset(
        version="2.3.1",
        description="my dataset",
        created_at="2026-01-01T00:00:00Z",
        author="jane",
        queries=(_query(id="q-1", query="q", expected=[("github", "a", 1)]),),
    )
    search_service = FakeSearchService(session, {"q": []})

    report = evaluate(dataset, search_service, k=5)

    assert report.dataset.version == "2.3.1"
    assert report.dataset.description == "my dataset"
    assert report.dataset.created_at == "2026-01-01T00:00:00Z"
    assert report.dataset.author == "jane"
    assert report.dataset.corpus_fingerprint.computed is False


def test_evaluate_report_corpus_statistics_counts_distinct_retrieved_ids(
    session: Session,
) -> None:
    shared = uuid.uuid4()
    other = uuid.uuid4()
    q1 = _query(id="q-1", query="q1", expected=[("github", "x", 1)])
    q2 = _query(id="q-2", query="q2", expected=[("github", "y", 1)])
    dataset = _dataset((q1, q2))
    search_service = FakeSearchService(
        session,
        {
            "q1": [_result(shared), _result(other)],
            "q2": [_result(shared)],  # shared id retrieved again
        },
    )

    report = evaluate(dataset, search_service, k=5)

    assert report.corpus_statistics.distinct_retrieved_incident_count == 2


def test_evaluate_report_has_execution_metadata(session: Session) -> None:
    dataset = _dataset((_query(id="q-1", query="q", expected=[("github", "a", 1)]),))
    search_service = FakeSearchService(session, {"q": []})

    report = evaluate(dataset, search_service, k=5)

    assert report.started_at
    assert report.finished_at
    assert report.duration_seconds >= 0.0


def test_evaluate_rejects_non_positive_k(session: Session) -> None:
    dataset = _dataset((_query(id="q-1", query="q", expected=[("github", "a", 1)]),))
    search_service = FakeSearchService(session, {"q": []})

    with pytest.raises(ValueError):
        evaluate(dataset, search_service, k=0)


def test_evaluate_report_is_immutable(session: Session) -> None:
    dataset = _dataset((_query(id="q-1", query="q", expected=[("github", "a", 1)]),))
    search_service = FakeSearchService(session, {"q": []})

    report = evaluate(dataset, search_service, k=5)

    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        report.num_evaluated = 99  # type: ignore[misc]
