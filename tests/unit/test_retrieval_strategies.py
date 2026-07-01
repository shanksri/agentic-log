from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

import app.evaluation.retrieval_strategies as retrieval_strategies_module
from app.evaluation.retrieval_strategies import (
    BM25RetrievalAdapter,
    HybridRetrievalAdapter,
    build_strategy,
    load_bm25_retriever,
)
from app.services.bm25_search import BM25Document, BM25Retriever
from app.services.hybrid_search import HybridRetriever
from app.services.search import IncidentSearchResult

# ── SQLite-backed fake Incident, used only to exercise load_bm25_retriever ──


class _TestBase(DeclarativeBase):
    pass


class _FakeIncident(_TestBase):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    canonical_text: Mapped[str] = mapped_column(Text, nullable=False)


@pytest.fixture(autouse=True)
def _patch_incident_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval_strategies_module, "Incident", _FakeIncident)


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    _TestBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        yield db


def _add_incident(db: Session, incident_id: str, text: str) -> None:
    db.add(_FakeIncident(id=incident_id, canonical_text=text))
    db.commit()


class FakeDenseService:
    def __init__(self, db, responses):
        self.db = db
        self._responses = responses

    def search(self, query, *, limit=10, call_site=None, **kwargs):
        return self._responses.get(query, [])


# ── load_bm25_retriever ───────────────────────────────────────────────────────


def test_load_bm25_retriever_indexes_every_incident_canonical_text(session: Session) -> None:
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    _add_incident(session, id_a, "scheduler crashloop ValidationError")
    _add_incident(session, id_b, "unrelated incident text")

    retriever = load_bm25_retriever(session)

    assert isinstance(retriever, BM25Retriever)
    assert retriever.index.size == 2
    results = retriever.retrieve("scheduler", limit=10)
    assert [r.document_id for r in results] == [id_a]


def test_load_bm25_retriever_empty_corpus_builds_empty_index(session: Session) -> None:
    retriever = load_bm25_retriever(session)
    assert retriever.index.size == 0
    assert retriever.retrieve("anything", limit=10) == []


# ── BM25RetrievalAdapter ───────────────────────────────────────────────────────


def test_bm25_adapter_converts_document_id_back_to_uuid(session: Session) -> None:
    incident_id = uuid.uuid4()
    bm25 = BM25Retriever.from_documents(
        [BM25Document(document_id=str(incident_id), text="memory leak crash")]
    )
    adapter = BM25RetrievalAdapter(session, bm25)

    [result] = adapter.retrieve("memory leak", limit=10)

    assert isinstance(result, IncidentSearchResult)
    assert result.incident.id == incident_id
    assert result.distance < 0  # negative score, per module docstring


def test_bm25_adapter_exposes_db() -> None:
    bm25 = BM25Retriever.from_documents([])
    adapter = BM25RetrievalAdapter("sentinel-db", bm25)
    assert adapter.db == "sentinel-db"


def test_bm25_adapter_rejects_expand() -> None:
    bm25 = BM25Retriever.from_documents([])
    adapter = BM25RetrievalAdapter("db", bm25)
    with pytest.raises(ValueError, match="expand"):
        adapter.retrieve("q", expand=True)


def test_bm25_adapter_rejects_rerank() -> None:
    bm25 = BM25Retriever.from_documents([])
    adapter = BM25RetrievalAdapter("db", bm25)
    with pytest.raises(ValueError, match="rerank"):
        adapter.retrieve("q", rerank=True)


# ── HybridRetrievalAdapter ─────────────────────────────────────────────────────


def test_hybrid_adapter_converts_document_id_back_to_uuid() -> None:
    incident_id = uuid.uuid4()
    dense = FakeDenseService("db", {})
    bm25 = BM25Retriever.from_documents(
        [BM25Document(document_id=str(incident_id), text="memory leak crash")]
    )
    hybrid = HybridRetriever(dense, bm25)
    adapter = HybridRetrievalAdapter("db", hybrid)

    [result] = adapter.retrieve("memory leak", limit=10)

    assert result.incident.id == incident_id


def test_hybrid_adapter_rejects_expand_and_rerank() -> None:
    hybrid = HybridRetriever(FakeDenseService("db", {}), BM25Retriever.from_documents([]))
    adapter = HybridRetrievalAdapter("db", hybrid)
    with pytest.raises(ValueError):
        adapter.retrieve("q", expand=True)
    with pytest.raises(ValueError):
        adapter.retrieve("q", rerank=True)


# ── build_strategy ─────────────────────────────────────────────────────────────


def test_build_strategy_dense_returns_search_service_unchanged() -> None:
    dense = FakeDenseService("db", {})
    assert build_strategy("dense", search_service=dense) is dense


def test_build_strategy_bm25_requires_bm25_retriever() -> None:
    dense = FakeDenseService("db", {})
    with pytest.raises(ValueError, match="bm25"):
        build_strategy("bm25", search_service=dense)


def test_build_strategy_bm25_returns_adapter() -> None:
    dense = FakeDenseService("db", {})
    bm25 = BM25Retriever.from_documents([])
    strategy = build_strategy("bm25", search_service=dense, bm25=bm25)
    assert isinstance(strategy, BM25RetrievalAdapter)
    assert strategy.db == "db"


def test_build_strategy_hybrid_requires_hybrid_factory() -> None:
    dense = FakeDenseService("db", {})
    bm25 = BM25Retriever.from_documents([])
    with pytest.raises(ValueError, match="hybrid_factory"):
        build_strategy("hybrid", search_service=dense, bm25=bm25)


def test_build_strategy_hybrid_returns_adapter() -> None:
    dense = FakeDenseService("db", {})
    bm25 = BM25Retriever.from_documents([])
    strategy = build_strategy(
        "hybrid", search_service=dense, bm25=bm25,
        hybrid_factory=lambda: HybridRetriever(dense, bm25),
    )
    assert isinstance(strategy, HybridRetrievalAdapter)


def test_build_strategy_unknown_name_raises() -> None:
    dense = FakeDenseService("db", {})
    with pytest.raises(ValueError, match="unknown"):
        build_strategy("nonexistent", search_service=dense)  # type: ignore[arg-type]
