from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

import app.services.identity as identity_module
from app.evaluation.gold_dataset import ExpectedIncident, GoldDataset, GoldQuery
from app.evaluation.gold_loader import (
    GoldDatasetParseError,
    GoldDatasetValidationError,
    load_gold_dataset,
    parse_gold_dataset,
    resolve_gold_dataset,
    summarize_resolution,
)
from app.services.identity import StableIdentity

SAMPLE_PATH = Path(__file__).parent.parent / "eval" / "gold" / "sample_gold_v2.json"


def _valid_raw() -> dict:
    return {
        "version": "2.0.0",
        "description": "d",
        "created_at": "2026-06-26T00:00:00Z",
        "queries": [
            {
                "id": "q-1",
                "query": "scheduler crashes",
                "category": "lexical-overlap",
                "difficulty": "easy",
                "expected_incidents": [
                    {
                        "source_type": "github",
                        "source_external_id": "acme/api#1",
                        "relevance": 3,
                    }
                ],
            }
        ],
    }


# ── parse_gold_dataset: structural parsing ───────────────────────────────────


def test_parse_gold_dataset_builds_dataset_from_valid_raw() -> None:
    dataset = parse_gold_dataset(_valid_raw())

    assert isinstance(dataset, GoldDataset)
    assert dataset.version == "2.0.0"
    assert len(dataset.queries) == 1
    assert dataset.queries[0] == GoldQuery(
        id="q-1",
        query="scheduler crashes",
        category="lexical-overlap",
        difficulty="easy",
        expected_incidents=(
            ExpectedIncident(source_type="github", source_external_id="acme/api#1", relevance=3),
        ),
    )
    assert dataset.author is None
    assert dataset.corpus_fingerprint.computed is False
    assert dataset.corpus_fingerprint.value is None


def test_parse_gold_dataset_rejects_non_dict_root() -> None:
    with pytest.raises(GoldDatasetParseError):
        parse_gold_dataset([])  # type: ignore[arg-type]


def test_parse_gold_dataset_rejects_missing_required_field() -> None:
    raw = _valid_raw()
    del raw["version"]
    with pytest.raises(GoldDatasetParseError, match="version"):
        parse_gold_dataset(raw)


def test_parse_gold_dataset_rejects_wrong_type_for_queries() -> None:
    raw = _valid_raw()
    raw["queries"] = "not-a-list"
    with pytest.raises(GoldDatasetParseError, match="queries"):
        parse_gold_dataset(raw)


def test_parse_gold_dataset_rejects_non_int_relevance() -> None:
    raw = _valid_raw()
    raw["queries"][0]["expected_incidents"][0]["relevance"] = "high"
    with pytest.raises(GoldDatasetParseError, match="relevance"):
        parse_gold_dataset(raw)


def test_parse_gold_dataset_parses_explicit_corpus_fingerprint() -> None:
    raw = _valid_raw()
    raw["corpus_fingerprint"] = {"computed": True, "value": "abc123"}
    dataset = parse_gold_dataset(raw)
    assert dataset.corpus_fingerprint.computed is True
    assert dataset.corpus_fingerprint.value == "abc123"


def test_parse_gold_dataset_rejects_non_object_corpus_fingerprint() -> None:
    raw = _valid_raw()
    raw["corpus_fingerprint"] = "not-an-object"
    with pytest.raises(GoldDatasetParseError, match="corpus_fingerprint"):
        parse_gold_dataset(raw)


def test_parse_gold_dataset_carries_optional_author() -> None:
    raw = _valid_raw()
    raw["author"] = "jane"
    dataset = parse_gold_dataset(raw)
    assert dataset.author == "jane"


# ── load_gold_dataset: file IO + validation ──────────────────────────────────


def test_load_gold_dataset_loads_and_validates_sample_fixture() -> None:
    dataset = load_gold_dataset(SAMPLE_PATH)

    assert dataset.version == "2.0.0"
    assert dataset.is_valid()
    assert {q.id for q in dataset.queries} == {"v2-lex-01", "v2-multi-01", "v2-neg-01"}
    neg_query = next(q for q in dataset.queries if q.id == "v2-neg-01")
    assert neg_query.category == "no-match-expected"
    assert neg_query.expected_incidents == ()
    multi_query = next(q for q in dataset.queries if q.id == "v2-multi-01")
    assert len(multi_query.expected_incidents) == 2


def test_load_gold_dataset_raises_validation_error_for_semantically_invalid_dataset(
    tmp_path: Path,
) -> None:
    raw = _valid_raw()
    raw["queries"][0]["category"] = "not-a-real-category"
    bad_path = tmp_path / "bad_gold.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(GoldDatasetValidationError) as exc_info:
        load_gold_dataset(bad_path)

    assert any("category" in issue for issue in exc_info.value.issues)


def test_load_gold_dataset_raises_parse_error_for_malformed_json(tmp_path: Path) -> None:
    raw = _valid_raw()
    del raw["description"]
    bad_path = tmp_path / "malformed_gold.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(GoldDatasetParseError, match="description"):
        load_gold_dataset(bad_path)


# ── Identity resolution against the current corpus ───────────────────────────


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


def _add_incident(db: Session, *, source_type: str, source_external_id: str) -> None:
    db.add(
        _FakeIncident(
            id=str(uuid.uuid4()), source_type=source_type, source_external_id=source_external_id
        )
    )
    db.commit()


def test_resolve_gold_dataset_marks_existing_incidents_resolved(session: Session) -> None:
    _add_incident(session, source_type="github", source_external_id="acme/api#1")
    dataset = parse_gold_dataset(_valid_raw())

    resolved = resolve_gold_dataset(session, dataset)

    assert len(resolved) == 1
    [resolved_query] = resolved
    assert resolved_query.query.id == "q-1"
    [resolved_incident] = resolved_query.resolved_incidents
    assert resolved_incident.is_resolved
    assert resolved_incident.resolved.source_type == "github"
    assert resolved_incident.resolved.source_external_id == "acme/api#1"
    assert resolved_query.unresolved_count == 0
    assert resolved_query.all_resolved is True


def test_resolve_gold_dataset_marks_missing_incidents_unresolved(session: Session) -> None:
    # No incidents added — the gold entry cannot resolve.
    dataset = parse_gold_dataset(_valid_raw())

    resolved = resolve_gold_dataset(session, dataset)

    [resolved_query] = resolved
    [resolved_incident] = resolved_query.resolved_incidents
    assert resolved_incident.is_resolved is False
    assert resolved_incident.resolved is None
    assert resolved_query.unresolved_count == 1
    assert resolved_query.all_resolved is False


def test_resolve_gold_dataset_handles_no_match_expected_query_with_zero_expected(
    session: Session,
) -> None:
    dataset = load_gold_dataset(SAMPLE_PATH)

    resolved = resolve_gold_dataset(session, dataset)

    neg_resolved = next(rq for rq in resolved if rq.query.id == "v2-neg-01")
    assert neg_resolved.resolved_incidents == ()
    assert neg_resolved.unresolved_count == 0
    assert neg_resolved.all_resolved is True


def test_resolve_gold_dataset_handles_multi_answer_query_partially_resolved(
    session: Session,
) -> None:
    _add_incident(session, source_type="github", source_external_id="microsoft/TypeScript#2001")
    # microsoft/TypeScript#2002 deliberately not added.
    dataset = load_gold_dataset(SAMPLE_PATH)

    resolved = resolve_gold_dataset(session, dataset)

    multi_resolved = next(rq for rq in resolved if rq.query.id == "v2-multi-01")
    assert multi_resolved.unresolved_count == 1
    assert multi_resolved.all_resolved is False
    resolved_flags = {
        entry.expected.source_external_id: entry.is_resolved
        for entry in multi_resolved.resolved_incidents
    }
    assert resolved_flags == {
        "microsoft/TypeScript#2001": True,
        "microsoft/TypeScript#2002": False,
    }


def test_summarize_resolution_aggregates_across_full_sample_dataset(session: Session) -> None:
    _add_incident(session, source_type="github", source_external_id="apache/airflow#1001")
    _add_incident(session, source_type="github", source_external_id="microsoft/TypeScript#2001")
    dataset = load_gold_dataset(SAMPLE_PATH)

    resolved = resolve_gold_dataset(session, dataset)
    summary = summarize_resolution(resolved)

    # 3 expected_incidents total across the sample dataset (1 + 2 + 0).
    assert summary.total_expected_incidents == 3
    assert summary.resolved_count == 2
    assert summary.unresolved_count == 1
    assert summary.unresolved_identities == (
        StableIdentity("github", "microsoft/TypeScript#2002"),
    )
    assert summary.fully_covered is False


def test_summarize_resolution_fully_covered_when_everything_resolves(session: Session) -> None:
    _add_incident(session, source_type="github", source_external_id="acme/api#1")
    dataset = parse_gold_dataset(_valid_raw())

    resolved = resolve_gold_dataset(session, dataset)
    summary = summarize_resolution(resolved)

    assert summary.fully_covered is True
    assert summary.unresolved_count == 0


# ── reference_answer parsing (Phase 22A) ─────────────────────────────────────


def test_parse_dataset_without_reference_answer_still_works() -> None:
    # Backward compatibility: a pre-22A dataset (no reference_answer field
    # anywhere) parses exactly as before, with None on every query.
    dataset = parse_gold_dataset(_valid_raw())
    assert dataset.queries[0].reference_answer is None


def test_parse_reference_answer_string_passes_through() -> None:
    raw = _valid_raw()
    raw["queries"][0]["reference_answer"] = "restart the kafka broker"
    dataset = parse_gold_dataset(raw)
    assert dataset.queries[0].reference_answer == "restart the kafka broker"


def test_parse_reference_answer_null_means_none() -> None:
    raw = _valid_raw()
    raw["queries"][0]["reference_answer"] = None
    dataset = parse_gold_dataset(raw)
    assert dataset.queries[0].reference_answer is None


def test_parse_reference_answer_non_string_rejected() -> None:
    raw = _valid_raw()
    raw["queries"][0]["reference_answer"] = 42
    with pytest.raises(GoldDatasetParseError, match="reference_answer"):
        parse_gold_dataset(raw)


def test_sample_gold_dataset_on_disk_still_loads() -> None:
    # The committed sample dataset predates Phase 22A and has no
    # reference_answer field — it must load unchanged.
    dataset = load_gold_dataset(SAMPLE_PATH)
    assert all(q.reference_answer is None for q in dataset.queries)
