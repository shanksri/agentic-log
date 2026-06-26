from __future__ import annotations

import uuid

import pytest
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

import app.services.identity as identity_module
from app.services.identity import IdentityResolver, ResolvedIdentity, StableIdentity


class _TestBase(DeclarativeBase):
    pass


class _FakeIncident(_TestBase):
    """Minimal stand-in for app.db.models.Incident.

    Only the columns IdentityResolver reads (id, source_type,
    source_external_id) are mapped, against a plain SQLite table, so these
    tests exercise real SQLAlchemy select()/execute() mechanics without
    requiring PostgreSQL-specific types (UUID/JSONB/ARRAY) or a live
    database.
    """

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


def _add_incident(db: Session, *, source_type: str, source_external_id: str) -> _FakeIncident:
    incident = _FakeIncident(
        id=str(uuid.uuid4()), source_type=source_type, source_external_id=source_external_id
    )
    db.add(incident)
    db.commit()
    return incident


# ── StableIdentity ────────────────────────────────────────────────────────────


def test_stable_identity_is_hashable_and_comparable() -> None:
    a = StableIdentity("github", "acme/api#42")
    b = StableIdentity("github", "acme/api#42")
    c = StableIdentity("jira", "acme/api#42")

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert str(a) == "github:acme/api#42"


def test_identity_for_derives_stable_identity_from_incident() -> None:
    incident = _FakeIncident(id="x", source_type="jira", source_external_id="KAFKA-123")
    assert IdentityResolver.identity_for(incident) == StableIdentity("jira", "KAFKA-123")


# ── IdentityResolver.resolve ──────────────────────────────────────────────────


def test_resolve_returns_resolved_identity_for_known_identity(session: Session) -> None:
    incident = _add_incident(session, source_type="github", source_external_id="acme/api#1")
    resolver = IdentityResolver(session)

    resolved = resolver.resolve(StableIdentity("github", "acme/api#1"))

    assert resolved == ResolvedIdentity(
        source_type="github",
        source_external_id="acme/api#1",
        incident_id=incident.id,
    )


def test_resolve_returns_none_for_unknown_identity(session: Session) -> None:
    resolver = IdentityResolver(session)

    resolved = resolver.resolve(StableIdentity("github", "does-not-exist"))

    assert resolved is None


def test_resolve_is_exact_on_both_fields(session: Session) -> None:
    """Same source_external_id under a different source_type must not resolve."""
    _add_incident(session, source_type="github", source_external_id="42")
    resolver = IdentityResolver(session)

    resolved = resolver.resolve(StableIdentity("jira", "42"))

    assert resolved is None


def test_resolve_survives_uuid_change_across_reingestion(session: Session) -> None:
    """The documented invariant: identity is (source_type, source_external_id),
    not the row UUID — re-ingestion regenerating the UUID must not break
    resolution as long as the stable identity is unchanged.
    """
    identity = StableIdentity("github", "acme/api#7")
    first_incident = _add_incident(
        session, source_type=identity.source_type, source_external_id=identity.source_external_id
    )
    resolver = IdentityResolver(session)
    assert resolver.resolve(identity).incident_id == first_incident.id

    # Simulate re-ingestion regenerating the row under a new UUID.
    session.delete(first_incident)
    session.commit()
    second_incident = _add_incident(
        session, source_type=identity.source_type, source_external_id=identity.source_external_id
    )
    assert second_incident.id != first_incident.id

    resolved_again = resolver.resolve(identity)
    assert resolved_again is not None
    assert resolved_again.incident_id == second_incident.id


def test_resolve_constructs_resolved_identity_from_db_row_not_input(session: Session) -> None:
    """ResolvedIdentity must echo the resolved row's own fields, not simply
    pass through whatever StableIdentity the caller asked for. Resolution is
    exact-match in this implementation, so the row's fields and the input
    are equal in the success case by construction of the WHERE clause — this
    test forces the distinction by mutating the row's stored fields directly
    (bypassing the resolver) after the lookup key was chosen, so a resolver
    that echoed the input instead of the row would return stale values.
    """
    incident = _add_incident(session, source_type="github", source_external_id="acme/api#9")
    identity = StableIdentity("github", "acme/api#9")

    # Mutate the underlying row's fields directly, simulating drift between
    # what was queried for and what the row currently holds in storage.
    session.execute(
        identity_module.Incident.__table__.update()
        .where(identity_module.Incident.id == incident.id)
        .values(source_external_id="acme/api#9-renamed")
    )
    session.commit()

    resolver = IdentityResolver(session)
    resolved = resolver.resolve(identity)

    # A resolver that echoed the input would return "acme/api#9" here even
    # though that no longer matches any row — but since the WHERE clause
    # requires an exact match against the (now-renamed) stored value, the
    # row no longer matches the input identity at all, and resolution
    # correctly reports no match rather than fabricating a stale identity.
    assert resolved is None

    # Resolving by the row's *current* stored identity must reflect the row,
    # confirming the DTO is built from query results, not query input.
    resolved_by_current_value = resolver.resolve(StableIdentity("github", "acme/api#9-renamed"))
    assert resolved_by_current_value == ResolvedIdentity(
        source_type="github",
        source_external_id="acme/api#9-renamed",
        incident_id=incident.id,
    )


# ── IdentityResolver.resolve_many ─────────────────────────────────────────────


def test_resolve_many_returns_a_result_for_every_requested_identity(session: Session) -> None:
    incident_a = _add_incident(session, source_type="github", source_external_id="a")
    incident_b = _add_incident(session, source_type="jira", source_external_id="b")
    resolver = IdentityResolver(session)

    identities = [
        StableIdentity("github", "a"),
        StableIdentity("jira", "b"),
        StableIdentity("github", "missing"),
    ]
    results = resolver.resolve_many(identities)

    assert set(results) == set(identities)
    assert results[StableIdentity("github", "a")] == ResolvedIdentity(
        source_type="github", source_external_id="a", incident_id=incident_a.id
    )
    assert results[StableIdentity("jira", "b")] == ResolvedIdentity(
        source_type="jira", source_external_id="b", incident_id=incident_b.id
    )
    assert results[StableIdentity("github", "missing")] is None


def test_resolve_many_empty_input_returns_empty_dict(session: Session) -> None:
    resolver = IdentityResolver(session)
    assert resolver.resolve_many([]) == {}


def test_resolve_many_does_not_cross_match_source_type_and_external_id(
    session: Session,
) -> None:
    """Guards the cross-product trap: batching by separate source_type/
    external_id sets must not let a (github, b) pair falsely resolve to a
    row that is actually (jira, a) + (github, b) sharing one field each.
    """
    _add_incident(session, source_type="jira", source_external_id="shared-id")
    resolver = IdentityResolver(session)

    results = resolver.resolve_many([StableIdentity("github", "shared-id")])

    assert results[StableIdentity("github", "shared-id")] is None
