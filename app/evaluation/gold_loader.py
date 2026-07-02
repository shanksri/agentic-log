"""Gold Dataset v2 — parser/loader and identity resolution (Phase 16B).

Two responsibilities, kept separate from schema (``gold_dataset.py``):

1. **Parse + validate** a v2 gold dataset from JSON into ``GoldDataset``.
2. **Resolve** each gold query's expected incidents against the *current*
   corpus via ``IdentityResolver`` (Phase 16A), reporting which gold entries
   currently resolve to a live incident and which do not.

This module does NOT compute retrieval metrics (Recall/MRR/NDCG), does not
run any search, and does not implement the harness/regression
runner/dashboard. Resolution here answers "does this gold entry's identity
still exist in the corpus" — a structural/referential validation, not a
retrieval-quality measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.evaluation.gold_dataset import (
    CorpusFingerprintPlaceholder,
    ExpectedIncident,
    GoldDataset,
    GoldQuery,
)
from app.services.identity import IdentityResolver, ResolvedIdentity, StableIdentity


class GoldDatasetParseError(Exception):
    """Raised when gold dataset JSON is structurally malformed.

    Distinct from ``GoldDatasetValidationError``: this is for JSON that
    cannot even be mapped onto the schema (missing required keys, wrong
    types) — semantic problems with otherwise well-shaped data (bad
    category, out-of-range relevance, duplicate ids, ...) are reported by
    validation instead.
    """


class GoldDatasetValidationError(Exception):
    """Raised by ``load_gold_dataset`` when the parsed dataset fails
    validation. Carries every issue found, not just the first.
    """

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        message = "Gold dataset failed validation:\n" + "\n".join(
            f"  - {issue}" for issue in issues
        )
        super().__init__(message)


def _require_str(obj: dict[str, Any], key: str, *, context: str) -> str:
    if key not in obj:
        raise GoldDatasetParseError(f"{context}: missing required field {key!r}")
    value = obj[key]
    if not isinstance(value, str):
        raise GoldDatasetParseError(
            f"{context}: field {key!r} must be a string, got {type(value)!r}"
        )
    return value


def _require_int(obj: dict[str, Any], key: str, *, context: str) -> int:
    if key not in obj:
        raise GoldDatasetParseError(f"{context}: missing required field {key!r}")
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise GoldDatasetParseError(f"{context}: field {key!r} must be an int, got {type(value)!r}")
    return value


def _parse_expected_incident(raw: dict[str, Any], *, context: str) -> ExpectedIncident:
    if not isinstance(raw, dict):
        raise GoldDatasetParseError(f"{context}: expected_incident entries must be objects")
    return ExpectedIncident(
        source_type=_require_str(raw, "source_type", context=context),
        source_external_id=_require_str(raw, "source_external_id", context=context),
        relevance=_require_int(raw, "relevance", context=context),
    )


def _parse_gold_query(raw: dict[str, Any]) -> GoldQuery:
    if not isinstance(raw, dict):
        raise GoldDatasetParseError("query entries must be objects")
    query_id = _require_str(raw, "id", context="query")
    context = f"query {query_id!r}"
    expected_raw = raw.get("expected_incidents", [])
    if not isinstance(expected_raw, list):
        raise GoldDatasetParseError(f"{context}: expected_incidents must be a list")
    # reference_answer (Phase 22A) is optional and backward compatible:
    # absent and null both mean "no reference answer" (generation evaluation
    # skips this query); when present it must be a string.
    reference_answer = raw.get("reference_answer")
    if reference_answer is not None and not isinstance(reference_answer, str):
        raise GoldDatasetParseError(
            f"{context}: field 'reference_answer' must be a string or null, "
            f"got {type(reference_answer)!r}"
        )
    return GoldQuery(
        id=query_id,
        query=_require_str(raw, "query", context=context),
        category=_require_str(raw, "category", context=context),
        difficulty=_require_str(raw, "difficulty", context=context),
        expected_incidents=tuple(
            _parse_expected_incident(entry, context=context) for entry in expected_raw
        ),
        reference_answer=reference_answer,
    )


def _parse_corpus_fingerprint(raw: Any) -> CorpusFingerprintPlaceholder:
    if raw is None:
        return CorpusFingerprintPlaceholder()
    if not isinstance(raw, dict):
        raise GoldDatasetParseError("dataset.corpus_fingerprint must be an object or null")
    return CorpusFingerprintPlaceholder(
        computed=bool(raw.get("computed", False)),
        value=raw.get("value"),
    )


def parse_gold_dataset(raw: dict[str, Any]) -> GoldDataset:
    """Parse a JSON-decoded dict into a ``GoldDataset``.

    Raises ``GoldDatasetParseError`` on structurally malformed input. Does
    NOT run semantic validation — call ``dataset.issues()`` (or use
    ``load_gold_dataset``, which validates automatically) for that.
    """
    if not isinstance(raw, dict):
        raise GoldDatasetParseError("dataset root must be a JSON object")

    queries_raw = raw.get("queries", [])
    if not isinstance(queries_raw, list):
        raise GoldDatasetParseError("dataset.queries must be a list")

    return GoldDataset(
        version=_require_str(raw, "version", context="dataset"),
        description=_require_str(raw, "description", context="dataset"),
        created_at=_require_str(raw, "created_at", context="dataset"),
        queries=tuple(_parse_gold_query(entry) for entry in queries_raw),
        corpus_fingerprint=_parse_corpus_fingerprint(raw.get("corpus_fingerprint")),
        author=raw.get("author"),
    )


def load_gold_dataset(path: Path) -> GoldDataset:
    """Load, parse, and validate a Gold Dataset v2 JSON file.

    Raises ``GoldDatasetParseError`` if the JSON is structurally malformed,
    or ``GoldDatasetValidationError`` (carrying every issue found) if it
    parses but fails semantic validation.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset = parse_gold_dataset(raw)
    issues = dataset.issues()
    if issues:
        raise GoldDatasetValidationError(issues)
    return dataset


# ── Identity resolution against the current corpus ──────────────────────────


@dataclass(frozen=True)
class ResolvedExpectedIncident:
    """One expected incident, paired with its resolution against the
    current corpus (``None`` if it no longer resolves to a live incident).
    """

    expected: ExpectedIncident
    resolved: ResolvedIdentity | None

    @property
    def is_resolved(self) -> bool:
        return self.resolved is not None


@dataclass(frozen=True)
class ResolvedGoldQuery:
    """One gold query, with each of its expected incidents resolved (or not)
    against the current corpus.
    """

    query: GoldQuery
    resolved_incidents: tuple[ResolvedExpectedIncident, ...]

    @property
    def unresolved_count(self) -> int:
        return sum(1 for entry in self.resolved_incidents if not entry.is_resolved)

    @property
    def all_resolved(self) -> bool:
        return self.unresolved_count == 0


@dataclass(frozen=True)
class GoldDatasetResolutionSummary:
    """Aggregate resolution coverage across an entire resolved dataset.

    This is a structural/referential coverage count — how many gold entries
    currently exist in the corpus — NOT a retrieval-quality metric. No
    search is run to produce this.
    """

    total_expected_incidents: int
    resolved_count: int
    unresolved_identities: tuple[StableIdentity, ...]

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved_identities)

    @property
    def fully_covered(self) -> bool:
        return self.unresolved_count == 0


def resolve_gold_dataset(db: Session, dataset: GoldDataset) -> list[ResolvedGoldQuery]:
    """Resolve every expected incident in ``dataset`` against the current
    corpus via ``IdentityResolver``. One batched lookup for the whole
    dataset, regardless of query count.
    """
    all_identities = [
        StableIdentity(expected.source_type, expected.source_external_id)
        for gold_query in dataset.queries
        for expected in gold_query.expected_incidents
    ]
    resolved_by_identity = IdentityResolver(db).resolve_many(all_identities)

    resolved_queries: list[ResolvedGoldQuery] = []
    for gold_query in dataset.queries:
        resolved_incidents = tuple(
            ResolvedExpectedIncident(
                expected=expected,
                resolved=resolved_by_identity.get(
                    StableIdentity(expected.source_type, expected.source_external_id)
                ),
            )
            for expected in gold_query.expected_incidents
        )
        resolved_queries.append(
            ResolvedGoldQuery(query=gold_query, resolved_incidents=resolved_incidents)
        )
    return resolved_queries


def summarize_resolution(resolved_queries: list[ResolvedGoldQuery]) -> GoldDatasetResolutionSummary:
    """Aggregate resolution coverage across all resolved queries."""
    total = 0
    resolved_count = 0
    unresolved: list[StableIdentity] = []
    for resolved_query in resolved_queries:
        for entry in resolved_query.resolved_incidents:
            total += 1
            if entry.is_resolved:
                resolved_count += 1
            else:
                unresolved.append(
                    StableIdentity(
                        entry.expected.source_type, entry.expected.source_external_id
                    )
                )
    return GoldDatasetResolutionSummary(
        total_expected_incidents=total,
        resolved_count=resolved_count,
        unresolved_identities=tuple(unresolved),
    )
