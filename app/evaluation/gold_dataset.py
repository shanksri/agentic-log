"""Gold Dataset v2 — schema and validation (Phase 16B).

Replaces the v1 gold format (``tests/eval/gold_queries.json``: a flat list of
``{id, query, query_type, expected_incident_ids}`` entries anchored on raw
UUID strings) with an identity-anchored, graded, versioned format, per
docs/architecture/15_evaluation_framework.md:

- **Identity-anchored, not UUID-anchored.** Each expected incident is a
  ``StableIdentity`` (``source_type`` + ``source_external_id``), not a UUID —
  UUIDs regenerate on re-ingestion (docs/architecture/05_deduplication.md);
  stable identity does not.
- **Graded, multi-answer relevance.** A query may have zero, one, or many
  expected incidents, each with its own relevance grade, rather than a single
  flat list of equally-weighted expected ids.
- **Versioned, with dataset-level metadata** (version, description,
  created_at, corpus fingerprint, optional author) so future runs can be
  attributed to a specific dataset version (doc 15's regression runner
  refuses to compare runs across incompatible gold versions/fingerprints).

This module is schema + validation ONLY. It does not compute retrieval
metrics (Recall/MRR/NDCG), does not run any search, and does not implement
the harness, regression runner, or dashboard — those are later phases (doc
15/17, Phase 16C+). It also does not implement corpus fingerprinting itself;
``corpus_fingerprint`` is carried as an explicit placeholder field only (see
``CorpusFingerprintPlaceholder``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Closed set of query categories, matching the buckets already used by the
# v1 gold set / Phase 0 plan (RETRIEVAL_V2.5_PLAN.md): lexical-overlap,
# paraphrase/semantic, multi-concept, and no-match-expected (negative
# controls, where an empty expected_incidents list is the correct shape).
VALID_CATEGORIES = frozenset(
    {"lexical-overlap", "paraphrase", "multi-concept", "no-match-expected"}
)

# Closed set of difficulty labels for v2. A new, explicit field — v1 had no
# difficulty concept.
VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})

# Graded relevance scale for v2. 1 = marginally relevant, 2 = relevant,
# 3 = exact/primary match. This is a deliberate, documented design choice
# for this phase (doc 15 specifies that relevance must be graded and feed
# NDCG, but does not fix a numeric scale) — chosen to be small enough to
# author by hand and wide enough to distinguish "the" answer from a merely
# acceptable alternative.
RELEVANCE_MIN = 1
RELEVANCE_MAX = 3


@dataclass(frozen=True)
class CorpusFingerprintPlaceholder:
    """Placeholder for the corpus fingerprint a dataset run was checked against.

    Phase 16B does not implement corpus fingerprinting (that is the
    Fingerprint component in doc 15's harness architecture, a later phase).
    This dataclass exists only so the dataset schema has a stable field to
    carry a fingerprint once one is computed — ``computed`` is always
    ``False`` and ``value`` is always ``None`` until that future phase fills
    it in. Do not interpret an absent/placeholder fingerprint as "the
    dataset has no corpus to fingerprint" — it means fingerprinting has not
    been implemented yet.
    """

    computed: bool = False
    value: str | None = None


@dataclass(frozen=True)
class ExpectedIncident:
    """One acceptable answer for a gold query, identity-anchored and graded."""

    source_type: str
    source_external_id: str
    relevance: int

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.source_type:
            problems.append("expected_incident.source_type must be non-empty")
        if not self.source_external_id:
            problems.append("expected_incident.source_external_id must be non-empty")
        if not (RELEVANCE_MIN <= self.relevance <= RELEVANCE_MAX):
            problems.append(
                f"expected_incident relevance {self.relevance!r} must be in "
                f"[{RELEVANCE_MIN}, {RELEVANCE_MAX}] "
                f"(source_type={self.source_type!r}, "
                f"source_external_id={self.source_external_id!r})"
            )
        return problems


@dataclass(frozen=True)
class GoldQuery:
    """One gold query entry: a natural-language query plus its expected,
    graded, identity-anchored answer set.
    """

    id: str
    query: str
    category: str
    difficulty: str
    expected_incidents: tuple[ExpectedIncident, ...] = field(default_factory=tuple)

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.id:
            problems.append("query.id must be non-empty")
        if not self.query:
            problems.append(f"query {self.id!r}: query text must be non-empty")
        if self.category not in VALID_CATEGORIES:
            problems.append(
                f"query {self.id!r}: category {self.category!r} not in {sorted(VALID_CATEGORIES)}"
            )
        if self.difficulty not in VALID_DIFFICULTIES:
            problems.append(
                f"query {self.id!r}: difficulty {self.difficulty!r} not in "
                f"{sorted(VALID_DIFFICULTIES)}"
            )
        if self.category != "no-match-expected" and not self.expected_incidents:
            problems.append(
                f"query {self.id!r}: category {self.category!r} requires at least one "
                "expected_incident (only 'no-match-expected' may have zero)"
            )
        if self.category == "no-match-expected" and self.expected_incidents:
            problems.append(
                f"query {self.id!r}: category 'no-match-expected' must have zero "
                "expected_incidents (negative control)"
            )
        seen_identities: set[tuple[str, str]] = set()
        for expected in self.expected_incidents:
            problems.extend(f"query {self.id!r}: {issue}" for issue in expected.issues())
            key = (expected.source_type, expected.source_external_id)
            if key in seen_identities:
                problems.append(
                    f"query {self.id!r}: duplicate expected_incident identity "
                    f"{expected.source_type}:{expected.source_external_id}"
                )
            seen_identities.add(key)
        return problems


@dataclass(frozen=True)
class GoldDataset:
    """A versioned Gold Dataset v2: dataset-level metadata plus gold queries."""

    version: str
    description: str
    created_at: str
    queries: tuple[GoldQuery, ...]
    corpus_fingerprint: CorpusFingerprintPlaceholder = field(
        default_factory=CorpusFingerprintPlaceholder
    )
    author: str | None = None

    def issues(self) -> list[str]:
        """Return all validation problems found, or an empty list if valid.

        Never raises. Callers that want fail-fast behavior should use
        ``GoldDatasetValidationError``/``load_gold_dataset`` in
        ``app.evaluation.gold_loader``.
        """
        problems: list[str] = []
        if not self.version:
            problems.append("dataset.version must be non-empty")
        if not self.description:
            problems.append("dataset.description must be non-empty")
        if not self.created_at:
            problems.append("dataset.created_at must be non-empty")
        if not self.queries:
            problems.append("dataset.queries must be non-empty")

        seen_ids: set[str] = set()
        for gold_query in self.queries:
            problems.extend(gold_query.issues())
            if gold_query.id in seen_ids:
                problems.append(f"duplicate query id {gold_query.id!r}")
            seen_ids.add(gold_query.id)
        return problems

    def is_valid(self) -> bool:
        return not self.issues()
