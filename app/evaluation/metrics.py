"""Retrieval Metric Engine (Phase 16C).

Scores a single query's already-retrieved results against its resolved gold
answer set. This module is a pure mathematical layer:

- It never performs retrieval (no ``IncidentSearchService``, no embedding,
  no LLM, no HTTP/network call of any kind).
- It never resolves identities (no ``IdentityResolver``, no database
  access). It consumes ``ResolvedGoldQuery`` (Phase 16B) as already-resolved
  input — resolution has already happened upstream.
- It never executes the evaluation harness (no multi-query orchestration,
  no aggregation across queries, no run/baseline/regression concerns). Those
  are Phase 16D+.

Inputs are intentionally primitive: retrieved results are a plain,
rank-ordered ``Sequence[uuid.UUID]`` — the same UUID space
``IncidentSearchResult.incident.id`` and ``ResolvedIdentity.incident_id``
already share — not an ``IncidentSearchResult`` or any other retrieval-layer
type. This is what keeps the engine decoupled from ``SearchService``: a
caller (the future harness) maps its search results down to a UUID list
before calling in, and this module never needs to know how that list was
produced.

# Metric definitions and formulas

Let ``R`` = the gold-derived relevant set: ``{incident_id: relevance grade}``,
built only from the *resolved* expected incidents of a ``ResolvedGoldQuery``
(see ``relevant_grades_from_resolved_gold``). Let ``D`` = the deduplicated,
rank-ordered list of retrieved incident ids (first occurrence kept; see
"Duplicate retrieved ids" below). Let ``k`` be the cutoff rank (``k >= 1``).

- **Recall@K** = ``|top_k(D) ∩ R| / |R|``, where ``top_k(D)`` is the first
  ``k`` ids of ``D``. Undefined (``None``) when ``R`` is empty — there is
  nothing to recall (see "Zero relevant items" below).
- **Reciprocal rank** (this query's contribution to Mean Reciprocal Rank;
  the *mean* across queries is computed by the harness, not here) =
  ``1 / rank`` of the first id in ``D`` that is in ``R``, or ``0.0`` if no
  such id appears anywhere in ``D``. Undefined (``None``) when ``R`` is
  empty.
- **DCG@K** (graded, exponential-gain form — Järvelin & Kekäläinen) =
  ``sum_{i=1..k} (2^rel(D[i]) - 1) / log2(i + 1)``, where ``rel(id) = R[id]``
  if ``id in R`` else ``0``. Always well-defined (``0.0`` when nothing
  retrieved is relevant or nothing was retrieved).
- **IDCG@K** = the same formula applied to the relevance grades of ``R``
  sorted descending and truncated to ``k`` — i.e. the DCG of the best
  possible ranking. Depends only on the gold relevance grades, never on
  what was actually retrieved. ``0.0`` when ``R`` is empty.
- **NDCG@K** = ``DCG@K / IDCG@K``. Undefined (``None``) when ``IDCG@K`` is
  ``0`` (equivalently, when ``R`` is empty) — mirrors Recall/reciprocal rank
  rather than adopting a 0/0 := 1.0 convention, since there is no "ranking
  quality" to score when nothing is relevant.

This module follows the v1 harness's own convention
(``tests/eval/run_retrieval_eval.py``) of returning ``None`` rather than
``0.0`` or ``1.0`` for metrics that are mathematically undefined when the
relevant set is empty, so "no signal" is never silently conflated with
"worst possible score."
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.evaluation.gold_loader import ResolvedGoldQuery


def _require_positive_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")


def _dedupe_keep_first(retrieved: Sequence[uuid.UUID]) -> list[uuid.UUID]:
    """Drop repeated ids, keeping each id's first (best-ranked) occurrence.

    Edge case: duplicate retrieved UUIDs. A later duplicate of an id already
    seen at a better rank must not count as a second hit, and must not
    shift other ids' effective rank by occupying an extra position — both
    would distort Recall/DCG/reciprocal-rank in a way that doesn't reflect
    genuine retrieval behavior. This mirrors the project's existing
    best-distance candidate-merge philosophy (docs/architecture/12) of
    keeping one slot per incident at its best standing. A duplicate
    appearing upstream is treated as a defect in the supplied retrieved
    list, not a metrics-engine concern to surface — it is silently
    normalized away here.
    """
    return list(dict.fromkeys(retrieved))


def relevant_grades_from_resolved_gold(
    resolved_gold_query: ResolvedGoldQuery,
) -> dict[uuid.UUID, int]:
    """Build the ``{incident_id: relevance}`` relevant set from a resolved
    gold query.

    Edge case: unresolved expected incidents. An expected incident whose
    ``ResolvedExpectedIncident.resolved`` is ``None`` (the stable identity
    no longer maps to a current incident — see Phase 16A/16B) is EXCLUDED
    from the returned set. This is deliberate: the metric engine cannot
    fault retrieval for failing to retrieve an incident that does not
    currently exist in the corpus. Treating an unresolved expected incident
    as "expected but missed" would permanently and unfairly depress
    Recall/NDCG for a query whose gold entry has simply gone stale (e.g. the
    referenced incident was deleted or re-identified), conflating a
    data-quality problem with a retrieval-quality one. Callers that care
    about resolution coverage as its own signal should consult
    ``ResolvedGoldQuery.unresolved_count`` /
    ``GoldDatasetResolutionSummary`` (Phase 16B) directly, not infer it from
    metric scores.

    Edge case: two distinct stable identities resolving to the same
    ``incident_id``. This should not occur given identity uniqueness
    (Phase 16A/16B validation), but defensively, if it ever did, the higher
    of the two relevance grades is kept rather than an arbitrary
    last-write-wins value, so no authored relevance signal is silently lost.
    """
    grades: dict[uuid.UUID, int] = {}
    for entry in resolved_gold_query.resolved_incidents:
        if entry.resolved is None:
            continue
        incident_id = entry.resolved.incident_id
        relevance = entry.expected.relevance
        if incident_id in grades:
            grades[incident_id] = max(grades[incident_id], relevance)
        else:
            grades[incident_id] = relevance
    return grades


def recall_at_k(
    retrieved: Sequence[uuid.UUID], relevant_ids: set[uuid.UUID] | frozenset[uuid.UUID], k: int
) -> float | None:
    """Fraction of relevant ids found within the top ``k`` retrieved.

    Returns ``None`` when ``relevant_ids`` is empty (zero relevant items —
    covers both "no-match-expected" queries and gold queries whose every
    expected incident is unresolved). Returns ``0.0`` (not ``None``) when
    retrieved is empty but relevant items exist — nothing was retrieved, so
    nothing was recalled. K larger than the retrieved count is handled by
    slicing (no error, no padding); K larger than the relevant count does
    not change the denominator, which is always ``len(relevant_ids)``.
    """
    _require_positive_k(k)
    if not relevant_ids:
        return None
    top_k = set(_dedupe_keep_first(retrieved)[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def reciprocal_rank(
    retrieved: Sequence[uuid.UUID], relevant_ids: set[uuid.UUID] | frozenset[uuid.UUID]
) -> float | None:
    """This query's contribution to Mean Reciprocal Rank: ``1 / rank`` of the
    first relevant id encountered in ``retrieved``, or ``0.0`` if none of
    ``retrieved`` is relevant. The *mean* across queries is the harness's
    responsibility (Phase 16D), not this function's.

    Returns ``None`` when ``relevant_ids`` is empty, for the same reason as
    ``recall_at_k``. Not bounded by any ``k`` — reciprocal rank is defined
    over the full retrieved list, since a relevant hit at rank 1000 is still
    a (very weak) hit, not a miss.
    """
    if not relevant_ids:
        return None
    relevant_set = set(relevant_ids)
    for rank, incident_id in enumerate(_dedupe_keep_first(retrieved), start=1):
        if incident_id in relevant_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(
    retrieved: Sequence[uuid.UUID], relevance_by_id: Mapping[uuid.UUID, int], k: int
) -> float:
    """Discounted Cumulative Gain at rank ``k``, graded exponential-gain form:
    ``sum_{i=1..k} (2^rel(D[i]) - 1) / log2(i + 1)``.

    Ids not present in ``relevance_by_id`` (retrieved incidents outside the
    gold set) contribute a gain of ``0`` — they neither help nor hurt DCG
    directly, though they do occupy a rank position and so can still push
    genuinely relevant ids to worse ranks, which IS reflected. Always
    well-defined; returns ``0.0`` for empty retrieved input or when nothing
    retrieved is relevant.
    """
    _require_positive_k(k)
    total = 0.0
    for rank, incident_id in enumerate(_dedupe_keep_first(retrieved)[:k], start=1):
        relevance = relevance_by_id.get(incident_id, 0)
        if relevance > 0:
            total += (2**relevance - 1) / math.log2(rank + 1)
    return total


def ideal_dcg_at_k(relevance_grades: Sequence[int], k: int) -> float:
    """The best-possible DCG@K given a multiset of relevance grades — i.e.
    the DCG of those grades sorted descending, truncated to ``k``.

    Depends only on the gold relevance grades, never on what was actually
    retrieved (the ideal ranking is a property of the gold set alone). When
    ``k`` exceeds the number of grades, all grades are used (no padding with
    zero-grade phantom items, since those would contribute nothing anyway).
    Returns ``0.0`` when ``relevance_grades`` is empty.
    """
    _require_positive_k(k)
    ordered = sorted(relevance_grades, reverse=True)[:k]
    return sum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ordered, start=1)
    )


def ndcg_at_k(
    retrieved: Sequence[uuid.UUID], relevance_by_id: Mapping[uuid.UUID, int], k: int
) -> float | None:
    """Normalized DCG@K = ``dcg_at_k / ideal_dcg_at_k``.

    Returns ``None`` when the ideal DCG is ``0`` — equivalently, when there
    are no relevant items at all. This mirrors ``recall_at_k`` and
    ``reciprocal_rank`` rather than adopting a 0/0 := 1.0 convention: there
    is no ranking quality to assess when nothing is relevant, so "undefined"
    is reported rather than a number that could be misread as a perfect or a
    zero score.
    """
    _require_positive_k(k)
    ideal = ideal_dcg_at_k(list(relevance_by_id.values()), k)
    if ideal == 0:
        return None
    return dcg_at_k(retrieved, relevance_by_id, k) / ideal


@dataclass(frozen=True)
class QueryMetricResult:
    """The complete, immutable metric outcome for one query at one cutoff
    ``k``. Every field is a plain primitive (str/int/float/None) — no
    reference to ``Incident``, ``GoldQuery``, or any ORM/retrieval type, so
    this result is trivially serializable (e.g. into a future run.json
    artifact, per docs/architecture/15_evaluation_framework.md) without
    pulling in any of this engine's input types.
    """

    query_id: str
    k: int
    num_relevant: int
    num_unresolved_expected: int
    num_retrieved: int
    num_duplicate_retrieved: int
    recall_at_k: float | None
    reciprocal_rank: float | None
    dcg_at_k: float
    idcg_at_k: float
    ndcg_at_k: float | None


def score_query(
    retrieved: Sequence[uuid.UUID], resolved_gold_query: ResolvedGoldQuery, *, k: int
) -> QueryMetricResult:
    """Score one query's retrieved results against its resolved gold answer
    set at cutoff ``k``. The single entry point this module is meant to be
    used through; the lower-level functions above remain public for direct
    testing and for harness composition (e.g. computing several ``k``
    values without recomputing the relevant set if that is ever a measured
    bottleneck — not optimized here, since at gold-set scale it is not one).
    """
    _require_positive_k(k)
    relevant_by_id = relevant_grades_from_resolved_gold(resolved_gold_query)
    relevant_ids = set(relevant_by_id)
    deduped = _dedupe_keep_first(retrieved)

    return QueryMetricResult(
        query_id=resolved_gold_query.query.id,
        k=k,
        num_relevant=len(relevant_by_id),
        num_unresolved_expected=resolved_gold_query.unresolved_count,
        num_retrieved=len(retrieved),
        num_duplicate_retrieved=len(retrieved) - len(deduped),
        recall_at_k=recall_at_k(deduped, relevant_ids, k),
        reciprocal_rank=reciprocal_rank(deduped, relevant_ids),
        dcg_at_k=dcg_at_k(deduped, relevant_by_id, k),
        idcg_at_k=ideal_dcg_at_k(list(relevant_by_id.values()), k),
        ndcg_at_k=ndcg_at_k(deduped, relevant_by_id, k),
    )
