"""Regression Runner (Phase 16E).

Compares two completed ``EvaluationReport``s (Phase 16D) and produces a
``RegressionReport`` classifying retrieval quality as improved, regressed,
unchanged, or mixed. This module is a pure comparison layer:

- It never re-runs evaluation, never calls ``IncidentSearchService``, never
  calls ``IdentityResolver``, never touches the database, and never invokes
  an LLM. It operates exclusively on two already-built ``EvaluationReport``
  objects (Phase 16D) — no other input.
- It never computes Recall/MRR/NDCG. It only reads the already-computed
  values off ``AggregateMetrics`` (Phase 16D, itself built from Phase 16C)
  and arithmetically diffs them.
- It never resolves identities and never re-derives resolution coverage —
  it reads ``AggregateMetrics.resolution_coverage`` and
  ``QueryEvaluationOutcome.num_unresolved_expected``, both already computed
  by Phase 16D from Phase 16B's resolution output.

# Regression lifecycle

```
compare(baseline, candidate) -> RegressionReport
  1. _check_compatibility(baseline, candidate) -> CompatibilityCheck
       - incompatible -> return immediately with verdict=INCOMPATIBLE,
         no deltas computed (see "Compatibility rules" below)
  2. overall metric deltas: Recall@K, MRR, NDCG, resolution coverage,
     queries-with-unresolved-incidents, queries skipped
  3. overall verdict, derived from Recall@K/MRR/NDCG deltas only (see
     "Regression detection" below)
  4. category_breakdown deltas, difficulty_breakdown deltas (same delta
     shape as step 2, scoped per bucket)
  5. newly skipped query ids, newly unresolved query ids (set difference
     over per_query outcomes, keyed by query_id)
  6. assemble and return an immutable RegressionReport
```

# Compatibility rules

A comparison is rejected (``CompatibilityCheck.compatible = False``, with
every applicable reason listed — never just the first) when:

1. **Gold dataset versions differ** (``baseline.dataset.version !=
   candidate.dataset.version``). Comparing across dataset versions conflates
   "did retrieval change" with "did the gold set change" — exactly the
   failure mode that invalidated the v1 gold baseline
   (docs/architecture/15_evaluation_framework.md).
2. **Corpus fingerprints differ** (``baseline.dataset.corpus_fingerprint !=
   candidate.dataset.corpus_fingerprint``). See the important caveat below.
3. **Evaluation K differs** (``baseline.config.k != candidate.config.k``).
   Recall@5 and Recall@10 are not the same metric; diffing them numerically
   would be comparing two different cutoffs, not measuring quality change.
4. **Query coverage differs** — the set of ``query_id``s present in
   ``baseline.per_query`` and ``candidate.per_query`` must be identical.
   This is not explicitly named in doc 15 but is the structural
   precondition every other check assumes: per-query, per-category, and
   per-difficulty deltas are meaningless if the two reports scored different
   query sets (e.g. someone accidentally diffs two unrelated gold files that
   happen to share a version string by coincidence).

**Deliberately NOT rejected: ``expand``/``rerank`` differing.** Comparing two
different retrieval configurations over the *same* gold set and corpus is a
legitimate, intended use of this runner — it is exactly how Phase 2 measured
query expansion's and reranking's effect. Rejecting on config difference
would make the regression runner unable to answer "what does enabling
rerank do," which is one of its two core use cases (the other being "did
retrieval regress over time with the same config"). ``ComparisonMetadata``
records both sides' ``expand``/``rerank`` values so a reader always knows
which kind of comparison they are looking at.

**Corpus fingerprint caveat.** Phase 16B's ``corpus_fingerprint`` is an
explicit, documented placeholder (``computed=False, value=None`` always, as
of this phase — real fingerprinting is unimplemented; see
``app.evaluation.gold_dataset.CorpusFingerprintPlaceholder``). Two reports
built from genuinely different corpus snapshots will currently both carry
the identical uncomputed placeholder and will NOT be rejected by rule 2 —
this rule is honored literally (it rejects when fingerprints differ) but
provides no real protection against undetected corpus drift until
fingerprinting itself is implemented. This is a structural limitation
inherited from upstream, not a bug introduced here — see this phase's
"Risks discovered" for the full implication.

# Regression detection (decision rules)

Every numeric comparison goes through one classification function with two
inputs: the two values being compared, and whether higher is better for
that particular metric (Recall/MRR/NDCG/resolution coverage: higher is
better; queries skipped/queries with unresolved incidents: lower is
better).

- Both values ``None`` → ``UNCHANGED`` (nothing measurable existed before
  or now; there is nothing to flag).
- Exactly one value ``None`` → ``UNDEFINED`` (a metric that was undefined
  became defined, or vice versa — this is not a numeric improvement or
  regression and is reported as its own state rather than guessed at).
- Both values present → ``delta = candidate − baseline`` (sign-flipped for
  lower-is-better metrics); ``|delta| <= EPSILON`` → ``UNCHANGED``;
  otherwise ``IMPROVED`` if the (direction-adjusted) delta is positive,
  else ``REGRESSED``.

``EPSILON = 1e-9`` exists ONLY to absorb floating-point noise from repeated
mean computation — it is not a "minimum meaningful change" threshold.
Statistical/practical significance is explicitly out of scope for this
phase ("do not implement statistical significance yet"); introducing a
larger epsilon here would smuggle in an undocumented significance threshold
under the guise of noise tolerance, which this phase deliberately avoids.

The **overall verdict** (``IMPROVED``/``REGRESSED``/``UNCHANGED``/``MIXED``)
is derived ONLY from the three core retrieval-quality metric
classifications — Recall@K, MRR, NDCG — ignoring ``UNDEFINED`` entries:
- all considered classifications are ``UNCHANGED`` (or all were
  ``UNDEFINED``) → ``UNCHANGED``
- at least one ``IMPROVED`` and none ``REGRESSED`` → ``IMPROVED``
- at least one ``REGRESSED`` and none ``IMPROVED`` → ``REGRESSED``
- at least one of each → ``MIXED``

Resolution coverage, queries-skipped, category, and difficulty deltas are
computed and reported alongside the verdict but do NOT drive it — they are
diagnostic context, not verdict inputs. The same four-way rule is reused,
unmodified, to produce each category/difficulty bucket's own
``BucketDelta.verdict`` (scoped to that bucket's three core metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.evaluation.harness import AggregateMetrics, EvaluationReport

EPSILON = 1e-9


class DeltaClassification(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    UNDEFINED = "undefined"


class Verdict(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    MIXED = "mixed"
    INCOMPATIBLE = "incompatible"


# ── Report data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompatibilityCheck:
    """Whether two reports may be compared, and why not if they may not be.

    ``reasons`` lists every applicable incompatibility, not just the first —
    a caller fixing one issue should immediately see the next one rather
    than re-running the comparison repeatedly to discover each in turn.
    """

    compatible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonMetadata:
    """The identifying configuration of both sides of the comparison —
    enough to know, at a glance, what kind of comparison this is (e.g. "same
    config, later run" vs. "same run, rerank enabled") without reaching into
    the full embedded reports.
    """

    baseline_dataset_version: str
    candidate_dataset_version: str
    baseline_k: int
    candidate_k: int
    baseline_expand: bool
    candidate_expand: bool
    baseline_rerank: bool
    candidate_rerank: bool
    baseline_started_at: str
    candidate_started_at: str


@dataclass(frozen=True)
class MetricDelta:
    """One metric's value on both sides, the signed delta (candidate −
    baseline, already direction-adjusted for lower-is-better metrics so a
    positive delta always means "better"), and its classification.
    """

    baseline: float | None
    candidate: float | None
    delta: float | None
    classification: DeltaClassification


@dataclass(frozen=True)
class BucketDelta:
    """The delta for one category or difficulty bucket — same metric shape
    as the overall comparison, scoped to that bucket's queries.
    """

    bucket: str
    num_queries_baseline: int
    num_queries_candidate: int
    recall_at_k: MetricDelta
    reciprocal_rank: MetricDelta
    ndcg_at_k: MetricDelta
    resolution_coverage: MetricDelta
    verdict: Verdict


@dataclass(frozen=True)
class OverallMetricDeltas:
    """Dataset-wide metric deltas. ``recall_at_k``/``reciprocal_rank``/
    ``ndcg_at_k`` drive the overall verdict; the remaining fields are
    reported but do not.
    """

    recall_at_k: MetricDelta
    reciprocal_rank: MetricDelta
    ndcg_at_k: MetricDelta
    resolution_coverage: MetricDelta
    queries_with_unresolved_incidents: MetricDelta
    num_skipped: MetricDelta


@dataclass(frozen=True)
class RegressionReport:
    """The complete, immutable result of comparing two ``EvaluationReport``s.

    ``baseline``/``candidate`` embed the full input reports by reference
    (not copied) — the same "denormalized for convenience" choice made for
    ``ResolvedGoldQuery.query`` in Phase 16B: a reader can drill into either
    side's full detail directly, while ``comparison`` provides a small,
    pre-extracted summary for callers that only want the identifying
    configuration without traversing two nested reports.

    When ``compatibility.compatible`` is ``False``, ``overall`` is ``None``
    and ``category_deltas``/``difficulty_deltas``/``newly_skipped_query_ids``/
    ``newly_unresolved_query_ids`` are all empty — no comparison was
    performed, per "do not silently compare incompatible reports."
    """

    baseline: EvaluationReport
    candidate: EvaluationReport
    comparison: ComparisonMetadata
    compatibility: CompatibilityCheck
    verdict: Verdict
    overall: OverallMetricDeltas | None
    category_deltas: dict[str, BucketDelta]
    difficulty_deltas: dict[str, BucketDelta]
    newly_skipped_query_ids: tuple[str, ...]
    newly_unresolved_query_ids: tuple[str, ...]
    summary: str


# ── Comparison ─────────────────────────────────────────────────────────────────


def compare(baseline: EvaluationReport, candidate: EvaluationReport) -> RegressionReport:
    """Compare ``baseline`` against ``candidate`` and return a
    ``RegressionReport``. Never raises for an incompatible pair — it returns
    a report with ``verdict=Verdict.INCOMPATIBLE`` and the reasons listed,
    so a caller (e.g. a future CI gate, out of scope for this phase) can
    inspect the result uniformly instead of needing a try/except around
    every comparison.
    """
    compatibility = _check_compatibility(baseline, candidate)
    comparison = ComparisonMetadata(
        baseline_dataset_version=baseline.dataset.version,
        candidate_dataset_version=candidate.dataset.version,
        baseline_k=baseline.config.k,
        candidate_k=candidate.config.k,
        baseline_expand=baseline.config.expand,
        candidate_expand=candidate.config.expand,
        baseline_rerank=baseline.config.rerank,
        candidate_rerank=candidate.config.rerank,
        baseline_started_at=baseline.started_at,
        candidate_started_at=candidate.started_at,
    )

    if not compatibility.compatible:
        return RegressionReport(
            baseline=baseline,
            candidate=candidate,
            comparison=comparison,
            compatibility=compatibility,
            verdict=Verdict.INCOMPATIBLE,
            overall=None,
            category_deltas={},
            difficulty_deltas={},
            newly_skipped_query_ids=(),
            newly_unresolved_query_ids=(),
            summary=f"Reports are not comparable: {'; '.join(compatibility.reasons)}",
        )

    overall = OverallMetricDeltas(
        recall_at_k=_metric_delta(
            baseline.aggregate_metrics.mean_recall_at_k,
            candidate.aggregate_metrics.mean_recall_at_k,
            higher_is_better=True,
        ),
        reciprocal_rank=_metric_delta(
            baseline.aggregate_metrics.mean_reciprocal_rank,
            candidate.aggregate_metrics.mean_reciprocal_rank,
            higher_is_better=True,
        ),
        ndcg_at_k=_metric_delta(
            baseline.aggregate_metrics.mean_ndcg_at_k,
            candidate.aggregate_metrics.mean_ndcg_at_k,
            higher_is_better=True,
        ),
        resolution_coverage=_metric_delta(
            baseline.aggregate_metrics.resolution_coverage,
            candidate.aggregate_metrics.resolution_coverage,
            higher_is_better=True,
        ),
        queries_with_unresolved_incidents=_metric_delta(
            float(baseline.aggregate_metrics.queries_with_unresolved_incidents),
            float(candidate.aggregate_metrics.queries_with_unresolved_incidents),
            higher_is_better=False,
        ),
        num_skipped=_metric_delta(
            float(baseline.num_skipped), float(candidate.num_skipped), higher_is_better=False
        ),
    )
    verdict = _verdict_from_classifications(
        [
            overall.recall_at_k.classification,
            overall.reciprocal_rank.classification,
            overall.ndcg_at_k.classification,
        ]
    )

    category_deltas = _bucket_deltas(baseline.category_breakdown, candidate.category_breakdown)
    difficulty_deltas = _bucket_deltas(
        baseline.difficulty_breakdown, candidate.difficulty_breakdown
    )
    newly_skipped, newly_unresolved = _per_query_drift(baseline, candidate)

    summary = _build_summary(
        verdict, overall, category_deltas, difficulty_deltas, newly_skipped, newly_unresolved
    )

    return RegressionReport(
        baseline=baseline,
        candidate=candidate,
        comparison=comparison,
        compatibility=compatibility,
        verdict=verdict,
        overall=overall,
        category_deltas=category_deltas,
        difficulty_deltas=difficulty_deltas,
        newly_skipped_query_ids=newly_skipped,
        newly_unresolved_query_ids=newly_unresolved,
        summary=summary,
    )


def _check_compatibility(
    baseline: EvaluationReport, candidate: EvaluationReport
) -> CompatibilityCheck:
    reasons: list[str] = []

    if baseline.dataset.version != candidate.dataset.version:
        reasons.append(
            "gold dataset version differs: "
            f"baseline={baseline.dataset.version!r} candidate={candidate.dataset.version!r}"
        )

    if baseline.dataset.corpus_fingerprint != candidate.dataset.corpus_fingerprint:
        reasons.append(
            "corpus fingerprint differs: "
            f"baseline={baseline.dataset.corpus_fingerprint!r} "
            f"candidate={candidate.dataset.corpus_fingerprint!r}"
        )

    if baseline.config.k != candidate.config.k:
        reasons.append(
            f"evaluation k differs: baseline={baseline.config.k!r} candidate={candidate.config.k!r}"
        )

    baseline_ids = {outcome.query_id for outcome in baseline.per_query}
    candidate_ids = {outcome.query_id for outcome in candidate.per_query}
    if baseline_ids != candidate_ids:
        missing_in_candidate = sorted(baseline_ids - candidate_ids)
        missing_in_baseline = sorted(candidate_ids - baseline_ids)
        reasons.append(
            "query coverage differs: "
            f"missing_in_candidate={missing_in_candidate} "
            f"missing_in_baseline={missing_in_baseline}"
        )

    return CompatibilityCheck(compatible=not reasons, reasons=tuple(reasons))


# ── Delta computation ──────────────────────────────────────────────────────────


def _classify(
    baseline: float | None, candidate: float | None, *, higher_is_better: bool
) -> DeltaClassification:
    if baseline is None and candidate is None:
        return DeltaClassification.UNCHANGED
    if baseline is None or candidate is None:
        return DeltaClassification.UNDEFINED
    delta = candidate - baseline
    if not higher_is_better:
        delta = -delta
    if abs(delta) <= EPSILON:
        return DeltaClassification.UNCHANGED
    return DeltaClassification.IMPROVED if delta > 0 else DeltaClassification.REGRESSED


def _metric_delta(
    baseline: float | None, candidate: float | None, *, higher_is_better: bool
) -> MetricDelta:
    classification = _classify(baseline, candidate, higher_is_better=higher_is_better)
    delta = candidate - baseline if (baseline is not None and candidate is not None) else None
    return MetricDelta(
        baseline=baseline, candidate=candidate, delta=delta, classification=classification
    )


def _verdict_from_classifications(classifications: list[DeltaClassification]) -> Verdict:
    meaningful = [c for c in classifications if c != DeltaClassification.UNDEFINED]
    if not meaningful:
        return Verdict.UNCHANGED
    improved = DeltaClassification.IMPROVED in meaningful
    regressed = DeltaClassification.REGRESSED in meaningful
    if improved and regressed:
        return Verdict.MIXED
    if improved:
        return Verdict.IMPROVED
    if regressed:
        return Verdict.REGRESSED
    return Verdict.UNCHANGED


def _bucket_delta(
    bucket: str, baseline_agg: AggregateMetrics | None, candidate_agg: AggregateMetrics | None
) -> BucketDelta:
    b_recall = baseline_agg.mean_recall_at_k if baseline_agg else None
    c_recall = candidate_agg.mean_recall_at_k if candidate_agg else None
    b_rr = baseline_agg.mean_reciprocal_rank if baseline_agg else None
    c_rr = candidate_agg.mean_reciprocal_rank if candidate_agg else None
    b_ndcg = baseline_agg.mean_ndcg_at_k if baseline_agg else None
    c_ndcg = candidate_agg.mean_ndcg_at_k if candidate_agg else None
    b_cov = baseline_agg.resolution_coverage if baseline_agg else None
    c_cov = candidate_agg.resolution_coverage if candidate_agg else None

    recall_delta = _metric_delta(b_recall, c_recall, higher_is_better=True)
    rr_delta = _metric_delta(b_rr, c_rr, higher_is_better=True)
    ndcg_delta = _metric_delta(b_ndcg, c_ndcg, higher_is_better=True)
    coverage_delta = _metric_delta(b_cov, c_cov, higher_is_better=True)

    verdict = _verdict_from_classifications(
        [recall_delta.classification, rr_delta.classification, ndcg_delta.classification]
    )

    return BucketDelta(
        bucket=bucket,
        num_queries_baseline=baseline_agg.num_queries if baseline_agg else 0,
        num_queries_candidate=candidate_agg.num_queries if candidate_agg else 0,
        recall_at_k=recall_delta,
        reciprocal_rank=rr_delta,
        ndcg_at_k=ndcg_delta,
        resolution_coverage=coverage_delta,
        verdict=verdict,
    )


def _bucket_deltas(
    baseline_breakdown: dict[str, AggregateMetrics],
    candidate_breakdown: dict[str, AggregateMetrics],
) -> dict[str, BucketDelta]:
    keys = sorted(set(baseline_breakdown) | set(candidate_breakdown))
    return {
        key: _bucket_delta(key, baseline_breakdown.get(key), candidate_breakdown.get(key))
        for key in keys
    }


def _per_query_drift(
    baseline: EvaluationReport, candidate: EvaluationReport
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Query ids that newly became skipped or newly became (partially or
    fully) unresolved in ``candidate`` relative to ``baseline``. Only
    computed when compatibility already guarantees both reports cover the
    same query_id set, so every lookup below is safe.
    """
    baseline_by_id = {outcome.query_id: outcome for outcome in baseline.per_query}
    candidate_by_id = {outcome.query_id: outcome for outcome in candidate.per_query}

    newly_skipped = tuple(
        sorted(
            query_id
            for query_id, candidate_outcome in candidate_by_id.items()
            if candidate_outcome.skipped and not baseline_by_id[query_id].skipped
        )
    )
    newly_unresolved = tuple(
        sorted(
            query_id
            for query_id, candidate_outcome in candidate_by_id.items()
            if candidate_outcome.num_unresolved_expected > 0
            and baseline_by_id[query_id].num_unresolved_expected == 0
        )
    )
    return newly_skipped, newly_unresolved


# ── Summary ────────────────────────────────────────────────────────────────────


def _fmt_delta(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def _build_summary(
    verdict: Verdict,
    overall: OverallMetricDeltas,
    category_deltas: dict[str, BucketDelta],
    difficulty_deltas: dict[str, BucketDelta],
    newly_skipped: tuple[str, ...],
    newly_unresolved: tuple[str, ...],
) -> str:
    parts = [f"Overall verdict: {verdict.value}."]
    parts.append(
        f"Recall@K {overall.recall_at_k.classification.value} "
        f"({_fmt_delta(overall.recall_at_k.delta)}), "
        f"MRR {overall.reciprocal_rank.classification.value} "
        f"({_fmt_delta(overall.reciprocal_rank.delta)}), "
        f"NDCG {overall.ndcg_at_k.classification.value} "
        f"({_fmt_delta(overall.ndcg_at_k.delta)})."
    )

    regressed_categories = sorted(
        name for name, bucket in category_deltas.items() if bucket.verdict == Verdict.REGRESSED
    )
    if regressed_categories:
        parts.append(f"Categories regressed: {', '.join(regressed_categories)}.")

    regressed_difficulties = sorted(
        name for name, bucket in difficulty_deltas.items() if bucket.verdict == Verdict.REGRESSED
    )
    if regressed_difficulties:
        parts.append(f"Difficulties regressed: {', '.join(regressed_difficulties)}.")

    if newly_skipped:
        parts.append(f"{len(newly_skipped)} query(ies) newly skipped.")
    if newly_unresolved:
        parts.append(f"{len(newly_unresolved)} query(ies) newly unresolved.")

    return " ".join(parts)
