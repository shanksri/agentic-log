from __future__ import annotations

import pytest

from app.evaluation.gold_dataset import CorpusFingerprintPlaceholder
from app.evaluation.gold_loader import GoldDatasetResolutionSummary
from app.evaluation.harness import (
    AggregateMetrics,
    CorpusStatistics,
    CoverageBreakdown,
    EvaluationConfig,
    EvaluationDatasetInfo,
    EvaluationReport,
    QueryEvaluationOutcome,
)
from app.evaluation.metrics import QueryMetricResult
from app.evaluation.regression import (
    DeltaClassification,
    RegressionReport,
    Verdict,
    compare,
)


def _metric(
    *,
    query_id: str = "q-1",
    k: int = 10,
    recall: float | None = 1.0,
    rr: float | None = 1.0,
    ndcg: float | None = 1.0,
    num_relevant: int = 1,
    num_unresolved_expected: int = 0,
) -> QueryMetricResult:
    return QueryMetricResult(
        query_id=query_id,
        k=k,
        num_relevant=num_relevant,
        num_unresolved_expected=num_unresolved_expected,
        num_retrieved=1,
        num_duplicate_retrieved=0,
        recall_at_k=recall,
        reciprocal_rank=rr,
        dcg_at_k=1.0,
        idcg_at_k=1.0,
        ndcg_at_k=ndcg,
    )


def _outcome(
    *,
    query_id: str = "q-1",
    category: str = "lexical-overlap",
    difficulty: str = "easy",
    num_relevant: int = 1,
    num_unresolved_expected: int = 0,
    skipped: bool = False,
    skip_reason: str | None = None,
    metric: QueryMetricResult | None = None,
) -> QueryEvaluationOutcome:
    if metric is None and not skipped:
        metric = _metric(
            query_id=query_id,
            num_relevant=num_relevant,
            num_unresolved_expected=num_unresolved_expected,
        )
    return QueryEvaluationOutcome(
        query_id=query_id,
        category=category,
        difficulty=difficulty,
        num_relevant=num_relevant,
        num_unresolved_expected=num_unresolved_expected,
        skipped=skipped,
        skip_reason=skip_reason,
        metric=metric,
    )


def _aggregate(
    *,
    num_queries: int = 1,
    recall: float | None = 1.0,
    rr: float | None = 1.0,
    ndcg: float | None = 1.0,
    coverage: float | None = 1.0,
    unresolved: int = 0,
) -> AggregateMetrics:
    return AggregateMetrics(
        num_queries=num_queries,
        mean_recall_at_k=recall,
        mean_reciprocal_rank=rr,
        mean_ndcg_at_k=ndcg,
        resolution_coverage=coverage,
        queries_with_unresolved_incidents=unresolved,
    )


def make_report(
    *,
    version: str = "2.0.0",
    k: int = 10,
    expand: bool = False,
    rerank: bool = False,
    corpus_fingerprint: CorpusFingerprintPlaceholder | None = None,
    per_query: tuple[QueryEvaluationOutcome, ...] | None = None,
    aggregate_metrics: AggregateMetrics | None = None,
    category_breakdown: dict[str, AggregateMetrics] | None = None,
    difficulty_breakdown: dict[str, AggregateMetrics] | None = None,
    num_skipped: int = 0,
    started_at: str = "2026-06-26T00:00:00+00:00",
) -> EvaluationReport:
    per_query = per_query if per_query is not None else (_outcome(),)
    num_evaluated = sum(1 for o in per_query if not o.skipped)
    return EvaluationReport(
        dataset=EvaluationDatasetInfo(
            version=version,
            description="d",
            created_at="2026-01-01T00:00:00Z",
            author=None,
            corpus_fingerprint=corpus_fingerprint or CorpusFingerprintPlaceholder(),
        ),
        config=EvaluationConfig(k=k, expand=expand, rerank=rerank),
        corpus_statistics=CorpusStatistics(
            corpus_fingerprint=corpus_fingerprint or CorpusFingerprintPlaceholder(),
            distinct_retrieved_incident_count=len(per_query),
        ),
        num_evaluated=num_evaluated,
        num_skipped=num_skipped,
        aggregate_metrics=aggregate_metrics or _aggregate(num_queries=len(per_query)),
        per_query=per_query,
        coverage=CoverageBreakdown(
            total_queries=len(per_query),
            no_match_expected_queries=0,
            fully_resolved_queries=len(per_query),
            partially_resolved_queries=0,
            fully_unresolved_queries=0,
        ),
        resolution_summary=GoldDatasetResolutionSummary(
            total_expected_incidents=len(per_query), resolved_count=len(per_query),
            unresolved_identities=(),
        ),
        category_breakdown=category_breakdown or {},
        difficulty_breakdown=difficulty_breakdown or {},
        started_at=started_at,
        finished_at=started_at,
        duration_seconds=0.1,
    )


# ── Identical reports ──────────────────────────────────────────────────────────


def test_compare_identical_reports_is_unchanged() -> None:
    baseline = make_report()
    candidate = make_report()

    report = compare(baseline, candidate)

    assert isinstance(report, RegressionReport)
    assert report.compatibility.compatible is True
    assert report.verdict == Verdict.UNCHANGED
    assert report.overall.recall_at_k.classification == DeltaClassification.UNCHANGED
    assert report.overall.recall_at_k.delta == 0.0
    assert report.newly_skipped_query_ids == ()
    assert report.newly_unresolved_query_ids == ()


# ── Improved / regressed / mixed ──────────────────────────────────────────────


def test_compare_improved_metrics() -> None:
    baseline = make_report(aggregate_metrics=_aggregate(recall=0.5, rr=0.5, ndcg=0.5))
    candidate = make_report(aggregate_metrics=_aggregate(recall=0.8, rr=0.7, ndcg=0.75))

    report = compare(baseline, candidate)

    assert report.verdict == Verdict.IMPROVED
    assert report.overall.recall_at_k.classification == DeltaClassification.IMPROVED
    assert report.overall.recall_at_k.delta == pytest.approx(0.3)
    assert report.overall.reciprocal_rank.classification == DeltaClassification.IMPROVED
    assert report.overall.ndcg_at_k.classification == DeltaClassification.IMPROVED


def test_compare_regressed_metrics() -> None:
    baseline = make_report(aggregate_metrics=_aggregate(recall=0.8, rr=0.8, ndcg=0.8))
    candidate = make_report(aggregate_metrics=_aggregate(recall=0.5, rr=0.6, ndcg=0.55))

    report = compare(baseline, candidate)

    assert report.verdict == Verdict.REGRESSED
    assert report.overall.recall_at_k.classification == DeltaClassification.REGRESSED
    assert report.overall.recall_at_k.delta == pytest.approx(-0.3)


def test_compare_mixed_improvements_and_regressions() -> None:
    baseline = make_report(aggregate_metrics=_aggregate(recall=0.5, rr=0.8, ndcg=0.6))
    candidate = make_report(aggregate_metrics=_aggregate(recall=0.8, rr=0.5, ndcg=0.6))

    report = compare(baseline, candidate)

    assert report.overall.recall_at_k.classification == DeltaClassification.IMPROVED
    assert report.overall.reciprocal_rank.classification == DeltaClassification.REGRESSED
    assert report.overall.ndcg_at_k.classification == DeltaClassification.UNCHANGED
    assert report.verdict == Verdict.MIXED


def test_compare_lower_is_better_metrics_direction() -> None:
    baseline = make_report(num_skipped=5, aggregate_metrics=_aggregate(unresolved=4))
    candidate = make_report(num_skipped=2, aggregate_metrics=_aggregate(unresolved=1))

    report = compare(baseline, candidate)

    # Fewer skipped / fewer unresolved = improvement, even though the raw
    # numeric delta (candidate - baseline) is negative.
    assert report.overall.num_skipped.delta == -3.0
    assert report.overall.num_skipped.classification == DeltaClassification.IMPROVED
    assert report.overall.queries_with_unresolved_incidents.delta == -3.0
    assert (
        report.overall.queries_with_unresolved_incidents.classification
        == DeltaClassification.IMPROVED
    )


def test_compare_undefined_when_one_side_has_no_metric() -> None:
    baseline = make_report(aggregate_metrics=_aggregate(recall=None))
    candidate = make_report(aggregate_metrics=_aggregate(recall=0.8))

    report = compare(baseline, candidate)

    assert report.overall.recall_at_k.classification == DeltaClassification.UNDEFINED
    assert report.overall.recall_at_k.delta is None


def test_compare_unchanged_when_both_sides_undefined() -> None:
    baseline = make_report(aggregate_metrics=_aggregate(recall=None, rr=None, ndcg=None))
    candidate = make_report(aggregate_metrics=_aggregate(recall=None, rr=None, ndcg=None))

    report = compare(baseline, candidate)

    assert report.verdict == Verdict.UNCHANGED


# ── Incompatible reports ──────────────────────────────────────────────────────


def test_compare_rejects_changed_dataset_version() -> None:
    baseline = make_report(version="2.0.0")
    candidate = make_report(version="2.1.0")

    report = compare(baseline, candidate)

    assert report.compatibility.compatible is False
    assert any("version differs" in reason for reason in report.compatibility.reasons)
    assert report.verdict == Verdict.INCOMPATIBLE
    assert report.overall is None
    assert report.category_deltas == {}
    assert report.difficulty_deltas == {}
    assert "not comparable" in report.summary


def test_compare_rejects_changed_corpus_fingerprint() -> None:
    baseline = make_report(
        corpus_fingerprint=CorpusFingerprintPlaceholder(computed=True, value="a")
    )
    candidate = make_report(
        corpus_fingerprint=CorpusFingerprintPlaceholder(computed=True, value="b")
    )

    report = compare(baseline, candidate)

    assert report.compatibility.compatible is False
    assert any("corpus fingerprint differs" in reason for reason in report.compatibility.reasons)
    assert report.verdict == Verdict.INCOMPATIBLE


def test_compare_does_not_reject_matching_uncomputed_fingerprints() -> None:
    # Both sides carry the default, uncomputed placeholder -> structurally
    # equal -> not rejected by rule 2, even though neither corpus snapshot
    # was actually verified (documented limitation, see module docstring).
    baseline = make_report()
    candidate = make_report()

    report = compare(baseline, candidate)

    assert report.compatibility.compatible is True


def test_compare_rejects_changed_k() -> None:
    baseline = make_report(k=5)
    candidate = make_report(k=10)

    report = compare(baseline, candidate)

    assert report.compatibility.compatible is False
    assert any("evaluation k differs" in reason for reason in report.compatibility.reasons)
    assert report.verdict == Verdict.INCOMPATIBLE


def test_compare_rejects_differing_query_coverage() -> None:
    baseline = make_report(per_query=(_outcome(query_id="q-1"),))
    candidate = make_report(per_query=(_outcome(query_id="q-2"),))

    report = compare(baseline, candidate)

    assert report.compatibility.compatible is False
    assert any("query coverage differs" in reason for reason in report.compatibility.reasons)


def test_compare_does_not_reject_differing_expand_rerank() -> None:
    # Deliberate: comparing two configs over the same gold set/corpus is a
    # legitimate use case (e.g. "what does rerank do"), not an incompatibility.
    baseline = make_report(expand=False, rerank=False)
    candidate = make_report(expand=True, rerank=True)

    report = compare(baseline, candidate)

    assert report.compatibility.compatible is True
    assert report.comparison.baseline_expand is False
    assert report.comparison.candidate_expand is True
    assert report.comparison.baseline_rerank is False
    assert report.comparison.candidate_rerank is True


def test_compare_reports_every_incompatibility_reason_not_just_first() -> None:
    baseline = make_report(version="2.0.0", k=5, per_query=(_outcome(query_id="q-1"),))
    candidate = make_report(version="3.0.0", k=10, per_query=(_outcome(query_id="q-2"),))

    report = compare(baseline, candidate)

    assert len(report.compatibility.reasons) >= 3


# ── Category / difficulty deltas ──────────────────────────────────────────────


def test_compare_category_regression_only() -> None:
    baseline_categories = {
        "lexical-overlap": _aggregate(recall=1.0, rr=1.0, ndcg=1.0),
        "paraphrase": _aggregate(recall=0.9, rr=0.9, ndcg=0.9),
    }
    candidate_categories = {
        "lexical-overlap": _aggregate(recall=1.0, rr=1.0, ndcg=1.0),  # unchanged
        "paraphrase": _aggregate(recall=0.4, rr=0.4, ndcg=0.4),  # regressed
    }
    baseline = make_report(
        category_breakdown=baseline_categories,
        aggregate_metrics=_aggregate(recall=0.95, rr=0.95, ndcg=0.95),
    )
    candidate = make_report(
        category_breakdown=candidate_categories,
        aggregate_metrics=_aggregate(recall=0.7, rr=0.7, ndcg=0.7),
    )

    report = compare(baseline, candidate)

    assert report.category_deltas["lexical-overlap"].verdict == Verdict.UNCHANGED
    assert report.category_deltas["paraphrase"].verdict == Verdict.REGRESSED
    assert "paraphrase" in report.summary
    assert "Categories regressed" in report.summary


def test_compare_difficulty_regression_only() -> None:
    baseline_difficulties = {
        "easy": _aggregate(recall=1.0, rr=1.0, ndcg=1.0),
        "hard": _aggregate(recall=0.8, rr=0.8, ndcg=0.8),
    }
    candidate_difficulties = {
        "easy": _aggregate(recall=1.0, rr=1.0, ndcg=1.0),
        "hard": _aggregate(recall=0.2, rr=0.2, ndcg=0.2),
    }
    baseline = make_report(difficulty_breakdown=baseline_difficulties)
    candidate = make_report(difficulty_breakdown=candidate_difficulties)

    report = compare(baseline, candidate)

    assert report.difficulty_deltas["easy"].verdict == Verdict.UNCHANGED
    assert report.difficulty_deltas["hard"].verdict == Verdict.REGRESSED
    assert "hard" in report.summary
    assert "Difficulties regressed" in report.summary


def test_compare_bucket_present_only_on_one_side_treated_as_zero_baseline() -> None:
    baseline = make_report(category_breakdown={})
    candidate = make_report(category_breakdown={"multi-concept": _aggregate(recall=0.6)})

    report = compare(baseline, candidate)

    bucket = report.category_deltas["multi-concept"]
    assert bucket.num_queries_baseline == 0
    assert bucket.recall_at_k.baseline is None
    assert bucket.recall_at_k.classification == DeltaClassification.UNDEFINED


# ── Newly skipped / newly unresolved ──────────────────────────────────────────


def test_compare_detects_newly_skipped_query() -> None:
    baseline = make_report(per_query=(_outcome(query_id="q-1", skipped=False),))
    candidate = make_report(
        per_query=(_outcome(query_id="q-1", skipped=True, skip_reason="search_failed: boom"),)
    )

    report = compare(baseline, candidate)

    assert report.newly_skipped_query_ids == ("q-1",)
    assert "newly skipped" in report.summary


def test_compare_does_not_flag_already_skipped_query_as_newly_skipped() -> None:
    baseline = make_report(
        per_query=(_outcome(query_id="q-1", skipped=True, skip_reason="search_failed: x"),)
    )
    candidate = make_report(
        per_query=(_outcome(query_id="q-1", skipped=True, skip_reason="search_failed: y"),)
    )

    report = compare(baseline, candidate)

    assert report.newly_skipped_query_ids == ()


def test_compare_detects_newly_unresolved_query() -> None:
    baseline = make_report(
        per_query=(_outcome(query_id="q-1", num_relevant=1, num_unresolved_expected=0),)
    )
    candidate = make_report(
        per_query=(_outcome(query_id="q-1", num_relevant=0, num_unresolved_expected=1),)
    )

    report = compare(baseline, candidate)

    assert report.newly_unresolved_query_ids == ("q-1",)
    assert "newly unresolved" in report.summary


def test_compare_does_not_flag_already_unresolved_query_as_newly_unresolved() -> None:
    baseline = make_report(
        per_query=(_outcome(query_id="q-1", num_relevant=0, num_unresolved_expected=1),)
    )
    candidate = make_report(
        per_query=(_outcome(query_id="q-1", num_relevant=0, num_unresolved_expected=1),)
    )

    report = compare(baseline, candidate)

    assert report.newly_unresolved_query_ids == ()


# ── Report / comparison metadata correctness ──────────────────────────────────


def test_compare_comparison_metadata_carries_both_sides() -> None:
    baseline = make_report(k=5, version="2.0.0")
    candidate = make_report(k=5, version="2.0.0")

    report = compare(baseline, candidate)

    assert report.comparison.baseline_dataset_version == "2.0.0"
    assert report.comparison.candidate_dataset_version == "2.0.0"
    assert report.comparison.baseline_k == 5
    assert report.comparison.candidate_k == 5


def test_compare_embeds_full_input_reports() -> None:
    baseline = make_report()
    candidate = make_report()

    report = compare(baseline, candidate)

    assert report.baseline is baseline
    assert report.candidate is candidate


def test_compare_report_is_immutable() -> None:
    report = compare(make_report(), make_report())

    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        report.verdict = Verdict.IMPROVED  # type: ignore[misc]
