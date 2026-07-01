from __future__ import annotations

from app.evaluation.judge_agreement import (
    AgreementPair,
    BiasDirection,
    PromptVariantResult,
    ScoredRecord,
    analyze_bias,
    analyze_consistency,
    analyze_prompt_sensitivity,
    collect_repeated_scores,
    compute_agreement,
)

PLAN = "plan"
DECISION = "decision"


def _record(record_id, stage, *, human=None, rule=None, llm=None) -> ScoredRecord:
    return ScoredRecord(
        record_id=record_id, stage=stage, human_score=human, rule_score=rule, llm_score=llm,
    )


# ── Agreement ──────────────────────────────────────────────────────────────────


def test_compute_agreement_reports_per_stage_differences() -> None:
    records = (
        _record("r1", PLAN, human=7.0, llm=7.5),
        _record("r2", PLAN, human=8.0, llm=6.0),
        _record("r3", DECISION, human=5.0, llm=5.0),
    )
    results = compute_agreement(records, AgreementPair.HUMAN_VS_LLM)

    by_stage = {r.stage: r for r in results}
    assert by_stage[PLAN].n == 2
    assert by_stage[PLAN].differences == (0.5, 2.0)
    assert by_stage[PLAN].mean_absolute_difference == 1.25
    assert by_stage[DECISION].n == 1
    assert by_stage[DECISION].mean_absolute_difference == 0.0


def test_compute_agreement_within_tolerance_fraction() -> None:
    records = (
        _record("r1", PLAN, human=7.0, llm=7.5),  # diff 0.5 -> within tolerance 1.0
        _record("r2", PLAN, human=8.0, llm=6.0),  # diff 2.0 -> NOT within tolerance 1.0
    )
    results = compute_agreement(records, AgreementPair.HUMAN_VS_LLM, tolerance=1.0)
    assert results[0].agreement_within_tolerance == 0.5


def test_compute_agreement_skips_records_missing_either_side() -> None:
    records = (
        _record("r1", PLAN, human=7.0, llm=None),
        _record("r2", PLAN, human=None, llm=6.0),
        _record("r3", PLAN, human=8.0, llm=8.0),
    )
    results = compute_agreement(records, AgreementPair.HUMAN_VS_LLM)
    assert len(results) == 1
    assert results[0].n == 1


def test_compute_agreement_returns_nothing_for_no_data() -> None:
    assert compute_agreement((), AgreementPair.RULE_VS_LLM) == ()


def test_compute_agreement_supports_rule_vs_llm_pair() -> None:
    records = (_record("r1", PLAN, rule=6.0, llm=8.0),)
    results = compute_agreement(records, AgreementPair.RULE_VS_LLM)
    assert results[0].mean_absolute_difference == 2.0


# ── Consistency ────────────────────────────────────────────────────────────────


def test_analyze_consistency_computes_basic_statistics() -> None:
    result = analyze_consistency(PLAN, [6.0, 7.0, 8.0])
    assert result.n == 3
    assert result.mean == 7.0
    assert result.minimum == 6.0
    assert result.maximum == 8.0
    assert result.variance > 0
    assert result.std_dev == result.variance ** 0.5


def test_analyze_consistency_zero_variance_for_identical_scores() -> None:
    result = analyze_consistency(PLAN, [7.0, 7.0, 7.0])
    assert result.variance == 0.0
    assert result.std_dev == 0.0


def test_analyze_consistency_handles_empty_input() -> None:
    result = analyze_consistency(PLAN, [])
    assert result.n == 0
    assert result.mean == 0.0


def test_collect_repeated_scores_calls_evaluate_fn_n_times() -> None:
    calls = []

    def fake_evaluate():
        calls.append(1)
        return 7.0 + len(calls) * 0.1

    scores = collect_repeated_scores(fake_evaluate, n=5)
    assert len(scores) == 5
    assert len(calls) == 5


# ── Prompt sensitivity ──────────────────────────────────────────────────────────


def test_analyze_prompt_sensitivity_computes_stage_wise_drift() -> None:
    results = (
        PromptVariantResult("v1", PLAN, 7.0), PromptVariantResult("v2", PLAN, 9.0),
        PromptVariantResult("v1", DECISION, 5.0), PromptVariantResult("v2", DECISION, 5.5),
    )
    report = analyze_prompt_sensitivity(results)
    by_stage = {sd.stage: sd for sd in report.stage_drifts}
    assert by_stage[PLAN].drift == 2.0
    assert by_stage[DECISION].drift == 0.5
    assert report.mean_drift == 1.25
    assert report.max_drift == 2.0


def test_analyze_prompt_sensitivity_handles_empty_input() -> None:
    report = analyze_prompt_sensitivity(())
    assert report.stage_drifts == ()
    assert report.mean_drift is None
    assert report.max_drift is None


def test_analyze_prompt_sensitivity_single_variant_has_zero_drift() -> None:
    report = analyze_prompt_sensitivity((PromptVariantResult("v1", PLAN, 7.0),))
    assert report.stage_drifts[0].drift == 0.0


# ── Bias ───────────────────────────────────────────────────────────────────────


def test_analyze_bias_detects_systematic_difference_above_threshold() -> None:
    records = (
        _record("r1", PLAN, rule=5.0, llm=7.0),
        _record("r2", PLAN, rule=5.5, llm=7.5),
    )
    findings = analyze_bias(records, AgreementPair.RULE_VS_LLM)
    assert len(findings) == 1
    assert findings[0].direction == BiasDirection.SECOND_HIGHER
    assert "llm_judge" in findings[0].description


def test_analyze_bias_ignores_small_differences_below_threshold() -> None:
    records = (
        _record("r1", PLAN, rule=7.0, llm=7.2),
        _record("r2", PLAN, rule=6.9, llm=7.0),
    )
    findings = analyze_bias(records, AgreementPair.RULE_VS_LLM)
    assert findings == ()


def test_analyze_bias_first_higher_direction() -> None:
    records = (_record("r1", DECISION, human=8.0, rule=6.0),)
    findings = analyze_bias(records, AgreementPair.HUMAN_VS_RULE)
    assert findings[0].direction == BiasDirection.FIRST_HIGHER
    assert findings[0].mean_signed_difference == 2.0


def test_analyze_bias_handles_no_data() -> None:
    assert analyze_bias((), AgreementPair.HUMAN_VS_LLM) == ()


def test_analyze_bias_is_deterministic() -> None:
    records = (_record("r1", PLAN, rule=5.0, llm=7.0), _record("r2", PLAN, rule=5.0, llm=7.0))
    first = analyze_bias(records, AgreementPair.RULE_VS_LLM)
    second = analyze_bias(records, AgreementPair.RULE_VS_LLM)
    assert first == second
