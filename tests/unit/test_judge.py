from __future__ import annotations

import pytest

from app.evaluation.judge import (
    CRITERIA,
    SCORE_MAX,
    SCORE_MIN,
    STAGE_CRITIQUE,
    STAGE_DECISION,
    STAGE_HYPOTHESES,
    STAGE_PLAN,
    STAGE_SESSION,
    Judge,
    JudgeEvaluation,
    JudgeFinding,
    classify_score,
    make_judge_score,
)


def test_classify_score_bands_match_rubric() -> None:
    assert classify_score(1.0) == "Poor"
    assert classify_score(2.0) == "Poor"
    assert classify_score(3.0) == "Weak"
    assert classify_score(4.0) == "Weak"
    assert classify_score(5.0) == "Acceptable"
    assert classify_score(6.0) == "Acceptable"
    assert classify_score(7.0) == "Good"
    assert classify_score(8.0) == "Good"
    assert classify_score(9.0) == "Excellent"
    assert classify_score(10.0) == "Excellent"


def test_classify_score_clamps_out_of_range_values() -> None:
    assert classify_score(-5.0) == "Poor"
    assert classify_score(100.0) == "Excellent"


def test_make_judge_score_clamps_and_assigns_band() -> None:
    score = make_judge_score(15.0)
    assert score.value == SCORE_MAX
    assert score.band == "Excellent"

    score = make_judge_score(-3.0)
    assert score.value == SCORE_MIN
    assert score.band == "Poor"


def test_judge_evaluation_requires_non_empty_explanation() -> None:
    with pytest.raises(ValueError, match="non-empty explanation"):
        JudgeEvaluation(stage=STAGE_PLAN, score=make_judge_score(5.0), explanation="")


def test_judge_evaluation_rejects_whitespace_only_explanation() -> None:
    with pytest.raises(ValueError):
        JudgeEvaluation(stage=STAGE_PLAN, score=make_judge_score(5.0), explanation="   ")


def test_judge_evaluation_is_frozen() -> None:
    evaluation = JudgeEvaluation(
        stage=STAGE_PLAN, score=make_judge_score(5.0), explanation="fine",
    )
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        evaluation.stage = STAGE_DECISION  # type: ignore[misc]


def test_judge_evaluation_defaults_to_empty_finding_tuples() -> None:
    evaluation = JudgeEvaluation(
        stage=STAGE_PLAN, score=make_judge_score(5.0), explanation="fine",
    )
    assert evaluation.strengths == ()
    assert evaluation.weaknesses == ()
    assert evaluation.recommendations == ()


def test_judge_finding_is_frozen() -> None:
    finding = JudgeFinding(criterion="diversity", detail="two distinct hypotheses")
    with pytest.raises(Exception):  # noqa: PT011
        finding.detail = "changed"  # type: ignore[misc]


def test_criteria_defines_every_stage() -> None:
    assert set(CRITERIA) == {
        STAGE_PLAN, STAGE_HYPOTHESES, STAGE_DECISION, STAGE_CRITIQUE, STAGE_SESSION,
    }
    for stage, criteria in CRITERIA.items():
        assert criteria, f"stage {stage!r} must have at least one criterion"


def test_judge_is_abstract_and_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Judge()  # type: ignore[abstract]


def test_judge_subclass_must_implement_every_method() -> None:
    class IncompleteJudge(Judge):
        def evaluate_plan(self, problem, plan):
            return JudgeEvaluation(stage=STAGE_PLAN, score=make_judge_score(5.0), explanation="x")

    with pytest.raises(TypeError):
        IncompleteJudge()  # type: ignore[abstract]
