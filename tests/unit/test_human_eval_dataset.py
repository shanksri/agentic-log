from __future__ import annotations

from app.evaluation.human_eval_dataset import HumanEvaluationDataset, HumanEvaluationRecord


def _record(**overrides) -> HumanEvaluationRecord:
    defaults = dict(record_id="r1")
    defaults.update(overrides)
    return HumanEvaluationRecord(**defaults)


def test_minimal_record_with_no_scores_is_valid() -> None:
    assert _record().issues() == []


def test_record_with_all_scores_is_valid() -> None:
    record = _record(
        human_planner_score=7.0, human_hypotheses_score=6.0, human_decision_score=8.0,
        human_critique_score=5.0, human_overall_score=7.5, notes="solid investigation",
    )
    assert record.issues() == []


def test_empty_record_id_is_invalid() -> None:
    issues = _record(record_id="").issues()
    assert any("record_id must be non-empty" in issue for issue in issues)


def test_out_of_range_score_is_invalid() -> None:
    issues = _record(human_planner_score=15.0).issues()
    assert any("human_planner_score" in issue for issue in issues)

    issues = _record(human_decision_score=-1.0).issues()
    assert any("human_decision_score" in issue for issue in issues)


def test_boundary_scores_are_valid() -> None:
    assert _record(human_planner_score=0.0).issues() == []
    assert _record(human_critique_score=10.0).issues() == []


def test_dataset_with_valid_records_is_valid() -> None:
    dataset = HumanEvaluationDataset(
        version="v1", description="d", created_at="2026-01-01",
        records=(_record(record_id="r1"), _record(record_id="r2")),
    )
    assert dataset.is_valid()


def test_dataset_allows_zero_records() -> None:
    dataset = HumanEvaluationDataset(version="v1", description="d", created_at="2026-01-01")
    assert dataset.is_valid()


def test_dataset_rejects_duplicate_record_ids() -> None:
    dataset = HumanEvaluationDataset(
        version="v1", description="d", created_at="2026-01-01",
        records=(_record(record_id="r1"), _record(record_id="r1")),
    )
    assert any("duplicate record id" in issue for issue in dataset.issues())


def test_dataset_requires_version_description_created_at() -> None:
    dataset = HumanEvaluationDataset(version="", description="", created_at="")
    issues = dataset.issues()
    assert any("version" in issue for issue in issues)
    assert any("description" in issue for issue in issues)
    assert any("created_at" in issue for issue in issues)


def test_dataset_get_returns_matching_record() -> None:
    record = _record(record_id="r1")
    dataset = HumanEvaluationDataset(
        version="v1", description="d", created_at="2026-01-01", records=(record,),
    )
    assert dataset.get("r1") == record
    assert dataset.get("missing") is None
