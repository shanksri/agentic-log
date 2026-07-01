from __future__ import annotations

from app.evaluation.reasoning_dataset import InvestigationScenario, ReasoningGoldDataset


def _scenario(**overrides) -> InvestigationScenario:
    defaults = dict(
        id="s1", problem="login fails with an expired token",
        expected_strategy="authentication", expected_root_causes=("expired token",),
        expected_verdict="approved", expected_stopping_reason="critic_approved",
    )
    defaults.update(overrides)
    return InvestigationScenario(**defaults)


def test_valid_scenario_has_no_issues() -> None:
    assert _scenario().issues() == []


def test_negative_control_scenario_with_empty_root_causes_is_valid() -> None:
    scenario = _scenario(
        expected_root_causes=(), expected_verdict="inconclusive",
        expected_stopping_reason="max_iterations",
    )
    assert scenario.issues() == []


def test_empty_id_is_invalid() -> None:
    issues = _scenario(id="").issues()
    assert any("id must be non-empty" in issue for issue in issues)


def test_empty_problem_is_invalid() -> None:
    issues = _scenario(problem="").issues()
    assert any("problem must be non-empty" in issue for issue in issues)


def test_unknown_expected_strategy_is_invalid() -> None:
    issues = _scenario(expected_strategy="not_a_strategy").issues()
    assert any("expected_strategy" in issue for issue in issues)


def test_unknown_expected_verdict_is_invalid() -> None:
    issues = _scenario(expected_verdict="not_a_verdict").issues()
    assert any("expected_verdict" in issue for issue in issues)


def test_unknown_expected_stopping_reason_is_invalid() -> None:
    issues = _scenario(expected_stopping_reason="not_a_reason").issues()
    assert any("expected_stopping_reason" in issue for issue in issues)


def test_approved_verdict_with_no_expected_root_causes_is_invalid() -> None:
    issues = _scenario(expected_root_causes=(), expected_verdict="approved").issues()
    assert any("implies an accepted hypothesis" in issue for issue in issues)


def test_dataset_with_valid_scenarios_is_valid() -> None:
    dataset = ReasoningGoldDataset(
        version="v1", description="d", created_at="2026-01-01",
        scenarios=(_scenario(id="s1"), _scenario(id="s2")),
    )
    assert dataset.is_valid()


def test_dataset_rejects_duplicate_scenario_ids() -> None:
    dataset = ReasoningGoldDataset(
        version="v1", description="d", created_at="2026-01-01",
        scenarios=(_scenario(id="s1"), _scenario(id="s1")),
    )
    assert any("duplicate scenario id" in issue for issue in dataset.issues())


def test_dataset_requires_at_least_one_scenario() -> None:
    dataset = ReasoningGoldDataset(
        version="v1", description="d", created_at="2026-01-01", scenarios=(),
    )
    assert any("scenarios must be non-empty" in issue for issue in dataset.issues())
