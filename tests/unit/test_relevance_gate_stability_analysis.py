"""Tests for the stability experiment's own analysis logic
(scripts/probe_llm_relevance_gate_stability.py's `_rates`).

Experiment tooling only -- this exercises no production behavior. The gate
itself is deliberately NOT wired into production; these tests exist so the
numbers the stability report prints can be trusted, particularly the rule
that positives only count toward precision/recall when retrieval actually
surfaced the expected incident (a gate rejecting a genuinely wrong top-1 is
correct behavior, not a false rejection).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "probe_llm_relevance_gate_stability.py"


def _load():
    spec = importlib.util.spec_from_file_location("_stability_probe", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _entry(id_: str, group: str) -> dict:
    return {"id": id_, "group": group, "query": "q", "top1_score": 0.7, "expected_external_ids": []}


def test_perfect_run_scores_one(mod) -> None:
    test_set = [_entry("h1", "HARD_NEGATIVE"), _entry("p1", "POSITIVE")]
    decisions = {"h1": False, "p1": True}
    out = mod._rates(decisions, test_set, {"p1": True})
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (1, 0, 0, 1)
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["fpr"] == 0.0


def test_hard_negative_accepted_is_a_false_positive(mod) -> None:
    test_set = [_entry("h1", "HARD_NEGATIVE"), _entry("p1", "POSITIVE")]
    out = mod._rates({"h1": True, "p1": True}, test_set, {"p1": True})
    assert out["fp"] == 1
    assert out["fpr"] == 1.0
    assert out["precision"] == 0.5


def test_genuine_match_rejected_is_a_false_negative(mod) -> None:
    test_set = [_entry("p1", "POSITIVE"), _entry("p2", "POSITIVE")]
    out = mod._rates({"p1": True, "p2": False}, test_set, {"p1": True, "p2": True})
    assert (out["tp"], out["fn"]) == (1, 1)
    assert out["recall"] == 0.5


def test_positive_whose_top1_was_not_expected_is_excluded_entirely(mod) -> None:
    # The headline recall must not be penalized for a query where retrieval
    # surfaced the wrong incident -- the gate rejecting it is correct.
    test_set = [_entry("p1", "POSITIVE"), _entry("p2", "POSITIVE")]
    out = mod._rates({"p1": True, "p2": False}, test_set, {"p1": True, "p2": False})
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (1, 0, 0, 0)
    assert out["recall"] == 1.0


def test_none_decisions_are_skipped_not_counted(mod) -> None:
    test_set = [_entry("h1", "HARD_NEGATIVE"), _entry("p1", "POSITIVE")]
    out = mod._rates({"h1": None, "p1": None}, test_set, {"p1": True})
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (0, 0, 0, 0)
    assert out["precision"] is None
    assert out["recall"] is None
    assert out["fpr"] is None


def test_rates_are_deterministic_for_identical_input(mod) -> None:
    test_set = [_entry("h1", "HARD_NEGATIVE"), _entry("p1", "POSITIVE")]
    decisions = {"h1": False, "p1": True}
    assert mod._rates(decisions, test_set, {"p1": True}) == mod._rates(
        decisions, test_set, {"p1": True}
    )


def test_prompt_is_imported_from_the_original_probe_not_copied(mod) -> None:
    """The stability run must use the same prompt as the measured run, or the
    comparison is meaningless."""
    original = mod._load_original_probe()
    assert isinstance(original._SYSTEM, str) and original._SYSTEM.strip()
    assert callable(original._user_prompt)
    # The stability script must not define its own competing system prompt.
    assert not hasattr(mod, "_SYSTEM")
