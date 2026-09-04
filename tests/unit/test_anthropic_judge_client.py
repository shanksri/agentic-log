"""Tests for the Anthropic-backed JudgeLLMClient.

No network and no key required: the SDK client is replaced with a fake after
construction, or construction itself is exercised for its failure mode. The
point of this client is that answers written by OPENAI_MODEL are not also
graded by it, so the test that matters most is the structural one at the
bottom -- that it genuinely satisfies the protocol LLMJudge depends on.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from app.evaluation.anthropic_judge_client import (
    AnthropicJudgeClient,
    AnthropicJudgeError,
)
from app.evaluation.llm_judge import JudgeLLMClient, LLMJudge
from app.services.planner_agent import InvestigationPlan, PlanningStrategy


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


class FakeMessages:
    def __init__(self, response=None, *, raises=None):
        self._response = response
        self._raises = raises
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._response


def _client(monkeypatch, response=None, *, raises=None) -> AnthropicJudgeClient:
    monkeypatch.setattr(
        "app.evaluation.anthropic_judge_client.Anthropic",
        lambda **kwargs: SimpleNamespace(messages=FakeMessages(response, raises=raises)),
    )
    return AnthropicJudgeClient(api_key="test-key", model="claude-haiku-4-5")


# ── Construction ───────────────────────────────────────────────────────────────


def test_missing_key_fails_at_construction_not_mid_run(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.evaluation.anthropic_judge_client.settings",
        SimpleNamespace(anthropic_api_key=None, anthropic_model="claude-haiku-4-5"),
    )
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
        AnthropicJudgeClient()


def test_missing_key_message_explains_pro_is_not_api_access(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.evaluation.anthropic_judge_client.settings",
        SimpleNamespace(anthropic_api_key=None, anthropic_model="m"),
    )
    with pytest.raises(ValueError, match="does not include API access"):
        AnthropicJudgeClient()


def test_explicit_arguments_win_over_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.evaluation.anthropic_judge_client.settings",
        SimpleNamespace(anthropic_api_key="from-settings", anthropic_model="from-settings"),
    )
    monkeypatch.setattr(
        "app.evaluation.anthropic_judge_client.Anthropic",
        lambda **kwargs: SimpleNamespace(messages=FakeMessages(None)),
    )
    client = AnthropicJudgeClient(api_key="explicit", model="explicit-model")
    assert client.api_key == "explicit" and client.model == "explicit-model"


# ── complete() ─────────────────────────────────────────────────────────────────


def test_returns_the_text_of_the_response(monkeypatch) -> None:
    client = _client(monkeypatch, SimpleNamespace(content=[_text_block("verdict: good")]))
    assert client.complete("judge this") == "verdict: good"


def test_concatenates_multiple_text_blocks(monkeypatch) -> None:
    client = _client(
        monkeypatch, SimpleNamespace(content=[_text_block("part one "), _text_block("part two")])
    )
    assert client.complete("p") == "part one part two"


def test_ignores_non_text_blocks(monkeypatch) -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", text="ignored"), _text_block("kept")]
    )
    assert _client(monkeypatch, response).complete("p") == "kept"


def test_temperature_is_pinned_to_zero_for_run_to_run_comparability(monkeypatch) -> None:
    client = _client(monkeypatch, SimpleNamespace(content=[_text_block("x")]))
    client.complete("p")
    assert client.client.messages.calls[0]["temperature"] == 0


def test_prompt_is_passed_through_unmodified(monkeypatch) -> None:
    client = _client(monkeypatch, SimpleNamespace(content=[_text_block("x")]))
    client.complete("the exact prompt LLMJudge built")
    sent = client.client.messages.calls[0]["messages"][0]["content"]
    assert sent == "the exact prompt LLMJudge built"


# ── Failures ───────────────────────────────────────────────────────────────────


def test_api_error_is_wrapped_in_the_typed_error(monkeypatch) -> None:
    from anthropic import APIError

    error = APIError("boom", request=SimpleNamespace(), body=None)
    client = _client(monkeypatch, raises=error)
    with pytest.raises(AnthropicJudgeError, match="Anthropic request failed"):
        client.complete("p")


def test_empty_response_raises_rather_than_returning_blank(monkeypatch) -> None:
    response = SimpleNamespace(content=[_text_block("   ")], stop_reason="max_tokens")
    with pytest.raises(AnthropicJudgeError, match="no text content"):
        _client(monkeypatch, response).complete("p")


def test_no_content_blocks_at_all_raises(monkeypatch) -> None:
    with pytest.raises(AnthropicJudgeError):
        _client(monkeypatch, SimpleNamespace(content=[], stop_reason="end_turn")).complete("p")


# ── Protocol conformance ───────────────────────────────────────────────────────


def test_satisfies_the_judge_client_protocol(monkeypatch) -> None:
    """The whole design depends on LLMJudge accepting this without changes.

    ``JudgeLLMClient`` is not ``@runtime_checkable``, so this compares
    signatures structurally rather than calling ``isinstance``. Making the
    production protocol runtime-checkable purely to simplify a test would be
    the tail wagging the dog.
    """
    client = _client(monkeypatch, SimpleNamespace(content=[_text_block("x")]))
    assert callable(client.complete)
    expected = inspect.signature(JudgeLLMClient.complete)
    actual = inspect.signature(type(client).complete)
    assert list(actual.parameters) == list(expected.parameters)
    assert actual.return_annotation == expected.return_annotation


def test_llm_judge_accepts_this_client_and_calls_it_once(monkeypatch) -> None:
    """End-to-end through the real LLMJudge, with the API faked."""
    client = _client(
        monkeypatch,
        SimpleNamespace(content=[_text_block('{"score": 4, "explanation": "ok"}')]),
    )
    judge = LLMJudge(client)
    plan = InvestigationPlan(
        problem="p", strategy=PlanningStrategy.UNKNOWN, objective="o",
        priority_list=("a",), evidence_priorities=("b",), assumptions=("c",),
        expected_difficulty="medium", strategy_rationale="r",
    )
    judge.evaluate_plan("p", plan)
    assert len(client.client.messages.calls) == 1
