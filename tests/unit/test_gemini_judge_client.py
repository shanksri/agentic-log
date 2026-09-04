"""Tests for the Gemini-backed JudgeLLMClient.

No network and no key required: the SDK client is replaced with a fake, and
``time.sleep`` is stubbed so backoff behavior is asserted without waiting for
it. The retry tests exist because the free tier genuinely sheds load -- a
judge-sized prompt returned 503 five times in a row against a real key, then
succeeded unchanged -- so retry correctness is load-bearing here, not
theoretical.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.evaluation.gemini_judge_client import (
    GeminiJudgeClient,
    GeminiJudgeError,
    _strip_code_fences,
)
from app.evaluation.llm_judge import JudgeLLMClient, LLMJudge
from app.services.planner_agent import InvestigationPlan, PlanningStrategy


def _response(text: str, *, finish_reason: str = "STOP"):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))],
    )


def _api_error(status: int) -> genai_errors.APIError:
    error = genai_errors.APIError.__new__(genai_errors.APIError)
    error.code = status
    error.message = f"status {status}"
    error.args = (f"status {status}",)
    return error


class FakeModels:
    """Yields queued outcomes in order; an Exception is raised, anything else
    returned. The last outcome repeats once exhausted.
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("app.evaluation.gemini_judge_client.time.sleep", slept.append)
    return slept


def _client(monkeypatch, outcomes, **kwargs) -> GeminiJudgeClient:
    monkeypatch.setattr(
        "app.evaluation.gemini_judge_client.genai.Client",
        lambda **_: SimpleNamespace(models=FakeModels(outcomes)),
    )
    return GeminiJudgeClient(api_key="test-key", model="gemini-3.1-flash-lite", **kwargs)


# ── Construction ───────────────────────────────────────────────────────────────


def test_missing_key_fails_at_construction_not_mid_run(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.evaluation.gemini_judge_client.settings",
        SimpleNamespace(gemini_api_key=None, gemini_model="gemini-3.1-flash-lite"),
    )
    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        GeminiJudgeClient()


def test_missing_key_message_warns_against_pasting_into_chat(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.evaluation.gemini_judge_client.settings",
        SimpleNamespace(gemini_api_key=None, gemini_model="m"),
    )
    with pytest.raises(ValueError, match="do not paste it into a chat"):
        GeminiJudgeClient()


# ── complete() ─────────────────────────────────────────────────────────────────


def test_returns_the_response_text(monkeypatch) -> None:
    client = _client(monkeypatch, [_response('{"score": 4}')])
    assert client.complete("p") == '{"score": 4}'


def test_temperature_pinned_to_zero_and_json_mime_type_requested(monkeypatch) -> None:
    client = _client(monkeypatch, [_response("{}")])
    client.complete("p")
    config = client.client.models.calls[0]["config"]
    assert config.temperature == 0
    assert config.response_mime_type == "application/json"


def test_prompt_passed_through_unmodified(monkeypatch) -> None:
    client = _client(monkeypatch, [_response("{}")])
    client.complete("the exact prompt LLMJudge built")
    assert client.client.models.calls[0]["contents"] == "the exact prompt LLMJudge built"


def test_blank_response_raises_and_names_the_finish_reason(monkeypatch) -> None:
    client = _client(monkeypatch, [_response("   ", finish_reason="MAX_TOKENS")])
    with pytest.raises(GeminiJudgeError, match="MAX_TOKENS"):
        client.complete("p")


# ── Retry ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", [503, 429])
def test_retryable_statuses_are_retried_then_succeed(monkeypatch, status) -> None:
    client = _client(monkeypatch, [_api_error(status), _api_error(status), _response("{}")])
    assert client.complete("p") == "{}"
    assert len(client.client.models.calls) == 3


@pytest.mark.parametrize("status", [400, 404, 401, 500])
def test_non_retryable_statuses_fail_immediately(monkeypatch, status) -> None:
    client = _client(monkeypatch, [_api_error(status), _response("{}")])
    with pytest.raises(GeminiJudgeError, match="Gemini request failed"):
        client.complete("p")
    assert len(client.client.models.calls) == 1, "must not retry a client error"


def test_retries_are_bounded(monkeypatch) -> None:
    client = _client(monkeypatch, [_api_error(503)], max_attempts=3)
    with pytest.raises(GeminiJudgeError, match="after 3 attempts"):
        client.complete("p")
    assert len(client.client.models.calls) == 3


def test_backoff_grows_between_attempts(monkeypatch, no_sleep) -> None:
    client = _client(monkeypatch, [_api_error(503)], max_attempts=4)
    with pytest.raises(GeminiJudgeError):
        client.complete("p")
    assert len(no_sleep) == 3, "sleeps between attempts, not after the last one"
    assert no_sleep[0] < no_sleep[1] < no_sleep[2]


def test_no_sleep_when_the_first_attempt_succeeds(monkeypatch, no_sleep) -> None:
    _client(monkeypatch, [_response("{}")]).complete("p")
    assert no_sleep == []


# ── Code fences ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ("", ""),
    ],
)
def test_code_fences_are_stripped_when_present(raw, expected) -> None:
    assert _strip_code_fences(raw) == expected


def test_fenced_response_still_parses(monkeypatch) -> None:
    """The API config should prevent fences, but drift must not break a run."""
    client = _client(monkeypatch, [_response('```json\n{"score": 2}\n```')])
    assert client.complete("p") == '{"score": 2}'


# ── Protocol conformance ───────────────────────────────────────────────────────


def test_signature_matches_the_judge_client_protocol(monkeypatch) -> None:
    client = _client(monkeypatch, [_response("{}")])
    expected = inspect.signature(JudgeLLMClient.complete)
    actual = inspect.signature(type(client).complete)
    assert list(actual.parameters) == list(expected.parameters)
    assert actual.return_annotation == expected.return_annotation


def test_llm_judge_accepts_this_client_and_calls_it_once(monkeypatch) -> None:
    client = _client(monkeypatch, [_response('{"score": 4, "explanation": "ok"}')])
    plan = InvestigationPlan(
        problem="p", strategy=PlanningStrategy.UNKNOWN, objective="o",
        priority_list=("a",), evidence_priorities=("b",), assumptions=("c",),
        expected_difficulty="medium", strategy_rationale="r",
    )
    LLMJudge(client).evaluate_plan("p", plan)
    assert len(client.client.models.calls) == 1
