"""Phase 23: functional validation + resilience tests for ``LLMService``.

No real OpenAI calls — ``self.client`` is monkeypatched with a fake object
exposing ``chat.completions.create`` so these run offline and deterministically.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai import APIError

from app.services.llm_service import LLMResponseError, LLMService


def _service() -> LLMService:
    return LLMService(api_key="fake-key", model="fake-model")


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_missing_api_key_raises_value_error(monkeypatch) -> None:
    monkeypatch.setattr("app.services.llm_service.settings.openai_api_key", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        LLMService(api_key=None)


def test_client_configured_with_default_timeout() -> None:
    service = _service()
    # OpenAI's SDK client stores the configured timeout on `.timeout`.
    assert service.client.timeout == 30.0


def test_client_honors_explicit_timeout() -> None:
    service = LLMService(api_key="fake-key", timeout=5.0)
    assert service.client.timeout == 5.0


def test_generate_json_malformed_json_raises_llm_response_error() -> None:
    service = _service()
    service.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: _fake_response("{not valid json")
            )
        )
    )
    with pytest.raises(LLMResponseError, match="not valid JSON"):
        service.generate_json(system_prompt="sys", user_prompt="usr")


def test_generate_json_non_dict_json_raises_llm_response_error() -> None:
    service = _service()
    service.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: _fake_response("[1, 2, 3]"))
        )
    )
    with pytest.raises(LLMResponseError, match="JSON object"):
        service.generate_json(system_prompt="sys", user_prompt="usr")


def test_generate_json_empty_content_defaults_to_empty_object() -> None:
    service = _service()
    service.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: _fake_response(None))
        )
    )
    assert service.generate_json(system_prompt="sys", user_prompt="usr") == {}


def test_generate_json_api_error_raises_llm_response_error() -> None:
    service = _service()

    def _raise(**kwargs):
        raise APIError("boom", request=SimpleNamespace(), body=None)

    service.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_raise))
    )
    with pytest.raises(LLMResponseError, match="OpenAI request failed"):
        service.generate_json(system_prompt="sys", user_prompt="usr")


def test_generate_investigation_api_error_raises_llm_response_error() -> None:
    service = _service()

    def _raise(**kwargs):
        raise APIError("boom", request=SimpleNamespace(), body=None)

    service.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_raise))
    )
    with pytest.raises(LLMResponseError):
        service.generate_investigation(problem="p", context="c")


def test_generate_investigation_returns_content() -> None:
    service = _service()
    service.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: _fake_response("root cause: X"))
        )
    )
    assert service.generate_investigation(problem="p", context="c") == "root cause: X"


def test_generate_hypotheses_tolerates_malformed_payload_shape() -> None:
    """generate_json succeeds but 'hypotheses' isn't a list — the higher-level
    method degrades to an empty list rather than raising.
    """
    service = _service()
    service.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: _fake_response('{"hypotheses": "not-a-list"}')
            )
        )
    )
    assert service.generate_hypotheses(problem="p", context="c") == []
