"""Anthropic-backed ``JudgeLLMClient`` for the evaluation judge.

# Why a second model family

``LLMService`` generates hypotheses and answers with ``OPENAI_MODEL``
(``gpt-4o-mini``). If the judge that scores those answers is the same model,
the evaluation is a model agreeing with itself, and the resulting numbers
cannot be used to claim the system is correct -- only that it is
self-consistent. The README already carries this as an open caveat.

``LLMJudge`` was built against a one-method ``JudgeLLMClient`` protocol
precisely so a different backend could be substituted without touching the
judge, the prompts, or the parsing. This module is that substitution and
nothing more: it does not change what is judged, how prompts are worded, or
how responses are parsed, so a run with this client is directly comparable
to a run with an OpenAI-backed one.

# Billing

Anthropic API credits are billed separately from a Claude Pro subscription.
Pro covers claude.ai and Claude Code; it grants no API access. ``settings.
anthropic_api_key`` is therefore optional everywhere -- nothing in the
application requires it, and only evaluation runs that explicitly construct
this client will fail without it.

# Cost shape

One judge call per evaluated item, roughly 1-2k input tokens and ~200 output
tokens. A full pass over the 56-query gold set is on the order of 50k input
tokens, so the default model is the cheapest of the Claude line. Nothing here
retries, batches, or caches: at that volume the complexity would cost more
than the tokens.
"""

from __future__ import annotations

import logging

from anthropic import Anthropic, APIError

from app.core.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_TOKENS = 1024

_SYSTEM = (
    "You are an impartial evaluator of incident-investigation output. Judge "
    "only what you are shown. Do not assume facts that are not present, and "
    "do not reward fluent writing that is unsupported by the evidence given. "
    "Follow the requested response format exactly."
)


class AnthropicJudgeError(RuntimeError):
    """Raised when the Anthropic API call fails or returns nothing usable.
    Typed so an evaluation run can tell "the judge backend is broken" from "the
    judge returned an answer I could not parse", which is
    ``JudgeResponseError`` and lives in ``llm_judge``.
    """


class AnthropicJudgeClient:
    """``JudgeLLMClient`` implementation: one ``complete(prompt) -> str``.

    Deliberately mirrors ``LLMService.__init__`` -- explicit key and model
    arguments falling back to settings, and a hard failure at construction
    when no key is available, so a missing key surfaces before an evaluation
    run starts rather than partway through it.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self.max_tokens = max_tokens
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required to use the Anthropic judge. Note that a "
                "Claude Pro subscription does not include API access; credits are "
                "purchased separately at console.anthropic.com."
            )
        self.client = Anthropic(api_key=self.api_key, timeout=timeout)

    def complete(self, prompt: str) -> str:
        """Return the model's text response. Temperature is pinned to 0 so
        repeat runs over the same gold set differ as little as the API allows
        -- judge scores are compared across runs, so drift here would be
        indistinguishable from a real change in system quality.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except APIError as exc:
            raise AnthropicJudgeError(f"Anthropic request failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise AnthropicJudgeError(
                f"Anthropic returned no text content (stop_reason="
                f"{getattr(response, 'stop_reason', None)!r})"
            )
        return text
