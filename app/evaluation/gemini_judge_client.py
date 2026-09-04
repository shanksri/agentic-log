"""Gemini-backed ``JudgeLLMClient`` for the evaluation judge.

# Why a second model family

``LLMService`` generates hypotheses and answers with ``OPENAI_MODEL``
(``gpt-4o-mini``). If the judge scoring those answers is the same model, the
evaluation is a model agreeing with itself: the numbers show self-consistency,
not correctness. The README already carries this as an open caveat.

``LLMJudge`` was written against a one-method ``JudgeLLMClient`` protocol so a
different backend could be substituted without touching the judge, its
prompts, or its parsing. This module is that substitution and nothing more, so
a run with this client is directly comparable to a run with any other.

``AnthropicJudgeClient`` is the equivalent for Claude. Either satisfies the
requirement; only one is needed.

# Free tier and retries

Gemini's free tier sheds load aggressively. Measured against this project's
key, short prompts succeed while the longer judge prompt returned ``503
UNAVAILABLE`` five times in a row, then succeeded later unchanged. Without
retries a 56-query evaluation run cannot complete.

So this client retries ``503`` and ``429`` with exponential backoff and
jitter, up to ``_DEFAULT_MAX_ATTEMPTS``. Both statuses mean "try again", not
"your request was wrong". Every other status fails immediately, because
retrying a 400 or a 404 just wastes time and hides the real cause.

Retries are bounded on purpose. An unbounded retry loop turns a dead backend
into a hang, and a half-finished evaluation run that silently drops queries
would look like a quality change rather than an outage.

# Response format

``LLMJudge`` parses responses as raw JSON. Gemini wraps JSON in ```json code
fences by default, where OpenAI does not, so the identical prompt would parse
for one backend and fail for the other. This client sets
``response_mime_type="application/json"`` so the judge and its prompts stay
untouched and runs remain comparable. Normalising a backend's formatting quirk
is the adapter's job, not the judge's.

# Model naming

``gemini-2.0-flash`` is retired and returns 404. The default here is
``gemini-3.6-flash``, verified working against this project's key. Model
availability changes without warning, and a 404 from a retired model looks
nothing like an auth failure, so ``GeminiJudgeError`` preserves the API's own
message rather than flattening it.
"""

from __future__ import annotations

import logging
import random
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.core.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_MAX_OUTPUT_TOKENS = 1024
_DEFAULT_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 2.0
# 503 is the free tier shedding load and 429 is its rate limit; both clear on
# their own. Every other status is a real error and must not be retried.
_RETRYABLE_STATUS = (429, 503)

_SYSTEM = (
    "You are an impartial evaluator of incident-investigation output. Judge "
    "only what you are shown. Do not assume facts that are not present, and "
    "do not reward fluent writing that is unsupported by the evidence given. "
    "Follow the requested response format exactly."
)


class GeminiJudgeError(RuntimeError):
    """Raised when the Gemini API call fails or returns nothing usable. Typed
    so an evaluation run can tell "the judge backend is broken" from "the judge
    returned something I could not parse", which is ``JudgeResponseError`` in
    ``llm_judge``.
    """


class GeminiJudgeClient:
    """``JudgeLLMClient`` implementation: one ``complete(prompt) -> str``.

    Mirrors ``LLMService.__init__``: explicit key and model arguments falling
    back to settings, and a hard failure at construction when no key is
    available, so a missing key surfaces before an evaluation run starts
    rather than partway through one.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to use the Gemini judge. Get one from "
                "aistudio.google.com and put it in .env; do not paste it into a chat "
                "transcript."
            )
        self.client = genai.Client(api_key=self.api_key)

    def complete(self, prompt: str) -> str:
        """Return the model's text response. Temperature is pinned to 0 so
        repeat runs over the same gold set differ as little as the API allows:
        judge scores are compared across runs, so sampling drift here would be
        indistinguishable from a real change in system quality.
        """
        config = genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
            # LLMJudge parses the response as raw JSON. Gemini otherwise wraps
            # it in ```json fences, which OpenAI does not, so without this the
            # identical prompt parses for one backend and fails for the other.
            response_mime_type="application/json",
        )
        response = self._generate_with_retry(prompt, config)

        text = _strip_code_fences(response.text)
        if not text or not text.strip():
            raise GeminiJudgeError(
                "Gemini returned no text content "
                f"(finish reason: {_finish_reason(response)!r})"
            )
        return text


    def _generate_with_retry(self, prompt: str, config):
        """Call the API, retrying only statuses that mean "try again"."""
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except genai_errors.APIError as exc:
                if getattr(exc, "code", None) not in _RETRYABLE_STATUS:
                    raise GeminiJudgeError(f"Gemini request failed: {exc}") from exc
                last = exc
                if attempt == self.max_attempts:
                    break
                delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                delay += random.uniform(0, delay * 0.25)  # jitter, avoids lockstep
                logger.warning(
                    "gemini_judge.retrying",
                    extra={"attempt": attempt, "status": getattr(exc, "code", None),
                           "sleep_seconds": round(delay, 2)},
                )
                time.sleep(delay)
        raise GeminiJudgeError(
            f"Gemini request failed after {self.max_attempts} attempts: {last}"
        ) from last


def _strip_code_fences(text: str | None) -> str | None:
    """Remove a wrapping ```json ... ``` block if one is present.

    ``response_mime_type="application/json"`` should make this unnecessary,
    and normally does. It is kept as a fallback because model output formats
    drift between versions, and the failure it guards against is a parse error
    partway through an evaluation run -- expensive to re-run and easy to
    misread as a quality change.
    """
    if not text:
        return text
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    body = stripped[3:]
    if body.lower().startswith("json"):
        body = body[4:]
    return body.rsplit("```", 1)[0].strip() if "```" in body else body.strip()


def _finish_reason(response) -> str | None:
    """Best-effort extraction for the error message. A blank response is most
    often a safety block or a token-limit stop, and which one it was decides
    whether the prompt or the config needs changing.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return getattr(reason, "name", None) or (str(reason) if reason else None)
