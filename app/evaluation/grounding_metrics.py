"""Grounding Metric Engine — RAGAS-style RAG evaluation (Phase 22).

Implements the four core RAGAS metrics (Es et al., 2023) plus Context Entity
Recall, judging a generated answer against the retrieved context it was
grounded in — the questions BERTScore structurally cannot answer:

- **Faithfulness** — is every claim in the answer supported by the retrieved
  context? (hallucination detector; needs answer + context, NO reference)
- **Answer Relevancy** — does the answer actually address the question?
  (needs question + answer + a sentence embedder, NO reference)
- **Context Precision** — are the retrieved chunks that matter ranked ahead
  of the ones that don't? (needs question + contexts + reference-or-answer)
- **Context Recall** — does the retrieved context cover everything the
  reference answer needs? (REQUIRES a reference)
- **Context Entity Recall** — are the entities the reference mentions
  present in the retrieved context? (REQUIRES a reference)

# Why not the `ragas` library

The ``ragas`` package hard-depends on langchain + datasets and binds to
specific LLM providers. This codebase's standing convention (Phase 20B's
``JudgeLLMClient``, Phase 21C's ``AuthorLLMClient``) is to implement
LLM-backed evaluation against a minimal protocol so unit tests never require
OpenAI and the provider stays swappable. The two protocols here are satisfied
by EXISTING services with zero adapters:

- ``GroundingLLMClient`` = ``generate_json(*, system_prompt, user_prompt) ->
  dict`` — ``app.services.llm_service.LLMService`` already has exactly this
  method (pre-16, unmodified).
- ``SentenceEmbedder`` = ``embed_text(text) -> Sequence[float]`` —
  ``app.services.embedding_service.EmbeddingService`` already has exactly
  this method.

# Formulas (faithful to the RAGAS definitions)

- faithfulness           = supported_claims / total_claims
- answer_relevancy       = mean cosine(embed(question),
                                        embed(generated_question_i)),
                           clamped to [0, 1] (embeddings can produce small
                           negative cosines; a negative relevancy is
                           meaningless on our scale)
- context_precision@K    = Σ_k (precision@k × v_k) / |relevant chunks|,
                           v_k ∈ {0,1} per retrieved chunk in RANK ORDER —
                           rank-sensitive by construction
- context_recall         = attributed_reference_claims / total_reference_claims
- context_entity_recall  = |entities(reference) ∩ entities(contexts)|
                           / |entities(reference)|   (case-insensitive)

# Undefined (None), never fabricated

Every metric returns ``None`` when its inputs make it uncomputable — empty
contexts (all five), missing reference (recall + entity recall; precision
falls back to the generated answer), zero extracted claims/entities/questions
(the ratio's denominator would be 0). A malformed LLM response raises
``GroundingResponseError`` (a ``ValueError``, mirroring Phase 20B's
``JudgeResponseError``) — strict parsing, no retry, no self-repair; the
HARNESS decides to isolate that to per-metric ``None`` + a recorded note,
this module never guesses.

# Determinism

Given a deterministic client/embedder (as in unit tests) every function here
is deterministic. In production the LLM verdicts are the acknowledged source
of variance — the same caveat Phase 20B documents for ``LLMJudge``.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from app.evaluation.generation_metrics import (
    MetricSummary,
    cosine_similarity,
    summarize_values,
)

DEFAULT_RELEVANCY_QUESTION_COUNT = 3

# Evaluator-stability confidence bands (Phase 22B). These classify the
# STANDARD DEVIATION of a grounding metric across repeated executions of the
# SAME evaluation — i.e. how stable the LLM evaluator is, not how good the
# answer is. Boundary semantics: std < 0.05 → HIGH; 0.05 ≤ std ≤ 0.10 →
# MEDIUM; std > 0.10 → LOW.
STDDEV_HIGH_CONFIDENCE_MAX = 0.05
STDDEV_MEDIUM_CONFIDENCE_MAX = 0.10


# ── Protocols ─────────────────────────────────────────────────────────────────


class GroundingLLMClient(Protocol):
    """JSON-mode completion — ``LLMService.generate_json`` satisfies this
    directly (duck-typed), no adapter needed."""

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


class SentenceEmbedder(Protocol):
    """Whole-text embedding — ``EmbeddingService.embed_text`` satisfies this
    directly (duck-typed), no adapter needed."""

    def embed_text(self, text: str) -> Sequence[float]: ...


class GroundingResponseError(ValueError):
    """The grounding LLM's JSON did not have the required shape."""


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GroundingMetrics:
    """RAGAS-style scores for ONE (question, answer, contexts[, reference])
    tuple. Each field is ``None`` when that metric was uncomputable for this
    query (see module docstring's "Undefined") — the harness records why in
    the per-query notes.
    """

    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    context_entity_recall: float | None


@dataclass(frozen=True)
class GroundingAggregateMetrics:
    """Dataset-level statistics per grounding metric, over defined values
    only (same convention as ``GenerationAggregateMetrics``)."""

    num_scored: int
    faithfulness: MetricSummary | None
    answer_relevancy: MetricSummary | None
    context_precision: MetricSummary | None
    context_recall: MetricSummary | None
    context_entity_recall: MetricSummary | None


# ── Evaluator stability (Phase 22B) ───────────────────────────────────────────


class EvaluatorConfidence(str, Enum):
    """Qualitative classification of a grounding metric's run-to-run
    stability — a statement about the EVALUATOR, never about the answer.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def classify_evaluator_confidence(std_dev: float) -> EvaluatorConfidence:
    """std < 0.05 → HIGH; 0.05 ≤ std ≤ 0.10 → MEDIUM; std > 0.10 → LOW."""
    if std_dev < STDDEV_HIGH_CONFIDENCE_MAX:
        return EvaluatorConfidence.HIGH
    if std_dev <= STDDEV_MEDIUM_CONFIDENCE_MAX:
        return EvaluatorConfidence.MEDIUM
    return EvaluatorConfidence.LOW


@dataclass(frozen=True)
class MetricStability:
    """Run-to-run statistics for ONE grounding metric over N repeated
    executions of the same evaluation. ``mean`` is what gets reported as the
    metric value; the rest quantify evaluator variance (Phase 22B — the
    metric FORMULAS are unchanged, only how many times they run).
    """

    count: int
    mean: float
    std_dev: float
    minimum: float
    maximum: float
    confidence: EvaluatorConfidence


def measure_stability(samples: Sequence[float]) -> MetricStability:
    """Compute repetition statistics over ``samples`` (requires >= 2 — a
    single sample carries no variance information; the harness records
    stability as ``None`` in that case rather than fabricating std=0).
    Uses the SAMPLE standard deviation (``statistics.stdev``).
    """
    if len(samples) < 2:
        raise ValueError(
            f"measure_stability requires at least 2 samples, got {len(samples)}"
        )
    std_dev = statistics.stdev(samples)
    return MetricStability(
        count=len(samples),
        mean=statistics.mean(samples),
        std_dev=std_dev,
        minimum=min(samples),
        maximum=max(samples),
        confidence=classify_evaluator_confidence(std_dev),
    )


@dataclass(frozen=True)
class GroundingStability:
    """Per-metric run-to-run stability for ONE query, symmetric with
    ``GroundingMetrics``. A field is ``None`` when that metric was skipped,
    disabled, or produced fewer than 2 successful samples.
    """

    faithfulness: MetricStability | None
    answer_relevancy: MetricStability | None
    context_precision: MetricStability | None
    context_recall: MetricStability | None
    context_entity_recall: MetricStability | None


# ── Strict response parsing helpers ───────────────────────────────────────────


def _require_str_list(response: dict[str, Any], key: str) -> list[str]:
    value = response.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise GroundingResponseError(
            f"grounding LLM response field {key!r} must be a list of strings, "
            f"got {value!r}"
        )
    return value


def _require_bool_list(
    response: dict[str, Any], key: str, *, expected_length: int
) -> list[bool]:
    value = response.get(key)
    if not isinstance(value, list) or not all(isinstance(item, bool) for item in value):
        raise GroundingResponseError(
            f"grounding LLM response field {key!r} must be a list of booleans, "
            f"got {value!r}"
        )
    if len(value) != expected_length:
        raise GroundingResponseError(
            f"grounding LLM response field {key!r} must have exactly "
            f"{expected_length} entries, got {len(value)}"
        )
    return value


def _numbered(items: Sequence[str]) -> str:
    return "\n".join(f"[{index}] {item}" for index, item in enumerate(items, start=1))


# ── Faithfulness ──────────────────────────────────────────────────────────────


def faithfulness(
    answer: str,
    contexts: Sequence[str],
    *,
    llm: GroundingLLMClient,
) -> float | None:
    """supported_claims / total_claims. ``None`` when there are no contexts
    or the answer decomposes into zero claims.
    """
    if not contexts or not answer.strip():
        return None

    claims_response = llm.generate_json(
        system_prompt=(
            "You decompose an answer into atomic factual claims. Return ONLY a "
            'JSON object: {"claims": ["<claim>", ...]}. Each claim must be a '
            "single, self-contained factual statement from the answer. Do not "
            "add claims that are not in the answer."
        ),
        user_prompt=f"Answer:\n{answer}",
    )
    claims = _require_str_list(claims_response, "claims")
    if not claims:
        return None

    verdict_response = llm.generate_json(
        system_prompt=(
            "You verify claims against a context. For each claim, decide whether "
            "it can be directly inferred from the context alone. Return ONLY a "
            'JSON object: {"verdicts": [true, false, ...]} with exactly one '
            "boolean per claim, in order. Use false when the context does not "
            "support the claim, even partially."
        ),
        user_prompt=(
            f"Context:\n{_numbered(contexts)}\n\nClaims:\n{_numbered(claims)}"
        ),
    )
    verdicts = _require_bool_list(verdict_response, "verdicts", expected_length=len(claims))
    return sum(verdicts) / len(verdicts)


# ── Answer Relevancy ──────────────────────────────────────────────────────────


def answer_relevancy(
    question: str,
    answer: str,
    *,
    llm: GroundingLLMClient,
    embedder: SentenceEmbedder,
    n_questions: int = DEFAULT_RELEVANCY_QUESTION_COUNT,
) -> float | None:
    """Mean cosine between the original question and questions the answer
    would be a good answer to, clamped to [0, 1]. ``None`` when the answer
    is empty or the LLM produces zero questions.
    """
    if not answer.strip() or not question.strip():
        return None

    response = llm.generate_json(
        system_prompt=(
            f"Given an answer, write {n_questions} distinct questions that this "
            "answer would directly and completely answer. Return ONLY a JSON "
            'object: {"questions": ["<question>", ...]}.'
        ),
        user_prompt=f"Answer:\n{answer}",
    )
    generated_questions = _require_str_list(response, "questions")
    if not generated_questions:
        return None

    question_vector = embedder.embed_text(question)
    similarities = [
        cosine_similarity(question_vector, embedder.embed_text(generated))
        for generated in generated_questions
    ]
    mean_similarity = sum(similarities) / len(similarities)
    return max(0.0, min(1.0, mean_similarity))


# ── Context Precision ─────────────────────────────────────────────────────────


def context_precision(
    question: str,
    contexts: Sequence[str],
    ground_truth: str,
    *,
    llm: GroundingLLMClient,
) -> float | None:
    """Rank-weighted precision of the retrieved contexts:
    ``Σ_k (precision@k × v_k) / |relevant|`` with per-chunk usefulness
    verdicts in rank order. 0.0 when no chunk is relevant; ``None`` when
    there are no contexts. ``ground_truth`` is the reference answer when
    available, else the generated answer (the harness decides).
    """
    if not contexts:
        return None

    response = llm.generate_json(
        system_prompt=(
            "You judge retrieved context chunks. For each chunk, decide whether "
            "it was useful for arriving at the given ground-truth answer to the "
            'question. Return ONLY a JSON object: {"verdicts": [true, false, ...]} '
            "with exactly one boolean per chunk, in the order given."
        ),
        user_prompt=(
            f"Question:\n{question}\n\nGround-truth answer:\n{ground_truth}\n\n"
            f"Context chunks (in retrieval rank order):\n{_numbered(contexts)}"
        ),
    )
    verdicts = _require_bool_list(response, "verdicts", expected_length=len(contexts))

    total_relevant = sum(verdicts)
    if total_relevant == 0:
        return 0.0

    weighted_sum = 0.0
    relevant_so_far = 0
    for rank, verdict in enumerate(verdicts, start=1):
        if verdict:
            relevant_so_far += 1
            weighted_sum += relevant_so_far / rank
    return weighted_sum / total_relevant


# ── Context Recall ────────────────────────────────────────────────────────────


def context_recall(
    reference: str,
    contexts: Sequence[str],
    *,
    llm: GroundingLLMClient,
) -> float | None:
    """attributed_reference_claims / total_reference_claims. ``None`` when
    there are no contexts, no reference text, or the reference decomposes
    into zero claims.
    """
    if not contexts or not reference.strip():
        return None

    response = llm.generate_json(
        system_prompt=(
            "You decompose a reference answer into atomic factual claims, then "
            "decide for EACH claim whether it can be attributed to (supported "
            'by) the given context. Return ONLY a JSON object: {"claims": '
            '["<claim>", ...], "verdicts": [true, false, ...]} with exactly one '
            "boolean per claim, in order."
        ),
        user_prompt=(
            f"Reference answer:\n{reference}\n\nContext:\n{_numbered(contexts)}"
        ),
    )
    claims = _require_str_list(response, "claims")
    if not claims:
        return None
    verdicts = _require_bool_list(response, "verdicts", expected_length=len(claims))
    return sum(verdicts) / len(verdicts)


# ── Context Entity Recall ─────────────────────────────────────────────────────


def _extract_entities(text: str, *, llm: GroundingLLMClient, what: str) -> set[str]:
    response = llm.generate_json(
        system_prompt=(
            "You extract named entities (systems, components, services, error "
            "codes, versions, people, organizations) from text. Return ONLY a "
            'JSON object: {"entities": ["<entity>", ...]}. No duplicates.'
        ),
        user_prompt=f"{what}:\n{text}",
    )
    entities = _require_str_list(response, "entities")
    return {entity.strip().lower() for entity in entities if entity.strip()}


def context_entity_recall(
    reference: str,
    contexts: Sequence[str],
    *,
    llm: GroundingLLMClient,
) -> float | None:
    """|entities(reference) ∩ entities(contexts)| / |entities(reference)|,
    case-insensitive. ``None`` when there are no contexts, no reference, or
    the reference contains zero entities.
    """
    if not contexts or not reference.strip():
        return None

    reference_entities = _extract_entities(reference, llm=llm, what="Reference answer")
    if not reference_entities:
        return None
    context_entities = _extract_entities(
        "\n".join(contexts), llm=llm, what="Retrieved context"
    )
    return len(reference_entities & context_entities) / len(reference_entities)


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_grounding_metrics(
    metrics: Sequence[GroundingMetrics],
) -> GroundingAggregateMetrics:
    """Dataset-level statistics per grounding metric over defined values
    only — a query whose metric is ``None`` simply doesn't contribute.
    """

    def _defined(field: str) -> list[float]:
        return [
            getattr(m, field) for m in metrics if getattr(m, field) is not None
        ]

    return GroundingAggregateMetrics(
        num_scored=len(metrics),
        faithfulness=summarize_values(_defined("faithfulness")),
        answer_relevancy=summarize_values(_defined("answer_relevancy")),
        context_precision=summarize_values(_defined("context_precision")),
        context_recall=summarize_values(_defined("context_recall")),
        context_entity_recall=summarize_values(_defined("context_entity_recall")),
    )
