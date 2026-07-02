"""Generation Metric Engine — semantic answer quality (Phase 22).

Scores one LLM-generated answer against one human-authored reference answer
using **BERTScore** — the semantic-similarity half of Phase 22's generation
evaluation (the grounding half — RAGAS-style Faithfulness / Answer Relevancy /
Context Precision / Context Recall — lives in
``app.evaluation.grounding_metrics``).

Lexical n-gram metrics (BLEU / ROUGE / METEOR) are DELIBERATELY not
implemented: this is an enterprise Retrieval-Augmented Generation platform,
not a traditional NLP benchmark. Two answers that recommend the same fix in
different words are equivalent for our purposes; surface n-gram overlap would
punish exactly the paraphrasing an LLM answer is expected to produce.
Semantic similarity (BERTScore) and grounding (RAGAS) are the signals that
matter here, per the Phase 22 brief.

This module is a pure mathematical layer, the generation-side sibling of
``app.evaluation.metrics`` (Phase 16C, retrieval):

- **No retrieval.** Never imports ``IncidentSearchService`` or touches the DB.
- **No LLM calls.** BERTScore needs token embeddings, supplied through the
  ``TokenEmbedder`` protocol below — this module never constructs a model.
- **Deterministic** given a deterministic embedder.

# BERTScore

(Zhang et al., 2020) — greedy cosine matching between candidate and reference
token embeddings: recall = mean over reference tokens of the best cosine to
any candidate token; precision = the mirror; F1 = harmonic mean. Two
documented simplifications: no idf weighting and no baseline rescaling.
Token embeddings come from an injected ``TokenEmbedder`` — when none is
supplied every BERTScore field is ``None`` (undefined), never fabricated.
This mirrors Phase 20B's ``JudgeLLMClient`` pattern: the metric ships fully
implemented against a protocol; unit tests use a fake embedder and never
download a model.

# Edge-case contract

- Empty candidate OR empty reference (after tokenization) → 0.0 (defined).
  An empty answer earns zero credit; an empty reference gives nothing to
  match. "Missing reference" (``None``) is NOT this module's concern — the
  harness skips BERTScore for those queries before calling in here.
- BERTScore is ``None`` (undefined) when no embedder is supplied.

# Aggregation

``aggregate_generation_metrics`` reduces per-query ``GenerationMetrics`` to
dataset-level ``MetricSummary`` (count / mean / median / min / max) per
metric, computed over DEFINED values only — the same "mean over defined
values" convention as Phase 16D's ``AggregateMetrics`` and Phase 20B's
``aggregate_judge_evaluations``. A metric with zero defined values
aggregates to ``None``, never a fabricated 0.0. ``MetricSummary`` and
``summarize_values`` are shared with ``grounding_metrics`` so both halves
of generation evaluation aggregate identically.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# Same lowercase-\w+ convention as app.services.bm25_search.default_tokenizer
# and app.services.routing's signal extraction — one tokenization vocabulary
# across the codebase.
_TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization (``\\w+``)."""
    return _TOKEN_PATTERN.findall(text.lower())


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationMetrics:
    """Semantic-similarity scores for ONE (candidate, reference) pair.

    Fields are ``None`` when no ``TokenEmbedder`` was supplied (undefined,
    not zero).
    """

    bert_score_precision: float | None
    bert_score_recall: float | None
    bert_score_f1: float | None


@dataclass(frozen=True)
class MetricSummary:
    """Dataset-level statistics for one metric over its defined values.
    Shared by generation (this module) and grounding
    (``app.evaluation.grounding_metrics``) aggregation.
    """

    count: int
    mean: float
    median: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class GenerationAggregateMetrics:
    """Dataset-level statistics per metric. A summary is ``None`` when zero
    queries had a defined value for that metric.
    """

    num_scored: int
    bert_score_precision: MetricSummary | None
    bert_score_recall: MetricSummary | None
    bert_score_f1: MetricSummary | None


# ── TokenEmbedder protocol (BERTScore backend) ────────────────────────────────


class TokenEmbedder(Protocol):
    """Anything that can embed each token of a text into a vector.

    ``embed_tokens(text)`` returns one vector per token of ``text`` (the
    implementation may use its own tokenizer — BERTScore's greedy matching
    only needs the two vector sequences, not aligned token strings). An
    empty text must return an empty sequence.
    """

    def embed_tokens(self, text: str) -> Sequence[Sequence[float]]: ...


class SentenceTransformerTokenEmbedder:
    """Concrete ``TokenEmbedder`` over the project's existing
    sentence-transformers model (``EmbeddingService.model``) using
    ``output_value="token_embeddings"``. Constructed lazily so importing
    this module never loads a model; unit tests use fakes instead.
    """

    def __init__(self, embedding_service) -> None:  # duck-typed: needs .model
        self._embedding_service = embedding_service

    def embed_tokens(self, text: str) -> Sequence[Sequence[float]]:
        if not text.strip():
            return []
        token_embeddings = self._embedding_service.model.encode(
            text, output_value="token_embeddings"
        )
        return [[float(v) for v in row] for row in token_embeddings.tolist()]


# ── Cosine (shared with grounding_metrics) ────────────────────────────────────


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── BERTScore ─────────────────────────────────────────────────────────────────


def bert_score(
    candidate: str,
    reference: str,
    *,
    token_embedder: TokenEmbedder,
) -> tuple[float, float, float]:
    """BERTScore ``(precision, recall, f1)`` via greedy cosine matching over
    token embeddings; see module docstring for documented simplifications
    (no idf weighting, no baseline rescaling).
    """
    cand_vectors = list(token_embedder.embed_tokens(candidate))
    ref_vectors = list(token_embedder.embed_tokens(reference))
    if not cand_vectors or not ref_vectors:
        return (0.0, 0.0, 0.0)

    precision = sum(
        max(cosine_similarity(cv, rv) for rv in ref_vectors) for cv in cand_vectors
    ) / len(cand_vectors)
    recall = sum(
        max(cosine_similarity(rv, cv) for cv in cand_vectors) for rv in ref_vectors
    ) / len(ref_vectors)
    if precision + recall == 0.0:
        return (0.0, 0.0, 0.0)
    f1 = 2.0 * precision * recall / (precision + recall)
    return (precision, recall, f1)


# ── Top-level scoring + aggregation ───────────────────────────────────────────


def compute_generation_metrics(
    candidate: str,
    reference: str,
    *,
    token_embedder: TokenEmbedder | None = None,
) -> GenerationMetrics:
    """Score one generated ``candidate`` against one ``reference`` answer.

    All fields are ``None`` when ``token_embedder`` is ``None`` (undefined —
    the caller had no embedding backend), and defined floats (possibly 0.0)
    otherwise.
    """
    if token_embedder is None:
        return GenerationMetrics(
            bert_score_precision=None,
            bert_score_recall=None,
            bert_score_f1=None,
        )
    precision, recall, f1 = bert_score(
        candidate, reference, token_embedder=token_embedder
    )
    return GenerationMetrics(
        bert_score_precision=precision,
        bert_score_recall=recall,
        bert_score_f1=f1,
    )


def summarize_values(values: Sequence[float]) -> MetricSummary | None:
    """count/mean/median/min/max over ``values``; ``None`` for empty input.
    Shared by generation and grounding aggregation.
    """
    if not values:
        return None
    return MetricSummary(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
    )


def aggregate_generation_metrics(
    metrics: Sequence[GenerationMetrics],
) -> GenerationAggregateMetrics:
    """Dataset-level statistics over defined values only — see module
    docstring's "Aggregation".
    """
    return GenerationAggregateMetrics(
        num_scored=len(metrics),
        bert_score_precision=summarize_values(
            [m.bert_score_precision for m in metrics if m.bert_score_precision is not None]
        ),
        bert_score_recall=summarize_values(
            [m.bert_score_recall for m in metrics if m.bert_score_recall is not None]
        ),
        bert_score_f1=summarize_values(
            [m.bert_score_f1 for m in metrics if m.bert_score_f1 is not None]
        ),
    )
