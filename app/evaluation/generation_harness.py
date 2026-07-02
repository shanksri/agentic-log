"""Generation Evaluation Harness (Phase 22).

Orchestrates end-to-end generation evaluation for a Gold Dataset v2 against
a live search service and an answer generator, producing BOTH halves of
Phase 22's answer-quality signal:

- **Generation** (semantic similarity vs. the reference answer — BERTScore,
  ``app.evaluation.generation_metrics``)
- **Grounding** (RAGAS-style Faithfulness / Answer Relevancy / Context
  Precision / Context Recall / Context Entity Recall vs. the retrieved
  context — ``app.evaluation.grounding_metrics``)

```
GoldDataset (16B, optional reference_answer per query — Phase 22)
    │
    ▼
for each GoldQuery:
    │
    ├─ nothing computable (no reference AND no grounding backend)
    │        ──► GenerationQueryResult(skipped) — zero retrieval/LLM cost
    ▼
search_service.search(query, limit=k)     [dense or routed — duck-typed]
    │
    ▼
contexts = [render(incident) per result]  [per-chunk, rank-ordered — the
    │                                      unit RAGAS metrics operate on]
    ▼
answer_generator.generate_answer(query, joined contexts)
    │
    ├── reference present ──► compute_generation_metrics()   [BERTScore]
    │                          (None fields when no token_embedder)
    └── grounding_llm present ──► per-metric grounding calls
                                   (each isolated; a malformed LLM response
                                    downgrades that ONE metric to None with
                                    a recorded note, never fails the query)
    │
    ▼
GenerationQueryResult ──► aggregates ──► GenerationEvaluationReport
```

# Evaluation modes (Phase 22B — cost control)

``GenerationEvaluationConfig.mode`` gates WHICH grounding metrics execute
(BERTScore runs in every mode):

- ``FAST``     — BERTScore + Faithfulness                  (2 LLM calls/query)
- ``STANDARD`` — FAST + Answer Relevancy                   (3 LLM calls/query)
- ``FULL``     — STANDARD + Context Precision + Context
                 Recall + Context Entity Recall            (7 LLM calls/query)

The default is ``FAST`` — conservative for production cost. A disabled
metric is SKIPPED (``None`` + a note naming the mode), never reported as
zero.

# Repeatability (Phase 22B — evaluator stability)

``GenerationEvaluationConfig.evaluation_repetitions = N`` (default 1): for
N > 1 each enabled grounding metric is executed N independent times; the
REPORTED value is the mean, and per-query ``GroundingStability`` records
count/mean/std-dev/min/max plus a qualitative ``EvaluatorConfidence`` band
(std < 0.05 HIGH, 0.05–0.10 MEDIUM, > 0.10 LOW). This measures EVALUATOR
stability, not answer quality — the metric formulas themselves are
unchanged. Stability needs >= 2 successful samples; with fewer it is
``None`` (never a fabricated std of 0). BERTScore is deterministic and is
never repeated. Deterministic skips (no contexts / no reference / disabled
by mode) are decided once, not repeated.

# Skip semantics (per the Phase 22/22B briefs — neither case fails evaluation)

- ``reference_answer`` absent → **BERTScore skipped** (``generation=None``),
  and the reference-dependent grounding metrics (context precision, context
  recall, context entity recall) are ``None``; faithfulness / answer
  relevancy still compute. **Context precision no longer falls back to
  judging against the generated answer** (Phase 22B): that was circular —
  contexts "useful for arriving at" a hallucinated answer scored well — so
  it is now skipped with a recorded reason instead.
- retrieved context absent (search returned nothing) → **all grounding
  metrics ``None``** (RAGAS skipped); BERTScore still computes if a
  reference exists.
- A query where NOTHING could be computed (no reference and no grounding
  backend configured) is skipped before any retrieval or LLM cost.
- A raising answer-generation call → recorded as failed, run continues
  (same per-item isolation Phase 21E applies to judge calls).

# What this module does NOT implement

- No metrics (delegated to generation_metrics / grounding_metrics).
- No retrieval (delegated to the injected ``search_service``'s ``.search()``
  — the raw candidate-generation primitive, matching how Phase 19A's
  ``HypothesisEvaluator`` sources evidence).
- No LLM plumbing: ``LLMServiceAnswerGenerator`` adapts the existing
  ``LLMService.generate_investigation`` (pre-16, unmodified);
  ``LLMService.generate_json`` / ``EmbeddingService.embed_text`` satisfy the
  grounding protocols directly.
- No regression runner (future phase — the benchmark layer stores runs so a
  comparison has history to work with).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol

from app.evaluation.generation_metrics import (
    GenerationAggregateMetrics,
    GenerationMetrics,
    TokenEmbedder,
    aggregate_generation_metrics,
    compute_generation_metrics,
)
from app.evaluation.gold_dataset import GoldDataset
from app.evaluation.grounding_metrics import (
    GroundingAggregateMetrics,
    GroundingLLMClient,
    GroundingMetrics,
    GroundingStability,
    SentenceEmbedder,
    aggregate_grounding_metrics,
    answer_relevancy,
    classify_evaluator_confidence,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
    measure_stability,
)

DEFAULT_GENERATION_K = 5

_GROUNDING_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_entity_recall",
)


# ── Evaluation configuration (Phase 22B) ──────────────────────────────────────


class GenerationEvaluationMode(str, Enum):
    """Which grounding metrics execute — see module docstring's
    "Evaluation modes". BERTScore runs in every mode."""

    FAST = "fast"
    STANDARD = "standard"
    FULL = "full"


_MODE_GROUNDING_METRICS: dict[GenerationEvaluationMode, frozenset[str]] = {
    GenerationEvaluationMode.FAST: frozenset({"faithfulness"}),
    GenerationEvaluationMode.STANDARD: frozenset({"faithfulness", "answer_relevancy"}),
    GenerationEvaluationMode.FULL: frozenset(_GROUNDING_METRIC_NAMES),
}


@dataclass(frozen=True)
class GenerationEvaluationConfig:
    """Cost/reliability knobs for one generation-evaluation run.

    ``mode`` defaults to ``FAST`` (conservative for production cost);
    ``evaluation_repetitions`` defaults to 1 (no repetition — stability is
    opt-in because it multiplies LLM cost by N).
    """

    mode: GenerationEvaluationMode = GenerationEvaluationMode.FAST
    evaluation_repetitions: int = 1

    def __post_init__(self) -> None:
        if self.evaluation_repetitions < 1:
            raise ValueError(
                f"evaluation_repetitions must be >= 1, got "
                f"{self.evaluation_repetitions}"
            )

    @property
    def enabled_grounding_metrics(self) -> frozenset[str]:
        return _MODE_GROUNDING_METRICS[self.mode]


# ── Answer generation protocol + adapter ─────────────────────────────────────


class AnswerGenerator(Protocol):
    """Anything that can produce a natural-language answer for a query given
    a plain-text retrieval-context block. Mirrors Phase 20B's
    ``JudgeLLMClient``: the harness depends only on this protocol, never on
    a concrete LLM service, so unit tests never require OpenAI.
    """

    def generate_answer(self, query: str, context: str) -> str: ...


class LLMServiceAnswerGenerator:
    """Adapts the existing ``LLMService.generate_investigation(problem=,
    context=)`` (pre-16, unmodified) to the ``AnswerGenerator`` protocol.
    Duck-typed: any object exposing that method works.
    """

    def __init__(self, llm_service) -> None:
        self._llm_service = llm_service

    def generate_answer(self, query: str, context: str) -> str:
        return self._llm_service.generate_investigation(problem=query, context=context)


def render_incident_context(result: object) -> str:
    """Render ONE retrieved ``IncidentSearchResult`` into a plain-text
    context chunk. Fields are read defensively (``getattr`` with defaults)
    so both real ORM incidents and test stand-ins render. One chunk per
    incident — the granularity RAGAS's context precision/recall operate on.
    """
    incident = result.incident
    title = getattr(incident, "title", "") or ""
    resolution = getattr(incident, "resolution_summary", "") or ""
    status = getattr(incident, "status", "") or ""
    return f"{title}\nstatus: {status}\nresolution: {resolution}"


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationQueryResult:
    """One query's generation-evaluation outcome.

    - ``generation`` is ``None`` when the query has no reference_answer
      (BERTScore skipped per the brief).
    - ``grounding`` is ``None`` when no grounding LLM was configured;
      individual fields inside are ``None`` per that metric's own skip
      rules (no contexts, no reference, or a malformed LLM response —
      the reason is recorded in ``notes``).
    - ``skipped=True`` means nothing was computed at all (``skip_reason``
      says why) and the query cost no retrieval or LLM calls unless the
      failure happened mid-flight.
    """

    query_id: str
    query: str
    generated_answer: str | None
    reference_answer: str | None
    num_contexts: int
    generation: GenerationMetrics | None
    grounding: GroundingMetrics | None
    skipped: bool
    skip_reason: str | None
    notes: tuple[str, ...]
    # Phase 22B — per-metric run-to-run stability when
    # evaluation_repetitions > 1; None otherwise (defaulted for backward
    # compatibility with pre-22B construction sites).
    grounding_stability: GroundingStability | None = None


@dataclass(frozen=True)
class GenerationEvaluationReport:
    """The complete result of one generation-evaluation run: both the
    semantic-similarity (generation) and RAGAS-style (grounding) halves,
    with independent aggregates.
    """

    dataset_version: str
    dataset_description: str
    k: int
    num_answered: int
    num_generation_scored: int
    num_grounding_scored: int
    num_skipped: int
    num_failed: int
    results: tuple[GenerationQueryResult, ...]
    generation_aggregate: GenerationAggregateMetrics
    grounding_aggregate: GroundingAggregateMetrics
    started_at: str
    finished_at: str
    duration_seconds: float
    # Phase 22B — all defaulted so pre-22B construction sites are
    # unaffected. metric_variance/metric_confidence are dataset-level
    # roll-ups (mean per-query std-dev per metric, and its confidence
    # band); None when evaluation_repetitions == 1.
    evaluation_mode: str = GenerationEvaluationMode.FAST.value
    repetitions: int = 1
    metric_variance: dict[str, float] | None = None
    metric_confidence: dict[str, str] | None = None


# ── Grounding scoring (per-metric isolation) ──────────────────────────────────


def _score_grounding(
    *,
    question: str,
    answer: str,
    contexts: Sequence[str],
    reference: str | None,
    llm: GroundingLLMClient,
    sentence_embedder: SentenceEmbedder | None,
    config: GenerationEvaluationConfig,
) -> tuple[GroundingMetrics, GroundingStability | None, tuple[str, ...]]:
    """Compute the mode-enabled grounding metrics independently, each
    executed ``config.evaluation_repetitions`` times (Phase 22B).

    - A metric disabled by the mode is skipped with a note — never zero.
    - Deterministic skips (no contexts, no reference, no embedder) are
      decided ONCE, before any repetition, and never burn LLM calls.
    - Per repetition, a failure (e.g. malformed LLM response) is isolated:
      the successful samples still count. The reported value is the mean of
      successful samples; ``MetricStability`` is recorded only when >= 2
      samples exist AND repetitions > 1.
    """
    notes: list[str] = []
    values: dict[str, float | None] = {name: None for name in _GROUNDING_METRIC_NAMES}
    stability: dict[str, object | None] = {name: None for name in _GROUNDING_METRIC_NAMES}
    enabled = config.enabled_grounding_metrics
    repetitions = config.evaluation_repetitions

    if not contexts:
        notes.append("grounding skipped: no retrieved context")
        return GroundingMetrics(**values), None, tuple(notes)

    def _run(name: str, compute: Callable[[], float | None]) -> None:
        """Execute one metric ``repetitions`` times with per-repetition
        isolation; fold samples into value (mean) + stability."""
        samples: list[float] = []
        first_error: str | None = None
        failures = 0
        for _ in range(repetitions):
            try:
                sample = compute()
            except Exception as exc:  # noqa: BLE001 — per-metric isolation
                failures += 1
                if first_error is None:
                    first_error = repr(exc)
                continue
            if sample is not None:
                samples.append(sample)

        if failures:
            notes.append(
                f"{name} failed on {failures}/{repetitions} repetition(s): "
                f"{first_error}"
            )
        if not samples:
            return
        values[name] = statistics.mean(samples)
        if repetitions > 1:
            if len(samples) >= 2:
                stability[name] = measure_stability(samples)
            else:
                notes.append(
                    f"{name} stability unavailable: only {len(samples)}/"
                    f"{repetitions} repetition(s) produced a sample"
                )

    def _disabled(name: str) -> bool:
        if name not in enabled:
            notes.append(
                f"{name} skipped: disabled in {config.mode.value!r} mode"
            )
            return True
        return False

    if not _disabled("faithfulness"):
        _run("faithfulness", lambda: faithfulness(answer, contexts, llm=llm))

    if not _disabled("answer_relevancy"):
        if sentence_embedder is None:
            notes.append("answer_relevancy skipped: no sentence embedder configured")
        else:
            _run(
                "answer_relevancy",
                lambda: answer_relevancy(
                    question, answer, llm=llm, embedder=sentence_embedder
                ),
            )

    # Phase 22B: context precision REQUIRES a reference. The previous
    # fallback (judging usefulness toward the generated answer) was
    # circular — contexts "useful for arriving at" a hallucinated answer
    # scored well — so it is skipped, never fabricated.
    if not _disabled("context_precision"):
        if reference is None:
            notes.append(
                "context_precision skipped: no reference_answer (circular "
                "evaluation against the generated answer is disabled)"
            )
        else:
            _run(
                "context_precision",
                lambda: context_precision(question, contexts, reference, llm=llm),
            )

    if not _disabled("context_recall"):
        if reference is None:
            notes.append("context_recall skipped: no reference_answer")
        else:
            _run(
                "context_recall",
                lambda: context_recall(reference, contexts, llm=llm),
            )

    if not _disabled("context_entity_recall"):
        if reference is None:
            notes.append("context_entity_recall skipped: no reference_answer")
        else:
            _run(
                "context_entity_recall",
                lambda: context_entity_recall(reference, contexts, llm=llm),
            )

    grounding_stability = (
        GroundingStability(**stability)  # type: ignore[arg-type]
        if repetitions > 1 and any(s is not None for s in stability.values())
        else None
    )
    return GroundingMetrics(**values), grounding_stability, tuple(notes)


# ── Harness ───────────────────────────────────────────────────────────────────


def _rollup_stability(
    results: Sequence[GenerationQueryResult],
) -> tuple[dict[str, float] | None, dict[str, str] | None]:
    """Dataset-level variance roll-up (Phase 22B): for each metric, the mean
    per-query std-dev across queries that recorded stability, plus its
    confidence band. ``(None, None)`` when no query recorded any stability.
    """
    per_metric_std_devs: dict[str, list[float]] = {}
    for result in results:
        if result.grounding_stability is None:
            continue
        for name in _GROUNDING_METRIC_NAMES:
            metric_stability = getattr(result.grounding_stability, name)
            if metric_stability is not None:
                per_metric_std_devs.setdefault(name, []).append(
                    metric_stability.std_dev
                )
    if not per_metric_std_devs:
        return None, None
    variance = {
        name: statistics.mean(std_devs)
        for name, std_devs in per_metric_std_devs.items()
    }
    confidence = {
        name: classify_evaluator_confidence(mean_std).value
        for name, mean_std in variance.items()
    }
    return variance, confidence


def evaluate_generation(
    dataset: GoldDataset,
    search_service,  # duck-typed: needs .search(query, *, limit, call_site)
    answer_generator: AnswerGenerator,
    *,
    k: int = DEFAULT_GENERATION_K,
    token_embedder: TokenEmbedder | None = None,
    grounding_llm: GroundingLLMClient | None = None,
    sentence_embedder: SentenceEmbedder | None = None,
    config: GenerationEvaluationConfig | None = None,
) -> GenerationEvaluationReport:
    """Run generation + grounding evaluation over ``dataset``; see module
    docstring for the full lifecycle, evaluation modes, repetition, and
    skip semantics. ``config`` defaults to FAST mode with 1 repetition —
    conservative for production cost.
    """
    config = config or GenerationEvaluationConfig()
    started_perf = time.monotonic()
    started_at = datetime.now(UTC).isoformat()

    results: list[GenerationQueryResult] = []
    generation_scored: list[GenerationMetrics] = []
    grounding_scored: list[GroundingMetrics] = []
    num_answered = 0
    num_skipped = 0
    num_failed = 0

    for gold_query in dataset.queries:
        # Nothing computable for this query? Skip before ANY cost.
        if gold_query.reference_answer is None and grounding_llm is None:
            num_skipped += 1
            results.append(
                GenerationQueryResult(
                    query_id=gold_query.id,
                    query=gold_query.query,
                    generated_answer=None,
                    reference_answer=None,
                    num_contexts=0,
                    generation=None,
                    grounding=None,
                    skipped=True,
                    skip_reason=(
                        "no reference_answer and no grounding backend — "
                        "nothing to evaluate"
                    ),
                    notes=(),
                )
            )
            continue

        try:
            retrieved = search_service.search(
                gold_query.query, limit=k, call_site="generation_evaluation"
            )
            contexts = [render_incident_context(result) for result in retrieved]
            joined = (
                "\n\n".join(contexts) if contexts else "(no similar incidents retrieved)"
            )
            answer = answer_generator.generate_answer(gold_query.query, joined)
        except Exception as exc:  # noqa: BLE001 — per-query isolation, see docstring
            num_failed += 1
            results.append(
                GenerationQueryResult(
                    query_id=gold_query.id,
                    query=gold_query.query,
                    generated_answer=None,
                    reference_answer=gold_query.reference_answer,
                    num_contexts=0,
                    generation=None,
                    grounding=None,
                    skipped=True,
                    skip_reason=f"generation failed: {exc!r}",
                    notes=(),
                )
            )
            continue

        num_answered += 1
        notes: list[str] = []

        generation_metrics: GenerationMetrics | None = None
        if gold_query.reference_answer is not None:
            generation_metrics = compute_generation_metrics(
                answer, gold_query.reference_answer, token_embedder=token_embedder
            )
            generation_scored.append(generation_metrics)
            if token_embedder is None:
                notes.append("bert_score undefined: no token embedder configured")
        else:
            notes.append("bert_score skipped: no reference_answer")

        grounding_metrics: GroundingMetrics | None = None
        grounding_stability: GroundingStability | None = None
        if grounding_llm is not None:
            grounding_metrics, grounding_stability, grounding_notes = _score_grounding(
                question=gold_query.query,
                answer=answer,
                contexts=contexts,
                reference=gold_query.reference_answer,
                llm=grounding_llm,
                sentence_embedder=sentence_embedder,
                config=config,
            )
            grounding_scored.append(grounding_metrics)
            notes.extend(grounding_notes)
        else:
            notes.append("grounding skipped: no grounding LLM configured")

        results.append(
            GenerationQueryResult(
                query_id=gold_query.id,
                query=gold_query.query,
                generated_answer=answer,
                reference_answer=gold_query.reference_answer,
                num_contexts=len(contexts),
                generation=generation_metrics,
                grounding=grounding_metrics,
                skipped=False,
                skip_reason=None,
                notes=tuple(notes),
                grounding_stability=grounding_stability,
            )
        )

    finished_at = datetime.now(UTC).isoformat()
    metric_variance, metric_confidence = _rollup_stability(results)
    return GenerationEvaluationReport(
        dataset_version=dataset.version,
        dataset_description=dataset.description,
        k=k,
        num_answered=num_answered,
        num_generation_scored=len(generation_scored),
        num_grounding_scored=len(grounding_scored),
        num_skipped=num_skipped,
        num_failed=num_failed,
        results=tuple(results),
        generation_aggregate=aggregate_generation_metrics(generation_scored),
        grounding_aggregate=aggregate_grounding_metrics(grounding_scored),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(time.monotonic() - started_perf, 4),
        evaluation_mode=config.mode.value,
        repetitions=config.evaluation_repetitions,
        metric_variance=metric_variance,
        metric_confidence=metric_confidence,
    )
