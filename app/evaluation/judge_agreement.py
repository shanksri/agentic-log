"""Judge Validation — Agreement, Consistency, Prompt Sensitivity, and Bias
Analysis (Phase 21B).

A PURE analysis layer over already-collected judge scores. This module
never calls a ``Judge`` itself, never makes an LLM call, and never imports
``RuleJudge``/``LLMJudge`` — every analysis function takes plain,
already-computed ``float`` scores (or ``ScoredRecord``s wrapping them) as
input. Per the brief's "treat judges exactly like ML models... only
validation," this is statistics over outputs, never new reasoning.

# Validation workflow (this module's slice)

```
ScoredRecord (human / rule / llm scores, per stage, per investigation)
            │
            ▼
   compute_agreement(records, pair)        -> tuple[AgreementResult, ...]
   analyze_bias(records, pair)             -> tuple[BiasFinding, ...]

repeated judge-call scores for ONE artifact/stage
            │
            ▼
   analyze_consistency(scores)             -> ConsistencyResult

per-prompt-variant scores for ONE artifact/stage
            │
            ▼
   analyze_prompt_sensitivity(results)     -> PromptSensitivityReport
```

# Agreement methodology

For each ``(pair, stage)`` combination present in the supplied
``ScoredRecord``s, ``compute_agreement`` reports BOTH the raw
absolute-difference distribution (never just one aggregate number, per
the brief's "do not stop at one aggregate number") and the fraction of
pairs agreeing within a caller-supplied ``tolerance`` (default ``1.0`` on
the judge framework's own 1-10 scale - "within one point" is the simplest,
least-arbitrary default tolerance available, the same status Phase
16E/20A's ``EPSILON`` constants document for their own comparison
thresholds). A record missing EITHER side of a pair for a given stage
contributes nothing to that stage's agreement (not a synthetic 0
difference) - agreement can only be measured where both scores actually
exist.

# Consistency methodology

``analyze_consistency(scores)`` is a pure statistics function over an
already-collected sequence of repeated scores for ONE artifact/stage -
mean, population variance, standard deviation, minimum, maximum. This
module does NOT call a judge N times itself; a caller collects the N
scores (e.g. by invoking ``judge.evaluate_plan(...)`` N times against a
real or mocked, possibly non-deterministic ``JudgeLLMClient``) and passes
the resulting sequence in - "the framework should support N repeated
runs," not "the framework runs them," keeping this module pure and
trivially testable without any LLM dependency.

# Prompt sensitivity methodology

``analyze_prompt_sensitivity`` groups already-collected
``PromptVariantResult``s by stage and computes, per stage, ``drift =
max(scores) - min(scores)`` across that stage's prompt variants - the
simplest, non-arbitrary measure of "how much did changing only the prompt
move the score." ``mean_drift``/``max_drift`` aggregate across stages.
This module never constructs or evaluates a prompt variant itself; a
caller supplies the already-computed score for each ``(variant, stage)``
pair.

# Bias methodology

``analyze_bias`` computes, per ``(pair, stage)``, the MEAN SIGNED
difference (``first_source - second_source``, never absolute) across
every record where both sides have a score for that stage. A mean signed
difference whose absolute value is ``>= BIAS_THRESHOLD`` (``0.5``, half a
rubric-scale point - the smallest difference distinguishable from rounding
noise on a 1-10 scale used throughout this codebase, the same reasoning
Phase 19D's ``MARGIN_THRESHOLD`` documents for its own "smallest
meaningful difference on this scale" choice) is reported as a directional
``BiasFinding`` (e.g. "rule judge scores hypotheses 1.2 points higher than
llm judge on average" -> ``RULE_HIGHER``); below that threshold, no
finding is reported for that ``(pair, stage)`` - "evidence-backed," never
flagging noise as bias.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

BIAS_THRESHOLD = 0.5
DEFAULT_TOLERANCE = 1.0


class AgreementPair(str, Enum):
    HUMAN_VS_LLM = "human_vs_llm"
    HUMAN_VS_RULE = "human_vs_rule"
    RULE_VS_LLM = "rule_vs_llm"


class BiasDirection(str, Enum):
    FIRST_HIGHER = "first_higher"
    SECOND_HIGHER = "second_higher"


_PAIR_LABELS: dict[AgreementPair, tuple[str, str]] = {
    AgreementPair.HUMAN_VS_LLM: ("human", "llm_judge"),
    AgreementPair.HUMAN_VS_RULE: ("human", "rule_judge"),
    AgreementPair.RULE_VS_LLM: ("rule_judge", "llm_judge"),
}


# ── Score data model ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoredRecord:
    """One investigation artifact's score from up to three sources, for
    ONE stage. A caller builds one of these per ``(record_id, stage)``
    pair it wants to validate. Missing sources are ``None``, never a
    fabricated 0.
    """

    record_id: str
    stage: str
    human_score: float | None
    rule_score: float | None
    llm_score: float | None


def _score_for(record: ScoredRecord, source: str) -> float | None:
    return {
        "human": record.human_score, "rule_judge": record.rule_score,
        "llm_judge": record.llm_score,
    }[source]


# ── Agreement ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgreementResult:
    pair: AgreementPair
    stage: str
    n: int
    differences: tuple[float, ...]
    mean_absolute_difference: float | None
    agreement_within_tolerance: float | None
    tolerance: float


def compute_agreement(
    records: Sequence[ScoredRecord], pair: AgreementPair, *, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[AgreementResult, ...]:
    """Per-stage agreement for ``pair`` - see module docstring's
    "Agreement methodology". One ``AgreementResult`` per stage present in
    ``records``, in first-seen order.
    """
    first_label, second_label = _PAIR_LABELS[pair]
    by_stage: dict[str, list[float]] = {}
    order: list[str] = []
    for record in records:
        first = _score_for(record, first_label)
        second = _score_for(record, second_label)
        if first is None or second is None:
            continue
        if record.stage not in by_stage:
            by_stage[record.stage] = []
            order.append(record.stage)
        by_stage[record.stage].append(abs(first - second))

    results = []
    for stage in order:
        diffs = by_stage[stage]
        within = sum(1 for d in diffs if d <= tolerance) / len(diffs)
        results.append(
            AgreementResult(
                pair=pair, stage=stage, n=len(diffs), differences=tuple(diffs),
                mean_absolute_difference=sum(diffs) / len(diffs),
                agreement_within_tolerance=within, tolerance=tolerance,
            )
        )
    return tuple(results)


# ── Consistency ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConsistencyResult:
    stage: str
    n: int
    scores: tuple[float, ...]
    mean: float
    variance: float
    std_dev: float
    minimum: float
    maximum: float


def collect_repeated_scores(evaluate_fn, *, n: int) -> tuple[float, ...]:
    """Call ``evaluate_fn()`` (a zero-argument callable a caller builds,
    typically a closure over ``judge.evaluate_plan(problem, plan).score.
    value``) ``n`` times and return the raw scores - the one place this
    module touches a Judge at all, and only via a caller-supplied closure
    it has no knowledge of the internals of.
    """
    return tuple(evaluate_fn() for _ in range(n))


def analyze_consistency(stage: str, scores: Sequence[float]) -> ConsistencyResult:
    """Pure statistics over already-collected repeated scores - see
    module docstring's "Consistency methodology".
    """
    if not scores:
        return ConsistencyResult(
            stage=stage, n=0, scores=(), mean=0.0, variance=0.0, std_dev=0.0, minimum=0.0,
            maximum=0.0,
        )
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    return ConsistencyResult(
        stage=stage, n=n, scores=tuple(scores), mean=mean, variance=variance,
        std_dev=math.sqrt(variance), minimum=min(scores), maximum=max(scores),
    )


# ── Prompt sensitivity ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PromptVariantResult:
    variant: str
    stage: str
    score: float


@dataclass(frozen=True)
class StageDrift:
    stage: str
    drift: float
    scores_by_variant: tuple[PromptVariantResult, ...]


@dataclass(frozen=True)
class PromptSensitivityReport:
    stage_drifts: tuple[StageDrift, ...]
    mean_drift: float | None
    max_drift: float | None


def analyze_prompt_sensitivity(
    results: Sequence[PromptVariantResult],
) -> PromptSensitivityReport:
    """See module docstring's "Prompt sensitivity methodology"."""
    by_stage: dict[str, list[PromptVariantResult]] = {}
    order: list[str] = []
    for result in results:
        if result.stage not in by_stage:
            by_stage[result.stage] = []
            order.append(result.stage)
        by_stage[result.stage].append(result)

    stage_drifts = []
    for stage in order:
        variants = by_stage[stage]
        scores = [v.score for v in variants]
        stage_drifts.append(
            StageDrift(
                stage=stage, drift=max(scores) - min(scores), scores_by_variant=tuple(variants)
            )
        )

    drifts = [sd.drift for sd in stage_drifts]
    return PromptSensitivityReport(
        stage_drifts=tuple(stage_drifts),
        mean_drift=sum(drifts) / len(drifts) if drifts else None,
        max_drift=max(drifts) if drifts else None,
    )


# ── Bias ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BiasFinding:
    pair: AgreementPair
    stage: str
    mean_signed_difference: float
    direction: BiasDirection
    n: int
    description: str


def analyze_bias(records: Sequence[ScoredRecord], pair: AgreementPair) -> tuple[BiasFinding, ...]:
    """See module docstring's "Bias methodology". Only reports findings
    whose ``|mean_signed_difference| >= BIAS_THRESHOLD``.
    """
    first_label, second_label = _PAIR_LABELS[pair]
    by_stage: dict[str, list[float]] = {}
    order: list[str] = []
    for record in records:
        first = _score_for(record, first_label)
        second = _score_for(record, second_label)
        if first is None or second is None:
            continue
        if record.stage not in by_stage:
            by_stage[record.stage] = []
            order.append(record.stage)
        by_stage[record.stage].append(first - second)

    findings = []
    for stage in order:
        diffs = by_stage[stage]
        mean_diff = sum(diffs) / len(diffs)
        if abs(mean_diff) < BIAS_THRESHOLD:
            continue
        direction = BiasDirection.FIRST_HIGHER if mean_diff > 0 else BiasDirection.SECOND_HIGHER
        higher_label = first_label if mean_diff > 0 else second_label
        lower_label = second_label if mean_diff > 0 else first_label
        findings.append(
            BiasFinding(
                pair=pair, stage=stage, mean_signed_difference=mean_diff, direction=direction,
                n=len(diffs),
                description=(
                    f"for stage {stage!r}, {higher_label} scores {abs(mean_diff):.2f} points "
                    f"higher than {lower_label} on average across {len(diffs)} record(s)"
                ),
            )
        )
    return tuple(findings)
