"""Phase 4: retrieval confidence calibration.

Classifies a retrieval result set's confidence based on its top1 similarity
score. Thresholds are derived from the gold-query score distribution (see
PHASE_4_CONFIDENCE_CALIBRATION.md): on the 24-query gold set, all
no-match-expected queries scored top1 <= 0.344 and all genuine matches scored
top1 >= 0.422, leaving a 0.078 gap. The thresholds below sit inside that gap
(LOW boundary) and split the genuine-match range roughly at its median
(MEDIUM/HIGH boundary), so the LOW/MEDIUM/HIGH split degrades gracefully
rather than relying on a single brittle cutoff.
"""

from __future__ import annotations

LOW_CONFIDENCE_THRESHOLD = 0.40
HIGH_CONFIDENCE_THRESHOLD = 0.55

CONFIDENCE_LOW = "LOW"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_HIGH = "HIGH"


def classify_confidence(top1_score: float | None) -> str:
    """Classify retrieval confidence from the top1 similarity score.

    - score < LOW_CONFIDENCE_THRESHOLD (or no results): LOW
    - LOW_CONFIDENCE_THRESHOLD <= score < HIGH_CONFIDENCE_THRESHOLD: MEDIUM
    - score >= HIGH_CONFIDENCE_THRESHOLD: HIGH
    """
    if top1_score is None or top1_score < LOW_CONFIDENCE_THRESHOLD:
        return CONFIDENCE_LOW
    if top1_score < HIGH_CONFIDENCE_THRESHOLD:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_HIGH


# Phase 6A: hypothesis confidence propagation.
#
# Multiplies a hypothesis's raw, self-reported `confidence_score` by two
# evidence-quality factors so that the resulting composite score reflects not
# just "how confident the model sounds" but "how much retrieval evidence
# actually backs this up". Both factors are <= 1.0, so composite scores can
# only move *down* from the raw score, never up - a hypothesis can't become
# more credible than the model's own statement, only less.
RETRIEVAL_CONFIDENCE_WEIGHTS: dict[str, float] = {
    CONFIDENCE_HIGH: 1.0,
    CONFIDENCE_MEDIUM: 0.85,
    CONFIDENCE_LOW: 0.5,
}

# Applied when a hypothesis's validation_keywords fail to retrieve any
# relevant incident in the top-K (see run_hypothesis_eval.py /
# _collect_evidence): the hypothesis is unsupported by retrievable evidence.
VALIDATION_KEYWORD_SUCCESS_WEIGHT = 1.0
VALIDATION_KEYWORD_FAILURE_WEIGHT = 0.6


def composite_hypothesis_confidence(
    *,
    raw_confidence: float,
    retrieval_confidence_level: str,
    validation_keyword_recall_ok: bool | None,
) -> float:
    """Combine a hypothesis's self-reported confidence with retrieval-side
    signals into a single composite score in [0, 1].

    - `retrieval_confidence_level`: the *initial* retrieval confidence
      (LOW/MEDIUM/HIGH) for the investigation as a whole. A LOW initial
      retrieval means every hypothesis was generated from weak/irrelevant
      context, so all hypotheses for that case are discounted.
    - `validation_keyword_recall_ok`: whether this specific hypothesis's
      validation_keywords retrieve a relevant incident (None if no keywords
      were generated, treated the same as failure).
    """
    retrieval_weight = RETRIEVAL_CONFIDENCE_WEIGHTS.get(retrieval_confidence_level, 0.5)
    keyword_weight = (
        VALIDATION_KEYWORD_SUCCESS_WEIGHT
        if validation_keyword_recall_ok
        else VALIDATION_KEYWORD_FAILURE_WEIGHT
    )
    composite = raw_confidence * retrieval_weight * keyword_weight
    return max(0.0, min(1.0, composite))
