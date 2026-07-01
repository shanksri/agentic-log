# 14 — Confidence Calibration

# Purpose

To attach an interpretable LOW/MEDIUM/HIGH confidence label to every retrieval, and to propagate
retrieval confidence into downstream hypothesis scoring.

# Problem Statement

A ranked list without a quality signal forces consumers (UI, investigation agent) to guess whether
the top result is trustworthy. We need a single, calibrated signal that says "how much should you
trust this retrieval," and a way to discount hypotheses built on weak retrieval.

# High-Level Architecture

```
top-1 similarity ─► classify_confidence
   < 0.40            → LOW
   0.40 ≤ s < 0.55   → MEDIUM
   ≥ 0.55            → HIGH

hypothesis raw_confidence × retrieval_weight × keyword_weight ─► composite (≤ raw)
   retrieval_weight: HIGH 1.0 / MEDIUM 0.85 / LOW 0.5
   keyword_weight:   success 1.0 / failure 0.6
```

# Detailed Flow

`classify_confidence(top1_score)` maps the best result's similarity to a band. `confidence_for`
returns `(top1, level)` for a result set (LOW when empty). On the agent side,
`composite_hypothesis_confidence` multiplies a hypothesis's self-reported confidence by the
retrieval-confidence weight and a validation-keyword weight, so a hypothesis can only become *less*
credible than the model claims, never more. Enters: a similarity score (and, downstream, a raw
hypothesis confidence). Leaves: a confidence label / a composite score in [0,1].

# Design Decisions

- **Why top-1 similarity.** The strongest single match is the clearest signal of whether the corpus
  contains anything genuinely relevant; it is cheap, interpretable, and independent of pipeline config.
- **Why two thresholds (0.40 / 0.55).** Derived from the original gold-set score distribution: no-match
  queries scored ≤0.344 and genuine matches ≥0.422, leaving a gap; 0.40 sits inside the gap (LOW
  boundary) and 0.55 splits the genuine-match range (MEDIUM/HIGH). The three-band split degrades more
  gracefully than a single cutoff.
- **Why composite confidence only discounts.** Both weights are ≤1.0, so evidence quality can lower a
  hypothesis's score but never inflate it beyond the model's own statement — a conservative, honest
  propagation.

# Tradeoffs

- **Advantage:** one interpretable signal; principled hypothesis discounting; trivially computed.
- **Disadvantage:** thresholds were calibrated on a ~400-incident corpus; at ~8,000 the moderate-
  similarity floor rose, so the MEDIUM band now also catches generic/noise queries (e.g. "bug" 0.454,
  "memory" 0.440) — MEDIUM has lost discriminative power (doc 16). HIGH (≥0.55) remains meaningful.
- **Alternatives considered:** top-k mean (washes out a strong single match), score-gap (top1−top2)
  signals (a future enhancement), learned calibration (premature for v1).

# Failure Scenarios

- **Generic single-token query** → moderate top-1 → MEDIUM despite no real information need
  (overconfidence). Documented; addressed by recalibration (doc 17).
- **Real query with a buried answer** ("triggerer not starting" → 0.366 LOW) → underconfidence caused
  by corpus drift in retrieval, surfaced honestly as LOW.
- **No results** → LOW by construction.

# Sequence Diagram

```
SearchService → Confidence: classify_confidence(top1)
Confidence → SearchService: LOW | MEDIUM | HIGH
---
Agent → Confidence: composite(raw_conf, retrieval_level, keyword_ok)
Confidence → Agent: composite (≤ raw_conf)
```

# Component Diagram

```
confidence.py
 ├─ classify_confidence(top1) -> level
 ├─ confidence_for(results) -> (top1, level)
 └─ composite_hypothesis_confidence(...) -> [0,1]
```

# Database Interaction

None — operates on similarity scores already produced by retrieval.

# API Interaction

None directly. (It consumes scores; the agent that uses composite confidence calls the LLM elsewhere.)

# Performance Considerations

O(1). Pure arithmetic and comparisons.

# Operational Considerations

`confidence_level` is logged on both `search` and `retrieve`. Bands feed UI/agent behavior and gate
hypothesis escalation in the investigation agent.

# Future Improvements

Recalibrate thresholds for the grown corpus (likely raise the LOW boundary) and/or add a secondary
signal (top1−top2 gap, top-K source agreement) so MEDIUM regains meaning — measured via the framework's
per-bucket calibration metric (doc 15).

**Status:** the 0.40/0.55 thresholds and `classify_confidence` itself are unchanged by later work.
What shipped instead is a layer *in front of* this classifier: [doc 18C](18_adaptive_routing_and_hybrid_confidence.md#phase-18c--strategy-aware-confidence-normalization)
normalizes BM25's and Hybrid's native scores onto the same `[0, 1]` scale this module expects,
so all three retrieval strategies can share these same two thresholds — but the thresholds
themselves were not recalibrated, and no top1−top2 gap signal exists yet. Doc 18C's own docstring
frames statistical recalibration as still future work, deferred until enough labeled data exists.

# Interview Questions

- Why use top-1 similarity rather than the mean of the top-k?
- Why can composite hypothesis confidence only move a score downward?
- Why has the MEDIUM band lost meaning as the corpus grew, and which threshold would you revisit?
- How would you detect overconfidence quantitatively?
