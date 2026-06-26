# 17 — Future Roadmap

# Purpose

To list improvements that build on the current architecture without redesigning it, ordered by
expected impact, each tied to a measurable hypothesis.

# Problem Statement

The limitations in doc 16 are real but bounded. Each has a compatible improvement that the existing
component boundaries already accommodate. None requires re-architecting ingestion or retrieval.

# High-Level Architecture

```
Build the measurement platform first (Phase 16A–16H), THEN change retrieval:
  identity → metric engine → gold v2 → harness → fingerprint → regression → dashboard → CI
        └────────── every retrieval change below is gated by this ──────────┘
```

# Detailed Flow — ranked improvements

1. **Evaluation platform (Phase 16A–16H).** Prerequisite for everything else: durable, drift-resistant
   measurement so no future change can invalidate history (doc 15). Highest priority because it de-risks
   all the rest.
2. **Hybrid dense + lexical (BM25 / trigram) retrieval.** Directly fixes the largest dense failure class —
   rare jargon, exact error codes, the Airflow hijack. The trigram full-text index already exists; this is
   a candidate-generation enhancement merged at doc 12's boundary. Hypothesis: large NDCG gain on
   lexical/jargon/error-code categories.
3. **Confidence recalibration for the grown corpus.** Raise the LOW boundary and/or add a secondary signal
   (top1−top2 gap, top-K source agreement) so MEDIUM regains meaning. Pure threshold/measurement work,
   validated by the framework's per-bucket calibration metric (doc 14).
4. **Default-on query expansion.** Empirically the most reliable lever; promote from opt-in to default
   once the harness confirms the gain and bounds latency (doc 11).
5. **Hub-incident detection / downweighting.** Identify incidents that rank top-1 across many unrelated
   gold queries and downweight them to fix cross-contamination and drift artifacts (docs 10, 12).
6. **Reranker guardrails / cross-encoder.** Prevent the reranker from discarding much-higher-similarity
   candidates; evaluate a cross-encoder as a stronger reranker — both measured before adoption (doc 13).
7. **Embedding model upgrade path.** Larger/domain-tuned embeddings with a dual-write/shadow-index
   migration keyed on `model_name`; adopt only if the framework shows a win (doc 08).
8. **Source/schema generalizations** for non-issue sources (PagerDuty/Datadog/postmortems): adapter-owned
   gold rules and canonical templates, recurrence handling, decoupling search filters from GitHub legacy
   columns (docs 04, 16).

# Design Decisions

- **Measurement before algorithms.** Every retrieval change ships behind the evaluation platform so gains
  are proven and regressions are blocked — the explicit lesson of the v1 baseline invalidation.
- **Smallest compatible change.** Each item slots into an existing boundary (candidate generation,
  confidence, embedding, adapters) — no redesign.

# Tradeoffs

Sequencing the platform first delays visible retrieval gains slightly but prevents the recurrence of
unmeasurable, history-invalidating changes. Hybrid retrieval is the highest-value *algorithm* change but
must wait for the harness to quantify it.

# Failure Scenarios

Skipping the platform and jumping to hybrid retrieval would reproduce the v1 problem: improvements that
can't be trusted and regressions that can't be caught.

# Sequence Diagram / Component Diagram

See doc 15 (platform) and docs 12/13 (where hybrid/reranker changes attach).

# Database Interaction

Hybrid retrieval leverages the existing trigram/GIN indexes; embedding upgrades touch `embeddings`
(re-embed + index rebuild); schema generalizations are additive (JSONB `source_metadata`).

# API Interaction

No new external dependencies are required for items 1–5; cross-encoder/embedding upgrades may add model
inference; new sources add their APIs via new collectors.

# Performance Considerations

Hybrid adds a lexical query alongside the ANN probe (cheap, in-DB). Default-on expansion adds LLM latency
to every read — acceptable given its gain, but worth caching. Embedding upgrades are a one-time re-embed cost.

# Operational Considerations

Each change is a small, independently mergeable PR gated by CI regression (doc 15). The Phase 16 risk
ranking and "never combine" rules govern PR boundaries.

# Future Improvements

(This document is the future-improvements registry; keep it ordered by measured expected impact and prune
items as they ship.)

# Interview Questions

- Why build the evaluation platform before improving retrieval?
- Why is hybrid retrieval the highest-value algorithm change, and which limitation does it target?
- How would you migrate the embedding model without invalidating historical evaluation results?
- What makes each roadmap item "compatible with the current architecture" rather than a redesign?
