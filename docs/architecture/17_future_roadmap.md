# 17 — Future Roadmap

> **Update note:** items 1 and 2 below have shipped; see the ✅/⚠️ status notes inline under
> "Detailed Flow" and docs [18](18_adaptive_routing_and_hybrid_confidence.md)–[22](22_evaluation_api.md)
> for what was actually built. Items 4–8 remain open as of this writing (query expansion is still
> opt-in, no hub-incident downweighting or reranker guardrail exists, no embedding upgrade or
> non-issue source generalization has shipped).

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

1. **Evaluation platform (Phase 16A–16H).** ✅ **Shipped**, and then some — see doc 15 for the
   retrieval-side platform this item originally scoped, and docs 20–22 for the reasoning
   evaluation, LLM-as-judge, judge validation, dataset authoring/labeling, end-to-end pipeline,
   experiment tracking, and REST API built on top of it (Phases 20A–21H), none of which this
   roadmap entry anticipated.
2. **Hybrid dense + lexical (BM25 / trigram) retrieval.** ✅ **Shipped as an independent BM25
   engine + RRF fusion**, not a trigram-index candidate-generation merge at doc 12's boundary as
   originally scoped — see doc 18 (Phases 17A/17B). Also shipped alongside it: an adaptive router
   (18A/18B) that picks dense/BM25/hybrid per query, and a confidence-normalization layer (18C) so
   all three strategies can share doc 14's thresholds. **None of it is wired into an API route** —
   `app/api/routes/search.py` still only calls dense `IncidentSearchService`.
3. **Confidence recalibration for the grown corpus.** ⚠️ **Not done.** Doc 18C's normalization
   layer lets non-dense strategies share the existing 0.40/0.55 thresholds, but nobody has raised
   the LOW boundary or added a top1−top2-gap/source-agreement signal — still open, per doc 14's
   and doc 18C's own Future Work sections.
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
