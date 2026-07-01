# 15 — Evaluation Framework

> **Scope note:** this document covers *retrieval* evaluation only (Phases 16A–16H below). A
> parallel system for *reasoning*/agent-quality evaluation (Phase 20A), an LLM-as-judge layer
> (Phase 20B), and a full productionization layer on top of both — AI quality intelligence, judge
> validation, gold-dataset authoring/labeling tooling, an end-to-end pipeline, persistent
> experiment tracking, and a REST API — have since shipped and are **not** described here. See
> [doc 20](20_reasoning_evaluation_and_judges.md), [doc 21](21_evaluation_platform_productionization.md),
> and [doc 22](22_evaluation_api.md).

# Purpose

To measure retrieval quality in a way that stays trustworthy as the corpus grows, so every future
change (hybrid retrieval, reranker tweaks, embedding upgrades, confidence recalibration) can be
judged objectively and historical results are never invalidated.

# Problem Statement

Retrieval metrics are only meaningful relative to the corpus they were computed over. The v1 gold
set pinned each query to a single incident UUID and was measured on a ~400-incident corpus; after
growth to ~8,000 the baseline became invalid (rankings shifted because the neighborhood densified,
not because the system regressed). We need an evaluation architecture that survives corpus growth.

# High-Level Architecture

```
gold (versioned, graded, identity-anchored) ─┐
corpus fingerprint (size/composition/model) ─┼─► Harness (3 configs) ─► run.json
                                              │        │
                          Metric Engine (NDCG/Recall/MRR/...) ─► scored run
                                              │
                        Regression Runner (same gold+fingerprint only) ─► verdict
                                              │
                              HTML Dashboard / CI gate
```

# Detailed Flow

The gold set describes *what makes an answer correct* (a graded set of acceptable incidents keyed on
`(source_type, source_external_id)`), not which row currently exists. The harness runs dense / +expand
/ +rerank, resolves identities to current rows, and scores with the metric engine. Each run is stamped
with a corpus fingerprint. The regression runner compares only runs with matching gold version and
fingerprint. Enters: a query set + gold labels. Leaves: per-config, per-category metrics + a pass/fail
verdict + a report.

# Design Decisions

- **Identity-anchored gold (not UUIDs).** UUIDs regenerate on re-ingest; `(source_type,
  source_external_id)` survives, so the gold set doesn't silently rot (doc 05).
- **Graded, multi-answer relevance.** Corpus growth adds legitimately-relevant incidents; a single
  gold answer would mark these as misses. Grades feed NDCG.
- **Corpus fingerprinting + comparison guard.** Every result records corpus size/composition/model;
  the regression runner refuses cross-corpus or cross-gold comparisons — the structural fix for what
  broke the v1 baseline.
- **Same-corpus A/B for algorithm changes; versioned trends for corpus changes.** Never diff a new run
  against a stale smaller-corpus baseline.
- **NDCG@10 primary; Recall@5/10 + MRR co-primary; others diagnostic.** NDCG sees graded relevance and
  ordering; MRR is the sensitivity when recall saturates; precision/hit-rate/avg-similarity/latency and
  expansion/reranker gain are diagnostics; per-bucket confidence calibration is a first-class check.

# Tradeoffs

- **Advantage:** durable, source-aware, drift-resistant measurement; attributable stage gains; safe
  foundation for future retrieval work.
- **Disadvantage:** requires disciplined gold curation and re-judging on corpus change; LLM configs are
  non-deterministic (gated with tolerance; dense is the hard gate).
- **Alternatives considered:** single-UUID gold (v1 — rotted), accuracy-style single-answer metrics
  (collapse under valid competitors). Rejected for the reasons growth exposed.

# Failure Scenarios

- **Corpus grew since baseline** → fingerprint mismatch → comparison refused, not silently wrong.
- **New valid incident displaces a gold answer** → reviewer admits it to gold by relevance criteria
  (not because the system ranked it), preventing phantom regressions (doc 16/roadmap).
- **Metric bug** → isolated, exhaustively unit-tested pure functions prevent a metric defect from
  masquerading as a retrieval regression.

# Sequence Diagram

```
CI → Harness: run(gold vN, corpus) for [dense, expand, rerank]
Harness → IdentityResolver: resolve gold identities → current rows
Harness → SearchService: retrieve per query per config
Harness → MetricEngine: score(retrieved, graded_gold)
Harness → Fingerprint: stamp corpus
Harness → RegressionRunner: compare(run, baseline)  # same gold+fingerprint
RegressionRunner → CI: verdict + per-query diffs
```

# Component Diagram

```
gold/vN  Harness  MetricEngine  Fingerprint  RegressionRunner  Dashboard  CI
   └─ identity-anchored, graded, categorized ──────────────────────────────┘
```

# Database Interaction

- **Reads:** `incidents`/`embeddings` (via retrieval) and identity resolution; corpus aggregates for the
  fingerprint.
- **Writes:** none to product tables; evaluation artifacts are files (run.json, reports, baselines).

# API Interaction

OpenAI for the expand/rerank configs during evaluation; PostgreSQL for retrieval and fingerprinting.

# Performance Considerations

Cost = (queries × configs) retrievals; LLM configs dominate. Dense config is deterministic and cheap;
run it as the gating signal, LLM configs with tolerance.

# Operational Considerations

Datasets and baselines are versioned and immutable; runs are stamped; the regression runner is the
enforcement point and must refuse incompatible comparisons. (Implementation is roadmapped in phases
16A–16H; v1 ships `run_retrieval_eval.py` + a single-UUID gold set as the precursor.)

# Future Improvements

The full phased platform (identity resolver, gold v2, harness, metric engine, fingerprinting, regression
runner, dashboard, CI) — see doc 17 and the Phase 16 roadmap.

**Status:** the retrieval-side platform described above (Phases 16A–16H) has shipped as described.
On top of it, docs 20–22 add an entirely parallel reasoning-evaluation system, judges, judge
validation, dataset-authoring/labeling tools, an orchestration pipeline, persistent experiment
tracking, and a 15-endpoint REST API — none of which existed when this document was written and
none of which is described by the diagrams above. No CI gate or dashboard exists for either the
retrieval or reasoning side as of this writing; both remain manually invoked (CLI scripts, or the
REST API in doc 22).

# Interview Questions

- Why did the original UUID-anchored, single-answer gold set become invalid after corpus growth?
- Why must the regression runner refuse cross-corpus comparisons?
- Why is NDCG primary while Recall stays co-primary and MRR is the saturation sensor?
- When does a newly-retrieved incident earn a place in the gold set — and when must it not?
