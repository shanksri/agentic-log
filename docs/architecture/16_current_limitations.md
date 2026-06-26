# 16 — Current Limitations

# Purpose

To record, honestly and specifically, what Retrieval v1 does *not* do well — so future engineers
know the real edges of the system and don't mistake known limitations for bugs.

# Problem Statement

A system is only safe to build on if its weaknesses are documented. These limitations are observed
empirically (behavior study over the 8k corpus) or follow structurally from the design.

# High-Level Architecture

(Limitations span components; see each doc referenced below.)

# Detailed Flow — the limitations

1. **Dense-only retrieval misses lexical/jargon/exact-token queries.** Rare terms ("ISR", "triggerer")
   and exact error codes under-embed; dense buries the correct incident (e.g. "ISR shrink event" → LOW
   0.371). Lexical/hybrid would fix these (docs 09, 17).
2. **Reranking is near-neutral and occasionally harmful.** It can discard higher-similarity candidates
   by over-anchoring on topical affinity (ISR reverted 0.50→0.37). Expansion is the real lever (doc 13).
3. **Confidence thresholds are stale for the grown corpus.** Calibrated on ~400 incidents; at ~8,000 the
   MEDIUM band catches generic noise ("bug" 0.454, "memory" 0.440) — MEDIUM lost discriminative power.
   HIGH still meaningful (doc 14).
4. **Corpus drift & "hub" incidents.** Generic-text incidents become high-recall attractors as the corpus
   densifies; one Kafka incident ("Infinite loop trying to start a broker") is top-1 for several unrelated
   queries and hijacks "triggerer not starting" (Airflow) (docs 10, 12).
5. **Cross-source competition is sometimes wrong.** Jira (KAFKA/SPARK/CASSANDRA) wins ~46% of top-1 on
   19% of corpus — mostly earned, but it displaces the correct Airflow/other answer on some queries.
6. **Gold/eval v1 is single-UUID and 100% GitHub-centric.** It cannot show whether Jira *helped* (no
   Jira-targeted queries) and rotted under growth — the reason for the Phase 16 framework (doc 15).
7. **Canonical shape is issue-tracker-flavored.** Monitoring/non-issue sources (PagerDuty, Datadog,
   runbooks) won't fit cleanly (title/body/resolution assumptions) (docs 04, 17).
8. **Jira description assumes wiki string (Server/DC).** Jira Cloud ADF (dict) bodies are not flattened
   (doc 04).
9. **N+1 comment fetching** in the GitHub collector; **`MAX_SCANNED_ITEMS` caps healthy dense repos** at
   ~20 pages, limiting deep backfills (doc 03).
10. **`timeout_partial` runs still advance the watermark**, so un-scanned pages aren't auto-retried
    without `force_backfill` (doc 06).
11. **`source` column is redundant with `source_type`** (kept consistent post-fix) and the GitHub legacy
    columns (`owner`/`repo`/`state`) make non-repo sources second-class in some search filters (doc 09).

# Design Decisions (why these are acceptable for v1)

Each limitation is either a deliberate scope cut (lexical/hybrid, larger models, eval platform) or a
known edge with a safe fallback (timeouts, ADF). None causes data loss or incorrect persistence; they
bound *retrieval quality* and *measurement*, which the roadmap addresses without redesign.

# Tradeoffs

Shipping a dense-only v1 with honest limitations bought speed and a clean architecture to build hybrid
retrieval and a durable evaluation platform on top of — at the cost of known recall gaps on lexical/jargon
queries and a stale confidence calibration.

# Failure Scenarios

See each numbered item; all are observed or structural, and none are silent data-corruption risks.

# Sequence Diagram / Component Diagram

N/A (cross-cutting; refer to the per-component docs).

# Database Interaction

Relevant schema debt: redundant `source`/`source_type`; GitHub-specific `owner`/`repo`/`state` columns;
`raw_documents.source_id` NOT NULL (forces an auto-managed row for ad-hoc ingestion).

# API Interaction

LLM non-determinism in expand/rerank limits exact reproducibility of those configs (dense is
deterministic).

# Performance Considerations

N+1 comment fetches and unbatched embedding are the main throughput limits on the write path; LLM stages
are the main latency cost on the read path.

# Operational Considerations

Most limitations are observable in existing logs (confidence distribution, exit_reason, candidate counts);
the evaluation framework (doc 15) is what turns them into tracked metrics.

# Future Improvements

Mapped one-to-one to doc 17's roadmap.

# Interview Questions

- Which queries does dense retrieval fail on, and why structurally?
- Why did the MEDIUM confidence band lose meaning, and is HIGH still trustworthy?
- What is a "hub incident" and how does best-distance merge partially mitigate it?
- Why can the current evaluation set not prove that Jira improved retrieval?
