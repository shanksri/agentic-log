# 04 — Normalization

# Purpose

Normalizers convert a source-specific raw payload into the single canonical
`NormalizedIncident`, including the `canonical_text` string that is later embedded.

# Problem Statement

GitHub issues and Jira issues have different shapes, vocabularies, and quality signals.
Everything downstream (dedup, embedding, retrieval, evaluation) must operate on one shape
with consistent semantics (severity, status, gold-labeling), or it would need per-source
branches everywhere.

# High-Level Architecture

```
raw GitHub issue ─► GitHubNormalizer ─┐
raw Jira issue   ─► JiraNormalizer   ─┴─► NormalizedIncident
   (source_type, source_external_id, title, description, severity, status,
    incident_type, environment, affected_components, tags, symptoms,
    resolution_summary, canonical_text, confidence_score, is_gold_labeled,
    created/updated_at_source, source_metadata)
```

# Detailed Flow

Enters: one raw issue dict. The normalizer: cleans text (control whitespace collapsed,
markup stripped), derives `severity` and `incident_type` (from labels/priority + content
heuristics), extracts `symptoms` (title + error-bearing lines), derives
`resolution_summary`, computes a heuristic `confidence_score`, sets `is_gold_labeled`,
and assembles `canonical_text`. Source-specific extras (owner/repo/state for GitHub;
project_key/priority/resolution/components for Jira) go into `source_metadata`. Leaves: a
frozen `NormalizedIncident`.

**`canonical_text` template** (both sources, parallel structure):
```
{title}

{label} | {incident_type} | severity {severity}

Symptoms: {s1}; {s2}

What happened: {description excerpt ≤280 chars}

Resolution: {resolution excerpt ≤150 chars}   (only if resolved)
```

# Design Decisions

- **Why `canonical_text` exists.** The embedding model needs one coherent, information-dense
  string per incident. Embedding raw bodies would include markup, stack-trace noise, and
  variable structure; the template foregrounds the highest-signal fields (title, type,
  severity, symptoms, resolution) in a fixed order so embeddings are comparable across sources.
- **Severity from Jira `priority` first, then labels.** Apache Jira encodes severity in the
  priority field, not labels; reading priority is what makes Jira severity meaningful (vs.
  GitHub's near-100% `unknown`).
- **Resolution from structured Jira field first, comment scan as fallback.** The structured
  `resolution` field is a reliable, deliberate signal; comment keyword-scanning is the
  GitHub-style fallback.
- **`is_gold_labeled` is source-defined.** GitHub: resolution text + resolved. Jira: a
  *positive* structured resolution (Fixed/Done/…) excludes Won't-Fix/Duplicate.
- **Jira normalizer emits `NormalizedIncident` directly**; GitHub's predates the adapter layer
  and emits a GitHub dataclass mapped by the adapter — both converge on the canonical shape.

# Tradeoffs

- **Advantage:** one shape, consistent cross-source semantics, embedding-optimized text.
- **Disadvantage:** truncation (280/150 chars, 2 symptoms) discards detail; the template is
  issue-shaped and would underfit monitoring events.
- **Alternative considered:** embedding raw title+body — rejected; noisier embeddings, worse
  retrieval, no severity/resolution signal.

# Failure Scenarios

- **Empty/markup-only body** → excerpt falls back gracefully; symptoms default to the title.
- **Jira description as ADF (Cloud) rather than wiki string** → current normalizer assumes a
  string (Apache Server/DC); a Cloud instance would need an ADF flattener (doc 16).
- **Non-fix Jira resolution** → recorded in `source_metadata` but not gold-labeled.

# Sequence Diagram

```
Adapter → Normalizer: normalize(raw)
Normalizer: clean text, strip markup
Normalizer: derive severity (priority→labels), incident_type
Normalizer: extract symptoms, resolution_summary
Normalizer: build canonical_text
Normalizer → Adapter: NormalizedIncident
```

# Component Diagram

```
GitHubNormalizer ─┐  helpers: clean_text, strip_markup, severity map,
JiraNormalizer   ─┘  type map, symptom extractor, canonical_text builder
                  → NormalizedIncident (frozen dataclass)
```

# Database Interaction

None directly. Normalizers are pure functions; the produced fields are persisted later by
`IngestionService`.

# API Interaction

None. (LLM is not used in normalization.)

# Performance Considerations

O(payload size) string processing per incident; regex-bounded. Cheap relative to embedding.

# Operational Considerations

Pure and deterministic → trivially unit-tested and re-runnable (the `backfill_canonical_text`
pattern re-normalizes from stored raw payloads and re-embeds when `text_hash` changes).

# Future Improvements

A shared `BaseNormalizer` to remove GitHub/Jira duplication; ADF flattening for Jira Cloud;
per-source canonical templates for non-issue sources.

# Interview Questions

- Why does `canonical_text` exist instead of embedding the raw body?
- Why read Jira `priority` for severity but GitHub `labels`?
- Why is `is_gold_labeled` defined per source rather than centrally?
- What breaks if two normalizers produce different `canonical_text` structure for similar incidents?
