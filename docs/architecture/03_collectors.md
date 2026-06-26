# 03 — Collectors

# Purpose

Collectors are the only components that talk to external source APIs. They paginate,
filter, enrich, and return raw payloads plus diagnostics — without normalizing or
persisting anything.

# Problem Statement

Each source API has its own pagination model, rate limits, failure modes, and noise
(GitHub's `/issues` endpoint returns pull requests mixed with issues; high-PR repos can
exhaust pagination yielding almost nothing). Collection must be resilient to timeouts,
low-yield repositories, and excessive pagination, and must never lose already-collected
data.

# High-Level Architecture

```
GitHubCollector                         JiraCollector
  GET /repos/{o}/{r}/issues               GET /rest/api/2/search?jql=...
  per_page=100, page loop                 startAt loop, maxResults=100
  filter pull_request items               JQL: project + status + updated>=since
  optional comment enrichment             fields incl. priority/resolution/components
  safeguards: page ceiling, MAX_PAGES,    ORDER BY updated ASC
   MAX_SCANNED_ITEMS, timeout handling
  → (issues[], CollectionDiagnostics)     → issues[]
```

# Detailed Flow

**GitHub (`collect_issues`).** Enters: owner/repo/state/limit/include_comments/since.
Loops pages of 100; per page it filters out `pull_request` entries, enriches real issues
with comments, and accumulates. It tracks `pages_traversed`, `raw_items_scanned`,
`prs_filtered`. It exits with one of: `limit_reached`, `empty_batch`, `page_ceiling`,
`low_yield_abort` (scan budget or low avg-yield), `timeout_partial`. Leaves: issue dicts
(repository block injected) + a `CollectionDiagnostics`.

**Jira (`collect_issues`).** Enters: project_key/limit/since/status_filter. Builds a JQL
query (`project = X AND status in (...) AND updated >= "ts" ORDER BY updated ASC`),
paginates with `startAt`/`maxResults`, requests fields including `priority`, `resolution`,
`components`, and injects the instance base URL so the normalizer can build browse URLs.
Leaves: raw Jira issue dicts.

# Design Decisions

- **Collectors are pure I/O, no normalization.** Keeps source-specific HTTP concerns out
  of the canonical model and makes collectors independently testable (via `httpx.MockTransport`).
- **GitHub page ceiling = 100.** GitHub returns HTTP 422 beyond page 100; the loop stops
  at the ceiling rather than erroring.
- **Low-yield safeguards (`MAX_PAGES=20`, `MAX_SCANNED_ITEMS=2000`, `LOW_YIELD_THRESHOLD=0.5`).**
  Protect against PR-heavy repos (e.g. apache/spark) that would otherwise scan 100 pages to
  collect a handful of issues. The collector aborts with partial results, never raises.
- **Timeout → partial, not failure.** `httpx.ReadTimeout`/`TimeoutException` on the
  pagination GET preserves everything collected and exits `timeout_partial`. Comment-fetch
  timeouts are swallowed (enrichment, not critical).
- **`since` → incremental.** GitHub uses the `since` query param; Jira maps it to a JQL
  `updated >=` clause (minute resolution).

# Tradeoffs

- **Advantage:** resilient, observable, bounded collection; corpus building survives bad repos.
- **Disadvantage:** `MAX_SCANNED_ITEMS` (2000 ≈ 20 pages) also caps healthy dense repos; deep
  backfills of large repos need a higher budget.
- **Alternatives considered:** GitHub Search API (`type:issue`) and GraphQL — Search caps at
  1000 results/10 pages; GraphQL gives pure issues but adds a second client. REST + filtering
  was retained for v1 simplicity; see doc 17.

# Failure Scenarios

- **PR-heavy repo (golang/go, apache/spark):** scan budget aborts at ~page 20 with partial
  issues and `low_yield_abort` rather than walking to page 100.
- **GitHub read timeout deep in pagination (observed on apache/kafka):** `timeout_partial`,
  partial issues preserved, no HTTP 500 surfaced to the caller.
- **Comment endpoint timeout:** that issue's comments come back partial; collection continues.

# Sequence Diagram

```
IngestionService → GitHubCollector: collect_issues(o, r, since)
loop pages
  GitHubCollector → GitHub: GET /issues?per_page=100&page=N[&since]
  alt timeout
     GitHubCollector: exit_reason=timeout_partial, break
  GitHubCollector: filter PRs, enrich comments, accumulate
  GitHubCollector: check scan budget / low-yield → maybe abort
GitHubCollector → IngestionService: issues[], diagnostics
```

# Component Diagram

```
GitHubCollector ── httpx.Client ── GitHub REST
   └─ CollectionDiagnostics(pages, scanned, collected, prs_filtered, yield, exit_reason)
JiraCollector   ── httpx.Client ── Jira REST/JQL
```

# Database Interaction

None. Collectors are stateless network components; they never read or write the DB.

# API Interaction

GitHub REST `/repos/{owner}/{repo}/issues` + comments; Jira REST `/rest/api/2/search`.
Auth via bearer token (GitHub) / optional token (Jira). A `client` may be injected for tests.

# Performance Considerations

O(pages) network round-trips; each page is ≤100 items. Comment enrichment adds one request
per commented issue (N+1 — see Future). Memory is O(collected issues) held in a list.

# Operational Considerations

One `collection_complete` INFO log per run (source, repo, pages, scanned, collected,
prs_filtered, yield, exit_reason). Per-page detail at DEBUG only. Timeouts are warnings, not
errors.

# Future Improvements

GraphQL/Search-API collectors for high-PR repos; batched/concurrent comment fetching; making
`MAX_SCANNED_ITEMS` configurable per source for deep backfills.

# Interview Questions

- Why does the collector stop at page 100, and what happens if it didn't?
- Why does a timeout return partial results instead of raising?
- Why are comment-fetch timeouts handled differently from pagination timeouts?
- How would you parallelize collectors safely given the watermark contract?
- Why is `low_yield_abort` preferable to walking the full pagination budget?
