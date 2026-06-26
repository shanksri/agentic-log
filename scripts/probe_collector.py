"""
Probe script: measure patched GitHubCollector.collect_issues() behaviour
for apache/spark at limit=500 WITHOUT touching the database.

Monkey-patches collect_issues to intercept per-page metrics before the
instrumentation logger (which requires DEBUG level to appear) and reports
a clean summary at the end.

Run:
    python scripts/probe_collector.py
"""
from __future__ import annotations

import os
import sys
import time
import types

# ── make sure project root is on the path ────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import httpx  # noqa: E402 (after sys.path fixup)

# Load token from .env manually so we don't need dotenv installed
token: str | None = None
env_path = os.path.join(ROOT, ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if line.startswith("GITHUB_TOKEN="):
            token = line.split("=", 1)[1].strip()
            break

if not token:
    print("ERROR: GITHUB_TOKEN not found in .env")
    sys.exit(1)

# ── import the real (patched) collector ───────────────────────────────────────
from app.ingestion.collectors.github_collector import GitHubCollector  # noqa: E402

OWNER = "apache"
REPO  = "spark"
LIMIT = 500
STATE = "closed"

# ── per-page counters (collected without changing collector logic) ────────────
page_records: list[dict] = []

# Wrap the real httpx.Client.get so we can intercept raw batch sizes
# without altering any GitHubCollector logic.
_original_get = httpx.Client.get

def _patched_get(self, url, *, params=None, **kwargs):
    response = _original_get(self, url, params=params, **kwargs)
    # Only intercept issue-list calls (not comment calls)
    if isinstance(url, str) and "/issues" in url and params and "page" in params:
        _last_call["page"]     = params.get("page", "?")
        _last_call["per_page"] = params.get("per_page", "?")
        _last_call["status"]   = response.status_code
    return response

_last_call: dict = {}
httpx.Client.get = _patched_get  # type: ignore[method-assign]

# ── also wrap collect_issues to record per-page stats ────────────────────────
_original_collect = GitHubCollector.collect_issues

def _instrumented_collect(
    self,
    owner: str,
    repo: str,
    *,
    state: str = "all",
    limit: int = 50,
    include_comments: bool = True,
):
    issues: list = []
    page = 1
    _MAX_PAGE = 100
    exit_reason = "unknown"

    while len(issues) < limit:
        if page >= _MAX_PAGE:   # GitHub rejects page=100; safe ceiling is 99
            exit_reason = "page_ceiling"
            break

        try:
            response = self._client.get(
                f"/repos/{owner}/{repo}/issues",
                params={
                    "state": state,
                    "per_page": 100,
                    "page": page,
                    "sort": "updated",
                    "direction": "desc",
                },
            )
            response.raise_for_status()
        except Exception as exc:
            exit_reason = f"http_error_page{page}:{exc}"
            break
        batch = response.json()

        if not batch:
            exit_reason = "empty_batch"
            break

        prs_in_batch   = sum(1 for item in batch if "pull_request" in item)
        issues_before  = len(issues)

        for item in batch:
            if "pull_request" in item:
                continue
            issue = dict(item)
            issue["repository"] = {
                "owner": owner,
                "name": repo,
                "full_name": f"{owner}/{repo}",
            }
            issue["comments_payload"] = []   # skip comments for speed
            issues.append(issue)
            if len(issues) >= limit:
                break

        issues_accepted = len(issues) - issues_before

        page_records.append({
            "page":            page,
            "per_page":        100,
            "raw":             len(batch),
            "prs_filtered":    prs_in_batch,
            "issues_accepted": issues_accepted,
            "cumulative":      len(issues),
        })

        if len(issues) >= limit:
            exit_reason = "limit_reached"

        page += 1

    return issues, exit_reason, page - 1   # return extra info alongside issues

with GitHubCollector(token=token) as collector:
    print(f"Probing {OWNER}/{REPO}  state={STATE}  limit={LIMIT}")
    print("─" * 60)
    t0 = time.perf_counter()
    result, exit_reason, last_page = _instrumented_collect(
        collector, OWNER, REPO, state=STATE, limit=LIMIT, include_comments=False
    )
    elapsed = time.perf_counter() - t0

issues_collected = len(result)
total_pages      = len(page_records)
total_raw        = sum(r["raw"]          for r in page_records)
total_prs        = sum(r["prs_filtered"] for r in page_records)
total_accepted   = sum(r["issues_accepted"] for r in page_records)

# ── per-page table ────────────────────────────────────────────────────────────
print(f"{'page':>4}  {'raw':>4}  {'prs':>4}  {'accepted':>8}  {'cumulative':>10}  {'yield%':>7}")
print(f"{'─'*4}  {'─'*4}  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*7}")
for r in page_records:
    yield_pct = (r["issues_accepted"] / r["raw"] * 100) if r["raw"] else 0.0
    print(
        f"{r['page']:>4}  {r['raw']:>4}  {r['prs_filtered']:>4}  "
        f"{r['issues_accepted']:>8}  {r['cumulative']:>10}  {yield_pct:>6.1f}%"
    )

# ── summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  pages traversed        : {total_pages}")
print(f"  last page number       : {last_page}")
print(f"  page 100 reached       : {'YES' if last_page >= 100 else 'no'}")
print(f"  raw items fetched      : {total_raw}")
print(f"  PRs filtered           : {total_prs}")
print(f"  issues collected       : {issues_collected}")
print(f"  effective yield        : {issues_collected/total_raw*100:.1f}%")
print(f"  avg issues/page        : {issues_collected/total_pages:.2f}")
print(f"  exit reason            : {exit_reason}")
print(f"  elapsed                : {elapsed:.1f}s")
print("=" * 60)
