from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Phase 14B/14C hardening safeguards ───────────────────────────────────────
# GitHub hard-caps /issues pagination at page 100 (HTTP 422 beyond that).
_PAGE_CEILING = 100
# Abort a low-yield scan rather than walking the full pagination budget.
MAX_PAGES = 20
MAX_SCANNED_ITEMS = 2000
LOW_YIELD_THRESHOLD = 0.5


@dataclass
class CollectionDiagnostics:
    pages_traversed: int
    raw_items_scanned: int
    issues_collected: int
    prs_filtered: int
    effective_yield: float
    exit_reason: str


# Allowed exit reasons (documented contract):
#   limit_reached | empty_batch | page_ceiling | low_yield_abort
#   timeout_partial | error


class GitHubCollector:
    """Collects GitHub issues and optionally their comments."""

    def __init__(
        self,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        if client is not None:
            self._client = client  # injected (tests)
        else:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "enterprise-incident-intelligence-platform",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            self._client = httpx.Client(
                base_url="https://api.github.com",
                headers=headers,
                timeout=timeout_seconds,
                follow_redirects=True,
            )
        # Populated for every collect_issues() run; read by the adapter.
        self.last_diagnostics: CollectionDiagnostics | None = None

    def collect_issues(
        self,
        owner: str,
        repo: str,
        *,
        state: str = "all",
        limit: int = 50,
        include_comments: bool = True,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        page = 1
        pages_traversed = 0
        raw_items_scanned = 0
        prs_filtered = 0
        exit_reason = "empty_batch"

        # Incremental runs pass `since`: the GitHub API then returns only issues
        # with updated_at >= since, so pagination terminates naturally on an
        # empty batch. Backfill runs pass since=None (no filter).
        since_param = (
            since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if since is not None
            else None
        )

        while True:
            if len(issues) >= limit:
                exit_reason = "limit_reached"
                break
            if page > _PAGE_CEILING:
                exit_reason = "page_ceiling"
                break

            params: dict[str, Any] = {
                "state": state,
                "per_page": 100,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            }
            if since_param is not None:
                params["since"] = since_param

            try:
                response = self._client.get(f"/repos/{owner}/{repo}/issues", params=params)
                response.raise_for_status()
            except (httpx.ReadTimeout, httpx.TimeoutException):
                # Preserve everything collected so far; do not re-raise.
                exit_reason = "timeout_partial"
                logger.warning(
                    "GitHub read timeout on %s/%s page %d — returning %d partial issues",
                    owner, repo, page, len(issues),
                )
                break

            batch = response.json()
            if not batch:
                exit_reason = "empty_batch"
                break

            pages_traversed += 1
            raw_items_scanned += len(batch)
            prs_filtered += sum(1 for item in batch if "pull_request" in item)

            for item in batch:
                if "pull_request" in item:
                    continue
                issue = dict(item)
                issue["repository"] = {"owner": owner, "name": repo, "full_name": f"{owner}/{repo}"}
                if include_comments and issue.get("comments", 0) > 0:
                    issue["comments_payload"] = self._collect_comments(issue["comments_url"])
                else:
                    issue["comments_payload"] = []
                issues.append(issue)
                if len(issues) >= limit:
                    break

            logger.debug(
                "[collect_issues] page=%d raw=%d prs=%d cumulative=%d repo=%s/%s",
                page, len(batch), sum(1 for i in batch if "pull_request" in i),
                len(issues), owner, repo,
            )

            # limit reached on this page takes precedence over abort heuristics
            if len(issues) >= limit:
                exit_reason = "limit_reached"
                break

            # Scan-budget guard (Part 3): stop scanning regardless of yield.
            if raw_items_scanned >= MAX_SCANNED_ITEMS:
                exit_reason = "low_yield_abort"
                break

            # Low-yield guard (Part 2): too many pages for too few issues.
            avg_issues_per_page = len(issues) / pages_traversed
            if pages_traversed >= MAX_PAGES and avg_issues_per_page < LOW_YIELD_THRESHOLD:
                exit_reason = "low_yield_abort"
                break

            page += 1

        effective_yield = (len(issues) / pages_traversed) if pages_traversed else 0.0
        self.last_diagnostics = CollectionDiagnostics(
            pages_traversed=pages_traversed,
            raw_items_scanned=raw_items_scanned,
            issues_collected=len(issues),
            prs_filtered=prs_filtered,
            effective_yield=round(effective_yield, 2),
            exit_reason=exit_reason,
        )

        logger.info(
            "collection_complete source=github repo=%s/%s pages_traversed=%d "
            "raw_items_scanned=%d issues_collected=%d prs_filtered=%d "
            "effective_yield=%.2f exit_reason=%s",
            owner, repo, pages_traversed, raw_items_scanned, len(issues),
            prs_filtered, round(effective_yield, 2), exit_reason,
        )
        return issues

    def _collect_comments(self, comments_url: str) -> list[dict[str, Any]]:
        comments: list[dict[str, Any]] = []
        page = 1
        while True:
            try:
                response = self._client.get(comments_url, params={"per_page": 100, "page": page})
                response.raise_for_status()
            except (httpx.ReadTimeout, httpx.TimeoutException):
                # Comments are enrichment, not critical: return what we have
                # rather than aborting the whole collection for one issue.
                logger.debug("Comment fetch timeout at %s page %d — partial", comments_url, page)
                break
            batch = response.json()
            if not batch:
                break
            comments.extend(batch)
            page += 1
        return comments

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubCollector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
