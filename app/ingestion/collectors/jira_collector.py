"""Collects Jira issues via the REST v2 search (JQL) endpoint.

Mirrors the GitHubCollector contract: returns a list of raw issue dicts and
supports ``since`` for incremental runs (mapped to a JQL ``updated >=`` clause).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_FIELDS = (
    "summary,description,status,labels,created,updated,project,comment,"
    "priority,resolution,components"
)


class JiraCollector:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        if client is not None:
            # Injected client (used by tests).
            self._client = client
            return
        headers = {
            "Accept": "application/json",
            "User-Agent": "enterprise-incident-intelligence-platform",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=True,
        )

    def collect_issues(
        self,
        project_key: str,
        *,
        limit: int = 50,
        since: datetime | None = None,
        status_filter: list[str] | None = None,
        extra_jql: str | None = None,
    ) -> list[dict[str, Any]]:
        jql = self._build_jql(project_key, since, status_filter, extra_jql)
        issues: list[dict[str, Any]] = []
        start_at = 0

        while len(issues) < limit:
            response = self._client.get(
                "/rest/api/2/search",
                params={
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": 100,
                    "fields": _FIELDS,
                },
            )
            response.raise_for_status()
            data = response.json()
            batch = data.get("issues", [])
            if not batch:
                break

            for item in batch:
                # Inject the instance base_url so the normalizer can build a
                # browse URL without being stateful (mirrors GitHubCollector
                # injecting the repository block).
                item["_base_url"] = self._base_url
                issues.append(item)
                if len(issues) >= limit:
                    break

            start_at += len(batch)
            total = data.get("total", 0)
            if start_at >= total:
                break

        logger.info(
            "Collected %d Jira issues from project %s (jql=%s)",
            len(issues),
            project_key,
            jql,
        )
        return issues

    def _build_jql(
        self,
        project_key: str,
        since: datetime | None,
        status_filter: list[str] | None,
        extra_jql: str | None,
    ) -> str:
        clauses = [f"project = {project_key}"]
        if status_filter:
            quoted = ", ".join(f'"{status}"' for status in status_filter)
            clauses.append(f"status in ({quoted})")
        if since is not None:
            # Jira JQL has minute resolution; truncate and quote.
            since_str = since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
            clauses.append(f'updated >= "{since_str}"')
        if extra_jql:
            clauses.append(f"({extra_jql})")
        return " AND ".join(clauses) + " ORDER BY updated ASC"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JiraCollector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
