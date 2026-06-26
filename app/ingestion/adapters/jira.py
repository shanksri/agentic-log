"""JiraAdapter — wraps JiraCollector + JiraNormalizer behind SourceAdapter."""

from __future__ import annotations

from typing import Any, Iterator

from app.ingestion.adapters.base import NormalizedIncident, SourceAdapter
from app.ingestion.collectors.jira_collector import JiraCollector
from app.ingestion.normalizers.jira_normalizer import JiraNormalizer


class JiraAdapter(SourceAdapter):
    source_type = "jira"

    def collect(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Collect raw Jira issue dicts.

        Expected config keys:
          base_url, project_key, limit, status_filter (optional), token (optional)
        kwargs may include ``since`` (datetime) for incremental runs.
        """
        base_url = config["base_url"]
        project_key = config["project_key"]
        limit = config.get("limit", 50)
        status_filter = config.get("status_filter")
        token = config.get("token")
        since = kwargs.get("since")

        with JiraCollector(base_url, token) as collector:
            issues = collector.collect_issues(
                project_key,
                limit=limit,
                since=since,
                status_filter=status_filter,
            )
        yield from issues

    def normalize(self, raw: dict[str, Any]) -> NormalizedIncident:
        return JiraNormalizer().normalize(raw)
