"""GitHubAdapter — wraps GitHubCollector + GitHubNormalizer.

No logic lives here beyond the thin adaptation. Both underlying classes are
unchanged; this module satisfies the SourceAdapter interface by delegating
to them and mapping the GitHub-specific dataclass to NormalizedIncident.
"""

from __future__ import annotations

from typing import Any, Iterator

from app.ingestion.adapters.base import NormalizedIncident, SourceAdapter
from app.ingestion.collectors.github_collector import GitHubCollector
from app.ingestion.normalizers.github_normalizer import GitHubNormalizer


class GitHubAdapter(SourceAdapter):
    source_type = "github"

    def __init__(self) -> None:
        # Diagnostics from the most recent collect() run; read by
        # ingest_with_adapter() to surface in the result payload.
        self.last_diagnostics: Any | None = None

    def collect(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Collect raw GitHub issue dicts.

        Expected config keys:
          owner, repo, state, limit, include_comments, token (optional)
        kwargs may include ``since`` (datetime) for incremental runs.
        """
        self.last_diagnostics = None
        owner = config["owner"]
        repo = config["repo"]
        state = config.get("state", "all")
        limit = config.get("limit", 50)
        include_comments = config.get("include_comments", True)
        token = config.get("token")
        since = kwargs.get("since")

        with GitHubCollector(token) as collector:
            issues = collector.collect_issues(
                owner,
                repo,
                state=state,
                limit=limit,
                include_comments=include_comments,
                since=since,
            )
            self.last_diagnostics = getattr(collector, "last_diagnostics", None)
        yield from issues

    def normalize(self, raw: dict[str, Any]) -> NormalizedIncident:
        """Normalize one raw GitHub issue dict to NormalizedIncident."""
        gh = GitHubNormalizer().normalize(raw)

        source_metadata: dict[str, Any] = {
            "owner": gh.owner,
            "repo": gh.repo,
            "state": gh.state,
        }
        if raw.get("number") is not None:
            source_metadata["issue_number"] = raw["number"]

        return NormalizedIncident(
            source_type=gh.source_type,
            source_external_id=gh.source_external_id,
            source_url=gh.source_url,
            title=gh.title,
            description=gh.description,
            severity=gh.severity,
            status=gh.status,
            incident_type=gh.incident_type,
            environment=gh.environment,
            affected_components=gh.affected_components,
            tags=gh.tags,
            symptoms=gh.symptoms,
            root_cause_summary=gh.root_cause_summary,
            resolution_summary=gh.resolution_summary,
            canonical_text=gh.canonical_text,
            confidence_score=gh.confidence_score,
            is_gold_labeled=gh.is_gold_labeled,
            created_at_source=gh.created_at_source,
            updated_at_source=gh.updated_at_source,
            source_metadata=source_metadata,
        )
