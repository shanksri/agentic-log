from __future__ import annotations

from app.ingestion.adapters.github import GitHubAdapter
from app.ingestion.adapters.jira import JiraAdapter
from app.ingestion.adapters.registry import SourceRegistry

__all__ = ["GitHubAdapter", "JiraAdapter", "SourceRegistry"]

# Register built-in adapters so importing this package is sufficient to
# populate the registry.
SourceRegistry.register(GitHubAdapter())
SourceRegistry.register(JiraAdapter())
