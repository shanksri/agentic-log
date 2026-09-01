"""Directly fetch the specific GitHub issues the gold dataset references that
the broad "top 500 most-recently-updated" re-ingestion (scripts/
reingest_full_corpus.py) missed -- high-traffic repos where a months-old
issue falls outside the 500 most-recently-updated window.

Cheap and precise: ~12 single-issue GET requests (+ their comments) instead
of thousands of list-pagination requests. Reuses GitHubAdapter.normalize()
and the full existing ingest_with_adapter() pipeline (dedup/embed/persist/
watermark) unchanged -- only collection is replaced.
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import settings
from app.db.session import SessionLocal
from app.ingestion.adapters.github import GitHubAdapter
from app.services.incident_ingestion import IncidentIngestionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("fetch_missing")

# (owner, repo, issue_number) -- derived from resolve_gold_dataset()'s
# unresolved_identities against tests/eval/gold/phase17c_benchmark_v1.json.
MISSING_ISSUES = [
    ("kubernetes", "kubernetes", 137685),
    ("kubernetes", "kubernetes", 138622),
    ("kubernetes", "kubernetes", 115699),
    ("kubernetes", "kubernetes", 137741),
    ("istio", "istio", 59021),
    ("istio", "istio", 60122),
    ("langchain-ai", "langchain", 29768),
    ("ray-project", "ray", 61456),
    ("ray-project", "ray", 60150),
    ("ray-project", "ray", 62535),
    ("getsentry", "sentry", 63082),
    ("getsentry", "sentry", 100943),
]


class _SingleIssueAdapter(GitHubAdapter):
    """Fetches one specific issue (+ comments) instead of listing/paginating."""

    def __init__(self, issue_number: int) -> None:
        super().__init__()
        self._issue_number = issue_number

    def collect(self, config, **kwargs):
        owner = config["owner"]
        repo = config["repo"]
        token = config.get("token")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "enterprise-incident-intelligence-platform",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=30.0) as client:
            resp = client.get(f"/repos/{owner}/{repo}/issues/{self._issue_number}")
            resp.raise_for_status()
            issue = resp.json()
            issue["repository"] = {"owner": owner, "name": repo, "full_name": f"{owner}/{repo}"}
            comments_payload = []
            if issue.get("comments", 0) > 0:
                comments_url = issue["comments_url"]
                page = 1
                while True:
                    cresp = client.get(comments_url, params={"per_page": 100, "page": page})
                    cresp.raise_for_status()
                    batch = cresp.json()
                    if not batch:
                        break
                    comments_payload.extend(batch)
                    page += 1
            issue["comments_payload"] = comments_payload
        yield issue


def main() -> None:
    for owner, repo, number in MISSING_ISSUES:
        db = SessionLocal()
        try:
            service = IncidentIngestionService(db)
            source = service._get_or_create_source(owner, repo)
            config = {
                "owner": owner, "repo": repo, "state": "all",
                "limit": 1, "include_comments": True, "token": settings.github_token,
            }
            adapter = _SingleIssueAdapter(number)
            result = service.ingest_with_adapter(source, adapter, config, force_backfill=True)
            logger.info("%s/%s#%d: %s", owner, repo, number, result)
        except httpx.HTTPError as exc:
            logger.error("%s/%s#%d FAILED: %s", owner, repo, number, exc)
        finally:
            db.close()


if __name__ == "__main__":
    main()
