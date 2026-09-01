"""Re-ingest the full original evaluation corpus after the 2026-08 laptop
reset wiped the database down to a 5-incident smoke test.

Repo list: the 13 GitHub repos and 3 Jira projects (KAFKA/SPARK/CASSANDRA,
issues.apache.org) that gold datasets under tests/eval/ actually reference by
source_external_id (confirmed by scanning every gold/results JSON file), plus
three extra GitHub repos named in the corpus description embedded in
tests/eval/gold/phase17c_benchmark_v1.json ("kubernetes, terraform,
langchain, istio, redis, helm, ray, sentry, go, cockroach, kafka, spark")
that aren't otherwise gold-referenced, to get closer to the documented
16-repo / ~8,000-incident corpus scale. The first 13 are load-bearing (gold
queries won't resolve without them); the last 3 are best-effort padding, not
required for correctness.

Idempotent: safe to re-run (dedup by (source_type, source_external_id), per
doc 05).
"""

from __future__ import annotations

import logging
import sys
import time

import httpx

from app.core.config import settings
from app.db.session import SessionLocal
from app.services.incident_ingestion import IncidentIngestionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("reingest")

GOLD_REFERENCED_REPOS = [
    ("istio", "istio"),
    ("kubernetes", "kubernetes"),
    ("langchain-ai", "langchain"),
    ("microsoft", "TypeScript"),
    ("prometheus", "prometheus"),
    ("ray-project", "ray"),
    ("redis", "redis"),
]

BEST_EFFORT_REPOS = [
    ("cockroachdb", "cockroach"),
    ("apache", "kafka"),
    ("apache", "spark"),
]

JIRA_PROJECTS: list[str] = []
JIRA_BASE_URL = "https://issues.apache.org/jira"

LIMIT = 500


def ingest_repo(owner: str, repo: str, *, required: bool) -> None:
    db = SessionLocal()
    try:
        started = time.monotonic()
        result = IncidentIngestionService(db).ingest_github_repo(
            owner, repo, state="all", limit=LIMIT, include_comments=True
        )
        elapsed = time.monotonic() - started
        logger.info(
            "github %s/%s: fetched=%s inserted=%s updated=%s skipped=%s (%.1fs)",
            owner, repo, result.get("fetched"), result.get("inserted"),
            result.get("updated"), result.get("skipped"), elapsed,
        )
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        level = logger.error if required else logger.warning
        level("github %s/%s FAILED: %s", owner, repo, exc)
    finally:
        db.close()


def ingest_jira(project_key: str) -> None:
    db = SessionLocal()
    try:
        started = time.monotonic()
        result = IncidentIngestionService(db).ingest_jira_project(
            base_url=JIRA_BASE_URL, project_key=project_key, limit=LIMIT
        )
        elapsed = time.monotonic() - started
        logger.info(
            "jira %s: fetched=%s inserted=%s updated=%s skipped=%s (%.1fs)",
            project_key, result.get("fetched"), result.get("inserted"),
            result.get("updated"), result.get("skipped"), elapsed,
        )
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.error("jira %s FAILED: %s", project_key, exc)
    finally:
        db.close()


def main() -> None:
    if not settings.github_token:
        logger.warning("GITHUB_TOKEN not set — unauthenticated GitHub rate limits apply (60/hr).")

    logger.info("=== Ingesting %d gold-referenced GitHub repos (required) ===", len(GOLD_REFERENCED_REPOS))
    for owner, repo in GOLD_REFERENCED_REPOS:
        ingest_repo(owner, repo, required=True)

    logger.info("=== Ingesting %d best-effort GitHub repos (corpus scale, optional) ===", len(BEST_EFFORT_REPOS))
    for owner, repo in BEST_EFFORT_REPOS:
        ingest_repo(owner, repo, required=False)

    logger.info("=== Ingesting %d Jira projects ===", len(JIRA_PROJECTS))
    for project_key in JIRA_PROJECTS:
        ingest_jira(project_key)

    logger.info("Done.")


if __name__ == "__main__":
    sys.exit(main() or 0)
