"""Phase 22F (Part 2) — probe candidate hard-negative queries against the
live corpus so each one's "why it is NOT actually represented by the
corpus" justification is grounded in real retrieval output, not guesswork.
Read-only, no persistence. Prints top-5 dense search results per candidate.
"""

from __future__ import annotations

import json

from app.db.session import SessionLocal
from app.services.search import IncidentSearchService

CANDIDATES = {
    "hard-neg-payment-01": "Stripe webhook signature verification fails intermittently for subscription renewal events",
    "hard-neg-auth-01": "OAuth2 refresh token silently expires causing repeated re-login prompts in the web console",
    "hard-neg-monitoring-01": "PagerDuty alert routing sends duplicate incident pages to the same on-call engineer",
    "hard-neg-k8s-01": "Container restarts constantly due to an OOMKilled event triggered by a memory leak in a sidecar proxy",
    "hard-neg-networking-01": "DNS resolution intermittently fails for pods running in a different availability zone",
    "hard-neg-integrations-01": "Slack integration fails to post messages after workspace token rotation",
    "hard-neg-database-01": "PostgreSQL replication lag spikes during nightly vacuum causing read replica staleness",
    "hard-neg-api-01": "GraphQL query depth limit rejects legitimate nested queries from the mobile client",
    "hard-neg-config-01": "Helm chart values override silently ignored when using multiple values files with the same key",
    "hard-neg-deployment-01": "Kubernetes cluster upgrade from 1.28 to 1.29 breaks a custom admission webhook due to API version deprecation",
    "hard-neg-payment-02": "Recurring invoice generation skips a billing cycle for annual subscription plans",
    "hard-neg-auth-02": "Multi-factor authentication SMS codes arrive after the expiration window in the login flow",
    "hard-neg-monitoring-02": "Grafana dashboard panels show flat zero metrics after a Prometheus remote_write endpoint migration",
    "hard-neg-k8s-02": "HorizontalPodAutoscaler fails to scale down replicas even after CPU utilization drops below target",
    "hard-neg-networking-02": "Istio Envoy sidecar injection fails silently for pods in a namespace with a custom PodSecurityPolicy",
    "hard-neg-integrations-02": "LangChain agent tool-calling loop times out when chained with a custom retriever that wraps an external vector store",
    "hard-neg-database-02": "Redis cluster failover takes over 30 seconds causing client connection timeouts during primary election",
    "hard-neg-api-02": "TypeScript compiler emits incorrect type narrowing for a discriminated union inside a generic function",
    "hard-neg-config-02": "Terraform state lock is not released after a CI pipeline is cancelled mid-apply",
    "hard-neg-deployment-02": "Airflow DAG scheduler silently skips a task instance after a timezone configuration change",
}


def main() -> None:
    db = SessionLocal()
    search = IncidentSearchService(db)
    out = {}
    for cand_id, query in CANDIDATES.items():
        results = search.search(query, limit=10)
        out[cand_id] = {
            "query": query,
            "results": [
                {
                    "rank": rank,
                    "score": r.similarity_score,
                    "source_type": r.incident.source_type,
                    "source_external_id": getattr(r.incident, "source_external_id", None),
                    "title": r.incident.title,
                }
                for rank, r in enumerate(results, start=1)
            ],
        }
    with open("scripts/phase22f_hard_negative_probe_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote scripts/phase22f_hard_negative_probe_results.json")


if __name__ == "__main__":
    main()
