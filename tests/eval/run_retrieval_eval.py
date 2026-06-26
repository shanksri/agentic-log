"""Retrieval evaluation harness (Phase 0).

Runs the gold query set against IncidentSearchService.search() and reports
Recall@5, Recall@10, MRR, and NDCG@10, broken out by query_type bucket, plus
the raw similarity score distribution.

This harness exercises whatever IncidentSearchService.search() currently
does in production. It does not enable expansion or reranking on its own;
those are controlled by the search_config passed in (see --rerank/--expand).

Usage:
    python -m tests.eval.run_retrieval_eval
    python -m tests.eval.run_retrieval_eval --output tests/eval/results/baseline_v2.4.json
    python -m tests.eval.run_retrieval_eval --rerank --expand --label v2.5-rerank
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from app.db.session import SessionLocal
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchResult, IncidentSearchService

GOLD_QUERIES_PATH = Path(__file__).parent / "gold_queries.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"

K_RECALL = (5, 10)
K_NDCG = 10


def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float | None:
    if not expected_ids:
        return None
    top_k = set(retrieved_ids[:k])
    return len(top_k & expected_ids) / len(expected_ids)


def mrr(retrieved_ids: list[str], expected_ids: set[str]) -> float | None:
    if not expected_ids:
        return None
    for rank, incident_id in enumerate(retrieved_ids, start=1):
        if incident_id in expected_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float | None:
    if not expected_ids:
        return None
    dcg = 0.0
    for rank, incident_id in enumerate(retrieved_ids[:k], start=1):
        relevance = 1.0 if incident_id in expected_ids else 0.0
        if relevance:
            dcg += relevance / math.log2(rank + 1)
    ideal_hits = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_query(
    search_service: IncidentSearchService,
    entry: dict,
    *,
    limit: int = 10,
    use_debug: bool = False,
) -> dict:
    query = entry["query"]
    expected_ids = set(entry["expected_incident_ids"])

    if use_debug:
        results: list[IncidentSearchResult] = search_service.search_debug(query)
    else:
        results = search_service.search(query, limit=limit)

    retrieved_ids = [str(result.incident.id) for result in results]
    scores = [result.similarity_score for result in results]

    return {
        "id": entry["id"],
        "query": query,
        "query_type": entry["query_type"],
        "expected_incident_ids": sorted(expected_ids),
        "retrieved_incident_ids": retrieved_ids,
        "recall_at_5": recall_at_k(retrieved_ids, expected_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, expected_ids, 10),
        "mrr": mrr(retrieved_ids, expected_ids),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, expected_ids, K_NDCG),
        "top1_score": scores[0] if scores else None,
        "top5_mean_score": (
            sum(scores[:5]) / len(scores[:5]) if scores[:5] else None
        ),
    }


def aggregate_bucket(query_results: list[dict]) -> dict:
    def mean(values: list[float]) -> float | None:
        clean = [value for value in values if value is not None]
        if not clean:
            return None
        return sum(clean) / len(clean)

    return {
        "count": len(query_results),
        "recall_at_5": mean([row["recall_at_5"] for row in query_results]),
        "recall_at_10": mean([row["recall_at_10"] for row in query_results]),
        "mrr": mean([row["mrr"] for row in query_results]),
        "ndcg_at_10": mean([row["ndcg_at_10"] for row in query_results]),
        "top1_score_mean": mean([row["top1_score"] for row in query_results]),
        "top5_mean_score_mean": mean([row["top5_mean_score"] for row in query_results]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval evaluation harness")
    parser.add_argument(
        "--gold-queries",
        type=Path,
        default=GOLD_QUERIES_PATH,
        help="Path to gold_queries.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the results JSON (default: tests/eval/results/<label>_<timestamp>.json)",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="search_config label, e.g. 'baseline', 'v2.5-rerank' (default: baseline)",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Use search_debug() (query expansion + LLM reranking) instead of search()",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Alias for --expand; search_debug() performs both expansion and reranking",
    )
    args = parser.parse_args()

    use_debug = args.expand or args.rerank

    gold = json.loads(args.gold_queries.read_text(encoding="utf-8"))
    queries = gold["queries"]

    db = SessionLocal()
    try:
        embedding_service = EmbeddingService()
        llm_service = LLMService() if use_debug else None
        search_service = IncidentSearchService(
            db, embedding_service=embedding_service, llm_service=llm_service
        )

        query_results = [
            evaluate_query(search_service, entry, use_debug=use_debug) for entry in queries
        ]
    finally:
        db.close()

    buckets: dict[str, list[dict]] = {}
    for row in query_results:
        buckets.setdefault(row["query_type"], []).append(row)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "embedding_model_name": embedding_service.model_name,
        "search_config": {
            "expansion": use_debug,
            "reranking": use_debug,
            "hybrid": False,
            "label": args.label,
        },
        "gold_queries_file": str(args.gold_queries),
        "overall": aggregate_bucket(query_results),
        "by_query_type": {
            query_type: aggregate_bucket(rows) for query_type, rows in buckets.items()
        },
        "queries": query_results,
    }

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_OUTPUT_DIR / f"{args.label}_{timestamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"embedding_model_name: {report['embedding_model_name']}")
    print(f"search_config: {report['search_config']}")
    print("\nOverall:")
    for key, value in report["overall"].items():
        print(f"  {key}: {value}")
    print("\nBy query type:")
    for query_type, metrics in report["by_query_type"].items():
        print(f"  {query_type}:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
