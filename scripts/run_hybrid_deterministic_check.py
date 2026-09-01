"""Deterministic (no-LLM) recheck of the Phase 18D Dense/Hybrid/Routed
comparison.

Phase 18D's persisted benchmark (.benchmarks/phase18d/) ran all three
configs with expand=True, rerank=True, each config wired to its own
independently-constructed LLMService making live, separately-timed OpenAI
calls. Query expansion samples at non-zero temperature, so the same nominal
query can be expanded into different phrasings across configs -- meaning a
single-repetition, per-query difference between "Always Hybrid" and "Routed"
(both of which pick the hybrid strategy for the same query, per
routing_records.json) is not distinguishable from LLM sampling noise.

This script re-runs the same three configs (Dense / Always-Hybrid / Routed)
against the same 36-query gold set with expand=False, rerank=False --
zero LLM calls, fully deterministic -- to see whether Hybrid's aggregate
recall/MRR regression versus Dense (0.929 vs 0.964 in the original run)
holds up once LLM-driven variance is removed from the picture.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.db.session import SessionLocal
from app.evaluation.gold_loader import load_gold_dataset
from app.evaluation.harness import evaluate
from app.evaluation.retrieval_strategies import load_bm25_retriever
from app.services.hybrid_search import HybridRetriever
from app.services.routed_search import RoutedSearchConfig, RoutedSearchService
from app.services.routing import DefaultRuleBasedRoutingPolicy, RoutingEngine
from app.services.search import IncidentSearchService

GOLD_PATH = Path("tests/eval/gold/phase17c_benchmark_v1.json")
K = 10


def main() -> None:
    dataset = load_gold_dataset(GOLD_PATH)
    db = SessionLocal()

    print("Building BM25 index over the live corpus...")
    bm25 = load_bm25_retriever(db)

    dense = IncidentSearchService(db)
    hybrid_retriever = HybridRetriever(dense, bm25)

    service_dense = RoutedSearchService(
        dense, routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=False),
    )
    service_hybrid_always = RoutedSearchService(
        dense, bm25=bm25, hybrid=hybrid_retriever,
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=True),
    )
    service_routed = RoutedSearchService(
        dense, bm25=bm25, hybrid=hybrid_retriever,
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    # Force "Always Hybrid": monkeypatch the routing engine's policy to
    # always return HYBRID, so this config is genuinely hybrid-for-every-
    # query (matching Phase 18D's Config B intent) rather than adaptive.
    from app.services.routing import RoutingDecision, RoutingStrategy

    class _AlwaysHybridPolicy(DefaultRuleBasedRoutingPolicy):
        def decide(self, query, signals):
            return RoutingDecision(
                strategy=RoutingStrategy.HYBRID,
                reason="forced hybrid for deterministic recheck",
                signals=signals,
            )

    service_hybrid_always._routing_engine = RoutingEngine(_AlwaysHybridPolicy())

    print("Evaluating Dense (expand=False, rerank=False)...")
    report_dense = evaluate(dataset, service_dense, k=K, expand=False, rerank=False)
    print("Evaluating Always-Hybrid (expand=False, rerank=False)...")
    report_hybrid = evaluate(dataset, service_hybrid_always, k=K, expand=False, rerank=False)
    print("Evaluating Adaptive Routing (expand=False, rerank=False)...")
    report_routed = evaluate(dataset, service_routed, k=K, expand=False, rerank=False)

    per_query = {}
    for name, report in [
        ("dense", report_dense), ("hybrid_always", report_hybrid), ("routed", report_routed),
    ]:
        m = report.aggregate_metrics
        print(f"\n=== {name} ===")
        print(f"  evaluated={report.num_evaluated} skipped={report.num_skipped}")
        if report.num_skipped:
            for outcome in report.per_query:
                if outcome.skipped:
                    print(f"    SKIPPED {outcome.query_id}: {outcome.skip_reason}")
        print(
            f"  recall@{K}={m.mean_recall_at_k}  MRR={m.mean_reciprocal_rank}  "
            f"NDCG@{K}={m.mean_ndcg_at_k}"
        )
        for outcome in report.per_query:
            if outcome.metric is None:
                continue
            per_query.setdefault(outcome.query_id, {})[name] = {
                "recall_at_k": outcome.metric.recall_at_k,
                "reciprocal_rank": outcome.metric.reciprocal_rank,
                "ndcg_at_k": outcome.metric.ndcg_at_k,
            }

    print("\n=== Per-query divergence (dense vs hybrid_always vs routed) ===")
    for qid, results in per_query.items():
        recalls = {name: r["recall_at_k"] for name, r in results.items()}
        if len(set(recalls.values())) > 1:
            print(f"  {qid}: {recalls}")

    out_path = Path(".benchmarks/hybrid_deterministic_recheck.json")
    out_path.write_text(json.dumps(per_query, indent=2), encoding="utf-8")
    print(f"\nSaved per-query detail to {out_path}")


if __name__ == "__main__":
    main()
