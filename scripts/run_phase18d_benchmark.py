"""Phase 18D: evaluate Adaptive Routing (18A/18B) and Confidence
Normalization (18C) against the live corpus and Phase 17C's gold dataset.

Configurations (identical gold dataset, harness, K=10, expand=True,
rerank=True for all three — only routing behavior differs):
  A: Dense, routing disabled
  B: Hybrid, routing disabled ("Always Hybrid")
  C: Adaptive Routing enabled (DefaultRuleBasedRoutingPolicy)

Read-only against the corpus. Makes real OpenAI calls (expand+rerank are
ON for all three configs, deliberately, so there is real LLM cost data to
compare — see the script's own printed methodology note). Writes
artifacts to .benchmarks/phase18d/.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import openai

from app.db.session import SessionLocal
from app.evaluation.benchmark import FileBenchmarkRepository, compare_runs, create_benchmark_run
from app.evaluation.gold_loader import load_gold_dataset
from app.evaluation.harness import evaluate
from app.evaluation.production_pipeline import HybridProductionAdapter
from app.evaluation.retrieval_strategies import load_bm25_retriever
from app.services.confidence_normalization import normalize_confidence
from app.services.embedding_service import EmbeddingService
from app.services.hybrid_search import HybridRetriever
from app.services.llm_service import LLMService
from app.services.routed_search import RoutedSearchConfig, RoutedSearchService
from app.services.routing import DefaultRuleBasedRoutingPolicy, RoutingEngine, RoutingStrategy
from app.services.search import IncidentSearchService

GOLD_PATH = Path("tests/eval/gold/phase17c_benchmark_v1.json")
REPO_DIR = Path(".benchmarks/phase18d")
K = 10

# Illustrative, approximate published gpt-4o-mini rates at time of writing —
# NOT guaranteed current; used only to give an order-of-magnitude cost estimate.
PROMPT_COST_PER_1K = 0.00015
COMPLETION_COST_PER_1K = 0.0006


class _CostTrackingLLMService:
    """Wraps a real LLMService, monkeypatching its OpenAI client's
    ``chat.completions.create`` in place to record token usage per call.
    Delegates every method to the real instance unchanged.
    """

    def __init__(self, inner: LLMService) -> None:
        self._inner = inner
        self.calls: list[dict] = []
        original_create = inner.client.chat.completions.create

        def tracked_create(*args, **kwargs):
            response = original_create(*args, **kwargs)
            usage = response.usage
            self.calls.append(
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }
            )
            return response

        inner.client.chat.completions.create = tracked_create

    def expand_search_query(self, query):
        return self._inner.expand_search_query(query)

    def rerank_incident_search_results(self, *, query, candidates, limit):
        return self._inner.rerank_incident_search_results(
            query=query, candidates=candidates, limit=limit
        )

    def summary(self) -> dict:
        prompt = sum(c["prompt_tokens"] or 0 for c in self.calls)
        completion = sum(c["completion_tokens"] or 0 for c in self.calls)
        cost = prompt / 1000 * PROMPT_COST_PER_1K + completion / 1000 * COMPLETION_COST_PER_1K
        return {
            "num_calls": len(self.calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "estimated_cost_usd": cost,
        }


class _RetryingLLMService:
    """Evaluation-only retry/backoff wrapper — same pattern as Phase 17D's
    benchmark runner — so transient rate-limit errors don't get recorded as
    search failures by the harness.
    """

    def __init__(self, inner, *, max_retries: int = 6) -> None:
        self._inner = inner
        self._max_retries = max_retries

    def _call(self, fn, *args, **kwargs):
        delay = 2.0
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except openai.RateLimitError:
                if attempt == self._max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    def expand_search_query(self, query):
        return self._call(self._inner.expand_search_query, query)

    def rerank_incident_search_results(self, *, query, candidates, limit):
        return self._call(
            self._inner.rerank_incident_search_results,
            query=query, candidates=candidates, limit=limit,
        )


def _mean(values):
    return statistics.mean(values) if values else None


def _time_each(fn, items):
    durations = []
    results = []
    for item in items:
        start = time.perf_counter()
        results.append(fn(item))
        durations.append(time.perf_counter() - start)
    return results, durations


def main() -> None:
    dataset = load_gold_dataset(GOLD_PATH)
    db = SessionLocal()
    embedding_service = EmbeddingService()

    print("Building BM25 index over the live corpus...")
    bm25 = load_bm25_retriever(db)

    cost_trackers: dict[str, _CostTrackingLLMService] = {}

    def build_dense(name: str) -> tuple[IncidentSearchService, _RetryingLLMService]:
        tracker = _CostTrackingLLMService(LLMService())
        cost_trackers[name] = tracker
        retrying = _RetryingLLMService(tracker)
        dense = IncidentSearchService(db, embedding_service=embedding_service, llm_service=retrying)
        return dense, retrying

    dense_a, _ = build_dense("A")
    service_a = RoutedSearchService(
        dense_a, routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=False),
    )

    dense_b, llm_b = build_dense("B")
    service_b = HybridProductionAdapter(db, HybridRetriever(dense_b, bm25), llm_b)

    dense_c, _ = build_dense("C")
    service_c = RoutedSearchService(
        dense_c, bm25=bm25, hybrid=HybridRetriever(dense_c, bm25),
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=True),
    )

    print("Evaluating Config A: Dense (routing disabled)...")
    report_a = evaluate(dataset, service_a, k=K, expand=True, rerank=True)
    print("Evaluating Config B: Hybrid (routing disabled, Always Hybrid)...")
    report_b = evaluate(dataset, service_b, k=K, expand=True, rerank=True)
    print("Evaluating Config C: Adaptive Routing enabled...")
    report_c = evaluate(dataset, service_c, k=K, expand=True, rerank=True)

    for name, report in [("A (Dense)", report_a), ("B (Always Hybrid)", report_b), ("C (Routed)", report_c)]:
        m = report.aggregate_metrics
        print(f"\n=== {name} ===")
        print(f"  evaluated={report.num_evaluated} skipped={report.num_skipped}")
        print(f"  mean_recall@{K}={m.mean_recall_at_k:.4f} mean_MRR={m.mean_reciprocal_rank:.4f} "
              f"mean_NDCG@{K}={m.mean_ndcg_at_k:.4f} resolution_coverage={m.resolution_coverage:.4f}")

    print("\n=== Cost Analysis (one full evaluation pass, 36 queries, expand+rerank ON) ===")
    print("  (call/token counts theoretically identical across configs, since expand/rerank")
    print("   fire once per query regardless of which retrieval strategy backs candidate")
    print("   generation -- verifying that prediction empirically below)")
    for name in ("A", "B", "C"):
        summary = cost_trackers[name].summary()
        print(
            f"  Config {name}: calls={summary['num_calls']:3d}  "
            f"prompt_tokens={summary['prompt_tokens']:6d}  "
            f"completion_tokens={summary['completion_tokens']:5d}  "
            f"est_cost=${summary['estimated_cost_usd']:.4f}"
        )

    # ── Routing analysis (pure, free, no LLM, no DB) ────────────────────────────
    routing_engine = RoutingEngine(DefaultRuleBasedRoutingPolicy())
    routing_records = []
    routing_latencies = []
    for gold_query in dataset.queries:
        start = time.perf_counter()
        decision = routing_engine.route(gold_query.query)
        routing_latencies.append(time.perf_counter() - start)
        routing_records.append(
            {
                "query_id": gold_query.id, "strategy": decision.strategy.value,
                "reason": decision.reason, "token_count": decision.signals.token_count,
            }
        )

    strategy_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    tokens_by_strategy: dict[str, list[int]] = {}
    for record in routing_records:
        strategy_counts[record["strategy"]] = strategy_counts.get(record["strategy"], 0) + 1
        rule_counts[record["reason"]] = rule_counts.get(record["reason"], 0) + 1
        tokens_by_strategy.setdefault(record["strategy"], []).append(record["token_count"])

    total_queries = len(routing_records)
    print("\n=== Routing Statistics ===")
    for strategy, count in sorted(strategy_counts.items()):
        avg_tokens = _mean(tokens_by_strategy[strategy])
        print(
            f"  {strategy:8s} {count:2d}/{total_queries} ({100 * count / total_queries:.1f}%)  "
            f"avg_token_count={avg_tokens:.2f}"
        )
    print("  rule utilization:")
    for reason, count in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {count:2d}x  {reason}")
    print(
        f"  routing decision latency: mean={_mean(routing_latencies) * 1e6:.2f}us  "
        f"total={sum(routing_latencies) * 1e3:.3f}ms over {total_queries} queries"
    )
    print("  filter-forced dense routing: 0/36 (gold dataset queries carry no filters)")

    # ── Latency: retrieval-only (expand=F, rerank=F; zero LLM cost) ────────────
    print("\nMeasuring retrieval-only latency (expand=F, rerank=F)...")
    queries = [gold_query.query for gold_query in dataset.queries]
    _, latency_a = _time_each(
        lambda q: service_a.retrieve(q, limit=K, expand=False, rerank=False, call_site="p18d_lat"), queries
    )
    _, latency_b = _time_each(
        lambda q: service_b.retrieve(q, limit=K, expand=False, rerank=False, call_site="p18d_lat"), queries
    )
    _, latency_c = _time_each(
        lambda q: service_c.retrieve(q, limit=K, expand=False, rerank=False, call_site="p18d_lat"), queries
    )

    print("\n=== Latency: retrieval-only (no expand/rerank) ===")
    for name, latencies in [("Dense", latency_a), ("Hybrid (always)", latency_b), ("Routed", latency_c)]:
        print(f"  {name:18s} mean={_mean(latencies) * 1000:.2f}ms  total={sum(latencies) * 1000:.1f}ms")

    # ── Latency: normalization (pure, free) ─────────────────────────────────────
    norm_durations = []
    for _ in range(1000):
        start = time.perf_counter()
        normalize_confidence(RoutingStrategy.DENSE, 0.3)
        norm_durations.append(time.perf_counter() - start)
    print(f"\n=== Latency: confidence normalization === mean={_mean(norm_durations) * 1e6:.3f}us (n=1000)")

    # ── Confidence analysis (Config C only, expand+rerank ON, second pass) ─────
    print("\nComputing per-query normalized confidence for Config C (second LLM pass)...")
    metric_by_id = {
        outcome.query_id: outcome.metric for outcome in report_c.per_query if outcome.metric is not None
    }
    confidence_records = []
    confidence_latencies = []
    for gold_query in dataset.queries:
        start = time.perf_counter()
        results = service_c.retrieve(
            gold_query.query, limit=K, expand=True, rerank=True, call_site="p18d_confidence"
        )
        confidence_latencies.append(time.perf_counter() - start)
        observation = service_c.last_observation
        strategy = observation.effective_strategy
        if not results:
            raw_score = None
        else:
            top1 = results[0]
            raw_score = top1.distance if strategy == RoutingStrategy.DENSE else -top1.distance
        normalized = normalize_confidence(strategy, raw_score)
        metric = metric_by_id.get(gold_query.id)
        confidence_records.append(
            {
                "query_id": gold_query.id, "strategy": strategy.value, "value": normalized.value,
                "level": normalized.level,
                "recall_at_k": metric.recall_at_k if metric else None,
                "reciprocal_rank": metric.reciprocal_rank if metric else None,
            }
        )

    print(
        f"=== Latency: full pipeline, Config C (expand+rerank ON) === "
        f"mean={_mean(confidence_latencies) * 1000:.2f}ms total={sum(confidence_latencies) * 1000:.1f}ms"
    )

    level_counts: dict[str, int] = {}
    for record in confidence_records:
        level_counts[record["level"]] = level_counts.get(record["level"], 0) + 1
    print("\n=== Confidence Statistics (Config C) ===")
    for level, count in sorted(level_counts.items()):
        print(f"  {level:8s} {count:2d}/{len(confidence_records)}")

    values_by_strategy: dict[str, list[float]] = {}
    for record in confidence_records:
        values_by_strategy.setdefault(record["strategy"], []).append(record["value"])
    print("  mean normalized confidence by strategy:")
    for strategy, values in sorted(values_by_strategy.items()):
        print(f"    {strategy:8s} mean={_mean(values):.4f} n={len(values)}")

    with_recall = [r for r in confidence_records if r["recall_at_k"] is not None]
    high_conf = [r for r in with_recall if r["level"] == "HIGH"]
    medium_conf = [r for r in with_recall if r["level"] == "MEDIUM"]
    low_conf = [r for r in with_recall if r["level"] == "LOW"]
    print("\n  confidence vs Recall@K:")
    for label, bucket in [("HIGH", high_conf), ("MEDIUM", medium_conf), ("LOW", low_conf)]:
        recall_mean = _mean([r["recall_at_k"] for r in bucket])
        print(f"    {label:8s} n={len(bucket):2d}  mean_recall@K={recall_mean}")
    print("  confidence vs MRR:")
    for label, bucket in [("HIGH", high_conf), ("MEDIUM", medium_conf), ("LOW", low_conf)]:
        mrr_mean = _mean([r["reciprocal_rank"] for r in bucket])
        print(f"    {label:8s} n={len(bucket):2d}  mean_MRR={mrr_mean}")

    # ── Regression analysis ─────────────────────────────────────────────────────
    run_a = create_benchmark_run(experiment_name="phase18d", report=report_a, run_id="dense")
    run_b = create_benchmark_run(experiment_name="phase18d", report=report_b, run_id="hybrid")
    run_c = create_benchmark_run(experiment_name="phase18d", report=report_c, run_id="routed")

    dense_to_routing = compare_runs(run_a, run_c)
    hybrid_to_routing = compare_runs(run_b, run_c)
    print("\n=== Regression: Dense -> Routing ===")
    print(f"  {dense_to_routing.summary}")
    print("\n=== Regression: Hybrid -> Routing ===")
    print(f"  {hybrid_to_routing.summary}")

    REPO_DIR.mkdir(parents=True, exist_ok=True)
    repository = FileBenchmarkRepository(REPO_DIR)
    for run in (run_a, run_b, run_c):
        try:
            repository.save(run)
        except ValueError:
            repository.delete(run.run_id)
            repository.save(run)

    (REPO_DIR / "routing_records.json").write_text(json.dumps(routing_records, indent=2), encoding="utf-8")
    (REPO_DIR / "confidence_records.json").write_text(
        json.dumps(confidence_records, indent=2), encoding="utf-8"
    )
    print(f"\nSaved artifacts to {REPO_DIR}/")


if __name__ == "__main__":
    main()
