"""Phase 17D: evaluate the full production pipeline matrix (Dense/Hybrid x
expand x rerank) against the live corpus and the Phase 17C gold dataset.

Read-only against the corpus. Makes real OpenAI calls for expand_search_query
/ rerank_incident_search_results (configs A, B, D, E) via the unmodified
LLMService. Writes benchmark run artifacts to .benchmarks/phase17d/ via
FileBenchmarkRepository (Phase 16F).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import openai

from app.db.session import SessionLocal
from app.evaluation.benchmark import FileBenchmarkRepository, compare_runs, create_benchmark_run
from app.evaluation.gold_loader import load_gold_dataset
from app.evaluation.harness import EvaluationReport, evaluate
from app.evaluation.production_pipeline import HybridProductionAdapter
from app.evaluation.regression import RegressionReport
from app.evaluation.retrieval_strategies import HybridRetrievalAdapter, load_bm25_retriever
from app.services.embedding_service import EmbeddingService
from app.services.hybrid_search import HybridRetriever
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchService

GOLD_PATH = Path("tests/eval/gold/phase17c_benchmark_v1.json")
REPO_DIR = Path(".benchmarks/phase17d")
K = 10


class _RetryingLLMService:
    """Evaluation-only wrapper around LLMService that retries on
    OpenAI rate-limit errors with backoff, so a benchmark run's results
    are never contaminated by transient TPM exhaustion getting recorded
    as a search failure (a skipped query) by the harness. Delegates every
    other behavior to the real, unmodified LLMService.
    """

    def __init__(self, inner: LLMService, *, max_retries: int = 6) -> None:
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

    def expand_search_query(self, query: str) -> list[str]:
        return self._call(self._inner.expand_search_query, query)

    def rerank_incident_search_results(self, *, query, candidates, limit):
        return self._call(
            self._inner.rerank_incident_search_results,
            query=query, candidates=candidates, limit=limit,
        )


def _print_report(name: str, report: EvaluationReport) -> None:
    m = report.aggregate_metrics
    print(f"\n=== {name} ===")
    print(f"  evaluated={report.num_evaluated} skipped={report.num_skipped}")
    print(f"  mean_recall@{K}={m.mean_recall_at_k:.4f}")
    print(f"  mean_MRR={m.mean_reciprocal_rank:.4f}")
    print(f"  mean_NDCG@{K}={m.mean_ndcg_at_k:.4f}")
    print("  by category:")
    for category, agg in sorted(report.category_breakdown.items()):
        recall = f"{agg.mean_recall_at_k:.4f}" if agg.mean_recall_at_k is not None else "n/a"
        mrr = f"{agg.mean_reciprocal_rank:.4f}" if agg.mean_reciprocal_rank is not None else "n/a"
        ndcg = f"{agg.mean_ndcg_at_k:.4f}" if agg.mean_ndcg_at_k is not None else "n/a"
        print(f"    {category:20s} n={agg.num_queries:2d}  recall={recall} mrr={mrr} ndcg={ndcg}")


def _print_regression(name: str, regression: RegressionReport) -> None:
    print(f"\n=== Regression: {name} ===")
    print(f"  {regression.summary}")


def _query_level_deltas(baseline: EvaluationReport, candidate: EvaluationReport) -> dict:
    baseline_by_id = {o.query_id: o for o in baseline.per_query}
    candidate_by_id = {o.query_id: o for o in candidate.per_query}
    improved, harmed, unchanged = [], [], []
    for query_id, b in baseline_by_id.items():
        c = candidate_by_id[query_id]
        b_recall = b.metric.recall_at_k if b.metric else None
        c_recall = c.metric.recall_at_k if c.metric else None
        if b_recall is None or c_recall is None:
            continue
        if c_recall > b_recall:
            improved.append(query_id)
        elif c_recall < b_recall:
            harmed.append(query_id)
        else:
            unchanged.append(query_id)
    return {"improved": improved, "harmed": harmed, "unchanged": unchanged}


def main() -> None:
    dataset = load_gold_dataset(GOLD_PATH)
    db = SessionLocal()
    embedding_service = EmbeddingService()
    llm_service = _RetryingLLMService(LLMService())
    dense = IncidentSearchService(db, embedding_service=embedding_service, llm_service=llm_service)

    print("Building BM25 index over the live corpus...")
    bm25 = load_bm25_retriever(db)
    hybrid_retriever = HybridRetriever(dense, bm25)

    hybrid_baseline_adapter = HybridRetrievalAdapter(db, hybrid_retriever)
    hybrid_production_adapter = HybridProductionAdapter(db, hybrid_retriever, llm_service)

    print("Evaluating Baseline: Dense (expand=F, rerank=F)...")
    baseline = evaluate(dataset, dense, k=K, expand=False, rerank=False)

    print("Evaluating Config A: Dense + Expansion (expand=T, rerank=F)...")
    config_a = evaluate(dataset, dense, k=K, expand=True, rerank=False)

    print("Evaluating Config B: Dense + Expansion + Rerank (PRODUCTION today)...")
    config_b = evaluate(dataset, dense, k=K, expand=True, rerank=True)

    print("Evaluating Config C: Hybrid (expand=F, rerank=F)...")
    config_c = evaluate(dataset, hybrid_baseline_adapter, k=K, expand=False, rerank=False)

    print("Evaluating Config D: Hybrid + Expansion (expand=T, rerank=F)...")
    config_d = evaluate(dataset, hybrid_production_adapter, k=K, expand=True, rerank=False)

    print("Evaluating Config E: Hybrid + Expansion + Rerank (candidate production)...")
    config_e = evaluate(dataset, hybrid_production_adapter, k=K, expand=True, rerank=True)

    for name, report in [
        ("Baseline (Dense)", baseline), ("A (Dense+Expand)", config_a),
        ("B (Dense+Expand+Rerank)", config_b), ("C (Hybrid)", config_c),
        ("D (Hybrid+Expand)", config_d), ("E (Hybrid+Expand+Rerank)", config_e),
    ]:
        _print_report(name, report)

    runs = {
        "baseline": create_benchmark_run(experiment_name="phase17d", report=baseline, run_id="baseline"),
        "a": create_benchmark_run(experiment_name="phase17d", report=config_a, run_id="a"),
        "b": create_benchmark_run(experiment_name="phase17d", report=config_b, run_id="b"),
        "c": create_benchmark_run(experiment_name="phase17d", report=config_c, run_id="c"),
        "d": create_benchmark_run(experiment_name="phase17d", report=config_d, run_id="d"),
        "e": create_benchmark_run(experiment_name="phase17d", report=config_e, run_id="e"),
    }

    comparisons = [
        ("Dense -> Dense+Expansion", runs["baseline"], runs["a"]),
        ("Dense+Expansion -> Dense+Expansion+Rerank", runs["a"], runs["b"]),
        ("Dense -> Hybrid", runs["baseline"], runs["c"]),
        ("Hybrid -> Hybrid+Expansion", runs["c"], runs["d"]),
        ("Hybrid+Expansion -> Hybrid+Expansion+Rerank", runs["d"], runs["e"]),
        ("Dense Production -> Hybrid Production", runs["b"], runs["e"]),
    ]
    regressions = {}
    for name, baseline_run, candidate_run in comparisons:
        regression = compare_runs(baseline_run, candidate_run)
        regressions[name] = regression
        _print_regression(name, regression)

    print("\n=== Query-level recall deltas (failure analysis) ===")
    for name, baseline_report, candidate_report in [
        ("Expansion effect (Dense -> A)", baseline, config_a),
        ("Rerank effect (A -> B)", config_a, config_b),
        ("Hybrid+Expansion effect (C -> D)", config_c, config_d),
        ("Hybrid+Rerank effect (D -> E)", config_d, config_e),
    ]:
        deltas = _query_level_deltas(baseline_report, candidate_report)
        print(f"  {name}: improved={deltas['improved']} harmed={deltas['harmed']}")

    REPO_DIR.mkdir(parents=True, exist_ok=True)
    repository = FileBenchmarkRepository(REPO_DIR)
    for run in runs.values():
        try:
            repository.save(run)
        except ValueError:
            repository.delete(run.run_id)
            repository.save(run)
    print(f"\nSaved benchmark runs to {REPO_DIR}/")

    summary_path = REPO_DIR / "regressions_summary.json"
    summary_path.write_text(
        json.dumps({name: r.summary for name, r in regressions.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved regression summaries to {summary_path}")


if __name__ == "__main__":
    main()
