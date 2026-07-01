"""Phase 17C: run the Dense / BM25 / Hybrid evaluation matrix against the
live corpus and the Phase 17C gold dataset, then print metrics, regressions,
and retrieval-characteristics (overlap) results.

Read-only against the corpus (only SELECTs via IncidentSearchService /
load_bm25_retriever / IdentityResolver). Writes benchmark run artifacts to
.benchmarks/phase17c/ via FileBenchmarkRepository (Phase 16F) so results are
reproducible/inspectable after the fact, not just printed once and lost.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.db.session import SessionLocal
from app.evaluation.benchmark import FileBenchmarkRepository, compare_runs, create_benchmark_run
from app.evaluation.gold_loader import load_gold_dataset
from app.evaluation.harness import EvaluationReport, evaluate
from app.evaluation.overlap_analysis import compute_overlap
from app.evaluation.regression import RegressionReport
from app.evaluation.retrieval_strategies import (
    BM25RetrievalAdapter,
    HybridRetrievalAdapter,
    load_bm25_retriever,
)
from app.services.embedding_service import EmbeddingService
from app.services.hybrid_search import HybridRetriever
from app.services.search import IncidentSearchService

GOLD_PATH = Path("tests/eval/gold/phase17c_benchmark_v1.json")
REPO_DIR = Path(".benchmarks/phase17c")
K = 10


def _print_report(name: str, report: EvaluationReport) -> None:
    m = report.aggregate_metrics
    print(f"\n=== {name} ===")
    print(f"  evaluated={report.num_evaluated} skipped={report.num_skipped}")
    print(f"  mean_recall@{K}={m.mean_recall_at_k:.4f}")
    print(f"  mean_MRR={m.mean_reciprocal_rank:.4f}")
    print(f"  mean_NDCG@{K}={m.mean_ndcg_at_k:.4f}")
    print(f"  resolution_coverage={m.resolution_coverage:.4f}")
    print(f"  distinct_retrieved_incidents={report.corpus_statistics.distinct_retrieved_incident_count}")
    print("  by category:")
    for category, agg in sorted(report.category_breakdown.items()):
        recall = f"{agg.mean_recall_at_k:.4f}" if agg.mean_recall_at_k is not None else "n/a"
        mrr = f"{agg.mean_reciprocal_rank:.4f}" if agg.mean_reciprocal_rank is not None else "n/a"
        ndcg = f"{agg.mean_ndcg_at_k:.4f}" if agg.mean_ndcg_at_k is not None else "n/a"
        print(f"    {category:20s} n={agg.num_queries:2d}  recall={recall} mrr={mrr} ndcg={ndcg}")
    print("  by difficulty:")
    for difficulty, agg in sorted(report.difficulty_breakdown.items()):
        recall = f"{agg.mean_recall_at_k:.4f}" if agg.mean_recall_at_k is not None else "n/a"
        mrr = f"{agg.mean_reciprocal_rank:.4f}" if agg.mean_reciprocal_rank is not None else "n/a"
        ndcg = f"{agg.mean_ndcg_at_k:.4f}" if agg.mean_ndcg_at_k is not None else "n/a"
        print(f"    {difficulty:20s} n={agg.num_queries:2d}  recall={recall} mrr={mrr} ndcg={ndcg}")


def _print_regression(name: str, regression: RegressionReport) -> None:
    print(f"\n=== Regression: {name} ===")
    print(f"  {regression.summary}")


def main() -> None:
    dataset = load_gold_dataset(GOLD_PATH)
    db = SessionLocal()
    embedding_service = EmbeddingService()
    dense = IncidentSearchService(db, embedding_service=embedding_service)

    print("Building BM25 index over the live corpus...")
    bm25 = load_bm25_retriever(db)
    print(f"  BM25 corpus size: {bm25.index.size} documents")

    bm25_adapter = BM25RetrievalAdapter(db, bm25)
    hybrid_adapter = HybridRetrievalAdapter(db, HybridRetriever(dense, bm25))

    print("Evaluating Dense...")
    dense_report = evaluate(dataset, dense, k=K, expand=False, rerank=False)
    print("Evaluating BM25...")
    bm25_report = evaluate(dataset, bm25_adapter, k=K, expand=False, rerank=False)
    print("Evaluating Hybrid...")
    hybrid_report = evaluate(dataset, hybrid_adapter, k=K, expand=False, rerank=False)

    _print_report("Dense", dense_report)
    _print_report("BM25", bm25_report)
    _print_report("Hybrid", hybrid_report)

    dense_run = create_benchmark_run(experiment_name="phase17c", report=dense_report, run_id="dense")
    bm25_run = create_benchmark_run(experiment_name="phase17c", report=bm25_report, run_id="bm25")
    hybrid_run = create_benchmark_run(experiment_name="phase17c", report=hybrid_report, run_id="hybrid")

    dense_vs_bm25 = compare_runs(dense_run, bm25_run)
    dense_vs_hybrid = compare_runs(dense_run, hybrid_run)
    bm25_vs_hybrid = compare_runs(bm25_run, hybrid_run)

    _print_regression("Dense (baseline) vs BM25", dense_vs_bm25)
    _print_regression("Dense (baseline) vs Hybrid", dense_vs_hybrid)
    _print_regression("BM25 (baseline) vs Hybrid", bm25_vs_hybrid)

    print("\n=== Retrieval Characteristics: Dense vs BM25 candidate overlap ===")
    overlap = compute_overlap(dataset, dense, bm25, limit=K)
    print(f"  mean_jaccard={overlap.mean_jaccard:.4f}")
    print(f"  mean_overlap_count={overlap.mean_overlap_count:.4f}")
    print(f"  mean_dense_only_count={overlap.mean_dense_only_count:.4f}")
    print(f"  mean_bm25_only_count={overlap.mean_bm25_only_count:.4f}")
    print("  per-query:")
    for result in overlap.per_query:
        print(
            f"    {result.query_id:14s} overlap={result.overlap_count} "
            f"dense_only={result.dense_only_count} bm25_only={result.bm25_only_count} "
            f"jaccard={result.jaccard:.3f}"
        )

    REPO_DIR.mkdir(parents=True, exist_ok=True)
    repository = FileBenchmarkRepository(REPO_DIR)
    for run in (dense_run, bm25_run, hybrid_run):
        try:
            repository.save(run)
        except ValueError:
            repository.delete(run.run_id)
            repository.save(run)
    print(f"\nSaved benchmark runs to {REPO_DIR}/")

    overlap_path = REPO_DIR / "overlap_report.json"
    overlap_path.write_text(
        json.dumps(
            {
                "mean_jaccard": overlap.mean_jaccard,
                "mean_overlap_count": overlap.mean_overlap_count,
                "mean_dense_only_count": overlap.mean_dense_only_count,
                "mean_bm25_only_count": overlap.mean_bm25_only_count,
                "per_query": [
                    {
                        "query_id": r.query_id,
                        "overlap_count": r.overlap_count,
                        "dense_only_count": r.dense_only_count,
                        "bm25_only_count": r.bm25_only_count,
                        "jaccard": r.jaccard,
                    }
                    for r in overlap.per_query
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved overlap report to {overlap_path}")


if __name__ == "__main__":
    main()
