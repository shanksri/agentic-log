#!/usr/bin/env python
"""Full Evaluation Pipeline CLI (Phase 21E).

Wires the complete evaluation ecosystem (Phases 16-21) into a single
command.  Supply dataset paths and choose a judge; the pipeline handles
all orchestration.

Usage examples
--------------

  # Retrieval-only evaluation (no reasoning dataset, no judge):
  python scripts/run_full_evaluation.py \\
      --retrieval-dataset tests/eval/gold/phase17c_benchmark_v1.json

  # Full pipeline with rule-based judge (no OpenAI key required):
  python scripts/run_full_evaluation.py \\
      --retrieval-dataset tests/eval/gold/phase17c_benchmark_v1.json \\
      --reasoning-dataset tests/eval/reasoning/demo.json \\
      --judge rule \\
      --experiment nightly

  # Reasoning + judge only (skip retrieval):
  python scripts/run_full_evaluation.py \\
      --reasoning-dataset tests/eval/reasoning/demo.json \\
      --judge rule

Notes
-----
- Retrieval evaluation requires a live database and embedding service.
  If those are unavailable, retrieval is automatically skipped with a
  warning.  All other stages that do not depend on retrieval continue.
- ``--judge rule`` uses the deterministic RuleJudge (Phase 20B, no LLM
  call, no OpenAI key required).  ``--judge none`` skips judging entirely.
- Results are stored in InMemory repositories for this run only.  Pass
  ``--no-persist`` to suppress even that (useful in dry-run mode).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import traceback
from pathlib import Path


def _load_gold_dataset(path: str) -> object:
    """Load and parse a Gold Dataset v2 JSON file.  Returns None on error."""
    from app.evaluation.gold_loader import (
        GoldDatasetParseError,
        GoldDatasetValidationError,
        load_gold_dataset,
    )
    try:
        return load_gold_dataset(Path(path))
    except (GoldDatasetParseError, GoldDatasetValidationError, FileNotFoundError) as exc:
        print(f"[ERROR] Could not load retrieval dataset {path!r}: {exc}", file=sys.stderr)
        return None


def _load_reasoning_dataset(path: str) -> object:
    """Load a ReasoningGoldDataset from a JSON file.  Returns None on error."""
    from app.evaluation.reasoning_dataset import ReasoningGoldDataset, InvestigationScenario
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        scenarios = tuple(
            InvestigationScenario(**s) for s in raw.get("scenarios", [])
        )
        return ReasoningGoldDataset(
            version=raw["version"],
            description=raw["description"],
            created_at=raw["created_at"],
            scenarios=scenarios,
            author=raw.get("author"),
        )
    except (FileNotFoundError, KeyError, TypeError) as exc:
        print(f"[ERROR] Could not load reasoning dataset {path!r}: {exc}", file=sys.stderr)
        return None


def _build_judge(judge_arg: str) -> object:
    """Return a Judge instance for the given CLI argument, or None."""
    if judge_arg == "none":
        return None
    if judge_arg == "rule":
        from app.evaluation.rule_judge import RuleJudge
        return RuleJudge()
    print(f"[ERROR] Unknown judge type {judge_arg!r}. Use 'rule' or 'none'.", file=sys.stderr)
    return None


def _build_search_service() -> object:
    """Attempt to construct an IncidentSearchService.  Returns None if
    the database or embedding service is unavailable.
    """
    try:
        from app.core.config import settings
        from app.db.session import SessionLocal
        from app.services.embedding_service import EmbeddingService
        from app.services.search import IncidentSearchService

        db = SessionLocal()
        return IncidentSearchService(db=db, embedding_service=EmbeddingService())
    except Exception:  # noqa: BLE001
        return None


def _build_orchestrator() -> object:
    """Attempt to construct a MultiAgentInvestigationOrchestrator.
    Returns None if required services (LLM, DB) are unavailable.
    """
    try:
        from app.db.session import SessionLocal
        from app.services.investigation_orchestrator import (
            MultiAgentInvestigationOrchestrator,
            OrchestratorConfig,
        )
        from app.services.llm_service import LLMService
        from app.services.search import IncidentSearchService
        from app.services.embedding_service import EmbeddingService

        db = SessionLocal()
        llm = LLMService()
        search = IncidentSearchService(db=db, embedding_service=EmbeddingService())
        return MultiAgentInvestigationOrchestrator(search, llm)
    except Exception:  # noqa: BLE001
        return None


def _fmt_opt(value: float | None, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}" if value is not None else "n/a"


def _print_results(result: object) -> None:
    from app.evaluation.evaluation_pipeline import EvaluationPipelineResult

    r: EvaluationPipelineResult = result  # type: ignore[assignment]
    s = r.execution_summary

    print()
    print("=" * 60)
    print(f"  Evaluation complete in {s.duration_seconds:.2f}s")
    print("=" * 60)

    # ── Retrieval ───────────────────────────────────────────────────────────────
    if r.retrieval_report is not None:
        agg = r.retrieval_report.aggregate_metrics
        reg = r.retrieval_regression
        verdict = reg.verdict.value if reg is not None else "no baseline"
        print()
        print("  Retrieval")
        print(f"    Queries evaluated : {r.retrieval_report.num_evaluated}")
        print(f"    Queries skipped   : {r.retrieval_report.num_skipped}")
        print(f"    Recall@K          : {_fmt_opt(agg.mean_recall_at_k)}")
        print(f"    MRR               : {_fmt_opt(agg.mean_reciprocal_rank)}")
        print(f"    NDCG@K            : {_fmt_opt(agg.mean_ndcg_at_k)}")
        print(f"    Regression verdict: {verdict}")

    # ── Reasoning ───────────────────────────────────────────────────────────────
    if r.reasoning_report is not None:
        m = r.reasoning_report.metrics
        reg_r = r.reasoning_regression
        verdict_r = reg_r.verdict.value if reg_r is not None else "no baseline"
        print()
        print("  Reasoning")
        print(f"    Scenarios         : {m.num_scenarios}")
        print(f"    Planner accuracy  : {_fmt_opt(m.planner_accuracy)}")
        print(f"    Hypothesis recall : {_fmt_opt(m.hypothesis_recall)}")
        print(f"    Decision accuracy : {_fmt_opt(m.decision_accuracy)}")
        print(f"    Convergence rate  : {_fmt_opt(m.convergence_rate)}")
        print(f"    Regression verdict: {verdict_r}")

    # ── Judge ───────────────────────────────────────────────────────────────────
    if r.judge_report is not None and r.judge_report.judge_aggregate is not None:
        agg_j = r.judge_report.judge_aggregate
        trust = (
            r.judge_validation_report.overall_trustworthiness.value
            if r.judge_validation_report is not None else "n/a"
        )
        print()
        print("  Judge")
        print(f"    Evaluations       : {agg_j.num_evaluations}")
        print(f"    Mean session score: {_fmt_opt(agg_j.mean_session_score, 2)}")
        print(f"    Trustworthiness   : {trust}")

    # ── Quality ─────────────────────────────────────────────────────────────────
    if r.quality_report is not None:
        qr = r.quality_report
        print()
        print("  Quality")
        print(f"    {qr.overall_summary}")
        if qr.recommendations:
            print("    Top recommendations:")
            for rec in qr.recommendations[:3]:
                short = textwrap.shorten(rec.recommended_action, 60)
                print(f"      [{rec.priority.value}] {short}")

    # ── Warnings / Errors ───────────────────────────────────────────────────────
    if s.warnings:
        print()
        print("  Warnings:")
        for w in s.warnings:
            print(f"    - {w}")
    if s.errors:
        print()
        print("  Errors:")
        for e in s.errors:
            print(f"    ! {e}")

    print()
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full evaluation pipeline (Phase 21E).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--retrieval-dataset", metavar="PATH",
        help="Path to a Gold Dataset v2 JSON file (Phase 16B format).",
    )
    parser.add_argument(
        "--reasoning-dataset", metavar="PATH",
        help="Path to a ReasoningGoldDataset JSON file (Phase 20A format).",
    )
    parser.add_argument(
        "--judge", choices=["rule", "none"], default="none",
        help="Judge implementation to use.  'rule' requires no LLM key.",
    )
    parser.add_argument(
        "--experiment", default="default", metavar="NAME",
        help="Experiment name used for benchmark grouping (default: 'default').",
    )
    parser.add_argument(
        "--k", type=int, default=10, metavar="K",
        help="Retrieval cutoff K (default: 10).",
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Build result objects in memory but do not call repo.save().",
    )
    parser.add_argument(
        "--skip-retrieval", action="store_true",
        help="Disable retrieval evaluation even if a dataset is supplied.",
    )
    parser.add_argument(
        "--skip-reasoning", action="store_true",
        help="Disable reasoning evaluation even if a dataset is supplied.",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Disable judge validation step.",
    )

    args = parser.parse_args(argv)

    from app.evaluation.benchmark import InMemoryBenchmarkRepository
    from app.evaluation.reasoning_benchmark import InMemoryReasoningBenchmarkRepository
    from app.evaluation.judge_benchmark import InMemoryJudgedReasoningBenchmarkRepository
    from app.evaluation.evaluation_pipeline import (
        EvaluationPipeline,
        EvaluationPipelineConfig,
        PipelineInputs,
        PipelineRepositories,
    )

    gold_dataset = _load_gold_dataset(args.retrieval_dataset) if args.retrieval_dataset else None
    reasoning_dataset = (
        _load_reasoning_dataset(args.reasoning_dataset)
        if args.reasoning_dataset else None
    )
    judge = _build_judge(args.judge)

    # Services: attempt construction; pipeline will skip stages gracefully if
    # they return None.
    search_service = _build_search_service() if gold_dataset is not None else None
    orchestrator = _build_orchestrator() if reasoning_dataset is not None else None

    if gold_dataset is not None and search_service is None:
        print(
            "[WARNING] Retrieval dataset loaded but database/embedding service unavailable."
            " Retrieval stage will be skipped.",
            file=sys.stderr,
        )
    if reasoning_dataset is not None and orchestrator is None:
        print(
            "[WARNING] Reasoning dataset loaded but LLM/database unavailable."
            " Reasoning stage will be skipped.",
            file=sys.stderr,
        )

    config = EvaluationPipelineConfig(
        experiment_name=args.experiment,
        run_retrieval=not args.skip_retrieval,
        run_reasoning=not args.skip_reasoning,
        run_judge=args.judge != "none",
        run_failure_analysis=True,
        run_validation=not args.skip_validation,
        persist_results=not args.no_persist,
        retrieval_k=args.k,
    )

    repositories = PipelineRepositories(
        retrieval_repo=InMemoryBenchmarkRepository(),
        reasoning_repo=InMemoryReasoningBenchmarkRepository(),
        judged_repo=InMemoryJudgedReasoningBenchmarkRepository(),
    )

    pipeline = EvaluationPipeline(config=config, repositories=repositories)

    try:
        result = pipeline.run(PipelineInputs(
            gold_dataset=gold_dataset,
            search_service=search_service,
            reasoning_dataset=reasoning_dataset,
            orchestrator=orchestrator,
            judge=judge,
        ))
    except Exception:  # noqa: BLE001
        print("[FATAL] Pipeline raised an unexpected exception:", file=sys.stderr)
        traceback.print_exc()
        return 1

    _print_results(result)

    if result.execution_summary.errors:
        return 2  # completed with errors — callers can detect
    return 0


if __name__ == "__main__":
    sys.exit(main())
