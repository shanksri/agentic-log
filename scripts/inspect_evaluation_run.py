#!/usr/bin/env python
"""Inspection CLI for persisted evaluation runs (Phase 21F).

Usage
-----

  # Print a report for the most recent run:
  python scripts/inspect_evaluation_run.py latest

  # Print a report for a specific run by ID:
  python scripts/inspect_evaluation_run.py 20260701_213015_nightly

  # Change the storage directory (default: .evaluation_runs):
  python scripts/inspect_evaluation_run.py latest --dir /data/eval_runs

  # Show only failed queries:
  python scripts/inspect_evaluation_run.py latest --failed-queries

  # List all known runs:
  python scripts/inspect_evaluation_run.py --list

Exit codes
----------
  0 — success
  1 — run not found or other error
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path


def _fmt(value: float | None, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}" if value is not None else "n/a"


def _section(title: str) -> None:
    print()
    print(f"  {title}")
    print(f"  {'─' * len(title)}")


def _print_run_report(run: object) -> None:
    from app.evaluation.experiment_tracking import ExperimentRun
    r: ExperimentRun = run  # type: ignore[assignment]
    m = r.metadata

    print()
    print("=" * 62)
    print(f"  Run: {m.run_id}")
    print(f"  Experiment : {m.experiment_name}")
    print(f"  Timestamp  : {m.timestamp}")
    if m.git_commit:
        print(f"  Git commit : {m.git_commit}")
    print(f"  Duration   : {m.duration:.2f}s")
    if m.retrieval_dataset_version:
        print(f"  Retrieval dataset  : {m.retrieval_dataset_version}")
    if m.reasoning_dataset_version:
        print(f"  Reasoning dataset  : {m.reasoning_dataset_version}")
    if m.judge:
        print(f"  Judge      : {m.judge}")
    print("=" * 62)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    ret = r.retrieval_report
    if ret:
        _section("Retrieval")
        agg = ret.get("aggregate_metrics") or {}
        print(f"    Queries evaluated : {ret.get('num_evaluated', 'n/a')}")
        print(f"    Queries skipped   : {ret.get('num_skipped', 'n/a')}")
        print(f"    Recall@K          : {_fmt(agg.get('mean_recall_at_k'))}")
        print(f"    MRR               : {_fmt(agg.get('mean_reciprocal_rank'))}")
        print(f"    NDCG@K            : {_fmt(agg.get('mean_ndcg_at_k'))}")
        if r.regression_report:
            v = r.regression_report.get("verdict") or r.regression_report.get("overall", {}).get("verdict")
            print(f"    Regression        : {v or 'n/a'}")

        if r.failed_queries:
            _section(f"Top 10 Failed Retrieval Queries ({len(r.failed_queries)} total)")
            for q in r.failed_queries[:10]:
                qid = q.get("query_id") or q.get("id") or "?"
                metric = q.get("metric") or {}
                recall = _fmt(metric.get("recall_at_k"))
                skip = " (skipped)" if q.get("skipped") else ""
                print(f"    [{qid}]  recall={recall}{skip}")

    # ── Reasoning ──────────────────────────────────────────────────────────────
    reas = r.reasoning_report
    if reas:
        _section("Reasoning")
        m2 = reas.get("metrics") or {}
        print(f"    Scenarios         : {m2.get('num_scenarios', 'n/a')}")
        print(f"    Planner accuracy  : {_fmt(m2.get('planner_accuracy'))}")
        print(f"    Decision accuracy : {_fmt(m2.get('decision_accuracy'))}")
        print(f"    Convergence rate  : {_fmt(m2.get('convergence_rate'))}")

        if r.failed_reasoning:
            _section(f"Top Reasoning Failures ({len(r.failed_reasoning)} total)")
            for res in r.failed_reasoning[:10]:
                sid = res.get("scenario_id") or "?"
                conv = res.get("converged")
                dec = res.get("decision_correct")
                plan = res.get("planner_correct")
                tags = []
                if conv is False:
                    tags.append("no-convergence")
                if dec is False:
                    tags.append("wrong-decision")
                if plan is False:
                    tags.append("wrong-plan")
                print(f"    [{sid}]  {', '.join(tags) or 'failed'}")

    # ── Judge ──────────────────────────────────────────────────────────────────
    judge = r.judge_report
    if judge:
        _section("Judge")
        agg_j = judge.get("judge_aggregate") or {}
        print(f"    Evaluations       : {agg_j.get('num_evaluations', 'n/a')}")
        print(f"    Mean session score: {_fmt(agg_j.get('mean_session_score'), 2)}")

        if r.judge_disagreements:
            _section(f"Judge Disagreements ({len(r.judge_disagreements)} total)")
            for e in r.judge_disagreements[:10]:
                stage = e.get("stage") or "?"
                score_block = e.get("score") or {}
                score_val = _fmt(score_block.get("value"), 1)
                exp = e.get("explanation") or ""
                short = textwrap.shorten(exp, 60)
                print(f"    [{stage}]  score={score_val}  {short}")

        if r.validation_report:
            trust = r.validation_report.get("overall_trustworthiness", "n/a")
            print(f"    Trustworthiness   : {trust}")

    # ── Quality ────────────────────────────────────────────────────────────────
    qual = r.quality_report
    if qual:
        _section("Quality")
        print(f"    {qual.get('overall_summary', '')}")
        recs = qual.get("recommendations") or []
        if recs:
            print("    Top recommendations:")
            for rec in recs[:3]:
                action = rec.get("recommended_action") or ""
                prio = rec.get("priority") or "??"
                short = textwrap.shorten(action, 58)
                print(f"      [{prio}] {short}")

    # ── Warnings / Errors ──────────────────────────────────────────────────────
    summ = r.summary or {}
    warnings = summ.get("warnings") or []
    errors = summ.get("errors") or []
    if warnings:
        _section("Warnings")
        for w in warnings:
            print(f"    - {w}")
    if errors:
        _section("Errors")
        for e in errors:
            print(f"    ! {e}")

    print()
    print("=" * 62)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a persisted evaluation run (Phase 21F).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_id", nargs="?",
        help="Run ID to inspect, or 'latest'.  Required unless --list is passed.",
    )
    parser.add_argument(
        "--dir", default=".evaluation_runs", metavar="DIR",
        help="Base directory for experiment storage (default: .evaluation_runs).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all known run IDs (oldest first) and exit.",
    )
    parser.add_argument(
        "--failed-queries", action="store_true",
        help="Print failed queries section only.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print aggregate statistics across the full run history.",
    )

    args = parser.parse_args(argv)

    from app.evaluation.experiment_tracking import ExperimentRepository

    repo = ExperimentRepository(base_dir=Path(args.dir))

    if args.list:
        runs = repo.list_runs()
        if not runs:
            print("No runs found.")
            return 0
        print(f"{len(runs)} run(s):")
        for rid in runs:
            run = repo.load(rid)
            ts = run.metadata.timestamp if run else "?"
            print(f"  {rid}  ({ts})")
        return 0

    if args.stats:
        stats = repo.stats()
        print()
        print("  Experiment history statistics")
        print(f"    Total runs          : {stats.total_runs}")
        print(f"    Best MRR            : {_fmt(stats.best_mrr)}")
        print(f"    Best NDCG           : {_fmt(stats.best_ndcg)}")
        print(f"    Best reasoning acc. : {_fmt(stats.best_reasoning_accuracy)}")
        print(f"    Latest run          : {stats.latest_run or 'none'}")
        return 0

    if not args.run_id:
        parser.print_help()
        return 1

    if args.run_id == "latest":
        run = repo.latest()
        if run is None:
            print("[ERROR] No runs found in", args.dir, file=sys.stderr)
            return 1
    else:
        run = repo.load(args.run_id)
        if run is None:
            print(f"[ERROR] Run {args.run_id!r} not found in {args.dir}", file=sys.stderr)
            return 1

    if args.failed_queries:
        if not run.failed_queries:
            print("No failed queries in this run.")
        else:
            print(f"{len(run.failed_queries)} failed query/queries:")
            for q in run.failed_queries:
                qid = q.get("query_id") or q.get("id") or "?"
                metric = q.get("metric") or {}
                recall = _fmt(metric.get("recall_at_k"))
                print(f"  [{qid}]  recall={recall}")
        return 0

    _print_run_report(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
