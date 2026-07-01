"""End-to-End Evaluation Pipeline (Phase 21E).

Orchestration layer that wires every existing evaluation component into one
callable sequence.  This module introduces NO new metrics, NO new retrieval
algorithms, NO reasoning logic, and NO prompt tuning — it only calls the
existing public APIs in the correct order.

# Architecture

```
PipelineInputs  (datasets + services — passed at call time)
EvaluationPipelineConfig  (flags + tuning — set at construction time)
PipelineRepositories      (persistence — injected at construction time)
        │
        ▼
EvaluationPipeline.run(inputs)
        │
        ├─ Step 1-4  Retrieval
        │     harness.evaluate()
        │     → regression.compare()        (if prior run exists)
        │     → benchmark.create_*()  +  repo.save()
        │
        ├─ Step 5-7  Reasoning
        │     evaluate_reasoning_dataset()
        │     → compare_reasoning()         (if prior run exists)
        │     → create_reasoning_*()  +  repo.save()
        │
        ├─ Step 8    Judge
        │     judge.evaluate_session()  per InvestigationResult
        │     → create_judged_benchmark_run()  +  repo.save()
        │
        ├─ Step 9    Failure Analysis & Quality Report
        │     analyze_retrieval/reasoning/judge_failures()
        │     → cluster_failures()
        │     → generate_recommendations()
        │     → build_quality_report()
        │
        └─ Step 10   Validation
              build_validation_report_from_benchmarks()
        │
        ▼
EvaluationPipelineResult
```

# Design constraints

- The pipeline MUST NOT duplicate evaluation, regression, or benchmark logic —
  it only calls existing public APIs.
- The pipeline MUST NOT import ``IncidentSearchService`` internals or any
  agent/orchestrator/judge implementation class.  It receives services as
  opaque ``object`` arguments and forwards them verbatim to the existing
  harnesses (which know their own types).
- Every stage is individually skippable via ``EvaluationPipelineConfig``
  flags.  A stage that is skipped records a ``warning`` in
  ``ExecutionSummary`` rather than raising.
- A non-fatal error in one stage (e.g. retrieval fails) records an ``error``
  and lets downstream stages that do not depend on that stage's output
  continue.  A fatal error that cannot be recovered from is re-raised after
  recording.

# Risks discovered

- The pipeline passes ``search_service`` and ``orchestrator`` as ``object``
  at the type level; a misconfigured caller will get a runtime
  ``AttributeError`` deep inside the harness, not a clear type error at
  pipeline construction time.
- Retrieval regression requires the PREVIOUS benchmark's EvaluationReport
  to have the same gold version and K as the current run; an incompatible
  comparison produces ``verdict=INCOMPATIBLE`` (not an error) and is stored
  alongside the run with that flag set.
- Judge evaluation is applied only to ``evaluate_session`` (one call per
  InvestigationResult); the five per-stage evaluate_* calls are available
  to callers who construct bespoke pipelines but are not invoked here, to
  keep the number of LLM calls bounded and the default path fast.
- Validation (Step 10) is only meaningful with >= 2 judged benchmark runs;
  with fewer it returns ``INSUFFICIENT_DATA`` which is correct but
  potentially surprising to first-time users.
- ``persist_results=False`` skips all ``repo.save()`` calls but DOES still
  build BenchmarkRun / ReasoningBenchmarkRun / JudgedReasoningBenchmarkRun
  objects in memory (so Step 9/10 can still run in the same process); those
  objects are not written to any repository.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.evaluation.ai_quality_report import AIQualityReport, build_quality_report
from app.evaluation.benchmark import (
    BenchmarkRepository,
    BenchmarkRun,
    create_benchmark_run,
)
from app.evaluation.failure_analysis import (
    analyze_judge_failures,
    analyze_reasoning_failures,
    analyze_retrieval_failures,
    cluster_failures,
)
from app.evaluation.gold_dataset import GoldDataset
from app.evaluation.harness import EvaluationReport, evaluate
from app.evaluation.judge import Judge, JudgeEvaluation
from app.evaluation.judge_benchmark import (
    JudgedReasoningBenchmarkRepository,
    JudgedReasoningBenchmarkRun,
    aggregate_judge_evaluations,
    create_judged_benchmark_run,
)
from app.evaluation.judge_validation_report import (
    JudgeValidationReport,
    build_validation_report_from_benchmarks,
)
from app.evaluation.reasoning_benchmark import (
    ReasoningBenchmarkRepository,
    ReasoningBenchmarkRun,
    create_reasoning_benchmark_run,
)
from app.evaluation.reasoning_dataset import ReasoningGoldDataset
from app.evaluation.reasoning_harness import (
    InvestigationEvaluationReport,
    evaluate_reasoning_dataset,
)
from app.evaluation.reasoning_regression import ReasoningRegressionReport, compare_reasoning
from app.evaluation.recommendation_engine import generate_recommendations
from app.evaluation.regression import RegressionReport, compare


# ── Pipeline configuration ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationPipelineConfig:
    """Flags and tuning for one pipeline run.

    All boolean flags default to ``True`` so a caller that supplies all
    required inputs gets the full pipeline without explicitly opting into each
    stage.  Set a flag to ``False`` to skip that stage entirely.
    """

    experiment_name: str = "default"
    run_retrieval: bool = True
    run_reasoning: bool = True
    run_judge: bool = True
    run_failure_analysis: bool = True
    run_validation: bool = True
    persist_results: bool = True
    retrieval_k: int = 10
    retrieval_expand: bool = False
    retrieval_rerank: bool = False
    n_hypotheses: int = 3


# ── Repository bundle ────────────────────────────────────────────────────────────


@dataclass
class PipelineRepositories:
    """The three persistence backends the pipeline may write to.

    All are optional — ``None`` means "skip persistence for this layer."
    ``persist_results=False`` in the config overrides all three regardless
    of whether they are ``None``.
    """

    retrieval_repo: BenchmarkRepository | None = None
    reasoning_repo: ReasoningBenchmarkRepository | None = None
    judged_repo: JudgedReasoningBenchmarkRepository | None = None


# ── Pipeline inputs ──────────────────────────────────────────────────────────────


@dataclass
class PipelineInputs:
    """Runtime inputs: datasets and live services.

    ``search_service`` and ``orchestrator`` are typed as ``object`` so this
    module does not import ``IncidentSearchService`` or any agent class —
    see module docstring's "Design constraints".  The harnesses know their
    own parameter types and will raise ``AttributeError`` if the wrong object
    is passed.
    """

    gold_dataset: GoldDataset | None = None
    search_service: object = None
    reasoning_dataset: ReasoningGoldDataset | None = None
    orchestrator: object = None
    judge: Judge | None = None


# ── Execution summary ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionSummary:
    """Timing and count snapshot for one pipeline run."""

    start_time: str
    end_time: str
    duration_seconds: float
    retrieval_queries: int
    reasoning_scenarios: int
    judge_evaluations: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


# ── Pipeline result ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationPipelineResult:
    """The complete, immutable result of one end-to-end pipeline run.

    Every field is ``None`` when its stage was skipped or failed — callers
    must check for ``None`` before accessing nested attributes.
    ``execution_summary`` is always populated.
    """

    retrieval_report: EvaluationReport | None
    retrieval_regression: RegressionReport | None
    retrieval_benchmark: BenchmarkRun | None
    reasoning_report: InvestigationEvaluationReport | None
    reasoning_regression: ReasoningRegressionReport | None
    reasoning_benchmark: ReasoningBenchmarkRun | None
    judge_report: JudgedReasoningBenchmarkRun | None
    judge_validation_report: JudgeValidationReport | None
    quality_report: AIQualityReport | None
    execution_summary: ExecutionSummary


# ── Pipeline ─────────────────────────────────────────────────────────────────────


class EvaluationPipeline:
    """Orchestrates all evaluation phases in the correct order.

    Construct with a ``config`` and (optionally) ``repositories``, then call
    ``run(inputs)`` one or more times.  Each call is independent; state is
    not accumulated between calls.

    Usage::

        pipeline = EvaluationPipeline(
            config=EvaluationPipelineConfig(experiment_name="nightly"),
            repositories=PipelineRepositories(
                retrieval_repo=InMemoryBenchmarkRepository(),
                reasoning_repo=InMemoryReasoningBenchmarkRepository(),
                judged_repo=InMemoryJudgedReasoningBenchmarkRepository(),
            ),
        )
        result = pipeline.run(PipelineInputs(
            gold_dataset=my_gold_dataset,
            search_service=my_search_service,
            reasoning_dataset=my_reasoning_dataset,
            orchestrator=my_orchestrator,
            judge=RuleJudge(),
        ))
    """

    def __init__(
        self,
        config: EvaluationPipelineConfig | None = None,
        repositories: PipelineRepositories | None = None,
    ) -> None:
        self._config = config or EvaluationPipelineConfig()
        self._repos = repositories or PipelineRepositories()

    @property
    def config(self) -> EvaluationPipelineConfig:
        return self._config

    @property
    def repositories(self) -> PipelineRepositories:
        return self._repos

    def run(self, inputs: PipelineInputs) -> EvaluationPipelineResult:
        """Execute the full pipeline for ``inputs`` and return the result.

        Steps that are disabled in ``config`` or lack required inputs are
        skipped with a warning recorded in ``ExecutionSummary``.  Errors in
        one step are caught and recorded; downstream steps that do not depend
        on that step's output continue.
        """
        cfg = self._config
        repos = self._repos
        warnings: list[str] = []
        errors: list[str] = []
        start_perf = time.monotonic()
        start_time = datetime.now(UTC).isoformat()

        # ── Mutable accumulators ────────────────────────────────────────────────
        retrieval_report: EvaluationReport | None = None
        retrieval_regression: RegressionReport | None = None
        retrieval_benchmark: BenchmarkRun | None = None
        reasoning_report: InvestigationEvaluationReport | None = None
        reasoning_regression: ReasoningRegressionReport | None = None
        reasoning_benchmark: ReasoningBenchmarkRun | None = None
        judge_report: JudgedReasoningBenchmarkRun | None = None
        judge_validation: JudgeValidationReport | None = None
        quality_report: AIQualityReport | None = None
        n_judge_evals = 0

        # ── Step 1-4: Retrieval ─────────────────────────────────────────────────
        if not cfg.run_retrieval:
            warnings.append("Retrieval evaluation skipped (run_retrieval=False).")
        elif inputs.gold_dataset is None:
            warnings.append(
                "Retrieval evaluation skipped: no gold_dataset supplied in PipelineInputs."
            )
        elif inputs.search_service is None:
            warnings.append(
                "Retrieval evaluation skipped: no search_service supplied in PipelineInputs."
            )
        else:
            try:
                # Step 1-2: evaluate
                retrieval_report = evaluate(
                    inputs.gold_dataset,
                    inputs.search_service,  # type: ignore[arg-type]
                    k=cfg.retrieval_k,
                    expand=cfg.retrieval_expand,
                    rerank=cfg.retrieval_rerank,
                )
                # Step 3: regression against previous run
                if repos.retrieval_repo is not None:
                    previous = repos.retrieval_repo.latest(
                        experiment_name=cfg.experiment_name
                    )
                    if previous is not None:
                        retrieval_regression = compare(
                            previous.report, retrieval_report
                        )
                # Step 4: persist
                retrieval_benchmark = create_benchmark_run(
                    experiment_name=cfg.experiment_name,
                    report=retrieval_report,
                    regression=retrieval_regression,
                )
                if cfg.persist_results and repos.retrieval_repo is not None:
                    repos.retrieval_repo.save(retrieval_benchmark)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Retrieval evaluation failed: {exc!r}")

        # ── Step 5-7: Reasoning ─────────────────────────────────────────────────
        if not cfg.run_reasoning:
            warnings.append("Reasoning evaluation skipped (run_reasoning=False).")
        elif inputs.reasoning_dataset is None:
            warnings.append(
                "Reasoning evaluation skipped: no reasoning_dataset supplied."
            )
        elif inputs.orchestrator is None:
            warnings.append(
                "Reasoning evaluation skipped: no orchestrator supplied."
            )
        else:
            try:
                # Step 5: evaluate
                reasoning_report = evaluate_reasoning_dataset(
                    inputs.reasoning_dataset,
                    inputs.orchestrator,  # type: ignore[arg-type]
                    n_hypotheses=cfg.n_hypotheses,
                )
                # Step 6: regression against previous run
                if repos.reasoning_repo is not None:
                    previous_r = repos.reasoning_repo.latest(
                        experiment_name=cfg.experiment_name
                    )
                    if previous_r is not None:
                        reasoning_regression = compare_reasoning(
                            previous_r.report, reasoning_report
                        )
                # Step 7: persist
                reasoning_benchmark = create_reasoning_benchmark_run(
                    experiment_name=cfg.experiment_name,
                    report=reasoning_report,
                    regression=reasoning_regression,
                )
                if cfg.persist_results and repos.reasoning_repo is not None:
                    repos.reasoning_repo.save(reasoning_benchmark)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Reasoning evaluation failed: {exc!r}")

        # ── Step 8: Judge ───────────────────────────────────────────────────────
        if not cfg.run_judge:
            warnings.append("Judge evaluation skipped (run_judge=False).")
        elif inputs.judge is None:
            warnings.append(
                "Judge evaluation skipped: no judge supplied in PipelineInputs."
            )
        elif reasoning_benchmark is None:
            warnings.append(
                "Judge evaluation skipped: reasoning benchmark not available "
                "(reasoning evaluation did not produce a result)."
            )
        elif reasoning_report is None:
            warnings.append(
                "Judge evaluation skipped: reasoning report not available."
            )
        else:
            try:
                judge_evals: list[JudgeEvaluation] = []
                for result in reasoning_report.results:
                    try:
                        evaluation = inputs.judge.evaluate_session(
                            result.problem, result.session
                        )
                        judge_evals.append(evaluation)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            f"Judge.evaluate_session failed for "
                            f"scenario {result.scenario_id!r}: {exc!r}"
                        )
                n_judge_evals = len(judge_evals)
                judge_report = create_judged_benchmark_run(
                    experiment_name=cfg.experiment_name,
                    reasoning_run=reasoning_benchmark,
                    judge_evaluations=judge_evals,
                )
                if cfg.persist_results and repos.judged_repo is not None:
                    repos.judged_repo.save(judge_report)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Judge benchmark creation failed: {exc!r}")

        # ── Step 9: Failure Analysis & Quality Report ───────────────────────────
        if not cfg.run_failure_analysis:
            warnings.append(
                "Failure analysis skipped (run_failure_analysis=False)."
            )
        else:
            try:
                retrieval_reports = [retrieval_report] if retrieval_report else []
                reasoning_reports = [reasoning_report] if reasoning_report else []
                judge_evals_for_analysis: tuple[JudgeEvaluation, ...] = (
                    judge_report.judge_evaluations if judge_report else ()
                )
                regression_verdict: str | None = None
                if retrieval_regression is not None:
                    regression_verdict = retrieval_regression.verdict.value
                elif reasoning_regression is not None:
                    regression_verdict = reasoning_regression.verdict.value

                quality_report = build_quality_report(
                    retrieval_reports=retrieval_reports,
                    reasoning_reports=reasoning_reports,
                    judge_evaluations=judge_evals_for_analysis,
                    judge_errors=[e for e in errors if "Judge" in e],
                    regression_verdict=regression_verdict,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Failure analysis / quality report failed: {exc!r}")

        # ── Step 10: Judge Validation ───────────────────────────────────────────
        if not cfg.run_validation:
            warnings.append("Judge validation skipped (run_validation=False).")
        elif repos.judged_repo is None:
            warnings.append(
                "Judge validation skipped: no judged_repo in PipelineRepositories."
            )
        else:
            try:
                judge_validation = build_validation_report_from_benchmarks(
                    judged_repo=repos.judged_repo,
                    experiment_name=cfg.experiment_name,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Judge validation failed: {exc!r}")

        # ── Execution summary ───────────────────────────────────────────────────
        end_time = datetime.now(UTC).isoformat()
        duration = time.monotonic() - start_perf
        summary = ExecutionSummary(
            start_time=start_time,
            end_time=end_time,
            duration_seconds=round(duration, 4),
            retrieval_queries=(
                retrieval_report.num_evaluated + retrieval_report.num_skipped
                if retrieval_report else 0
            ),
            reasoning_scenarios=(
                reasoning_report.metrics.num_scenarios if reasoning_report else 0
            ),
            judge_evaluations=n_judge_evals,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

        return EvaluationPipelineResult(
            retrieval_report=retrieval_report,
            retrieval_regression=retrieval_regression,
            retrieval_benchmark=retrieval_benchmark,
            reasoning_report=reasoning_report,
            reasoning_regression=reasoning_regression,
            reasoning_benchmark=reasoning_benchmark,
            judge_report=judge_report,
            judge_validation_report=judge_validation,
            quality_report=quality_report,
            execution_summary=summary,
        )
