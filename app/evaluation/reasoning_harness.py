"""Reasoning Evaluation Harness (Phase 20A).

Evaluates investigation QUALITY — independent of retrieval quality, which
Phases 16-18 already cover. This module implements NONE of the reasoning
itself; it only runs an already-built investigation orchestrator
(``app.services.investigation_orchestrator.MultiAgentInvestigationOrchestrator``,
Phase 19D, unmodified — or any object satisfying the same
``investigate(problem, *, n_hypotheses=, routing_observation=) ->
InvestigationSession`` duck-typed contract, mirroring the way Phase 16D's
``evaluate()`` accepts anything satisfying ``IncidentSearchService``'s
shape) against an ``InvestigationScenario`` and DETERMINISTICALLY judges
the result against the scenario's expectations. No LLM call is made by
this module itself — every LLM call happens inside the orchestrator it is
handed, exactly once per investigation iteration, exactly as Phase 19D
already does.

# Updated architecture

```
              ReasoningGoldDataset (Phase 20A, this phase)
                          │
                          ▼
        for each InvestigationScenario:
                          │
                          ▼
        orchestrator.investigate(scenario.problem, ...)   [19D, unmodified
                          │                                  - the ONLY
                          ▼                                  reasoning that
                  InvestigationSession                       runs]
                          │
                          ▼
        _judge(scenario, session)    (THIS PHASE - pure comparison,
                          │            no reasoning, no LLM call)
                          ▼
                  InvestigationResult
                          │
                          ▼
        aggregate across every InvestigationResult
                          │
                          ▼
              InvestigationEvaluationReport
```

# Investigation evaluation lifecycle

```
evaluate_reasoning_dataset(dataset, orchestrator, *, n_hypotheses=)
  1. for each InvestigationScenario in dataset.scenarios:
       a. session = orchestrator.investigate(scenario.problem,
          n_hypotheses=n_hypotheses)               [19D, unmodified]
       b. actual_strategy = session.iterations[0].plan.strategy.value
          (the planner is a deterministic, pure function of
          problem+retrieved_incidents+routing_observation, Phase 19B — every
          iteration replans identically, so the first iteration's plan is
          representative; see "Metric definitions")
       c. all_hypotheses = every hypothesis generated across every
          recorded iteration (session.iterations[*].hypotheses)
       d. compare against scenario.expected_* (see "Metric definitions")
       e. -> InvestigationResult (immutable, includes ``session`` by
          reference - the same "denormalized for convenience" choice Phase
          16E/16F already make for embedded reports)
  2. aggregate every InvestigationResult into a ReasoningMetrics
  3. -> InvestigationEvaluationReport
```

# Metric definitions

- **Planner accuracy** — ``actual_strategy == scenario.expected_strategy``
  (string equality; ``actual_strategy`` is the FIRST iteration's
  ``plan.strategy.value``).
- **Hypothesis recall** — ``True`` if ``scenario.expected_root_causes`` is
  empty (nothing was expected to be found - vacuously satisfied, mirroring
  Phase 16C's treatment of no-match-expected queries), else ``True`` iff
  AT LEAST ONE generated hypothesis (across every iteration) matches at
  least one expected root cause via ``_root_cause_matches`` (see "Risks
  discovered" for why this is a heuristic, not exact-string, match).
- **Hypothesis precision** — the fraction of ALL generated hypotheses
  (across every iteration) that match at least one expected root cause.
  ``None`` (undefined, never coerced to 0.0) when zero hypotheses were
  generated at all, OR when ``scenario.expected_root_causes`` is empty
  (there is no defined "acceptable" set to measure against) — the same
  "undefined, not zero" convention Phase 16C/16D already use for
  metrics with no defined denominator.
- **Decision accuracy** — if ``expected_root_causes`` is empty, correct
  iff the final report's ``selected_hypothesis is None`` (the
  investigation correctly stayed uncertain); otherwise correct iff
  ``selected_hypothesis is not None`` AND its ``root_cause`` matches an
  expected root cause.
- **Critic accuracy** — ``session's final critique verdict.value ==
  scenario.expected_verdict``.
- **Stopping accuracy** — ``session.stopping_reason.value ==
  scenario.expected_stopping_reason``.
- **Iteration count** — ``session.total_iterations``, reported per
  scenario and averaged dataset-wide (``mean_iteration_count``); there is
  no "correct" iteration count, only a measured one.
- **Convergence rate** — the fraction of scenarios whose
  ``stopping_reason != StoppingReason.MAX_ITERATIONS`` — i.e. the
  orchestrator decided to stop on its own (approved, no progress, or no
  new hypotheses) rather than being cut off by the iteration budget. A
  scenario that legitimately SHOULD exhaust ``max_iterations`` (e.g. an
  intentionally unresolvable scenario) still counts as "did not converge"
  by this definition — convergence here measures orchestrator behavior,
  not whether that behavior was the scenario's expected outcome (stopping
  accuracy already measures that).

Dataset-wide ``ReasoningMetrics`` average ``planner_accuracy``,
``decision_accuracy``, ``critic_accuracy``, ``stopping_accuracy``, and
``convergence_rate`` over every scenario; ``hypothesis_recall``/
``hypothesis_precision`` are averaged only over scenarios where they are
defined (non-``None``) — the same "mean over defined values only"
convention as ``AggregateMetrics`` (Phase 16D).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import time
from typing import Protocol

from app.evaluation.reasoning_dataset import InvestigationScenario, ReasoningGoldDataset
from app.services.hypothesis_investigation import DEFAULT_HYPOTHESIS_COUNT
from app.services.investigation_orchestrator import InvestigationSession, StoppingReason


class _Orchestrator(Protocol):
    def investigate(
        self, problem: str, *, n_hypotheses: int = ..., routing_observation=...
    ) -> InvestigationSession: ...


# ── Report data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReasoningMetrics:
    """Aggregate reasoning-quality statistics over a set of
    ``InvestigationResult``s — see module docstring's "Metric definitions".
    """

    num_scenarios: int
    planner_accuracy: float | None
    hypothesis_recall: float | None
    hypothesis_precision: float | None
    decision_accuracy: float | None
    critic_accuracy: float | None
    stopping_accuracy: float | None
    convergence_rate: float | None
    mean_iteration_count: float | None


@dataclass(frozen=True)
class InvestigationResult:
    """The judged outcome of running ONE ``InvestigationScenario`` through
    an orchestrator. ``session`` is embedded by reference (denormalized,
    matching Phase 16E/16F's convention) so a reader can drill into the
    full iteration history without re-running anything.
    """

    scenario_id: str
    problem: str
    expected_strategy: str
    expected_root_causes: tuple[str, ...]
    expected_verdict: str
    expected_stopping_reason: str
    actual_strategy: str
    actual_root_causes: tuple[str, ...]
    actual_verdict: str
    actual_stopping_reason: str
    total_iterations: int
    planner_correct: bool
    hypothesis_recall_hit: bool
    hypothesis_precision: float | None
    decision_correct: bool
    critic_correct: bool
    stopping_correct: bool
    converged: bool
    session: InvestigationSession
    explanation: tuple[str, ...]


@dataclass(frozen=True)
class InvestigationEvaluationReport:
    """The complete, immutable result of evaluating a whole
    ``ReasoningGoldDataset`` — the reasoning-layer analogue of Phase 16D's
    ``EvaluationReport``.
    """

    dataset_version: str
    dataset_description: str
    n_hypotheses: int
    results: tuple[InvestigationResult, ...]
    metrics: ReasoningMetrics
    started_at: str
    finished_at: str
    duration_seconds: float


# ── Root-cause matching ─────────────────────────────────────────────────────────


def _root_cause_matches(root_cause: str, expected: Sequence[str]) -> bool:
    """Case-insensitive substring match, checked in both directions (the
    expected phrase may be a short keyword inside a longer generated
    sentence, or vice versa). A deliberate heuristic, not exact-string
    equality — see module docstring's "Risks discovered" for why this can
    both over- and under-match.
    """
    haystack = root_cause.lower()
    return any(
        candidate.lower() in haystack or haystack in candidate.lower()
        for candidate in expected
        if candidate
    )


# ── Per-scenario evaluation ─────────────────────────────────────────────────────


def evaluate_scenario(
    scenario: InvestigationScenario,
    orchestrator: _Orchestrator,
    *,
    n_hypotheses: int = DEFAULT_HYPOTHESIS_COUNT,
) -> InvestigationResult:
    """Run ``scenario`` through ``orchestrator`` and judge the result. See
    module docstring's "Metric definitions".
    """
    session = orchestrator.investigate(scenario.problem, n_hypotheses=n_hypotheses)

    actual_strategy = session.iterations[0].plan.strategy.value
    all_hypotheses = tuple(
        hypothesis
        for iteration in session.iterations
        for hypothesis in iteration.hypotheses
    )
    actual_root_causes = tuple(hypothesis.root_cause for hypothesis in all_hypotheses)
    actual_verdict = session.final_report.critique.verdict.value
    actual_stopping_reason = session.stopping_reason.value
    selected = session.final_report.investigation.selected_hypothesis

    planner_correct = actual_strategy == scenario.expected_strategy

    if not scenario.expected_root_causes:
        hypothesis_recall_hit = True
        hypothesis_precision = None
        decision_correct = selected is None
    else:
        matching = [
            root_cause for root_cause in actual_root_causes
            if _root_cause_matches(root_cause, scenario.expected_root_causes)
        ]
        hypothesis_recall_hit = bool(matching)
        hypothesis_precision = (
            len(matching) / len(actual_root_causes) if actual_root_causes else None
        )
        decision_correct = selected is not None and _root_cause_matches(
            selected.root_cause, scenario.expected_root_causes
        )

    critic_correct = actual_verdict == scenario.expected_verdict
    stopping_correct = actual_stopping_reason == scenario.expected_stopping_reason
    converged = session.stopping_reason != StoppingReason.MAX_ITERATIONS

    explanation = _explain_failures(
        scenario,
        actual_strategy=actual_strategy,
        actual_root_causes=actual_root_causes,
        actual_verdict=actual_verdict,
        actual_stopping_reason=actual_stopping_reason,
        planner_correct=planner_correct,
        hypothesis_recall_hit=hypothesis_recall_hit,
        decision_correct=decision_correct,
        critic_correct=critic_correct,
        stopping_correct=stopping_correct,
        selected_root_cause=selected.root_cause if selected else None,
    )

    return InvestigationResult(
        scenario_id=scenario.id,
        problem=scenario.problem,
        expected_strategy=scenario.expected_strategy,
        expected_root_causes=scenario.expected_root_causes,
        expected_verdict=scenario.expected_verdict,
        expected_stopping_reason=scenario.expected_stopping_reason,
        actual_strategy=actual_strategy,
        actual_root_causes=actual_root_causes,
        actual_verdict=actual_verdict,
        actual_stopping_reason=actual_stopping_reason,
        total_iterations=session.total_iterations,
        planner_correct=planner_correct,
        hypothesis_recall_hit=hypothesis_recall_hit,
        hypothesis_precision=hypothesis_precision,
        decision_correct=decision_correct,
        critic_correct=critic_correct,
        stopping_correct=stopping_correct,
        converged=converged,
        session=session,
        explanation=explanation,
    )


def _explain_failures(
    scenario: InvestigationScenario,
    *,
    actual_strategy: str,
    actual_root_causes: tuple[str, ...],
    actual_verdict: str,
    actual_stopping_reason: str,
    planner_correct: bool,
    hypothesis_recall_hit: bool,
    decision_correct: bool,
    critic_correct: bool,
    stopping_correct: bool,
    selected_root_cause: str | None,
) -> tuple[str, ...]:
    """Every failed investigation explains itself - see module docstring's
    "Explainability" expectations from the brief. Returns an empty tuple
    for a scenario with no failures.
    """
    explanations: list[str] = []
    if not planner_correct:
        explanations.append(
            f"planner mismatch: selected strategy {actual_strategy!r}, "
            f"expected {scenario.expected_strategy!r}"
        )
    if scenario.expected_root_causes and not hypothesis_recall_hit:
        explanations.append(
            f"missing hypotheses: none of {list(actual_root_causes)} matched any expected "
            f"root cause in {list(scenario.expected_root_causes)}"
        )
    if not decision_correct:
        if scenario.expected_root_causes and selected_root_cause is None:
            explanations.append(
                "incorrect rejection: investigation ended uncertain, but a hypothesis "
                f"matching {list(scenario.expected_root_causes)} was expected to be accepted"
            )
        elif scenario.expected_root_causes:
            explanations.append(
                f"incorrect acceptance: accepted {selected_root_cause!r}, which does not "
                f"match any expected root cause in {list(scenario.expected_root_causes)}"
            )
        else:
            explanations.append(
                f"incorrect acceptance: accepted {selected_root_cause!r}, but no hypothesis "
                "was expected to be accepted for this scenario"
            )
    if not critic_correct:
        explanations.append(
            f"critic verdict mismatch: got {actual_verdict!r}, "
            f"expected {scenario.expected_verdict!r}"
        )
    if not stopping_correct:
        explanations.append(
            f"incorrect stopping reason: got {actual_stopping_reason!r}, "
            f"expected {scenario.expected_stopping_reason!r}"
        )
        if (
            scenario.expected_stopping_reason != StoppingReason.MAX_ITERATIONS.value
            and actual_stopping_reason == StoppingReason.MAX_ITERATIONS.value
        ):
            explanations.append(
                "missing convergence: the orchestrator exhausted max_iterations instead of "
                "converging on its own"
            )
    return tuple(explanations)


# ── Dataset-wide evaluation ──────────────────────────────────────────────────────


def evaluate_reasoning_dataset(
    dataset: ReasoningGoldDataset,
    orchestrator: _Orchestrator,
    *,
    n_hypotheses: int = DEFAULT_HYPOTHESIS_COUNT,
) -> InvestigationEvaluationReport:
    """Run every scenario in ``dataset`` through ``orchestrator`` and
    return the aggregated ``InvestigationEvaluationReport``.
    """
    started_at = datetime.now(UTC)
    started_perf = time.monotonic()

    results = tuple(
        evaluate_scenario(scenario, orchestrator, n_hypotheses=n_hypotheses)
        for scenario in dataset.scenarios
    )

    finished_at = datetime.now(UTC)
    return InvestigationEvaluationReport(
        dataset_version=dataset.version,
        dataset_description=dataset.description,
        n_hypotheses=n_hypotheses,
        results=results,
        metrics=_aggregate(results),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=time.monotonic() - started_perf,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(results: Sequence[InvestigationResult]) -> ReasoningMetrics:
    recall_scope = [r for r in results if r.expected_root_causes]
    precision_values = [
        r.hypothesis_precision for r in results if r.hypothesis_precision is not None
    ]

    return ReasoningMetrics(
        num_scenarios=len(results),
        planner_accuracy=_mean([1.0 if r.planner_correct else 0.0 for r in results]),
        hypothesis_recall=_mean(
            [1.0 if r.hypothesis_recall_hit else 0.0 for r in recall_scope]
        ),
        hypothesis_precision=_mean(precision_values),
        decision_accuracy=_mean([1.0 if r.decision_correct else 0.0 for r in results]),
        critic_accuracy=_mean([1.0 if r.critic_correct else 0.0 for r in results]),
        stopping_accuracy=_mean([1.0 if r.stopping_correct else 0.0 for r in results]),
        convergence_rate=_mean([1.0 if r.converged else 0.0 for r in results]),
        mean_iteration_count=_mean([float(r.total_iterations) for r in results]),
    )
