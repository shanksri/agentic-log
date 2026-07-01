"""AI Quality Intelligence — Recommendation Engine (Phase 21A).

Generates engineering recommendations EXCLUSIVELY from already-clustered
failures (``app.evaluation.failure_analysis.FailureCluster``). Never
hard-codes a recommendation unrelated to an observed failure — every
``Recommendation`` is constructed from exactly one ``FailureCluster``, and
``generate_recommendations`` produces zero recommendations when given zero
clusters. No LLM call; fully deterministic given identical clusters.

# Recommendation workflow

```
tuple[FailureCluster, ...]
            │
            ▼
for each cluster: build_recommendation(cluster)
            │
            ▼
    Recommendation(problem=, root_cause=, estimated_impact=,
                    confidence=, recommended_action=, priority=)
            │
            ▼
sort by (priority descending, estimated_impact descending)  - see
"Priority ordering" below
            │
            ▼
    tuple[Recommendation, ...]   (already priority-ordered)
```

# Field derivation (every value traceable to the cluster, never invented)

- **problem** — a plain-text statement of the cluster's ``component`` +
  ``category`` + how many failures it covers (directly from
  ``len(cluster.failures)``).
- **root_cause** — ``cluster.common_cause`` (already computed by
  ``app.evaluation.failure_analysis.cluster_failures`` from the member
  failures' own systemic cause steps) — never re-derived here.
- **estimated_impact** — ``len(cluster.failures)`` (the literal count of
  affected queries/scenarios this cluster represents — the simplest
  non-arbitrary impact measure available without inventing a weighting
  scheme).
- **confidence** — ``min(1.0, 0.3 + 0.1 * len(cluster.failures))``: a
  cluster of 1 failure could be coincidence (confidence 0.4); a cluster of
  7+ failures sharing the same component/category/systemic cause is very
  unlikely to be coincidence (confidence caps at 1.0 from 7 onward) — the
  same "more repetition = more confidence this is a real pattern, not
  noise" reasoning Phase 19D's progress-detection module already applies
  to repeated signals, adapted here to cluster size instead of iteration
  count.
- **recommended_action** — a short, deterministic, FailureCategory-keyed
  action string keyed off ``cluster.category`` (e.g.
  ``STRATEGY_MISMATCH`` -> "review planner keyword priority ordering for
  this category") — listed exhaustively in ``_ACTION_BY_CATEGORY`` below,
  one entry per ``FailureCategory`` member, so every possible cluster
  category has a defined action (no silent fallback to a generic string
  unless a genuinely new, unhandled category appears — see "Risks
  discovered").
- **priority** — derived from ``cluster.severity`` via a fixed mapping
  (``Severity.CRITICAL -> Priority.CRITICAL``, etc. — i.e. priority IS
  severity, renamed for this output type's vocabulary, never computed by
  a second, independent formula that could disagree with the severity
  this phase already assigned).

# Priority ordering

Recommendations are sorted by ``priority`` descending, then
``estimated_impact`` descending, then ``problem`` (alphabetical) as a
final deterministic tie-break — so the result is always reproducible for
the same input clusters, satisfying "the orchestration loop must be
deterministic" (the same standing requirement Phase 19D establishes for
agent orchestration, applied here to this output's ordering).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from app.evaluation.failure_analysis import FailureCategory, FailureCluster, Severity

CONFIDENCE_BASE = 0.3
CONFIDENCE_PER_FAILURE = 0.1
CONFIDENCE_MAX = 1.0


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_PRIORITY_ORDER: dict[Priority, int] = {
    Priority.LOW: 0, Priority.MEDIUM: 1, Priority.HIGH: 2, Priority.CRITICAL: 3,
}

_PRIORITY_FROM_SEVERITY: dict[Severity, Priority] = {
    Severity.LOW: Priority.LOW,
    Severity.MEDIUM: Priority.MEDIUM,
    Severity.HIGH: Priority.HIGH,
    Severity.CRITICAL: Priority.CRITICAL,
}

_ACTION_BY_CATEGORY: dict[FailureCategory, str] = {
    FailureCategory.SEARCH_FAILURE: (
        "investigate retrieval infrastructure/connectivity stability"
    ),
    FailureCategory.INCOMPLETE_RECALL: (
        "review retrieval strategy/ranking for the affected category"
    ),
    FailureCategory.UNRESOLVED_GOLD_ENTRY: "refresh the gold dataset against the live corpus",
    FailureCategory.STRATEGY_MISMATCH: (
        "review planner keyword priority ordering for this category"
    ),
    FailureCategory.MISSING_HYPOTHESIS: (
        "improve hypothesis generation diversity/coverage prompting"
    ),
    FailureCategory.DUPLICATE_HYPOTHESIS: "add duplicate-detection to hypothesis generation",
    FailureCategory.INCORRECT_DECISION: (
        "recalibrate decision acceptance threshold or evidence weighting"
    ),
    FailureCategory.INCORRECT_CRITIQUE: "recalibrate critic contradiction/margin thresholds",
    FailureCategory.NO_CONVERGENCE: (
        "review orchestrator stopping-condition priority and thresholds"
    ),
    FailureCategory.LOW_CONFIDENCE: "review judge rubric criteria/prompting for this stage",
    FailureCategory.MALFORMED_EVALUATION: "harden judge response parsing or prompt format",
    FailureCategory.RULE_DISAGREEMENT: "reconcile RuleJudge and LLMJudge scoring criteria",
}


@dataclass(frozen=True)
class Recommendation:
    problem: str
    root_cause: str
    estimated_impact: int
    confidence: float
    recommended_action: str
    priority: Priority


def _build_recommendation(cluster: FailureCluster) -> Recommendation:
    count = len(cluster.failures)
    confidence = min(CONFIDENCE_MAX, CONFIDENCE_BASE + CONFIDENCE_PER_FAILURE * count)
    action = _ACTION_BY_CATEGORY.get(
        cluster.category, f"investigate {cluster.component.value} failures of category "
        f"{cluster.category.value}"
    )
    return Recommendation(
        problem=(
            f"{cluster.component.value} produced {count} {cluster.category.value} "
            f"failure(s)"
        ),
        root_cause=cluster.common_cause,
        estimated_impact=count,
        confidence=round(confidence, 4),
        recommended_action=action,
        priority=_PRIORITY_FROM_SEVERITY[cluster.severity],
    )


def generate_recommendations(clusters: Sequence[FailureCluster]) -> tuple[Recommendation, ...]:
    """Build and priority-order recommendations from already-clustered
    failures - see module docstring's "Recommendation workflow". Returns
    an empty tuple for an empty input, never fabricating a recommendation
    with no backing cluster.
    """
    recommendations = [_build_recommendation(cluster) for cluster in clusters]
    recommendations.sort(
        key=lambda r: (-_PRIORITY_ORDER[r.priority], -r.estimated_impact, r.problem)
    )
    return tuple(recommendations)
