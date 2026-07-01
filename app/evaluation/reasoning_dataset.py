"""Reasoning Evaluation Dataset — schema and validation (Phase 20A).

Mirrors the shape of Phase 16B's Gold Dataset v2 (``app.evaluation.
gold_dataset``) but for investigations, not retrieval queries: each
``InvestigationScenario`` represents one whole investigation run (problem
-> expected strategy/root causes/verdict/stopping reason), not one search
query.

Per this phase's explicit instruction ("avoid coupling this schema to
implementation classes"), every "expected" field is a plain string, never
``app.services.planner_agent.PlanningStrategy``,
``app.services.critic_agent.CritiqueVerdict``, or
``app.services.investigation_orchestrator.StoppingReason``. The closed
sets below (``VALID_STRATEGIES``/``VALID_VERDICTS``/
``VALID_STOPPING_REASONS``) are deliberately maintained as their own
string literals here, not derived from those enums' ``.value``s at import
time — a dataset file should remain readable/diffable/loadable without
importing the reasoning agents at all, the same reasoning Phase 16B's
schema module never imports ``IncidentSearchService``. The evaluation
harness (a separate module) is the only place these strings are translated
into the real enums for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_STRATEGIES = frozenset(
    {
        "infrastructure_failure", "configuration", "authentication",
        "network", "application_failure", "unknown",
    }
)
VALID_VERDICTS = frozenset(
    {"approved", "need_more_evidence", "alternative_hypothesis_plausible", "inconclusive"}
)
VALID_STOPPING_REASONS = frozenset(
    {"critic_approved", "max_iterations", "no_progress", "no_new_hypotheses"}
)


@dataclass(frozen=True)
class InvestigationScenario:
    """One gold investigation scenario.

    ``expected_root_causes`` is a tuple of plain-text keywords/phrases, not
    an identity-anchored answer set (Phase 16B's incidents are addressable
    by stable identity; a "correct root cause" has no such identity — it is
    free text the reasoning system itself generated). An empty tuple is a
    legitimate, intentional shape: a negative-control scenario (e.g. an
    unrelated problem) where no hypothesis SHOULD be confidently accepted —
    mirrors Phase 16B's ``no-match-expected`` category.
    """

    id: str
    problem: str
    expected_strategy: str
    expected_root_causes: tuple[str, ...] = field(default_factory=tuple)
    expected_verdict: str = "inconclusive"
    expected_stopping_reason: str = "max_iterations"
    notes: str = ""

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.id:
            problems.append("scenario.id must be non-empty")
        if not self.problem:
            problems.append(f"scenario {self.id!r}: problem must be non-empty")
        if self.expected_strategy not in VALID_STRATEGIES:
            problems.append(
                f"scenario {self.id!r}: expected_strategy {self.expected_strategy!r} "
                f"not in {sorted(VALID_STRATEGIES)}"
            )
        if self.expected_verdict not in VALID_VERDICTS:
            problems.append(
                f"scenario {self.id!r}: expected_verdict {self.expected_verdict!r} "
                f"not in {sorted(VALID_VERDICTS)}"
            )
        if self.expected_stopping_reason not in VALID_STOPPING_REASONS:
            problems.append(
                f"scenario {self.id!r}: expected_stopping_reason "
                f"{self.expected_stopping_reason!r} not in {sorted(VALID_STOPPING_REASONS)}"
            )
        if not self.expected_root_causes and self.expected_verdict not in {
            "inconclusive", "need_more_evidence",
        }:
            problems.append(
                f"scenario {self.id!r}: expected_verdict {self.expected_verdict!r} implies an "
                "accepted hypothesis, but expected_root_causes is empty"
            )
        return problems


@dataclass(frozen=True)
class ReasoningGoldDataset:
    """A versioned collection of ``InvestigationScenario``s — the reasoning-
    layer analogue of Phase 16B's ``GoldDataset``.
    """

    version: str
    description: str
    created_at: str
    scenarios: tuple[InvestigationScenario, ...]
    author: str | None = None

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.version:
            problems.append("dataset.version must be non-empty")
        if not self.description:
            problems.append("dataset.description must be non-empty")
        if not self.created_at:
            problems.append("dataset.created_at must be non-empty")
        if not self.scenarios:
            problems.append("dataset.scenarios must be non-empty")

        seen_ids: set[str] = set()
        for scenario in self.scenarios:
            problems.extend(scenario.issues())
            if scenario.id in seen_ids:
                problems.append(f"duplicate scenario id {scenario.id!r}")
            seen_ids.add(scenario.id)
        return problems

    def is_valid(self) -> bool:
        return not self.issues()
