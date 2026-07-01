"""Human Evaluation Dataset — schema (Phase 21B).

An immutable, versioned dataset of human-assigned scores for investigation
artifacts, mirroring the shape of Phase 16B's ``GoldDataset``/Phase 20A's
``ReasoningGoldDataset`` but for HUMAN judgments rather than retrieval or
heuristic-correctness gold answers. Every human score is OPTIONAL — a
record with every score field ``None`` (only ``notes`` populated, or
nothing at all beyond the artifact reference) is valid, since "human
labels should remain optional so synthetic datasets are still usable."

This module does NOT import ``app.evaluation.judge`` or any concrete
``Judge`` implementation — "do not couple the dataset to any concrete
Judge implementation." Scores are plain ``float | None``, the same
unitless 1-10 scale ``app.evaluation.judge.SCORE_MIN``/``SCORE_MAX``
already establish, but this module does not import those constants either
(a human rater's scale convention need not be enforced by code that
should remain usable even if a future Judge changes its own scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HumanEvaluationRecord:
    """One investigation artifact's human-assigned scores. ``record_id``
    is the caller's own identifier for the investigation/scenario this
    record describes (e.g. an ``InvestigationResult.scenario_id``) - this
    module does not require or inspect that artifact directly.
    """

    record_id: str
    human_planner_score: float | None = None
    human_hypotheses_score: float | None = None
    human_decision_score: float | None = None
    human_critique_score: float | None = None
    human_overall_score: float | None = None
    notes: str = ""

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.record_id:
            problems.append("record.record_id must be non-empty")
        for name, value in (
            ("human_planner_score", self.human_planner_score),
            ("human_hypotheses_score", self.human_hypotheses_score),
            ("human_decision_score", self.human_decision_score),
            ("human_critique_score", self.human_critique_score),
            ("human_overall_score", self.human_overall_score),
        ):
            if value is not None and not (0.0 <= value <= 10.0):
                problems.append(
                    f"record {self.record_id!r}: {name} {value!r} must be in [0, 10] or None"
                )
        return problems


@dataclass(frozen=True)
class HumanEvaluationDataset:
    version: str
    description: str
    created_at: str
    records: tuple[HumanEvaluationRecord, ...] = field(default_factory=tuple)
    author: str | None = None

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.version:
            problems.append("dataset.version must be non-empty")
        if not self.description:
            problems.append("dataset.description must be non-empty")
        if not self.created_at:
            problems.append("dataset.created_at must be non-empty")

        seen_ids: set[str] = set()
        for record in self.records:
            problems.extend(record.issues())
            if record.record_id in seen_ids:
                problems.append(f"duplicate record id {record.record_id!r}")
            seen_ids.add(record.record_id)
        return problems

    def is_valid(self) -> bool:
        return not self.issues()

    def get(self, record_id: str) -> HumanEvaluationRecord | None:
        for record in self.records:
            if record.record_id == record_id:
                return record
        return None
