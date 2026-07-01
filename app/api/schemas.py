from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str


class GitHubIngestRequest(BaseModel):
    owner: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    state: str = Field(default="all", pattern="^(open|closed|all)$")
    limit: int = Field(default=50, ge=1, le=500)
    include_comments: bool = True


class GitHubIngestResponse(BaseModel):
    source: str
    fetched: int
    inserted: int
    updated: int
    skipped: int


class JiraIngestRequest(BaseModel):
    base_url: str = Field(min_length=1)
    project_key: str = Field(min_length=1)
    limit: int = Field(default=50, ge=1, le=500)
    force_backfill: bool = False


class JiraIngestResponse(BaseModel):
    source: str
    fetched: int
    inserted: int
    updated: int
    skipped: int


class IncidentResponse(BaseModel):
    id: uuid.UUID
    source_type: str
    source_external_id: str
    source_url: str | None
    owner: str | None
    repo: str | None
    source: str | None
    state: str | None
    title: str
    description: str
    severity: str
    status: str
    incident_type: str
    environment: dict[str, Any]
    affected_components: list[str]
    tags: list[str]
    canonical_text: str
    created_at_source: datetime | None
    updated_at_source: datetime | None

    model_config = {"from_attributes": True}


class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    limit: int = Field(default=10, ge=1, le=50)
    source_type: str | None = None
    tags: list[str] | None = None
    owner: str | None = None
    repo: str | None = None
    source: str | None = None
    state: str | None = None


class SearchResult(BaseModel):
    incident: IncidentResponse
    similarity_score: float
    distance: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    top1_score: float | None = None
    confidence_level: str = "LOW"


class SearchDebugRequest(BaseModel):
    query: str = Field(min_length=3)
    owner: str | None = None
    repo: str | None = None
    source: str | None = None
    state: str | None = None


class SearchDebugResult(BaseModel):
    title: str
    repo: str | None
    similarity_score: float


class SearchDebugResponse(BaseModel):
    query: str
    filters: dict[str, str | None]
    results: list[SearchDebugResult]
    top1_score: float | None = None
    confidence_level: str = "LOW"


class InvestigationRequest(BaseModel):
    problem: str = Field(min_length=3)


class InvestigationResponse(BaseModel):
    analysis: str


class AdvancedInvestigationRequest(BaseModel):
    problem: str = Field(min_length=3)


class AdvancedHypothesis(BaseModel):
    root_cause: str
    confidence_score: float
    validation_keywords: list[str]
    rationale: str


class AdvancedEvidenceIncident(BaseModel):
    title: str
    symptoms: list[str]
    severity: str
    status: str
    resolution_summary: str
    similarity_score: float


class AdvancedHypothesisEvidence(BaseModel):
    hypothesis: AdvancedHypothesis
    query: str
    supporting_incidents: list[AdvancedEvidenceIncident]


class AdvancedInvestigationReport(BaseModel):
    executive_summary: str
    ranked_hypotheses: list[str]
    supporting_evidence: list[str]
    recommended_actions: list[str]
    confidence_assessment: str


class AdvancedInvestigationResponse(BaseModel):
    problem: str
    initial_incidents: list[AdvancedEvidenceIncident]
    hypotheses: list[AdvancedHypothesis]
    evidence: list[AdvancedHypothesisEvidence]
    report: AdvancedInvestigationReport


class OrchestratedInvestigationRequest(BaseModel):
    problem: str = Field(min_length=3)
    n_hypotheses: int = Field(default=3, ge=1, le=10)


class OrchestratedHypothesis(BaseModel):
    id: str
    root_cause: str
    rationale: str
    validation_keywords: list[str]
    raw_confidence: float


class OrchestratedCritique(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    findings: list[str]
    unresolved_questions: list[str]
    missing_evidence: list[str]
    recommended_actions: list[str]


class OrchestratedInvestigationResponse(BaseModel):
    """The canonical investigation endpoint, wiring Phase 19A-19D's
    ``MultiAgentInvestigationOrchestrator`` (planner, evidence-driven
    hypothesis generation, critic, iterative loop) — see
    docs/architecture/19_multi_agent_investigation.md. Reflects the
    orchestrator's final iteration only; use the evaluation API
    (docs/architecture/22_evaluation_api.md) to inspect full iteration
    history for a given problem.
    """

    problem: str
    selected_root_cause: str | None
    confidence: float
    confidence_level: str
    is_uncertain: bool
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    remaining_uncertainty: list[str]
    rejected_hypotheses: list[OrchestratedHypothesis]
    critique: OrchestratedCritique
    total_iterations: int
    stopping_reason: str
    stop_explanation: str


class GitHubIssueRef(BaseModel):
    url: HttpUrl
