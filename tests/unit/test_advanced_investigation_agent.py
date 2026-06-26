from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services.advanced_investigation_agent import (
    AdvancedInvestigationAgent,
    _ESCALATION_COMPOSITE_FLOOR,
    _POLICY_B_BASELINE_N,
)
from app.services.search import IncidentSearchResult


# ── helpers ──────────────────────────────────────────────────────────────────

def make_result(
    title: str,
    *,
    symptoms: list[str] | None = None,
    resolution_summary: str | None = "Restarted workers.",
    distance: float = 0.2,
) -> IncidentSearchResult:
    incident = SimpleNamespace(
        title=title,
        symptoms=[SimpleNamespace(text=symptom) for symptom in symptoms or ["timeout"]],
        severity="high",
        status="resolved",
        resolution_summary=resolution_summary,
    )
    return IncidentSearchResult(incident=incident, distance=distance)


# distance=0.2  → similarity=0.8  → HIGH retrieval confidence
HIGH_CONF_RESULT = make_result("Primary database timeout", symptoms=["timeout", "high latency"])
# distance=0.52 → similarity=0.48 → MEDIUM retrieval confidence
MEDIUM_CONF_RESULT = make_result("Triggerer not starting", distance=0.52)


class FakeSearchService:
    """Search service that returns configurable result sets per query."""

    def __init__(
        self,
        *,
        retrieve_results: list[IncidentSearchResult] | None = None,
        evidence_results: dict[str, list[IncidentSearchResult]] | None = None,
    ) -> None:
        self.retrieve_results = retrieve_results or [HIGH_CONF_RESULT]
        self.evidence_results: dict[str, list[IncidentSearchResult]] = evidence_results or {
            "connection pool timeout session leak": [
                make_result("Connection pool exhausted", symptoms=["pool exhausted"])
            ],
            "database lock contention deadlock": [
                make_result("Database lock contention", symptoms=["deadlock"])
            ],
        }
        self.search_calls: list[tuple[str, int]] = []
        self.retrieve_calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int = 10, **_: object) -> list[IncidentSearchResult]:
        self.search_calls.append((query, limit))
        return self.evidence_results.get(query, [])

    def retrieve(self, query: str, *, limit: int = 10, **_: object) -> list[IncidentSearchResult]:
        self.retrieve_calls.append((query, limit))
        return self.retrieve_results


class FakeLLMService:
    """LLM service that records calls and returns canned hypotheses."""

    def __init__(
        self,
        *,
        base_hypotheses: list[dict] | None = None,
        extra_hypotheses: list[dict] | None = None,
    ) -> None:
        self._base = base_hypotheses or [
            {
                "root_cause": "Connection pool exhaustion",
                "confidence_score": "0.82",
                "validation_keywords": ["connection pool", "timeout", "session leak"],
                "rationale": "Similar incidents mention pool exhaustion.",
            },
            {
                "root_cause": "Database lock contention",
                "confidence_score": 0.55,
                "validation_keywords": ["database lock", "contention", "deadlock"],
                "rationale": "Peak traffic can amplify lock waits.",
            },
        ]
        self._extra = extra_hypotheses or []
        self.generate_hypotheses_calls: list[dict[str, Any]] = []
        self.evidence_context: str | None = None
        self._call_count = 0

    def generate_hypotheses(
        self,
        *,
        problem: str,
        context: str,
        n: int = 2,
        existing_root_causes: list[str] | None = None,
    ) -> list[dict]:
        self.generate_hypotheses_calls.append(
            {
                "problem": problem,
                "n": n,
                "existing_root_causes": existing_root_causes,
            }
        )
        self._call_count += 1
        # First call → base hypotheses; subsequent calls → extra hypotheses
        if self._call_count == 1:
            return self._base[:n]
        return self._extra[:n]

    def evaluate_investigation_evidence(
        self,
        *,
        problem: str,
        initial_context: str,
        evidence_context: str,
    ) -> dict:
        self.evidence_context = evidence_context
        return {
            "executive_summary": "Traffic likely exhausted the database connection pool.",
            "ranked_hypotheses": ["Connection pool exhaustion", "Database lock contention"],
            "supporting_evidence": ["Connection pool exhausted"],
            "recommended_actions": ["Inspect pool metrics", "Check leaked sessions"],
            "confidence_assessment": "Moderate to high confidence.",
        }


def make_agent(
    *,
    retrieve_results: list[IncidentSearchResult] | None = None,
    evidence_results: dict[str, list[IncidentSearchResult]] | None = None,
    base_hypotheses: list[dict] | None = None,
    extra_hypotheses: list[dict] | None = None,
) -> tuple[AdvancedInvestigationAgent, FakeSearchService, FakeLLMService]:
    search_service = FakeSearchService(
        retrieve_results=retrieve_results,
        evidence_results=evidence_results,
    )
    llm_service = FakeLLMService(
        base_hypotheses=base_hypotheses,
        extra_hypotheses=extra_hypotheses,
    )
    agent = AdvancedInvestigationAgent(
        db=SimpleNamespace(),
        search_service=search_service,
        llm_service=llm_service,
    )
    return agent, search_service, llm_service


# ── baseline behaviour (HIGH confidence, no escalation) ──────────────────────

def test_hypothesis_generation_normalizes_llm_output() -> None:
    agent, _, _ = make_agent()

    hypotheses = agent._generate_hypotheses(
        "database timeout during peak traffic",
        "Incident 1\nTitle: Primary database timeout",
    )

    assert hypotheses == [
        {
            "root_cause": "Connection pool exhaustion",
            "confidence_score": 0.82,
            "validation_keywords": ["connection pool", "timeout", "session leak"],
            "rationale": "Similar incidents mention pool exhaustion.",
        },
        {
            "root_cause": "Database lock contention",
            "confidence_score": 0.55,
            "validation_keywords": ["database lock", "contention", "deadlock"],
            "rationale": "Peak traffic can amplify lock waits.",
        },
    ]


def test_evidence_collection_searches_keywords_for_each_hypothesis() -> None:
    agent, search_service, _ = make_agent()
    hypotheses = [
        {
            "root_cause": "Connection pool exhaustion",
            "confidence_score": 0.82,
            "validation_keywords": ["connection pool", "timeout", "session leak"],
            "rationale": "Similar incidents mention pool exhaustion.",
        },
        {
            "root_cause": "Database lock contention",
            "confidence_score": 0.55,
            "validation_keywords": ["database lock", "contention", "deadlock"],
            "rationale": "Peak traffic can amplify lock waits.",
        },
    ]

    evidence = agent._collect_evidence(hypotheses)

    assert search_service.search_calls == [
        ("connection pool timeout session leak", 5),
        ("database lock contention deadlock", 5),
    ]
    assert evidence[0]["query"] == "connection pool timeout session leak"
    assert evidence[0]["supporting_incidents"][0]["title"] == "Connection pool exhausted"
    assert evidence[1]["supporting_incidents"][0]["title"] == "Database lock contention"


def test_report_assembly_returns_structured_report() -> None:
    agent, search_service, llm_service = make_agent()

    result = agent.investigate("database timeout during peak traffic")

    assert search_service.retrieve_calls[0] == ("database timeout during peak traffic", 10)
    assert result["problem"] == "database timeout during peak traffic"
    assert result["initial_incidents"][0]["title"] == "Primary database timeout"
    assert result["report"] == {
        "executive_summary": "Traffic likely exhausted the database connection pool.",
        "ranked_hypotheses": ["Connection pool exhaustion", "Database lock contention"],
        "supporting_evidence": ["Connection pool exhausted"],
        "recommended_actions": ["Inspect pool metrics", "Check leaked sessions"],
        "confidence_assessment": "Moderate to high confidence.",
    }
    assert llm_service.evidence_context is not None
    assert "Connection pool exhausted" in llm_service.evidence_context


def test_investigate_returns_policy_metadata() -> None:
    agent, _, _ = make_agent()

    result = agent.investigate("database timeout during peak traffic")

    meta = result["policy_metadata"]
    assert meta["policy_used"] == "B"
    assert meta["hypothesis_count_generated"] == _POLICY_B_BASELINE_N
    assert isinstance(meta["latency_s"], float)


def test_high_confidence_does_not_escalate() -> None:
    """HIGH retrieval confidence must never trigger escalation regardless of
    composite scores."""
    agent, _, llm_service = make_agent()

    result = agent.investigate("database timeout during peak traffic")

    assert result["policy_metadata"]["escalation_triggered"] is False
    assert result["policy_metadata"]["retrieval_confidence"] == "HIGH"
    # Only one generate_hypotheses call (no escalation call)
    assert llm_service._call_count == 1
    assert len(result["hypotheses"]) == _POLICY_B_BASELINE_N


def test_low_confidence_does_not_escalate() -> None:
    """LOW retrieval confidence must never trigger escalation."""
    # distance=0.7 → similarity=0.3 → LOW
    low_result = make_result("Vague match", distance=0.7)
    agent, _, llm_service = make_agent(retrieve_results=[low_result])

    result = agent.investigate("something obscure")

    assert result["policy_metadata"]["escalation_triggered"] is False
    assert result["policy_metadata"]["retrieval_confidence"] == "LOW"
    assert llm_service._call_count == 1


# ── escalation behaviour (MEDIUM confidence + weak top-2 evidence) ────────────

def _medium_base_hypotheses() -> list[dict]:
    """Two hypotheses whose keywords return no evidence (LOW evidence confidence)."""
    return [
        {
            "root_cause": "Execution API startup failure",
            "confidence_score": 0.8,
            "validation_keywords": ["synchronous startup", "Execution API", "init failure"],
            "rationale": "Generic startup issue.",
        },
        {
            "root_cause": "Azure Blob connection missing",
            "confidence_score": 0.7,
            "validation_keywords": ["AirflowNotFoundException", "conn_id"],
            "rationale": "Log shows missing connection.",
        },
    ]


def _medium_extra_hypotheses() -> list[dict]:
    """Two escalation-rank hypotheses that do find evidence."""
    return [
        {
            "root_cause": "Remote logging not configured",
            "confidence_score": 0.6,
            "validation_keywords": ["remote logging", "not enabled"],
            "rationale": "Warning in logs.",
        },
        {
            "root_cause": "Inadequate error handling in triggerer startup",
            "confidence_score": 0.65,
            "validation_keywords": ["triggerer", "startup", "error handling"],
            "rationale": "Triggerer fails silently.",
        },
    ]


def test_medium_confidence_low_evidence_triggers_escalation() -> None:
    """MEDIUM retrieval + both top-2 hypotheses return no evidence → escalate."""
    # Both evidence searches return empty → LOW evidence confidence for both.
    agent, _, llm_service = make_agent(
        retrieve_results=[MEDIUM_CONF_RESULT],
        evidence_results={},  # all evidence searches miss
        base_hypotheses=_medium_base_hypotheses(),
        extra_hypotheses=_medium_extra_hypotheses(),
    )

    result = agent.investigate("triggerer not starting")

    meta = result["policy_metadata"]
    assert meta["escalation_triggered"] is True
    assert meta["retrieval_confidence"] == "MEDIUM"
    assert meta["hypothesis_count_generated"] == 4
    assert len(result["hypotheses"]) == 4
    assert len(result["evidence"]) == 4
    # Two LLM hypothesis-generation calls: baseline + escalation
    assert llm_service._call_count == 2
    # The escalation call passes existing root causes to avoid repetition
    escalation_call = llm_service.generate_hypotheses_calls[1]
    assert escalation_call["existing_root_causes"] == [
        "Execution API startup failure",
        "Azure Blob connection missing",
    ]


def test_medium_confidence_strong_evidence_no_escalation() -> None:
    """MEDIUM retrieval + at least one top-2 hypothesis reaches composite floor → no escalation."""
    # evidence search returns a HIGH-similarity result → evidence confidence HIGH
    # composite = 0.82 × 0.85 × 1.0 ≈ 0.697 > _ESCALATION_COMPOSITE_FLOOR
    strong_evidence = {
        "connection pool timeout session leak": [
            make_result("Strong match", distance=0.1)  # similarity=0.9 → HIGH
        ],
        "database lock contention deadlock": [],
    }
    agent, _, llm_service = make_agent(
        retrieve_results=[MEDIUM_CONF_RESULT],
        evidence_results=strong_evidence,
    )

    result = agent.investigate("triggerer not starting")

    assert result["policy_metadata"]["escalation_triggered"] is False
    assert llm_service._call_count == 1


def test_escalation_composite_floor_boundary() -> None:
    """Verify _should_escalate logic directly at the boundary."""
    agent, _, _ = make_agent()

    # Hypothesis with composite exactly at floor should block escalation
    hyp = [{"root_cause": "x", "confidence_score": _ESCALATION_COMPOSITE_FLOOR / 0.85}]
    ev_medium = [{"confidence_level": "MEDIUM"}]
    assert agent._should_escalate("MEDIUM", hyp, ev_medium) is False

    # Hypothesis with composite just below floor should trigger
    hyp_low = [{"root_cause": "x", "confidence_score": (_ESCALATION_COMPOSITE_FLOOR - 0.01) / 0.85}]
    assert agent._should_escalate("MEDIUM", hyp_low, [{"confidence_level": "MEDIUM"}]) is True

    # HIGH retrieval always returns False regardless of composite
    assert agent._should_escalate("HIGH", hyp_low, [{"confidence_level": "LOW"}]) is False
    assert agent._should_escalate("LOW", hyp_low, [{"confidence_level": "LOW"}]) is False


def test_generate_hypotheses_passes_n_to_llm() -> None:
    """_generate_hypotheses forwards n to llm_service.generate_hypotheses."""
    agent, _, llm_service = make_agent()

    agent._generate_hypotheses("p", "ctx", n=2)
    agent._generate_hypotheses("p", "ctx", n=2, existing_root_causes=["already seen"])

    assert llm_service.generate_hypotheses_calls[0]["n"] == 2
    assert llm_service.generate_hypotheses_calls[1]["existing_root_causes"] == ["already seen"]


def test_investigate_output_contains_all_evidence_after_escalation() -> None:
    """After escalation, evidence list spans all 4 hypotheses in order."""
    agent, _, _ = make_agent(
        retrieve_results=[MEDIUM_CONF_RESULT],
        evidence_results={},
        base_hypotheses=_medium_base_hypotheses(),
        extra_hypotheses=_medium_extra_hypotheses(),
    )

    result = agent.investigate("triggerer not starting")

    root_causes = [ev["hypothesis"]["root_cause"] for ev in result["evidence"]]
    assert root_causes == [
        "Execution API startup failure",
        "Azure Blob connection missing",
        "Remote logging not configured",
        "Inadequate error handling in triggerer startup",
    ]
