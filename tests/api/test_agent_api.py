"""API tests for /agent routes.

Phase 23A: ``/investigate`` is now the single canonical investigation
endpoint (previously ``/investigate-orchestrated``); the earlier
``/investigate`` (single-shot) and ``/investigate-advanced`` routes were
retired as historical implementations of the same capability — see
``app/api/routes/agent.py``'s module/route docstrings.

No database, no OpenAI, no retrieval — MultiAgentInvestigationOrchestrator
is monkeypatched; only the Pydantic/FastAPI routing layer is exercised.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.auth import require_api_key
from app.db.session import get_db
from app.main import app


def _client() -> TestClient:
    """Phase 23B: auth is bypassed here (dependency override to a no-op) —
    these tests exercise routing/orchestration, not authentication. See
    tests/api/test_authentication.py for the real auth behavior.
    """
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[require_api_key] = lambda: None
    return TestClient(app)


def _fake_session(*, uncertain: bool = False):
    from app.services.critic_agent import (
        CritiqueResult,
        CritiqueVerdict,
        CritiquedInvestigationReport,
    )
    from app.services.hypothesis_investigation import (
        InvestigationHypothesis,
        InvestigationReport,
    )
    from app.services.investigation_orchestrator import InvestigationSession, StoppingReason

    rejected = InvestigationHypothesis(
        id="h2",
        root_cause="a rejected cause",
        rationale="rejected rationale",
        validation_keywords=("kw1", "kw2"),
        raw_confidence=0.4,
    )
    accepted = None if uncertain else InvestigationHypothesis(
        id="h1",
        root_cause="expired auth token",
        rationale="token expiry matches symptom timing",
        validation_keywords=("token", "expired"),
        raw_confidence=0.9,
    )
    investigation = InvestigationReport(
        problem="users cannot log in",
        selected_hypothesis=accepted,
        confidence=0.0 if uncertain else 0.72,
        confidence_level="LOW" if uncertain else "HIGH",
        supporting_evidence=() if uncertain else ("Auth incident #123",),
        contradicting_evidence=(),
        remaining_uncertainty=("h2 rejected: composite score below floor",),
        is_uncertain=uncertain,
        rejected_hypotheses=(rejected,),
    )
    critique = CritiqueResult(
        verdict=CritiqueVerdict.INCONCLUSIVE if uncertain else CritiqueVerdict.APPROVED,
        confidence=0.0 if uncertain else 0.72,
        findings=("no accepted hypothesis",) if uncertain else ("evidence not challenged",),
        unresolved_questions=(),
        missing_evidence=(),
        recommended_actions=(),
        explanation="inconclusive: no hypothesis accepted" if uncertain else "approved: strong evidence",
    )
    return InvestigationSession(
        final_report=CritiquedInvestigationReport(investigation=investigation, critique=critique),
        iterations=(),
        stopping_reason=StoppingReason.MAX_ITERATIONS if uncertain else StoppingReason.CRITIC_APPROVED,
        total_iterations=1,
        stop_explanation="stopped after 1 iteration",
    )


def test_investigate_returns_accepted_hypothesis(monkeypatch) -> None:
    import app.api.routes.agent as agent_mod

    fake_orchestrator = MagicMock()
    fake_orchestrator.investigate.return_value = _fake_session()
    monkeypatch.setattr(
        agent_mod, "MultiAgentInvestigationOrchestrator", lambda db: fake_orchestrator
    )
    client = _client()
    try:
        resp = client.post("/agent/investigate", json={"problem": "users cannot log in"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["selected_root_cause"] == "expired auth token"
        assert body["confidence"] == 0.72
        assert body["is_uncertain"] is False
        assert body["critique"]["verdict"] == "approved"
        assert len(body["rejected_hypotheses"]) == 1
        assert body["rejected_hypotheses"][0]["id"] == "h2"
        assert body["stopping_reason"] == "critic_approved"
        assert body["total_iterations"] == 1
        fake_orchestrator.investigate.assert_called_once_with(
            "users cannot log in", n_hypotheses=3
        )
    finally:
        app.dependency_overrides.clear()


def test_investigate_uncertain_has_no_selected_cause(monkeypatch) -> None:
    import app.api.routes.agent as agent_mod

    fake_orchestrator = MagicMock()
    fake_orchestrator.investigate.return_value = _fake_session(uncertain=True)
    monkeypatch.setattr(
        agent_mod, "MultiAgentInvestigationOrchestrator", lambda db: fake_orchestrator
    )
    client = _client()
    try:
        resp = client.post("/agent/investigate", json={"problem": "users cannot log in"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["selected_root_cause"] is None
        assert body["is_uncertain"] is True
        assert body["critique"]["verdict"] == "inconclusive"
    finally:
        app.dependency_overrides.clear()


def test_investigate_respects_n_hypotheses(monkeypatch) -> None:
    import app.api.routes.agent as agent_mod

    fake_orchestrator = MagicMock()
    fake_orchestrator.investigate.return_value = _fake_session()
    monkeypatch.setattr(
        agent_mod, "MultiAgentInvestigationOrchestrator", lambda db: fake_orchestrator
    )
    client = _client()
    try:
        resp = client.post(
            "/agent/investigate",
            json={"problem": "users cannot log in", "n_hypotheses": 5},
        )
        assert resp.status_code == 200
        fake_orchestrator.investigate.assert_called_once_with(
            "users cannot log in", n_hypotheses=5
        )
    finally:
        app.dependency_overrides.clear()


def test_investigate_rejects_short_problem() -> None:
    client = _client()
    try:
        resp = client.post("/agent/investigate", json={"problem": "hi"})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_only_canonical_agent_route_registered() -> None:
    """Phase 23A: /investigate-advanced and /investigate-orchestrated must
    no longer exist — /investigate is the single canonical route.
    """
    client = _client()
    try:
        resp = client.get("/openapi.json")
        paths = set(resp.json()["paths"].keys())
        assert "/agent/investigate" in paths
        assert "/agent/investigate-advanced" not in paths
        assert "/agent/investigate-orchestrated" not in paths
    finally:
        app.dependency_overrides.clear()
