from __future__ import annotations

from types import SimpleNamespace

from app.services.investigation_agent import InvestigationAgent
from app.services.search import IncidentSearchResult


class FakeSearchService:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int = 10, **kwargs: object) -> list[IncidentSearchResult]:
        self.queries.append((query, limit))
        incident = SimpleNamespace(
            title="Database pool exhausted",
            symptoms=[
                SimpleNamespace(text="database timeout"),
                SimpleNamespace(text="connection pool saturated"),
            ],
            severity="high",
            status="resolved",
            resolution_summary="Increased pool size and fixed leaked sessions.",
        )
        return [IncidentSearchResult(incident=incident, distance=0.2)]

    def retrieve(self, query: str, *, limit: int = 10, **kwargs: object) -> list[IncidentSearchResult]:
        return self.search(query, limit=limit, **kwargs)


class FakeLLMService:
    def __init__(self) -> None:
        self.problem: str | None = None
        self.context: str | None = None

    def generate_investigation(self, *, problem: str, context: str) -> str:
        self.problem = problem
        self.context = context
        return "Probable root cause: connection pool exhaustion."


def test_investigation_agent_retrieves_context_and_calls_llm() -> None:
    search_service = FakeSearchService()
    llm_service = FakeLLMService()
    agent = InvestigationAgent(
        db=SimpleNamespace(),
        search_service=search_service,
        llm_service=llm_service,
    )

    analysis = agent.investigate("database timeout during peak traffic")

    assert analysis == "Probable root cause: connection pool exhaustion."
    assert search_service.queries == [("database timeout during peak traffic", 5)]
    assert llm_service.problem == "database timeout during peak traffic"
    assert llm_service.context is not None
    assert "Database pool exhausted" in llm_service.context
    assert "database timeout; connection pool saturated" in llm_service.context
    assert "Increased pool size and fixed leaked sessions." in llm_service.context


def test_investigation_agent_builds_empty_context_message() -> None:
    agent = InvestigationAgent(
        db=SimpleNamespace(),
        search_service=SimpleNamespace(search=lambda query, limit=10: []),
        llm_service=FakeLLMService(),
    )

    context = agent._build_context([])

    assert "Retrieval confidence: LOW" in context
    assert "no similar incidents were retrieved" in context.lower()

