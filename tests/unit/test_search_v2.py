from __future__ import annotations

import uuid
from types import SimpleNamespace

from app.services.search import IncidentSearchResult, IncidentSearchService


def make_result(
    title: str,
    *,
    distance: float,
    incident_id: uuid.UUID | None = None,
    owner: str = "apache",
    repo: str = "airflow",
    state: str = "closed",
) -> IncidentSearchResult:
    incident = SimpleNamespace(
        id=incident_id or uuid.uuid4(),
        title=title,
        owner=owner,
        repo=repo,
        source="github",
        state=state,
        symptoms=[],
        severity="high",
        status="resolved",
        resolution_summary="Fixed by changing configuration.",
    )
    return IncidentSearchResult(incident=incident, distance=distance)


class FakeLLMService:
    def __init__(self) -> None:
        self.rerank_candidates: list[dict] = []

    def expand_search_query(self, query: str) -> list[str]:
        assert query == "scheduler timeout"
        return [
            "airflow scheduler heartbeat timeout",
            "dag processor stalled",
            "database connection pool exhausted",
        ]

    def rerank_incident_search_results(
        self,
        *,
        query: str,
        candidates: list[dict],
        limit: int = 5,
    ) -> list[str]:
        assert query == "scheduler timeout"
        self.rerank_candidates = candidates
        return ["2", "1"]


class FakeSearchService(IncidentSearchService):
    def __init__(self, llm_service: FakeLLMService) -> None:
        self.llm_service = llm_service
        self.calls: list[dict] = []
        duplicate_id = uuid.uuid4()
        self.results_by_query = {
            "scheduler timeout": [
                make_result("Generic timeout", distance=0.3, incident_id=duplicate_id),
                make_result("Worker timeout", distance=0.4),
            ],
            "airflow scheduler heartbeat timeout": [
                make_result("Scheduler heartbeat missed", distance=0.1),
            ],
            "dag processor stalled": [
                make_result("Generic timeout", distance=0.2, incident_id=duplicate_id),
            ],
            "database connection pool exhausted": [],
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        source_type: str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        repo: str | None = None,
        source: str | None = None,
        state: str | None = None,
        call_site: str | None = None,
    ) -> list[IncidentSearchResult]:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "source_type": source_type,
                "tags": tags,
                "owner": owner,
                "repo": repo,
                "source": source,
                "state": state,
            }
        )
        return self.results_by_query[query][:limit]


def test_search_debug_expands_query_and_passes_metadata_filters() -> None:
    llm_service = FakeLLMService()
    service = FakeSearchService(llm_service)

    service.search_debug(
        "scheduler timeout",
        owner="apache",
        repo="airflow",
        source="github",
        state="closed",
    )

    assert [call["query"] for call in service.calls] == [
        "scheduler timeout",
        "airflow scheduler heartbeat timeout",
        "dag processor stalled",
        "database connection pool exhausted",
    ]
    assert all(call["limit"] == 25 for call in service.calls)
    assert all(call["owner"] == "apache" for call in service.calls)
    assert all(call["repo"] == "airflow" for call in service.calls)
    assert all(call["source"] == "github" for call in service.calls)
    assert all(call["state"] == "closed" for call in service.calls)


def test_search_debug_deduplicates_incidents_and_reranks_candidates() -> None:
    llm_service = FakeLLMService()
    service = FakeSearchService(llm_service)

    results = service.search_debug("scheduler timeout")

    assert [result.incident.title for result in results] == [
        "Generic timeout",
        "Scheduler heartbeat missed",
        "Worker timeout",
    ]
    assert [candidate["title"] for candidate in llm_service.rerank_candidates] == [
        "Scheduler heartbeat missed",
        "Generic timeout",
        "Worker timeout",
    ]
    generic_timeout = next(
        result for result in results if result.incident.title == "Generic timeout"
    )
    assert generic_timeout.distance == 0.2
