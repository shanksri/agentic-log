from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.planner_agent import (
    InvestigationPlan,
    PlannedInvestigationAgent,
    PlannerAgent,
    PlanningStrategy,
    RuleBasedPlanner,
    plan_then_generate_hypotheses,
)
from app.services.hypothesis_investigation import HypothesisGenerator
from app.services.search import IncidentSearchResult

# ── Fakes ──────────────────────────────────────────────────────────────────────


def _incident(title: str, symptoms=()):
    return SimpleNamespace(title=title, symptoms=[SimpleNamespace(text=s) for s in symptoms])


def _result(title: str, distance: float = 0.5, symptoms=()) -> IncidentSearchResult:
    return IncidentSearchResult(incident=_incident(title, symptoms), distance=distance)


def _routing_observation(*, has_stack_trace: bool):
    signals = SimpleNamespace(has_stack_trace=has_stack_trace)
    return SimpleNamespace(signals=signals)


class FakeLLMService:
    def __init__(self, hypotheses=None):
        self._hypotheses = hypotheses if hypotheses is not None else []
        self.calls: list[dict] = []

    def generate_hypotheses(self, *, problem, context, n=2, existing_root_causes=None):
        self.calls.append({"problem": problem, "context": context, "n": n})
        return self._hypotheses


class FakeSearchService:
    def __init__(self, *, retrieve_response=None, search_responses=None):
        self._retrieve_response = retrieve_response or []
        self._search_responses = search_responses or {}
        self.search_calls: list[dict] = []

    def retrieve(self, query, *, limit=10, expand=False, rerank=False, call_site=None):
        return self._retrieve_response

    def search(self, query, *, limit=10, call_site=None):
        self.search_calls.append({"query": query})
        return self._search_responses.get(query, [])


planner = RuleBasedPlanner()


# ── Planning strategies (keyword matching) ──────────────────────────────────────


@pytest.mark.parametrize(
    "problem, expected",
    [
        ("Getting a 401 unauthorized error when calling the API", PlanningStrategy.AUTHENTICATION),
        ("Login fails with an invalid token", PlanningStrategy.AUTHENTICATION),
        ("Connection refused when reaching the upstream service", PlanningStrategy.NETWORK),
        ("DNS resolution keeps timing out for the internal service", PlanningStrategy.NETWORK),
        (
            "Pod keeps getting killed due to OOM in the kubernetes cluster",
            PlanningStrategy.INFRASTRUCTURE_FAILURE,
        ),
        ("Disk pressure causing node eviction", PlanningStrategy.INFRASTRUCTURE_FAILURE),
        ("Feature flag misconfigured in the yaml settings file", PlanningStrategy.CONFIGURATION),
        ("Wrong environment variable value after deploy", PlanningStrategy.CONFIGURATION),
        ("NullPointerException causing a crash on startup", PlanningStrategy.APPLICATION_FAILURE),
        ("Service panics with a segfault under load", PlanningStrategy.APPLICATION_FAILURE),
    ],
)
def test_keyword_matching_selects_expected_strategy(problem, expected) -> None:
    plan = planner.plan(problem)
    assert plan.strategy == expected


def test_unrelated_problem_is_unknown_strategy() -> None:
    plan = planner.plan("the coffee machine in the break room is broken")
    assert plan.strategy == PlanningStrategy.UNKNOWN


def test_keyword_matching_uses_word_boundaries_not_bare_substrings() -> None:
    # "room" contains the substring "oom" but must not match the "oom" keyword.
    plan = planner.plan("the break room printer is jammed")
    assert plan.strategy == PlanningStrategy.UNKNOWN


def test_priority_order_picks_narrower_category_over_broader_one() -> None:
    # Mentions both an auth term ("token") and an app-failure term ("crash").
    plan = planner.plan("the token validation logic crashes intermittently")
    assert plan.strategy == PlanningStrategy.AUTHENTICATION


def test_strategy_rationale_names_the_matched_keyword() -> None:
    plan = planner.plan("login fails with an invalid token")
    assert "token" in plan.strategy_rationale or "login" in plan.strategy_rationale


def test_unknown_strategy_rationale_states_no_match() -> None:
    plan = planner.plan("the coffee machine is broken")
    assert "no strategy keyword matched" in plan.strategy_rationale


# ── Stack-trace routing observation overrides keyword matching ──────────────────


def test_routing_observation_stack_trace_forces_application_failure() -> None:
    observation = _routing_observation(has_stack_trace=True)
    plan = planner.plan("login fails with an invalid token", routing_observation=observation)
    assert plan.strategy == PlanningStrategy.APPLICATION_FAILURE
    assert "has_stack_trace" in plan.strategy_rationale


def test_routing_observation_without_stack_trace_falls_through_to_keywords() -> None:
    observation = _routing_observation(has_stack_trace=False)
    plan = planner.plan("login fails with an invalid token", routing_observation=observation)
    assert plan.strategy == PlanningStrategy.AUTHENTICATION


def test_retrieved_incidents_contribute_to_keyword_matching() -> None:
    incidents = [_result("Pod evicted due to OOM", symptoms=("kubelet restarted",))]
    plan = planner.plan("something is wrong", retrieved_incidents=incidents)
    assert plan.strategy == PlanningStrategy.INFRASTRUCTURE_FAILURE


# ── InvestigationPlan content ────────────────────────────────────────────────────


def test_plan_has_nonempty_objective_and_priorities_for_every_strategy() -> None:
    for strategy_problem in [
        "401 unauthorized", "connection refused", "kubernetes pod oom",
        "yaml config flag", "exception stack trace crash", "totally unrelated topic",
    ]:
        plan = planner.plan(strategy_problem)
        assert plan.objective
        assert len(plan.priority_list) >= 1
        assert len(plan.evidence_priorities) >= 1
        assert len(plan.assumptions) >= 1
        assert plan.expected_difficulty in {"low", "medium", "high"}


def test_plan_carries_original_problem_text() -> None:
    plan = planner.plan("the API returns a 403")
    assert plan.problem == "the API returns a 403"


def test_plan_is_frozen() -> None:
    plan = planner.plan("a 401 error")
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        plan.strategy = PlanningStrategy.UNKNOWN  # type: ignore[misc]


# ── Deterministic planning ───────────────────────────────────────────────────────


def test_planning_is_deterministic_across_repeated_calls() -> None:
    first = planner.plan("connection refused timeout")
    second = planner.plan("connection refused timeout")
    assert first == second


# ── Malformed / edge-case inputs ─────────────────────────────────────────────────


def test_empty_problem_string_does_not_crash() -> None:
    plan = planner.plan("")
    assert plan.strategy == PlanningStrategy.UNKNOWN


def test_no_retrieved_incidents_does_not_crash() -> None:
    plan = planner.plan("something broke", retrieved_incidents=())
    assert plan.strategy == PlanningStrategy.UNKNOWN


def test_no_routing_observation_does_not_crash() -> None:
    plan = planner.plan("something broke", routing_observation=None)
    assert plan.strategy == PlanningStrategy.UNKNOWN


# ── Planner replacement (interface independence) ────────────────────────────────


class _AlwaysNetworkPlanner(PlannerAgent):
    def plan(self, problem, *, retrieved_incidents=(), routing_observation=None):
        return InvestigationPlan(
            problem=problem, strategy=PlanningStrategy.NETWORK, objective="stub objective",
            priority_list=("stub",), evidence_priorities=("stub",), assumptions=("stub",),
            expected_difficulty="low", strategy_rationale="stub planner always picks network",
        )


def test_planner_can_be_replaced_without_changing_callers() -> None:
    stub_planner = _AlwaysNetworkPlanner()
    plan = stub_planner.plan("a totally unrelated kubernetes pod problem")
    assert plan.strategy == PlanningStrategy.NETWORK


def test_planned_investigation_agent_accepts_injected_planner() -> None:
    llm = FakeLLMService([])
    search = FakeSearchService(retrieve_response=[])
    agent = PlannedInvestigationAgent(
        db=None, planner=_AlwaysNetworkPlanner(), search_service=search, llm_service=llm
    )

    plan, _report = agent.investigate("kubernetes pod oom")

    assert plan.strategy == PlanningStrategy.NETWORK  # stub planner's choice, not keyword match


# ── Integration with Phase 19A (HypothesisGenerator unmodified) ─────────────────


def test_plan_then_generate_hypotheses_passes_problem_and_context_through() -> None:
    llm = FakeLLMService([
        {"root_cause": "x", "confidence_score": 0.5, "validation_keywords": [], "rationale": ""},
    ])
    generator = HypothesisGenerator(llm)
    plan = planner.plan("401 unauthorized on login")

    hypotheses = plan_then_generate_hypotheses(plan, generator, n=1)

    assert len(hypotheses) == 1
    assert llm.calls[0]["problem"] == "401 unauthorized on login"
    assert "authentication" in llm.calls[0]["context"]
    assert plan.objective in llm.calls[0]["context"]


def test_plan_then_generate_hypotheses_includes_retrieval_context() -> None:
    llm = FakeLLMService([])
    generator = HypothesisGenerator(llm)
    plan = planner.plan("connection refused")

    plan_then_generate_hypotheses(
        plan, generator, retrieval_context="Retrieval confidence: HIGH", n=2
    )

    assert "Retrieval confidence: HIGH" in llm.calls[0]["context"]


def test_generator_never_receives_strategy_enum_directly() -> None:
    """HypothesisGenerator.generate() only ever receives strings (problem,
    context) - the PlanningStrategy enum itself never crosses that boundary,
    preserving strategy independence at the type level.
    """
    llm = FakeLLMService([])
    generator = HypothesisGenerator(llm)
    plan = planner.plan("401 unauthorized")

    plan_then_generate_hypotheses(plan, generator, n=1)

    call = llm.calls[0]
    assert isinstance(call["problem"], str)
    assert isinstance(call["context"], str)
    assert not isinstance(call["problem"], PlanningStrategy)


# ── End-to-end PlannedInvestigationAgent ─────────────────────────────────────────


def test_planned_investigation_agent_end_to_end() -> None:
    llm = FakeLLMService([
        {
            "root_cause": "expired token", "confidence_score": 0.9,
            "validation_keywords": ["token", "expired"], "rationale": "seen before",
        },
    ])
    search = FakeSearchService(
        retrieve_response=[_result("similar incident", 0.2)],
        search_responses={"token expired": [_result("token expiry match", 0.1)]},
    )
    agent = PlannedInvestigationAgent(db=None, search_service=search, llm_service=llm)

    plan, report = agent.investigate("login fails with token error", n_hypotheses=1)

    assert plan.strategy == PlanningStrategy.AUTHENTICATION
    assert report.selected_hypothesis is not None
    assert report.selected_hypothesis.root_cause == "expired token"


def test_planned_investigation_agent_empty_hypotheses_is_uncertain() -> None:
    llm = FakeLLMService([])
    search = FakeSearchService(retrieve_response=[])
    agent = PlannedInvestigationAgent(db=None, search_service=search, llm_service=llm)

    plan, report = agent.investigate("totally unrelated coffee machine issue")

    assert plan.strategy == PlanningStrategy.UNKNOWN
    assert report.is_uncertain is True
