from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.critic_agent import (
    CriticAgent,
    CritiqueResult,
    CritiqueVerdict,
    HeuristicCriticAgent,
)
from app.services.hypothesis_investigation import (
    EvidenceEvaluation,
    HypothesisScore,
    InvestigationDecision,
    InvestigationHypothesis,
    make_investigation_decision,
)
from app.services.investigation_orchestrator import (
    DEFAULT_MAX_ITERATIONS,
    InvestigationIteration,
    MultiAgentInvestigationOrchestrator,
    OrchestratorConfig,
    StoppingReason,
    detect_progress,
)
from app.services.planner_agent import InvestigationPlan, PlanningStrategy
from app.services.search import IncidentSearchResult


# ── Fakes / builders ─────────────────────────────────────────────────────────────


def _incident(title: str, symptoms=()):
    return SimpleNamespace(title=title, symptoms=[SimpleNamespace(text=s) for s in symptoms])


def _result(title: str, distance: float = 0.5, symptoms=()) -> IncidentSearchResult:
    return IncidentSearchResult(incident=_incident(title, symptoms), distance=distance)


class FakeLLMService:
    """Returns one fixed batch of hypotheses per call, in order. The last
    batch repeats forever once exhausted, so tests can drive multiple
    iterations deterministically.
    """

    def __init__(self, batches: list[list[dict]]):
        self._batches = batches
        self.calls: list[dict] = []

    def generate_hypotheses(self, *, problem, context, n=2, existing_root_causes=None):
        index = min(len(self.calls), len(self._batches) - 1)
        self.calls.append(
            {"problem": problem, "context": context, "n": n,
             "existing_root_causes": existing_root_causes}
        )
        return self._batches[index]


class FakeSearchService:
    def __init__(self, *, retrieve_response=None, search_responses=None):
        self._retrieve_response = retrieve_response or []
        self._search_responses = search_responses or {}
        self.llm_service = None  # RoutedSearchService reads dense.llm_service at construction

    def retrieve(self, query, *, limit=10, expand=False, rerank=False, call_site=None):
        return self._retrieve_response

    def search(self, query, *, limit=10, call_site=None):
        return self._search_responses.get(query, [])


def _plan(problem: str = "p") -> InvestigationPlan:
    return InvestigationPlan(
        problem=problem, strategy=PlanningStrategy.UNKNOWN, objective="obj",
        priority_list=("p",), evidence_priorities=("e",), assumptions=("a",),
        expected_difficulty="medium", strategy_rationale="r",
    )


def _make_orchestrator(llm, search, **config_kwargs):
    config = OrchestratorConfig(**config_kwargs) if config_kwargs else None
    return MultiAgentInvestigationOrchestrator(
        db=None, config=config, search_service=search, llm_service=llm,
    )


# ── Approval on first iteration ─────────────────────────────────────────────────


def test_approval_on_first_iteration_stops_immediately() -> None:
    llm = FakeLLMService([
        [{
            "root_cause": "expired token", "confidence_score": 1.0,
            "validation_keywords": ["token"], "rationale": "seen before",
        }],
    ])
    search = FakeSearchService(
        retrieve_response=[_result("similar incident", 0.1)],
        search_responses={"token": [_result("token expiry match", 0.05)]},
    )
    orchestrator = _make_orchestrator(llm, search)

    session = orchestrator.investigate("login fails", n_hypotheses=1)

    assert session.stopping_reason == StoppingReason.CRITIC_APPROVED
    assert session.total_iterations == 1
    assert len(session.iterations) == 1
    assert session.final_report.critique.verdict == CritiqueVerdict.APPROVED
    assert len(llm.calls) == 1


# ── Multiple iterations ──────────────────────────────────────────────────────────


def test_multiple_iterations_when_first_pass_is_not_approved() -> None:
    # Iteration 1: weak hypothesis with no evidence -> NEED_MORE_EVIDENCE.
    # Iteration 2: a NEW, well-evidenced hypothesis -> APPROVED.
    llm = FakeLLMService([
        [{
            "root_cause": "weak guess", "confidence_score": 1.0,
            "validation_keywords": ["zzz_no_match_zzz"], "rationale": "",
        }],
        [{
            "root_cause": "expired token", "confidence_score": 1.0,
            "validation_keywords": ["token"], "rationale": "",
        }],
    ])
    search = FakeSearchService(
        retrieve_response=[_result("similar incident", 0.1)],
        search_responses={"token": [_result("token expiry match", 0.05)]},
    )
    orchestrator = _make_orchestrator(llm, search)

    session = orchestrator.investigate("login fails", n_hypotheses=1)

    assert session.stopping_reason == StoppingReason.CRITIC_APPROVED
    assert session.total_iterations == 2
    assert session.iterations[0].critique.verdict == CritiqueVerdict.NEED_MORE_EVIDENCE
    assert session.iterations[1].critique.verdict == CritiqueVerdict.APPROVED
    assert session.final_report.critique.verdict == CritiqueVerdict.APPROVED
    # existing_root_causes was passed forward on the second call
    assert list(llm.calls[1]["existing_root_causes"]) == ["weak guess"]


# ── Maximum iteration stop ──────────────────────────────────────────────────────


def test_max_iterations_stop_when_never_approved() -> None:
    # confidence_score rises each iteration so composite_score keeps improving
    # (real progress every round), but a high-similarity top1 result alongside
    # two low-similarity ("contradicting") results keeps the contradiction
    # ratio >= the critic's threshold, pinning the verdict at
    # NEED_MORE_EVIDENCE so it never reaches APPROVED.
    mixed_evidence = [
        _result("best match", 0.3),  # similarity 0.7 -> top1, HIGH, supporting
        _result("weak match 1", 0.8),  # similarity 0.2 -> contradicting
        _result("weak match 2", 0.9),  # similarity 0.1 -> contradicting
    ]
    llm = FakeLLMService([
        [{
            "root_cause": f"guess-{i}", "confidence_score": 0.5 + 0.2 * i,
            "validation_keywords": ["mixed"], "rationale": "",
        }]
        for i in range(DEFAULT_MAX_ITERATIONS)
    ])
    search = FakeSearchService(
        retrieve_response=[_result("similar incident", 0.1)],
        search_responses={"mixed": mixed_evidence},
    )
    orchestrator = _make_orchestrator(llm, search)

    session = orchestrator.investigate("login fails", n_hypotheses=1)

    assert session.stopping_reason == StoppingReason.MAX_ITERATIONS
    assert session.total_iterations == DEFAULT_MAX_ITERATIONS
    assert session.final_report.critique.verdict != CritiqueVerdict.APPROVED


def test_max_iterations_respects_custom_config() -> None:
    llm = FakeLLMService([
        [{
            "root_cause": f"guess-{i}", "confidence_score": 1.0,
            "validation_keywords": ["zzz_no_match_zzz"], "rationale": "",
        }]
        for i in range(5)
    ])
    search = FakeSearchService(retrieve_response=[_result("similar incident", 0.1)])
    orchestrator = _make_orchestrator(llm, search, max_iterations=2)

    session = orchestrator.investigate("login fails", n_hypotheses=1)

    assert session.stopping_reason == StoppingReason.MAX_ITERATIONS
    assert session.total_iterations == 2


# ── No-progress stop ─────────────────────────────────────────────────────────────


def test_no_progress_stop_when_repeated_passes_do_not_improve() -> None:
    # Every iteration generates a DIFFERENT root cause (so NO_NEW_HYPOTHESES
    # never fires) but with identical evidence/score/critique outcomes,
    # so detect_progress should report no improvement after iteration 1.
    llm = FakeLLMService([
        [{
            "root_cause": f"guess-{i}", "confidence_score": 1.0,
            "validation_keywords": ["zzz_no_match_zzz"], "rationale": "",
        }]
        for i in range(5)
    ])
    search = FakeSearchService(retrieve_response=[_result("similar incident", 0.1)])
    orchestrator = _make_orchestrator(llm, search, max_iterations=5, require_progress=True)

    session = orchestrator.investigate("login fails", n_hypotheses=1)

    assert session.stopping_reason == StoppingReason.NO_PROGRESS
    assert session.total_iterations == 2
    assert "no measurable improvement" in session.stop_explanation


def test_require_progress_false_disables_no_progress_stop() -> None:
    llm = FakeLLMService([
        [{
            "root_cause": f"guess-{i}", "confidence_score": 1.0,
            "validation_keywords": ["zzz_no_match_zzz"], "rationale": "",
        }]
        for i in range(3)
    ])
    search = FakeSearchService(retrieve_response=[_result("similar incident", 0.1)])
    orchestrator = _make_orchestrator(
        llm, search, max_iterations=3, require_progress=False,
    )

    session = orchestrator.investigate("login fails", n_hypotheses=1)

    assert session.stopping_reason == StoppingReason.MAX_ITERATIONS
    assert session.total_iterations == 3


# ── No-new-hypothesis stop ───────────────────────────────────────────────────────


def test_no_new_hypotheses_stop_when_generator_repeats_itself() -> None:
    llm = FakeLLMService([
        [{
            "root_cause": "weak guess", "confidence_score": 1.0,
            "validation_keywords": ["zzz_no_match_zzz"], "rationale": "",
        }],
        [{
            "root_cause": "weak guess", "confidence_score": 1.0,
            "validation_keywords": ["zzz_no_match_zzz"], "rationale": "",
        }],
    ])
    search = FakeSearchService(retrieve_response=[_result("similar incident", 0.1)])
    orchestrator = _make_orchestrator(llm, search, max_iterations=5)

    session = orchestrator.investigate("login fails", n_hypotheses=1)

    assert session.stopping_reason == StoppingReason.NO_NEW_HYPOTHESES
    assert session.total_iterations == 1
    assert len(llm.calls) == 2  # the LLM call for the discarded attempt still ran


def test_no_new_hypotheses_stop_when_generator_returns_nothing_twice() -> None:
    llm = FakeLLMService([[], []])
    search = FakeSearchService(retrieve_response=[])
    orchestrator = _make_orchestrator(llm, search, max_iterations=5)

    session = orchestrator.investigate("totally unrelated coffee machine issue")

    assert session.stopping_reason == StoppingReason.NO_NEW_HYPOTHESES
    assert session.total_iterations == 1
    assert session.final_report.investigation.is_uncertain is True


# ── Deterministic execution ──────────────────────────────────────────────────────


def test_orchestrator_is_deterministic_given_identical_agent_outputs() -> None:
    def build():
        llm = FakeLLMService([
            [{
                "root_cause": "expired token", "confidence_score": 1.0,
                "validation_keywords": ["token"], "rationale": "",
            }],
        ])
        search = FakeSearchService(
            retrieve_response=[_result("similar incident", 0.1)],
            search_responses={"token": [_result("token expiry match", 0.05)]},
        )
        return _make_orchestrator(llm, search)

    session_a = build().investigate("login fails", n_hypotheses=1)
    session_b = build().investigate("login fails", n_hypotheses=1)

    assert session_a.stopping_reason == session_b.stopping_reason
    assert session_a.total_iterations == session_b.total_iterations
    assert session_a.final_report.critique.verdict == session_b.final_report.critique.verdict


# ── Immutable state ──────────────────────────────────────────────────────────────


def test_investigation_iteration_is_frozen() -> None:
    iteration = InvestigationIteration(
        iteration_number=1, plan=_plan(), hypotheses=(), evaluations={},
        decision=make_investigation_decision(()),
        critique=CritiqueResult(
            verdict=CritiqueVerdict.INCONCLUSIVE, confidence=0.0, findings=(),
            unresolved_questions=(), missing_evidence=(), recommended_actions=(),
            explanation="x",
        ),
        progress_note="baseline", rationale="baseline",
    )
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        iteration.iteration_number = 2  # type: ignore[misc]


def test_session_is_frozen() -> None:
    llm = FakeLLMService([
        [{
            "root_cause": "expired token", "confidence_score": 1.0,
            "validation_keywords": ["token"], "rationale": "",
        }],
    ])
    search = FakeSearchService(
        retrieve_response=[_result("similar incident", 0.1)],
        search_responses={"token": [_result("token expiry match", 0.05)]},
    )
    session = _make_orchestrator(llm, search).investigate("login fails", n_hypotheses=1)

    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        session.total_iterations = 99  # type: ignore[misc]


# ── Progress detection (unit-level) ──────────────────────────────────────────────


def _iteration(
    n: int, *, composite: float, accepted_id: str | None, supporting: int,
    verdict: CritiqueVerdict,
) -> InvestigationIteration:
    if accepted_id is None:
        decision = InvestigationDecision(
            accepted=None, accepted_score=None, rejected=(), is_uncertain=True, rationale="x",
        )
        evaluations = {}
    else:
        hypothesis = InvestigationHypothesis(
            id=accepted_id, root_cause="x", rationale="x", validation_keywords=(),
            raw_confidence=0.9,
        )
        score = HypothesisScore(
            hypothesis_id=accepted_id, raw_confidence=0.9, retrieval_confidence_level="HIGH",
            evidence_confidence_level="HIGH", supporting_count=supporting, contradicting_count=0,
            missing_count=0, composite_score=composite,
        )
        decision = InvestigationDecision(
            accepted=hypothesis, accepted_score=score, rejected=(), is_uncertain=False,
            rationale="x",
        )
        evaluations = {
            accepted_id: EvidenceEvaluation(
                hypothesis_id=accepted_id, query="q",
                supporting_evidence=tuple(f"s{i}" for i in range(supporting)),
                contradicting_evidence=(), missing_evidence=(),
                evidence_confidence_level="HIGH", evidence_top1_score=0.9,
            ),
        }
    critique = CritiqueResult(
        verdict=verdict, confidence=0.5, findings=(), unresolved_questions=(),
        missing_evidence=(), recommended_actions=(), explanation="x",
    )
    return InvestigationIteration(
        iteration_number=n, plan=_plan(), hypotheses=(), evaluations=evaluations,
        decision=decision, critique=critique, progress_note="", rationale="",
    )


def test_detect_progress_true_when_composite_score_improves() -> None:
    prev = _iteration(
        1, composite=0.5, accepted_id="h1", supporting=1,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    curr = _iteration(
        2, composite=0.7, accepted_id="h1", supporting=1,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    made_progress, note = detect_progress(prev, curr)
    assert made_progress is True
    assert "composite score improved" in note


def test_detect_progress_true_when_evidence_increases() -> None:
    prev = _iteration(
        1, composite=0.6, accepted_id="h1", supporting=1,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    curr = _iteration(
        2, composite=0.6, accepted_id="h1", supporting=3,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    made_progress, note = detect_progress(prev, curr)
    assert made_progress is True
    assert "supporting evidence increased" in note


def test_detect_progress_true_when_critique_verdict_improves() -> None:
    prev = _iteration(
        1, composite=0.6, accepted_id="h1", supporting=1,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    curr = _iteration(
        2, composite=0.6, accepted_id="h1", supporting=1, verdict=CritiqueVerdict.APPROVED,
    )
    made_progress, note = detect_progress(prev, curr)
    assert made_progress is True
    assert "critique verdict improved" in note


def test_detect_progress_false_when_nothing_changes() -> None:
    prev = _iteration(
        1, composite=0.6, accepted_id="h1", supporting=1,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    curr = _iteration(
        2, composite=0.6, accepted_id="h1", supporting=1,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    made_progress, note = detect_progress(prev, curr)
    assert made_progress is False
    assert "no measurable improvement" in note


def test_detect_progress_does_not_count_hypothesis_swap_to_lower_score() -> None:
    prev = _iteration(
        1, composite=0.7, accepted_id="h1", supporting=1,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    curr = _iteration(
        2, composite=0.6, accepted_id="h2", supporting=1,
        verdict=CritiqueVerdict.NEED_MORE_EVIDENCE,
    )
    made_progress, _note = detect_progress(prev, curr)
    assert made_progress is False


# ── Stopping reasons are a real enum, never collapsed to booleans ──────────────


def test_all_stopping_reasons_are_distinct_enum_members() -> None:
    assert {
        StoppingReason.CRITIC_APPROVED,
        StoppingReason.MAX_ITERATIONS,
        StoppingReason.NO_PROGRESS,
        StoppingReason.NO_NEW_HYPOTHESES,
    } == set(StoppingReason)


# ── Integration with Phases 19A-19C (real agents, no stubbing) ─────────────────


def test_orchestrator_uses_real_planner_and_critic_by_default() -> None:
    llm = FakeLLMService([
        [{
            "root_cause": "expired token", "confidence_score": 1.0,
            "validation_keywords": ["token"], "rationale": "",
        }],
    ])
    search = FakeSearchService(
        retrieve_response=[_result("similar incident", 0.1)],
        search_responses={"token": [_result("token expiry match", 0.05)]},
    )
    orchestrator = _make_orchestrator(llm, search)
    assert isinstance(orchestrator._critic, HeuristicCriticAgent)

    session = orchestrator.investigate("login fails with token error", n_hypotheses=1)

    assert session.iterations[0].plan.strategy == PlanningStrategy.AUTHENTICATION
    assert session.final_report.investigation.selected_hypothesis is not None


def test_orchestrator_accepts_injected_planner_and_critic() -> None:
    class _AlwaysApprovingCritic(CriticAgent):
        def critique(self, plan, decision, evaluations) -> CritiqueResult:
            return CritiqueResult(
                verdict=CritiqueVerdict.APPROVED, confidence=1.0, findings=(),
                unresolved_questions=(), missing_evidence=(), recommended_actions=(),
                explanation="stub critic always approves",
            )

    llm = FakeLLMService([
        [{
            "root_cause": "anything", "confidence_score": 0.5,
            "validation_keywords": [], "rationale": "",
        }],
    ])
    search = FakeSearchService(retrieve_response=[])
    config = OrchestratorConfig()
    orchestrator = MultiAgentInvestigationOrchestrator(
        db=None, config=config, critic=_AlwaysApprovingCritic(),
        search_service=search, llm_service=llm,
    )

    session = orchestrator.investigate("totally unrelated coffee machine issue")

    assert session.stopping_reason == StoppingReason.CRITIC_APPROVED
    assert session.total_iterations == 1


# ── Adaptive routing adoption (RoutedSearchService default) ─────────────────
#
# Requirement 6: MultiAgentInvestigationOrchestrator's default search_service
# is now a fully-wired RoutedSearchService, not a plain dense-only
# IncidentSearchService, when no explicit search_service is passed. A caller
# that DOES pass its own search_service (every other test in this file) is
# unaffected -- these tests cover only the default-construction path.


def test_default_search_service_is_a_routed_search_service(monkeypatch) -> None:
    import app.services.investigation_orchestrator as orch_mod
    from app.services.routed_search import RoutedSearchConfig, RoutedSearchService

    fake_dense = FakeSearchService(retrieve_response=[])
    fake_routed = RoutedSearchService(
        fake_dense, config=RoutedSearchConfig(routing_enabled=False)
    )
    monkeypatch.setattr(
        orch_mod, "build_routed_search_service", lambda db, **kw: fake_routed
    )
    llm = FakeLLMService([[{
        "root_cause": "x", "confidence_score": 0.9,
        "validation_keywords": [], "rationale": "",
    }]])

    orchestrator = MultiAgentInvestigationOrchestrator(db=None, llm_service=llm)

    assert orchestrator.search_service is fake_routed
    assert isinstance(orchestrator.search_service, RoutedSearchService)


def test_explicit_search_service_override_bypasses_routed_default(monkeypatch) -> None:
    """A caller passing its own search_service (e.g. evaluation.py's
    _build_orchestrator, which pins dense-only retrieval) is unaffected by
    the routed default -- proves backward compatibility for existing
    explicit-override callers.
    """
    import app.services.investigation_orchestrator as orch_mod

    build_calls: list[object] = []
    monkeypatch.setattr(
        orch_mod, "build_routed_search_service",
        lambda db, **kw: build_calls.append(db) or None,
    )
    explicit_dense = FakeSearchService(retrieve_response=[])
    llm = FakeLLMService([[{
        "root_cause": "x", "confidence_score": 0.9,
        "validation_keywords": [], "rationale": "",
    }]])

    orchestrator = MultiAgentInvestigationOrchestrator(
        db=None, search_service=explicit_dense, llm_service=llm
    )

    assert orchestrator.search_service is explicit_dense
    assert build_calls == []  # the routed factory was never even called


def test_routing_enabled_executes_bm25_for_initial_retrieval_and_evidence_search(
    monkeypatch,
) -> None:
    """End-to-end: with a real RoutedSearchService/RoutingEngine wired as
    the orchestrator's default, a short problem statement (<=3 tokens, no
    filters) is routed to BM25 for BOTH the initial retrieve() call and each
    hypothesis's evidence search() call -- proving investigations genuinely
    execute adaptive retrieval, not just construct a RoutedSearchService
    that happens to default to dense.
    """
    import uuid

    import app.services.investigation_orchestrator as orch_mod
    from app.services.routed_search import RoutedSearchConfig, RoutedSearchService
    from app.services.routing import DefaultRuleBasedRoutingPolicy, RoutingEngine

    class _FakeDense:
        def __init__(self):
            self.db = SimpleNamespace(get=lambda model, iid: None)
            self.llm_service = None
            self.retrieve_calls: list[dict] = []

        def retrieve(self, query, **kwargs):
            self.retrieve_calls.append({"query": query, **kwargs})
            return []

    class _FakeBM25:
        def __init__(self):
            self.calls: list[str] = []

        def retrieve(self, query, *, limit=10):
            self.calls.append(query)
            incident = SimpleNamespace(
                id=uuid.uuid4(), title="db pool exhausted", symptoms=[],
            )
            return [SimpleNamespace(document_id=str(incident.id), score=3.0, _incident=incident)]

    fake_dense = _FakeDense()
    fake_bm25 = _FakeBM25()
    # RoutedSearchService resolves BM25 hits via dense.db.get(Incident, id) --
    # wire it to return the same incident the fake BM25 retriever reports.
    returned_incidents: dict = {}

    def _bm25_retrieve(query, *, limit=10):
        fake_bm25.calls.append(query)
        incident = SimpleNamespace(id=uuid.uuid4(), title="db pool exhausted", symptoms=[])
        returned_incidents[str(incident.id)] = incident
        return [SimpleNamespace(document_id=str(incident.id), score=3.0)]

    fake_bm25.retrieve = _bm25_retrieve
    fake_dense.db = SimpleNamespace(get=lambda model, iid: returned_incidents.get(str(iid)))

    routed = RoutedSearchService(
        fake_dense, bm25=fake_bm25,
        routing_engine=RoutingEngine(DefaultRuleBasedRoutingPolicy()),
        config=RoutedSearchConfig(routing_enabled=True),
    )
    monkeypatch.setattr(orch_mod, "build_routed_search_service", lambda db, **kw: routed)

    llm = FakeLLMService([[{
        "root_cause": "db pool exhausted", "confidence_score": 0.9,
        "validation_keywords": ["db"], "rationale": "matches evidence",
    }]])

    orchestrator = MultiAgentInvestigationOrchestrator(db=None, llm_service=llm)
    session = orchestrator.investigate("db slow", n_hypotheses=1)

    # Initial retrieve() (problem="db slow", 2 tokens) and the hypothesis's
    # evidence search() (query="db", 1 token) both route to BM25 -- neither
    # ever reached dense.retrieve().
    assert fake_dense.retrieve_calls == []
    assert fake_bm25.calls == ["db slow", "db"]
    assert routed.last_observation.effective_strategy.value == "bm25"
    assert routed.last_observation.routing_enabled is True
    assert session.final_report.investigation.selected_hypothesis is not None
    assert session.final_report.investigation.selected_hypothesis.root_cause == "db pool exhausted"
