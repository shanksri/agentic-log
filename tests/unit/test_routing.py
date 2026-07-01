from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.routing import (
    DefaultRuleBasedRoutingPolicy,
    RoutingDecision,
    RoutingEngine,
    RoutingPolicy,
    RoutingSignals,
    RoutingStrategy,
    extract_routing_signals,
)

# ── Signal extraction ────────────────────────────────────────────────────────────


def test_extract_signals_detects_python_traceback() -> None:
    query = 'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: bad'
    signals = extract_routing_signals(query)
    assert signals.has_stack_trace is True


def test_extract_signals_detects_java_style_frame() -> None:
    query = "crash at com.example.Foo.bar(Foo.java:42)"
    signals = extract_routing_signals(query)
    assert signals.has_stack_trace is True


def test_extract_signals_detects_python_file_line_frame() -> None:
    query = 'see File "app/main.py", line 10 for the failure'
    signals = extract_routing_signals(query)
    assert signals.has_stack_trace is True


def test_extract_signals_no_stack_trace_for_plain_query() -> None:
    signals = extract_routing_signals("scheduler crashes during startup")
    assert signals.has_stack_trace is False


def test_extract_signals_detects_camelcase_error_signature() -> None:
    signals = extract_routing_signals("scheduler crashloop ValidationError dag_version_id is NULL")
    assert signals.has_exact_error_signature is True


def test_extract_signals_detects_exception_suffix() -> None:
    signals = extract_routing_signals("NullPointerException during startup")
    assert signals.has_exact_error_signature is True


def test_extract_signals_detects_ticket_like_identifier() -> None:
    signals = extract_routing_signals("ZookeeperConsumerConnectorMBean issue KAFKA-17")
    assert signals.has_exact_error_signature is True


def test_extract_signals_no_error_signature_for_plain_query() -> None:
    signals = extract_routing_signals("credit card payment processing failure")
    assert signals.has_exact_error_signature is False


def test_extract_signals_detects_backtick_identifier() -> None:
    signals = extract_routing_signals("what does `dag_version_id` mean")
    assert signals.has_quoted_identifier is True


def test_extract_signals_detects_double_quoted_identifier() -> None:
    signals = extract_routing_signals('error containing "memory.high" cgroup setting')
    assert signals.has_quoted_identifier is True


def test_extract_signals_plain_quoted_sentence_is_not_an_identifier() -> None:
    # Spaces inside the quotes disqualify it as an "identifier" quote.
    signals = extract_routing_signals('the error says "something went wrong here"')
    assert signals.has_quoted_identifier is False


def test_extract_signals_token_count() -> None:
    signals = extract_routing_signals("one two three four")
    assert signals.token_count == 4


def test_extract_signals_empty_query_has_zero_tokens_and_zero_density() -> None:
    signals = extract_routing_signals("")
    assert signals.token_count == 0
    assert signals.lexical_density == 0.0


def test_extract_signals_lexical_density_hand_computed() -> None:
    # tokens: ["memory", "leak", "memory", "leak"] -> 2 unique / 4 total = 0.5
    signals = extract_routing_signals("memory leak memory leak")
    assert signals.lexical_density == pytest.approx(0.5)


def test_extract_signals_preserves_original_query() -> None:
    signals = extract_routing_signals("Some Query")
    assert signals.query == "Some Query"


def test_routing_signals_is_frozen() -> None:
    signals = extract_routing_signals("q")
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        signals.token_count = 99  # type: ignore[misc]


# ── DefaultRuleBasedRoutingPolicy: one rule at a time ────────────────────────────


def _decide(query: str) -> RoutingDecision:
    policy = DefaultRuleBasedRoutingPolicy()
    return policy.decide(query, extract_routing_signals(query))


def test_stack_trace_routes_to_bm25() -> None:
    decision = _decide('Traceback (most recent call last):\n  File "x.py", line 1')
    assert decision.strategy == RoutingStrategy.BM25
    assert "stack trace" in decision.reason


def test_exact_error_signature_routes_to_bm25() -> None:
    decision = _decide("scheduler crashloop ValidationError UUID dag_version_id is NULL")
    assert decision.strategy == RoutingStrategy.BM25
    assert "error signature" in decision.reason


def test_quoted_identifier_routes_to_bm25() -> None:
    decision = _decide("what is the purpose of the `canonical_text` field in this long query")
    assert decision.strategy == RoutingStrategy.BM25
    assert "quoted identifier" in decision.reason


def test_short_query_routes_to_bm25() -> None:
    decision = _decide("memory leak")
    assert decision.strategy == RoutingStrategy.BM25
    assert "short query" in decision.reason


def test_long_query_routes_to_hybrid() -> None:
    decision = _decide(
        "Kubernetes memory management issues combining MemoryQoS not setting limits "
        "for BestEffort pods alongside high memory usage in the controller manager process"
    )
    assert decision.strategy == RoutingStrategy.HYBRID
    assert "long query" in decision.reason


def test_medium_query_with_no_signals_routes_to_dense() -> None:
    decision = _decide("background scheduler refuses to launch after upgrade")
    assert decision.strategy == RoutingStrategy.DENSE
    assert "no strong lexical signal" in decision.reason


# ── Priority order / ambiguous queries ───────────────────────────────────────────


def test_stack_trace_wins_over_short_query_rule() -> None:
    # Only 2 tokens after the colon, but the traceback marker should win
    # over the short-query rule, not the other way around.
    decision = _decide("Traceback (most recent call last):")
    assert decision.strategy == RoutingStrategy.BM25
    assert "stack trace" in decision.reason


def test_error_signature_wins_over_long_query_rule() -> None:
    long_query_with_error = (
        "investigating a recurring issue where the background scheduler process "
        "throws a ValidationError during startup after the latest deployment rollout"
    )
    signals = extract_routing_signals(long_query_with_error)
    assert signals.token_count >= DefaultRuleBasedRoutingPolicy.LONG_QUERY_TOKEN_THRESHOLD
    assert signals.has_exact_error_signature is True

    decision = _decide(long_query_with_error)
    assert decision.strategy == RoutingStrategy.BM25
    assert "error signature" in decision.reason


def test_quoted_identifier_wins_over_short_query_rule() -> None:
    decision = _decide("`foo`")
    assert decision.strategy == RoutingStrategy.BM25
    assert "quoted identifier" in decision.reason


# ── Deterministic behavior ───────────────────────────────────────────────────────


def test_decide_is_deterministic_across_repeated_calls() -> None:
    query = "background scheduler refuses to launch after upgrade"
    first = _decide(query)
    second = _decide(query)
    assert first.strategy == second.strategy
    assert first.reason == second.reason


def test_engine_route_is_deterministic_across_repeated_calls() -> None:
    engine = RoutingEngine(DefaultRuleBasedRoutingPolicy())
    query = "memory leak"
    assert engine.route(query).strategy == engine.route(query).strategy


# ── RoutingEngine: depends only on the policy interface ─────────────────────────


class _AlwaysHybridPolicy(RoutingPolicy):
    def decide(self, query: str, signals: RoutingSignals) -> RoutingDecision:
        return RoutingDecision(
            strategy=RoutingStrategy.HYBRID, reason="always hybrid (test stub)", signals=signals
        )


class _AlwaysDensePolicy(RoutingPolicy):
    def decide(self, query: str, signals: RoutingSignals) -> RoutingDecision:
        return RoutingDecision(
            strategy=RoutingStrategy.DENSE, reason="always dense (test stub)", signals=signals
        )


def test_engine_uses_injected_policy() -> None:
    engine = RoutingEngine(_AlwaysHybridPolicy())
    decision = engine.route("memory leak")  # would be BM25 under the default policy
    assert decision.strategy == RoutingStrategy.HYBRID


def test_swapping_policy_changes_decision_without_changing_engine() -> None:
    query = "memory leak"
    hybrid_engine = RoutingEngine(_AlwaysHybridPolicy())
    dense_engine = RoutingEngine(_AlwaysDensePolicy())

    assert hybrid_engine.route(query).strategy == RoutingStrategy.HYBRID
    assert dense_engine.route(query).strategy == RoutingStrategy.DENSE


def test_engine_exposes_its_policy() -> None:
    policy = _AlwaysHybridPolicy()
    engine = RoutingEngine(policy)
    assert engine.policy is policy


def test_engine_computes_signals_once_and_passes_to_policy() -> None:
    captured: list[RoutingSignals] = []

    class _CapturingPolicy(RoutingPolicy):
        def decide(self, query: str, signals: RoutingSignals) -> RoutingDecision:
            captured.append(signals)
            return RoutingDecision(strategy=RoutingStrategy.DENSE, reason="r", signals=signals)

    engine = RoutingEngine(_CapturingPolicy())
    engine.route("memory leak issue")

    assert len(captured) == 1
    assert captured[0].query == "memory leak issue"
    assert captured[0].token_count == 3


def test_routing_decision_is_frozen() -> None:
    signals = extract_routing_signals("q")
    decision = RoutingDecision(strategy=RoutingStrategy.DENSE, reason="r", signals=signals)
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        decision.reason = "changed"  # type: ignore[misc]


# ── RoutingStrategy values match Phase 17C's build_strategy vocabulary ─────────


def test_routing_strategy_values_match_phase17c_strategy_names() -> None:
    assert RoutingStrategy.DENSE.value == "dense"
    assert RoutingStrategy.BM25.value == "bm25"
    assert RoutingStrategy.HYBRID.value == "hybrid"


# ── Integration with retrieval strategies (Phase 17C, not modified) ─────────────


def test_routing_decision_dispatches_via_build_strategy() -> None:
    """Demonstrates that a RoutingDecision can drive Phase 17C's
    build_strategy() purely through its .value — without RoutingEngine or
    any policy importing app.evaluation.retrieval_strategies. The wiring
    lives entirely in this test (the caller's responsibility), matching
    the module docstring's "Why RoutingEngine never touches a retriever".
    """
    from app.evaluation.retrieval_strategies import build_strategy
    from app.services.bm25_search import BM25Retriever

    decision = _decide("memory leak")  # routes to BM25
    assert decision.strategy == RoutingStrategy.BM25

    dense_stub = SimpleNamespace(db="sentinel-db")
    bm25 = BM25Retriever.from_documents([])
    strategy = build_strategy(decision.strategy.value, search_service=dense_stub, bm25=bm25)
    assert strategy is not dense_stub  # got a BM25RetrievalAdapter, not the dense passthrough


def test_dense_decision_dispatches_to_search_service_unchanged() -> None:
    from app.evaluation.retrieval_strategies import build_strategy

    decision = _decide("background scheduler refuses to launch after upgrade")
    assert decision.strategy == RoutingStrategy.DENSE

    dense_stub = object()
    strategy = build_strategy(decision.strategy.value, search_service=dense_stub)
    assert strategy is dense_stub
