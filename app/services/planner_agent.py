"""Planner Agent (Phase 19B).

Introduces the first true reasoning agent ahead of hypothesis generation:
given a problem (and, optionally, the incidents already retrieved for it
and a Phase 18B ``RoutingObservation``), the ``PlannerAgent`` decides *how*
the investigation should proceed — objective, strategy, priorities,
evidence-collection priorities, assumptions, expected difficulty — and
records that decision as an immutable ``InvestigationPlan``. The Planner
does NOT generate hypotheses; Phase 19A's ``HypothesisGenerator`` remains
the only thing that does, completely unmodified, and remains strategy-
independent (it never branches on ``InvestigationPlan.strategy`` — see
"Planner integration").

This phase does NOT modify ``app.services.hypothesis_investigation``
(Phase 19A) or anything in ``app.services.routing``/``app.services.routed_search``
(Phase 18A/18B) — every type/field this module reads from those modules is
read-only. It does NOT introduce a Critic agent, an Evidence agent, or any
multi-agent orchestration, and makes ZERO LLM calls — ``RuleBasedPlanner``
is fully deterministic, the same architectural choice Phase 18A's
``DefaultRuleBasedRoutingPolicy`` already made for the same reason (a
simple, explainable, first-version policy; tuning/ML/LLM planning is
explicitly deferred).

# Updated architecture

```
                                  Problem
                                     │
                                     ▼
                  search_service.retrieve(problem, ...)   [pre-16,
                                     │                      unmodified]
                                     ▼
                            retrieved incidents
                                     │
                                     ▼
                    PlannerAgent.plan(problem, retrieved_incidents=,
                                       routing_observation=)
                          (THIS PHASE — exactly one new agent)
                                     │
                                     ▼
                            InvestigationPlan
                       (objective, strategy, priority_list,
                        evidence_priorities, assumptions,
                        expected_difficulty, strategy_rationale)
                                     │
                                     ▼
              plan_then_generate_hypotheses(plan, generator)
                  (NEW — folds the plan into the SAME context string
                   HypothesisGenerator.generate() already accepts;
                   HypothesisGenerator itself, Phase 19A, UNMODIFIED)
                                     │
                                     ▼
                  tuple[InvestigationHypothesis, ...]   [19A, unmodified type]
                                     │
                                     ▼
            HypothesisEvaluator.evaluate() per hypothesis   [19A, unmodified]
                                     │
                                     ▼
                    score_hypothesis() per hypothesis        [19A, unmodified]
                                     │
                                     ▼
                  make_investigation_decision(scored)         [19A, unmodified]
                                     │
                                     ▼
                  build_investigation_report(...)             [19A, unmodified]
                                     │
                                     ▼
                          InvestigationReport                  [19A, unmodified type]
```

# Planning lifecycle

```
PlannerAgent.plan(problem, *, retrieved_incidents=(), routing_observation=None)
  1. (RuleBasedPlanner only) check routing_observation.signals.has_stack_trace
     first, if a routing_observation was supplied — an unambiguous, already-
     computed signal (Phase 18A) that overrides keyword matching entirely
     (see "Strategy selection methodology")
  2. else: match problem text + retrieved incidents' titles/symptoms against
     each strategy's keyword set, in a fixed, documented priority order —
     first match wins, exactly Phase 18A's "first matching rule wins"
     philosophy
  3. no match at all -> PlanningStrategy.UNKNOWN
  4. look up that strategy's deterministic plan template (objective,
     priority_list, evidence_priorities, assumptions, expected_difficulty)
  5. -> InvestigationPlan(problem=, strategy=, ..., strategy_rationale=
     <which signal/keyword matched, in plain text>)
```

# InvestigationPlan design

A frozen dataclass, never a dict (per this phase's explicit requirement):
``problem``, ``strategy`` (a ``PlanningStrategy``), ``objective`` (one
sentence describing what this investigation is trying to establish),
``priority_list`` (ordered investigation steps, most important first),
``evidence_priorities`` (what kind of retrieved evidence matters most for
this strategy), ``assumptions`` (what the plan is assuming to be true —
useful for the eventual report's "remaining uncertainty"), ``expected_difficulty``
(``"low"``/``"medium"``/``"high"``, a coarse, deterministic-per-strategy
estimate, not a model prediction), and ``strategy_rationale`` (see
"Explainability" below).

# Strategy selection methodology

``RuleBasedPlanner`` checks, in this fixed order, the FIRST of which
matches wins (mirrors ``DefaultRuleBasedRoutingPolicy``'s "priority order,
first match wins" structure — Phase 18A):

```
0. routing_observation.signals.has_stack_trace (if a routing_observation
   was supplied at all)         -> APPLICATION_FAILURE
   (an unambiguous signal already computed by Phase 18A; checked before
    any keyword matching because it is more specific than any keyword set
    below)
1. AUTHENTICATION keywords      ("auth", "token", "credential", "401",
                                  "403", "permission", "oauth", "jwt",
                                  "certificate", "tls", "ssl", ...)
2. NETWORK keywords             ("timeout", "connection refused", "dns",
                                  "network", "firewall", "latency",
                                  "socket", "proxy", ...)
3. INFRASTRUCTURE_FAILURE kw    ("kubernetes", "pod", "node", "cluster",
                                  "oom", "disk", "cpu", "kubelet",
                                  "container", "memory pressure", ...)
4. CONFIGURATION keywords       ("config", "yaml", "environment
                                  variable", "env var", "flag",
                                  "misconfigured", "settings", ...)
5. APPLICATION_FAILURE keywords ("exception", "stack trace", "crash",
                                  "panic", "segfault", "null pointer",
                                  "bug", "unhandled", ...)
6. (none matched)               -> UNKNOWN
```

This order is deliberately **most-specific/narrow first, broadest last**:
authentication and network terms are rarely used loosely (a problem
mentioning "401" is almost certainly about auth), whereas application-
failure terms ("crash", "exception") are broad enough to co-occur with
almost any other category, so they are checked last among the keyword
rules — a problem that mentions both "401" and "crash" is more usefully
routed to AUTHENTICATION (the more actionable, specific category) than to
APPLICATION_FAILURE. None of these keyword sets, thresholds, or the
priority order were tuned against any dataset — they are illustrative,
explainable v1 defaults, the same status Phase 18A's routing thresholds
have.

# Planner integration

``HypothesisGenerator`` (Phase 19A) is NOT modified and does not gain a
new parameter. Per this phase's "the hypothesis generator should remain
strategy-independent," ``plan_then_generate_hypotheses()`` is the new
integration point: it renders the ``InvestigationPlan`` into a plain-text
block (objective, strategy name, priorities, evidence priorities,
assumptions) and PREPENDS it to the ``context`` string already passed to
``HypothesisGenerator.generate(problem=, context=, n=)`` — the same
parameter Phase 19A already exposed for exactly this kind of enrichment.
``HypothesisGenerator`` itself never sees a ``PlanningStrategy`` value or
branches on it; it only ever sees a (now richer) string, preserving
strategy independence at the type level, not just by convention.

# Explainability

``InvestigationPlan.strategy_rationale`` is a plain-text sentence stating
exactly which signal or keyword caused ``RuleBasedPlanner`` to select its
strategy (e.g. ``"matched authentication keyword 'token' in the problem
text"`` or ``"routing observation reported has_stack_trace=True"``) —
intended for observability/logging today and for a future evaluation
phase that wants to audit planner decisions the same way Phase 18B logs
routing decisions and Phase 18A's ``RoutingDecision.reason`` already
works. Recording *why*, not just *what*, is the same explainability
contract every rule-based component in this codebase has followed since
Phase 18A.

# Risks discovered

- **Keyword sets are unvalidated.** Like Phase 18A's routing thresholds,
  these keyword lists were written by inspection, not derived from any
  gold dataset of investigation problems — they may both over- and under-
  match on real production problem text.
- **The "broadest category last" ordering can mis-route mixed-signal
  problems.** A problem mentioning both an authentication term and a
  clear stack trace (and no ``routing_observation``) is routed to
  AUTHENTICATION, not APPLICATION_FAILURE, purely because of priority
  order, not because authentication is actually the more likely category
  for that specific problem.
- **``routing_observation`` is a best-effort optional input, not a
  guarantee.** Most callers of ``PlannerAgent.plan()`` will not have one
  (Phase 18B's ``RoutedSearchService`` is not wired into any agent yet),
  so in practice keyword matching is the only signal exercised in
  production today — the stack-trace override exists but will rarely
  fire until a future phase wires routing observability into the
  investigation pipeline.
- **`expected_difficulty` is a static per-strategy label, not computed
  from anything about the specific problem.** Two very different
  AUTHENTICATION problems get the same difficulty label today.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.services.hypothesis_investigation import (
    DEFAULT_HYPOTHESIS_COUNT,
    HypothesisEvaluator,
    HypothesisGenerator,
    InvestigationHypothesis,
    InvestigationReport,
    build_investigation_report,
    make_investigation_decision,
    score_hypothesis,
)
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchResult, IncidentSearchService

if TYPE_CHECKING:
    from app.services.routed_search import RoutingObservation


class PlanningStrategy(str, Enum):
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    APPLICATION_FAILURE = "application_failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InvestigationPlan:
    """Immutable record of how the planner decided an investigation should
    proceed. See module docstring's "InvestigationPlan design".
    """

    problem: str
    strategy: PlanningStrategy
    objective: str
    priority_list: tuple[str, ...]
    evidence_priorities: tuple[str, ...]
    assumptions: tuple[str, ...]
    expected_difficulty: str
    strategy_rationale: str


# ── Planner interface ──────────────────────────────────────────────────────────


class PlannerAgent(ABC):
    """The swappable extension point. Future implementations
    (``MLPlanner``, ``LLMPlanner``, ``HybridPlanner``) implement this
    single method — the investigation pipeline depends only on this
    interface, never on ``RuleBasedPlanner`` directly.
    """

    @abstractmethod
    def plan(
        self,
        problem: str,
        *,
        retrieved_incidents: Sequence[IncidentSearchResult] = (),
        routing_observation: "RoutingObservation | None" = None,
    ) -> InvestigationPlan:
        """Return an ``InvestigationPlan`` for ``problem``. Must NOT
        generate hypotheses — only decide how the investigation should
        proceed.
        """


# ── Per-strategy plan templates ─────────────────────────────────────────────────


@dataclass(frozen=True)
class _StrategyTemplate:
    objective: str
    priority_list: tuple[str, ...]
    evidence_priorities: tuple[str, ...]
    assumptions: tuple[str, ...]
    expected_difficulty: str


_TEMPLATES: dict[PlanningStrategy, _StrategyTemplate] = {
    PlanningStrategy.AUTHENTICATION: _StrategyTemplate(
        objective=(
            "Determine whether the failure originates in authentication/authorization "
            "(credentials, tokens, permissions) rather than elsewhere in the request path."
        ),
        priority_list=(
            "Check token/credential validity and expiry",
            "Check permission/role assignment",
            "Check for recent identity-provider or certificate changes",
        ),
        evidence_priorities=(
            "Incidents mentioning 401/403/token/credential errors",
            "Incidents involving the same auth provider or service",
        ),
        assumptions=(
            "The failure is reproducible at the authentication boundary, not downstream.",
        ),
        expected_difficulty="medium",
    ),
    PlanningStrategy.NETWORK: _StrategyTemplate(
        objective=(
            "Determine whether the failure is caused by network-layer issues "
            "(connectivity, DNS, latency, routing) rather than application or "
            "infrastructure logic."
        ),
        priority_list=(
            "Check connectivity between the affected components",
            "Check DNS resolution and routing paths",
            "Check for recent network/firewall/proxy changes",
        ),
        evidence_priorities=(
            "Incidents mentioning timeouts, connection resets, or DNS failures",
            "Incidents involving the same network path or service boundary",
        ),
        assumptions=("The failure manifests as connectivity/latency, not a logic error.",),
        expected_difficulty="medium",
    ),
    PlanningStrategy.INFRASTRUCTURE_FAILURE: _StrategyTemplate(
        objective=(
            "Determine whether the failure stems from a resource/scheduling constraint "
            "in the underlying infrastructure rather than application logic."
        ),
        priority_list=(
            "Check for resource exhaustion (CPU/memory/disk)",
            "Check for node/pod scheduling or restart events",
            "Check for infrastructure-level error signatures",
        ),
        evidence_priorities=(
            "Incidents mentioning resource limits or OOM",
            "Incidents mentioning node/pod lifecycle events",
        ),
        assumptions=(
            "The symptom is observable at the infrastructure layer, not only in app logs.",
        ),
        expected_difficulty="medium",
    ),
    PlanningStrategy.CONFIGURATION: _StrategyTemplate(
        objective=(
            "Determine whether the failure was introduced by a configuration or "
            "environment change rather than a code defect."
        ),
        priority_list=(
            "Identify what configuration or environment variable changed recently",
            "Check for misconfigured flags or settings",
            "Compare against a known-good configuration baseline",
        ),
        evidence_priorities=(
            "Incidents referencing the same configuration key/flag",
            "Incidents following a deployment or settings change",
        ),
        assumptions=(
            "A configuration/environment change occurred near when the symptom appeared.",
        ),
        expected_difficulty="low",
    ),
    PlanningStrategy.APPLICATION_FAILURE: _StrategyTemplate(
        objective=(
            "Determine whether the failure is caused by an application-level defect "
            "(exception, crash, unhandled case) rather than infrastructure, "
            "configuration, auth, or network."
        ),
        priority_list=(
            "Identify the exact exception/error signature and stack location",
            "Check for a recent code change touching the affected path",
            "Check for known similar defects in history",
        ),
        evidence_priorities=(
            "Incidents sharing the same exception/error signature",
            "Incidents in the same module/component",
        ),
        assumptions=("The failure reproduces consistently given the same input/code path.",),
        expected_difficulty="medium",
    ),
    PlanningStrategy.UNKNOWN: _StrategyTemplate(
        objective=(
            "No strong signal pointed to a specific investigation category; proceed with "
            "a broad, unconstrained hypothesis search rather than a targeted one."
        ),
        priority_list=(
            "Gather more context before narrowing scope",
            "Treat all plausible root-cause categories as open",
        ),
        evidence_priorities=("Any incident with notable textual or symptom overlap",),
        assumptions=("Insufficient signal exists yet to assume a specific failure category.",),
        expected_difficulty="high",
    ),
}

_KEYWORDS: dict[PlanningStrategy, tuple[str, ...]] = {
    PlanningStrategy.AUTHENTICATION: (
        "auth", "login", "token", "credential", "permission", "403", "401",
        "unauthorized", "oauth", "jwt", "ssl", "certificate", "tls",
    ),
    PlanningStrategy.NETWORK: (
        "timeout", "connection refused", "dns", "network", "firewall", "latency",
        "socket", "tcp", "proxy", "unreachable",
    ),
    PlanningStrategy.INFRASTRUCTURE_FAILURE: (
        "kubernetes", "pod", "node", "cluster", "container", "crashloopbackoff",
        "oom", "disk", "cpu", "memory pressure", "kubelet", "docker",
    ),
    PlanningStrategy.CONFIGURATION: (
        "config", "yaml", "environment variable", "env var", "misconfigured",
        "settings", "feature flag", "helm values",
    ),
    PlanningStrategy.APPLICATION_FAILURE: (
        "exception", "stack trace", "null pointer", "crash", "bug", "panic",
        "segfault", "null reference", "unhandled",
    ),
}

# Most-specific/narrow first, broadest last — see module docstring's
# "Strategy selection methodology".
_KEYWORD_PRIORITY_ORDER: tuple[PlanningStrategy, ...] = (
    PlanningStrategy.AUTHENTICATION,
    PlanningStrategy.NETWORK,
    PlanningStrategy.INFRASTRUCTURE_FAILURE,
    PlanningStrategy.CONFIGURATION,
    PlanningStrategy.APPLICATION_FAILURE,
)


def _incident_text(result: IncidentSearchResult) -> str:
    incident = result.incident
    symptoms = " ".join(symptom.text for symptom in incident.symptoms)
    return f"{incident.title} {symptoms}"


class RuleBasedPlanner(PlannerAgent):
    """Deterministic, keyword-based ``PlannerAgent`` — see module
    docstring's "Strategy selection methodology". Makes zero LLM calls.
    """

    def plan(
        self,
        problem: str,
        *,
        retrieved_incidents: Sequence[IncidentSearchResult] = (),
        routing_observation: "RoutingObservation | None" = None,
    ) -> InvestigationPlan:
        if routing_observation is not None and routing_observation.signals.has_stack_trace:
            strategy = PlanningStrategy.APPLICATION_FAILURE
            rationale = "routing observation reported has_stack_trace=True"
        else:
            haystack = " ".join(
                [problem] + [_incident_text(result) for result in retrieved_incidents]
            ).lower()
            strategy, rationale = self._match_keywords(haystack)

        template = _TEMPLATES[strategy]
        return InvestigationPlan(
            problem=problem,
            strategy=strategy,
            objective=template.objective,
            priority_list=template.priority_list,
            evidence_priorities=template.evidence_priorities,
            assumptions=template.assumptions,
            expected_difficulty=template.expected_difficulty,
            strategy_rationale=rationale,
        )

    def _match_keywords(self, haystack: str) -> tuple[PlanningStrategy, str]:
        for strategy in _KEYWORD_PRIORITY_ORDER:
            for keyword in _KEYWORDS[strategy]:
                if re.search(rf"\b{re.escape(keyword)}\b", haystack):
                    return (
                        strategy,
                        f"matched {strategy.value} keyword {keyword!r} in problem text",
                    )
        return PlanningStrategy.UNKNOWN, "no strategy keyword matched the problem text or evidence"


# ── Integration with Phase 19A (unmodified) ──────────────────────────────────────


def _render_plan_context(plan: InvestigationPlan, retrieval_context: str) -> str:
    plan_block = "\n".join(
        [
            f"Investigation strategy: {plan.strategy.value} ({plan.strategy_rationale})",
            f"Objective: {plan.objective}",
            "Priorities:\n" + "\n".join(f"  - {item}" for item in plan.priority_list),
            "Evidence priorities:\n"
            + "\n".join(f"  - {item}" for item in plan.evidence_priorities),
            "Assumptions:\n" + "\n".join(f"  - {item}" for item in plan.assumptions),
            f"Expected difficulty: {plan.expected_difficulty}",
        ]
    )
    return f"{plan_block}\n\n{retrieval_context}"


def plan_then_generate_hypotheses(
    plan: InvestigationPlan,
    generator: HypothesisGenerator,
    *,
    retrieval_context: str = "",
    n: int = DEFAULT_HYPOTHESIS_COUNT,
    existing_root_causes: Sequence[str] | None = None,
) -> tuple[InvestigationHypothesis, ...]:
    """The integration point: folds ``plan`` into the ``context`` string
    ``HypothesisGenerator.generate()`` (Phase 19A, unmodified) already
    accepts. ``HypothesisGenerator`` never sees ``plan.strategy`` directly
    — only this rendered text — preserving strategy independence at the
    type level (see module docstring's "Planner integration").
    """
    context = _render_plan_context(plan, retrieval_context)
    return generator.generate(
        problem=plan.problem, context=context, n=n, existing_root_causes=existing_root_causes
    )


# ── Orchestrator: Planner -> 19A pipeline, end to end ───────────────────────────


class PlannedInvestigationAgent:
    """Wires ``PlannerAgent`` ahead of the unmodified Phase 19A pipeline —
    see module docstring's "Updated architecture". The planner is injected
    (constructor parameter), defaulting to ``RuleBasedPlanner``, so a
    future ``LLMPlanner``/``HybridPlanner`` can be substituted without
    changing this class.
    """

    def __init__(
        self,
        db: Session,
        *,
        planner: PlannerAgent | None = None,
        search_service: IncidentSearchService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.search_service = search_service or IncidentSearchService(db)
        self.llm_service = llm_service or LLMService()
        self._planner = planner or RuleBasedPlanner()
        self._generator = HypothesisGenerator(self.llm_service)
        self._evaluator = HypothesisEvaluator(self.search_service)

    def investigate(
        self,
        problem: str,
        *,
        n_hypotheses: int = DEFAULT_HYPOTHESIS_COUNT,
        routing_observation: "RoutingObservation | None" = None,
    ) -> tuple[InvestigationPlan, InvestigationReport]:
        initial_results = self.search_service.retrieve(
            problem, limit=10, expand=True, rerank=True,
            call_site="planner_agent.investigate",
        )
        _, retrieval_confidence_level = IncidentSearchService.confidence_for(initial_results)

        plan = self._planner.plan(
            problem, retrieved_incidents=initial_results, routing_observation=routing_observation
        )

        retrieval_context = f"Retrieval confidence: {retrieval_confidence_level}"
        hypotheses = plan_then_generate_hypotheses(
            plan, self._generator, retrieval_context=retrieval_context, n=n_hypotheses
        )
        if not hypotheses:
            decision = make_investigation_decision(())
            return plan, build_investigation_report(problem, decision, {})

        evaluations = {
            hypothesis.id: self._evaluator.evaluate(hypothesis) for hypothesis in hypotheses
        }
        scored = [
            (
                hypothesis,
                score_hypothesis(
                    hypothesis, evaluations[hypothesis.id],
                    retrieval_confidence_level=retrieval_confidence_level,
                ),
            )
            for hypothesis in hypotheses
        ]
        decision = make_investigation_decision(scored)
        report = build_investigation_report(problem, decision, evaluations)
        return plan, report
