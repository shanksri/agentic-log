"""Hypothesis-Driven Investigation Framework (Phase 19A).

Replaces today's single-pass reasoning shape —

    Question -> Retrieve candidates -> One LLM prompt -> Final answer

(``InvestigationAgent``, pre-16) or its two-LLM-call elaboration
(``AdvancedInvestigationAgent``'s Policy B, Phase 6A: generate hypotheses,
then ask the LLM a SECOND time to synthesize a final report from them) —
with an explicit reasoning architecture that generates competing
hypotheses, evaluates each independently against retrieved evidence,
scores them from that evidence (not from a second LLM judgment call), and
makes a deterministic, programmatic accept/reject/uncertain decision
before ever producing a report.

This phase does NOT modify ``InvestigationAgent`` or
``AdvancedInvestigationAgent`` (both pre-existing, both untouched — every
prior phase's behavior is exactly as it was). It introduces a new,
parallel module and a new agent class
(``HypothesisDrivenInvestigationAgent``) that future phases may adopt as a
replacement, without forcing that adoption now.

This phase does NOT build a multi-agent orchestrator, a planner agent, a
verifier agent, or a critic agent. It uses exactly ONE LLM call per
investigation (hypothesis generation, via the existing, unmodified
``LLMService.generate_hypotheses``) — one fewer than
``AdvancedInvestigationAgent``'s two, because hypothesis SELECTION here is
deterministic and evidence-driven rather than synthesized by a second LLM
call (``evaluate_investigation_evidence`` is never called by this module).

# Updated investigation architecture

```
                                  problem
                                     │
                                     ▼
                    search_service.retrieve(problem, expand=True,
                                              rerank=True)        [pre-16,
                                     │                             unmodified]
                                     ▼
                            initial candidates
                                     │
                                     ▼
                   HypothesisGenerator.generate(problem, context)
                          (ONE LLM call: LLMService.generate_hypotheses,
                           Phase 6A, unmodified)
                                     │
                                     ▼
                    tuple[InvestigationHypothesis, ...]     (N >= 1 —
                                     │                        "do not assume
                                     │                         only one
                                     │                         hypothesis
                                     │                         exists")
                                     ▼
            for each hypothesis, independently:
                   HypothesisEvaluator.evaluate(hypothesis)
                     (search_service.search() against the hypothesis's
                      own validation keywords — pre-16, unmodified;
                      NO additional LLM call)
                                     ▼
                          EvidenceEvaluation
                (supporting / contradicting / missing evidence,
                 plus this hypothesis's OWN evidence-retrieval
                 confidence level)
                                     ▼
                   score_hypothesis(hypothesis, evaluation, ...)
                     (composite_hypothesis_confidence, Phase 6A,
                      unmodified — see "Scoring methodology")
                                     ▼
                            HypothesisScore
                                     ▼
                  make_investigation_decision(scored hypotheses)
                     (deterministic — see "Decision workflow")
                                     ▼
                          InvestigationDecision
                  (accepted | rejected* | uncertain)
                                     ▼
                  build_investigation_report(problem, decision, evidence)
                                     ▼
                          InvestigationReport
```

# Hypothesis lifecycle

```
HypothesisGenerator(llm_service).generate(problem=, context=, n=3)
  1. llm_service.generate_hypotheses(problem, context, n, existing_root_causes)
     [Phase 6A, unmodified — same JSON contract: root_cause, confidence_score,
      validation_keywords, rationale]
  2. each raw dict -> InvestigationHypothesis(id=f"h{i}", root_cause=, rationale=,
     validation_keywords=tuple(...), raw_confidence=clamp(confidence_score, 0, 1))
  3. -> tuple[InvestigationHypothesis, ...] — an immutable, typed record per
     hypothesis, never a raw dict (per this phase's "avoid generic
     dictionaries" requirement)
```

``n`` defaults to ``DEFAULT_HYPOTHESIS_COUNT = 3`` — enough to represent
genuinely competing explanations (the explicit goal: "do not assume only
one hypothesis exists") without being so many that evidence evaluation
(one retrieval call per hypothesis) becomes expensive for a phase that
explicitly must not add LLM calls.

# Evidence evaluation workflow

``HypothesisEvaluator.evaluate(hypothesis)`` runs ONE retrieval call (the
existing, unmodified ``IncidentSearchService.search()`` — never
``.retrieve()``, since expansion/reranking are dense-retrieval refinements
unrelated to evidence-gathering, and never a new LLM call) against the
hypothesis's own ``validation_keywords`` (falling back to ``root_cause`` if
no keywords were generated), then classifies each retrieved incident
deterministically by its own ``similarity_score`` against the EXISTING,
unmodified ``LOW_CONFIDENCE_THRESHOLD`` (0.40, ``app.services.confidence``):

- ``similarity_score >= LOW_CONFIDENCE_THRESHOLD`` -> **supporting evidence**
  (the incident is judged plausibly relevant to this specific hypothesis)
- ``similarity_score < LOW_CONFIDENCE_THRESHOLD`` -> **contradicting evidence**
  (the search for this hypothesis's own keywords surfaced only weakly-related
  incidents — read as "the evidence search did not actually find grounding
  for this specific explanation," not as semantic logical contradiction,
  which would require an LLM judgment call this phase does not make; see
  "Risks discovered" for why this is a heuristic proxy, not true
  contradiction detection)
- no results retrieved at all -> **missing evidence** (a single recorded
  entry stating what was searched for and found nothing)

This is a genuine change from ``AdvancedInvestigationAgent._collect_evidence``
(Phase 6A), which records only ONE confidence level for an entire evidence
set and never distinguishes individual incidents from each other. Here,
every retrieved incident is independently classified — "do not simply
assign a score; capture the reasoning inputs," per this phase's explicit
requirement.

# Scoring methodology

``score_hypothesis()`` does NOT invent a new formula. It computes
``composite_score`` via ``app.services.confidence.composite_hypothesis_confidence``
(Phase 6A, unmodified) — ``raw_confidence * retrieval_weight * keyword_weight``,
already documented and justified in that module — using:

- ``raw_confidence``: the hypothesis's own self-reported confidence
  (from the LLM's hypothesis-generation response).
- ``retrieval_confidence_level``: the INITIAL retrieval's confidence level
  for the whole investigation (computed once, shared by every hypothesis —
  a weak initial retrieval discounts every hypothesis generated from it
  equally, the same reasoning Phase 6A's docstring already gives).
- ``validation_keyword_recall_ok``: ``True`` unless this hypothesis's OWN
  ``EvidenceEvaluation.evidence_confidence_level == CONFIDENCE_LOW`` — the
  exact derivation ``AdvancedInvestigationAgent._should_escalate`` already
  uses for its own ``keyword_ok`` check, kept consistent here rather than
  reinvented.

Reusing this exact, already-documented formula is a deliberate choice: per
this phase's own instruction ("do not use arbitrary numeric constants...
the exact formula may be heuristic"), the LEAST arbitrary choice available
is the formula this codebase already committed to and justified in Phase
6A, not a new one invented for this phase alone. ``supporting_count``/
``contradicting_count``/``missing_count`` are recorded on ``HypothesisScore``
as diagnostic context for the eventual report (see "Investigation Report"
below) — they inform ``remaining_uncertainty`` but are deliberately NOT
folded into ``composite_score`` itself, since doing so would require
inventing a new weighting constant this phase explicitly avoids.

# Decision workflow

``make_investigation_decision()`` is a pure function over already-computed
scores — no LLM call, fully deterministic, fully explainable:

```
1. No hypotheses at all -> uncertain (nothing to decide)
2. Partition hypotheses into "eligible" (composite_score >=
   ACCEPTANCE_COMPOSITE_FLOOR) and the rest
3. No eligible hypothesis -> uncertain; EVERY hypothesis is recorded as
   rejected (none cleared the bar) — "do not force every investigation to
   produce a single confident answer"
4. >= 1 eligible hypothesis -> accept the one with the HIGHEST
   composite_score (ties broken by generation order — the first-generated
   hypothesis among those tied for highest wins, a deterministic,
   reproducible rule); every other hypothesis (eligible or not) is
   rejected — "select best hypothesis" + "reject weak hypotheses" together
   mean exactly one accepted hypothesis, never more than one, even if
   several cleared the floor
```

``ACCEPTANCE_COMPOSITE_FLOOR = 0.60`` matches the floor
``AdvancedInvestigationAgent``'s own Policy B escalation check already uses
(``_ESCALATION_COMPOSITE_FLOOR``, Phase 6A) — defined independently here
(this module does not import a private symbol from another agent module)
but set to the same value *for consistency of meaning*: a composite score
of 0.60 already has a documented interpretation elsewhere in this codebase
(Phase 6A's own derivation: a MEDIUM-confidence-retrieval hypothesis with
successful keyword evidence should reach at least ``~0.85 * 0.71 ≈ 0.60``),
and reusing that interpretation is less arbitrary than picking a fresh
number.

# Investigation Report

``build_investigation_report()`` assembles the final, immutable
``InvestigationReport`` from already-computed objects — no LLM call:

- **accepted case**: ``selected_hypothesis`` = the accepted hypothesis,
  ``confidence``/``confidence_level`` from its score (reusing
  ``classify_confidence``, unmodified), ``supporting_evidence``/
  ``contradicting_evidence`` copied directly from its own
  ``EvidenceEvaluation``, and ``remaining_uncertainty`` listing its missing
  evidence (if any) plus a one-line summary of every rejected alternative
  (so a reader can see what else was considered, not just what won).
- **uncertain case**: ``selected_hypothesis=None``, ``confidence=0.0``,
  empty supporting/contradicting evidence, and ``remaining_uncertainty``
  listing why every hypothesis was rejected — the report explicitly
  represents "we do not have a confident answer" as a first-class outcome,
  not as an error or an empty report.

This is the stable public output type future agents (a Critic, a Final
Synthesizer) will consume — see module docstring's closing note on
forward evolution.

# Forward evolution (architecture, not yet built)

This phase's "Important" instruction is that the architecture evolve into
``Planner -> Hypothesis Generator -> Evidence Evaluator -> Critic -> Final
Synthesizer`` WITHOUT changing public interfaces. Concretely, that means:

- ``HypothesisGenerator.generate()`` could later be driven by a Planner
  agent's output instead of the raw ``problem`` string — its signature
  already accepts a ``context`` string, so a Planner that produces a
  richer context plugs in without a signature change.
- ``HypothesisEvaluator.evaluate()`` already evaluates one hypothesis at a
  time, independently — an Evidence Evaluator agent (a future phase) can
  replace its internals (e.g. adding LLM-based contradiction detection
  instead of the similarity-threshold heuristic — see "Risks discovered")
  without changing its ``(hypothesis) -> EvidenceEvaluation`` signature.
- ``make_investigation_decision()`` is where a future Critic agent would
  insert itself — today it is a pure function of already-computed scores;
  a Critic could supply additional, LLM-derived signals as additional
  fields on a richer ``HypothesisScore`` without changing
  ``InvestigationDecision``'s shape.
- ``build_investigation_report()`` is where a future Final Synthesizer
  would insert itself — it already consumes exactly the
  ``InvestigationDecision`` + evidence map a Synthesizer would need.

None of this is built in this phase. It is structurally possible because
every stage already has its own typed input/output and no stage reaches
into another stage's internals.

# Risks discovered

- **Contradiction detection is a heuristic proxy, not semantic
  contradiction.** "Contradicting evidence" here means "the evidence
  search for this hypothesis's own keywords surfaced only weakly-related
  incidents" — a retrieval-confidence signal, not a logical check that a
  retrieved incident actually argues against the hypothesis. A retrieved
  incident with low similarity could be irrelevant-but-harmless, not
  actively contradictory; this phase cannot tell the difference without
  an LLM judgment call, which is explicitly out of scope.
- **Ties in `make_investigation_decision` are broken by generation
  order**, an arbitrary (if deterministic) tiebreak — two hypotheses with
  identical composite scores are not actually equally good explanations in
  general, but this phase has no further signal to break the tie with.
- **`ACCEPTANCE_COMPOSITE_FLOOR` is reused, not re-validated.** It carries
  over Phase 6A's escalation floor by analogy (same scale, same
  derivation), not because this phase independently confirmed 0.60 is the
  right acceptance threshold for *this* decision (a different decision
  than Policy B's escalation check). A future calibration phase should
  treat this as a starting point, not a validated constant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.services.confidence import (
    CONFIDENCE_LOW,
    LOW_CONFIDENCE_THRESHOLD,
    classify_confidence,
    composite_hypothesis_confidence,
)
from app.services.llm_service import LLMService
from app.services.routed_search import RoutedSearchService
from app.services.search import IncidentSearchResult, IncidentSearchService

DEFAULT_HYPOTHESIS_COUNT = 3
ACCEPTANCE_COMPOSITE_FLOOR = 0.60


# ── Core data model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InvestigationHypothesis:
    """One candidate explanation. ``id`` is stable within a single
    investigation (``"h1"``, ``"h2"``, ...), used to join hypotheses with
    their ``EvidenceEvaluation``/``HypothesisScore`` without re-passing the
    full object everywhere.
    """

    id: str
    root_cause: str
    rationale: str
    validation_keywords: tuple[str, ...]
    raw_confidence: float


@dataclass(frozen=True)
class EvidenceEvaluation:
    """The independent evidence evaluation for ONE hypothesis. See module
    docstring's "Evidence evaluation workflow" for how each tuple is
    populated.
    """

    hypothesis_id: str
    query: str
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    evidence_confidence_level: str
    evidence_top1_score: float | None


@dataclass(frozen=True)
class HypothesisScore:
    """The explicit, derived score for one hypothesis. ``composite_score``
    is computed by ``score_hypothesis()`` via the unmodified, already-
    documented ``composite_hypothesis_confidence`` (Phase 6A) — see module
    docstring's "Scoring methodology".
    """

    hypothesis_id: str
    raw_confidence: float
    retrieval_confidence_level: str
    evidence_confidence_level: str
    supporting_count: int
    contradicting_count: int
    missing_count: int
    composite_score: float


@dataclass(frozen=True)
class InvestigationDecision:
    """The decision stage's output — see module docstring's "Decision
    workflow". ``accepted``/``accepted_score`` are both ``None`` exactly
    when ``is_uncertain`` is ``True``.
    """

    accepted: InvestigationHypothesis | None
    accepted_score: HypothesisScore | None
    rejected: tuple[tuple[InvestigationHypothesis, HypothesisScore], ...]
    is_uncertain: bool
    rationale: str


@dataclass(frozen=True)
class InvestigationReport:
    """The final, immutable output — see module docstring's "Investigation
    Report".
    """

    problem: str
    selected_hypothesis: InvestigationHypothesis | None
    confidence: float
    confidence_level: str
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    remaining_uncertainty: tuple[str, ...]
    is_uncertain: bool
    rejected_hypotheses: tuple[InvestigationHypothesis, ...]


# ── Hypothesis generation ───────────────────────────────────────────────────────


def _coerce_unit_interval(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


class HypothesisGenerator:
    """Wraps ``LLMService.generate_hypotheses`` (Phase 6A, unmodified) and
    converts its raw dicts into immutable ``InvestigationHypothesis``
    records. The ONE LLM call this entire module makes.
    """

    def __init__(self, llm_service: LLMService) -> None:
        self._llm_service = llm_service

    def generate(
        self,
        *,
        problem: str,
        context: str,
        n: int = DEFAULT_HYPOTHESIS_COUNT,
        existing_root_causes: Sequence[str] | None = None,
    ) -> tuple[InvestigationHypothesis, ...]:
        raw = self._llm_service.generate_hypotheses(
            problem=problem,
            context=context,
            n=n,
            existing_root_causes=list(existing_root_causes) if existing_root_causes else None,
        )
        return tuple(
            self._to_hypothesis(index, item) for index, item in enumerate(raw[:n], start=1)
        )

    def _to_hypothesis(self, index: int, item: dict[str, Any]) -> InvestigationHypothesis:
        keywords = item.get("validation_keywords", [])
        if not isinstance(keywords, list):
            keywords = [keywords]
        return InvestigationHypothesis(
            id=f"h{index}",
            root_cause=str(item.get("root_cause", "")),
            rationale=str(item.get("rationale", "")),
            validation_keywords=tuple(str(k) for k in keywords if str(k).strip()),
            raw_confidence=_coerce_unit_interval(item.get("confidence_score", 0.0)),
        )


# ── Evidence evaluation ──────────────────────────────────────────────────────────


class HypothesisEvaluator:
    """Evaluates ONE hypothesis at a time against retrieved evidence — see
    module docstring's "Evidence evaluation workflow". No LLM call; reuses
    ``search_service.search()`` (never ``.retrieve()``) and the existing
    confidence-classification machinery. ``search_service`` may be a plain
    ``IncidentSearchService`` (dense-only) or a ``RoutedSearchService``
    (Phase 18B, production default since the orchestrator's adoption pass —
    see ``MultiAgentInvestigationOrchestrator``) — both expose the same
    ``.search()`` contract, so evidence search benefits from adaptive
    routing automatically when the latter is used.
    """

    def __init__(self, search_service: IncidentSearchService | RoutedSearchService) -> None:
        self._search_service = search_service

    def evaluate(
        self, hypothesis: InvestigationHypothesis, *, limit: int = 5
    ) -> EvidenceEvaluation:
        query = " ".join(hypothesis.validation_keywords) or hypothesis.root_cause
        results: list[IncidentSearchResult] = (
            self._search_service.search(
                query, limit=limit, call_site="hypothesis_investigation.evaluate_evidence"
            )
            if query
            else []
        )
        top1_score, confidence_level = IncidentSearchService.confidence_for(results)

        supporting = tuple(
            self._describe(result)
            for result in results
            if result.similarity_score >= LOW_CONFIDENCE_THRESHOLD
        )
        contradicting = tuple(
            self._describe(result)
            for result in results
            if result.similarity_score < LOW_CONFIDENCE_THRESHOLD
        )
        missing = () if results else (f"no incidents found for validation query {query!r}",)

        return EvidenceEvaluation(
            hypothesis_id=hypothesis.id,
            query=query,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            missing_evidence=missing,
            evidence_confidence_level=confidence_level,
            evidence_top1_score=top1_score,
        )

    def _describe(self, result: IncidentSearchResult) -> str:
        return f"{result.incident.title} (similarity={result.similarity_score:.3f})"


# ── Scoring ────────────────────────────────────────────────────────────────────


def score_hypothesis(
    hypothesis: InvestigationHypothesis,
    evaluation: EvidenceEvaluation,
    *,
    retrieval_confidence_level: str,
) -> HypothesisScore:
    """See module docstring's "Scoring methodology" — delegates the
    composite formula entirely to the unmodified
    ``composite_hypothesis_confidence`` (Phase 6A).
    """
    validation_keyword_recall_ok = evaluation.evidence_confidence_level != CONFIDENCE_LOW
    composite = composite_hypothesis_confidence(
        raw_confidence=hypothesis.raw_confidence,
        retrieval_confidence_level=retrieval_confidence_level,
        validation_keyword_recall_ok=validation_keyword_recall_ok,
    )
    return HypothesisScore(
        hypothesis_id=hypothesis.id,
        raw_confidence=hypothesis.raw_confidence,
        retrieval_confidence_level=retrieval_confidence_level,
        evidence_confidence_level=evaluation.evidence_confidence_level,
        supporting_count=len(evaluation.supporting_evidence),
        contradicting_count=len(evaluation.contradicting_evidence),
        missing_count=len(evaluation.missing_evidence),
        composite_score=composite,
    )


# ── Decision ───────────────────────────────────────────────────────────────────


def make_investigation_decision(
    scored: Sequence[tuple[InvestigationHypothesis, HypothesisScore]],
) -> InvestigationDecision:
    """See module docstring's "Decision workflow". Pure, deterministic, no
    LLM call.
    """
    if not scored:
        return InvestigationDecision(
            accepted=None, accepted_score=None, rejected=(), is_uncertain=True,
            rationale="no hypotheses were generated; nothing to decide",
        )

    eligible = [pair for pair in scored if pair[1].composite_score >= ACCEPTANCE_COMPOSITE_FLOOR]
    if not eligible:
        best = max(score.composite_score for _, score in scored)
        rationale = (
            f"no hypothesis reached the acceptance floor "
            f"({ACCEPTANCE_COMPOSITE_FLOOR:.2f}); highest composite_score was {best:.2f}"
        )
        return InvestigationDecision(
            accepted=None, accepted_score=None, rejected=tuple(scored), is_uncertain=True,
            rationale=rationale,
        )

    accepted_hypothesis, accepted_score = max(eligible, key=lambda pair: pair[1].composite_score)
    rejected = tuple(pair for pair in scored if pair[0].id != accepted_hypothesis.id)
    rationale = (
        f"accepted {accepted_hypothesis.id!r} "
        f"(composite_score={accepted_score.composite_score:.2f}), "
        f"the highest of {len(scored)} candidate(s) clearing the "
        f"{ACCEPTANCE_COMPOSITE_FLOOR:.2f} acceptance floor"
    )
    return InvestigationDecision(
        accepted=accepted_hypothesis, accepted_score=accepted_score, rejected=rejected,
        is_uncertain=False, rationale=rationale,
    )


# ── Report ─────────────────────────────────────────────────────────────────────


def build_investigation_report(
    problem: str,
    decision: InvestigationDecision,
    evaluations: Mapping[str, EvidenceEvaluation],
) -> InvestigationReport:
    """See module docstring's "Investigation Report". No LLM call —
    assembled entirely from already-computed objects.
    """
    if decision.is_uncertain or decision.accepted is None:
        remaining = tuple(
            f"hypothesis {hypothesis.id!r} ({hypothesis.root_cause}) rejected: "
            f"composite_score={score.composite_score:.2f} below acceptance floor"
            for hypothesis, score in decision.rejected
        )
        remaining = remaining or (decision.rationale,)
        return InvestigationReport(
            problem=problem,
            selected_hypothesis=None,
            confidence=0.0,
            confidence_level=classify_confidence(0.0),
            supporting_evidence=(),
            contradicting_evidence=(),
            remaining_uncertainty=remaining,
            is_uncertain=True,
            rejected_hypotheses=tuple(hypothesis for hypothesis, _ in decision.rejected),
        )

    accepted_evaluation = evaluations.get(decision.accepted.id)
    supporting = accepted_evaluation.supporting_evidence if accepted_evaluation else ()
    contradicting = accepted_evaluation.contradicting_evidence if accepted_evaluation else ()
    missing = accepted_evaluation.missing_evidence if accepted_evaluation else ()

    remaining_uncertainty = missing + tuple(
        f"alternative hypothesis {hypothesis.id!r} ({hypothesis.root_cause}) was considered "
        f"and rejected (composite_score={score.composite_score:.2f})"
        for hypothesis, score in decision.rejected
    )

    confidence = decision.accepted_score.composite_score
    return InvestigationReport(
        problem=problem,
        selected_hypothesis=decision.accepted,
        confidence=confidence,
        confidence_level=classify_confidence(confidence),
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        remaining_uncertainty=remaining_uncertainty,
        is_uncertain=False,
        rejected_hypotheses=tuple(hypothesis for hypothesis, _ in decision.rejected),
    )


# ── Orchestrator ───────────────────────────────────────────────────────────────


class HypothesisDrivenInvestigationAgent:
    """The new, parallel agent — see module docstring. Does not replace
    ``InvestigationAgent``/``AdvancedInvestigationAgent`` (both untouched);
    a future phase decides whether/how to adopt this one.
    """

    def __init__(
        self,
        db: Session,
        *,
        search_service: IncidentSearchService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.search_service = search_service or IncidentSearchService(db)
        self.llm_service = llm_service or LLMService()
        self._generator = HypothesisGenerator(self.llm_service)
        self._evaluator = HypothesisEvaluator(self.search_service)

    def investigate(
        self, problem: str, *, n_hypotheses: int = DEFAULT_HYPOTHESIS_COUNT
    ) -> InvestigationReport:
        initial_results = self.search_service.retrieve(
            problem, limit=10, expand=True, rerank=True,
            call_site="hypothesis_investigation.investigate",
        )
        _, retrieval_confidence_level = IncidentSearchService.confidence_for(initial_results)
        context = self._build_context(initial_results, retrieval_confidence_level)

        hypotheses = self._generator.generate(problem=problem, context=context, n=n_hypotheses)
        if not hypotheses:
            decision = make_investigation_decision(())
            return build_investigation_report(problem, decision, {})

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
        return build_investigation_report(problem, decision, evaluations)

    def _build_context(self, results: list[IncidentSearchResult], confidence_level: str) -> str:
        if not results:
            return (
                "Retrieval confidence: LOW (no similar incidents were retrieved).\n"
                "No historical evidence is available - rely on general reasoning."
            )
        sections = []
        for index, result in enumerate(results, start=1):
            incident = result.incident
            symptoms = "; ".join(symptom.text for symptom in incident.symptoms) or "Unknown"
            sections.append(
                f"Incident {index}\nTitle: {incident.title}\nSymptoms: {symptoms}\n"
                f"Similarity score: {result.similarity_score:.3f}"
            )
        return f"Retrieval confidence: {confidence_level}\n\n" + "\n\n".join(sections)
