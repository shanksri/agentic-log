"""Gold Dataset Authoring Framework (Phase 21C).

Human-in-the-loop authoring workflow: an LLM proposes diverse candidate
queries and investigation scenarios from incident descriptions; a human
reviewer accepts, edits, or rejects each candidate; accepted items export
directly into the existing GoldDataset / ReasoningGoldDataset schemas.

# Architecture

```
IncidentSummary
      │
      ▼
LLMDatasetAuthor.generate_queries()        ─→ tuple[CandidateQuery, ...]
LLMDatasetAuthor.generate_investigation()  ─→ CandidateScenario
      │
      ▼
Human Review  .review(id, decision, edited_text=...)
      │
      ├─ ACCEPTED  ─→ export_gold_dataset()       → GoldDataset
      ├─ EDITED    ─→ (original AI text preserved)  export_reasoning_dataset()
      └─ REJECTED  ─→ excluded from all exports         → ReasoningGoldDataset
```

No retrieval. No evaluation. No benchmark execution. Only dataset creation.

# LLM prompt diversity

``generate_queries()`` requests five stylistically distinct query variants
per incident, covering:

- ``exact_keyword``    — verbatim terms from the incident title/description
- ``paraphrase``       — same meaning, different wording
- ``symptom_description`` — what an on-call engineer would observe/type
- ``novice_wording``   — plain English, no ops jargon
- ``multi_concept``    — combines two or more aspects of the incident

Each variant maps onto the Gold Dataset v2 ``VALID_CATEGORIES`` set and
carries a ``generation_method`` label so reviewers can see how it was
generated (and filter by method when the queue grows large).

# Review workflow

Every generated item starts as ``ReviewDecision.PENDING``.  A reviewer
calls ``.review(candidate_id, decision, edited_text=...)`` to transition
it.  ``EDITED`` items preserve the original AI-generated text in the
``query`` / ``problem`` field and store the reviewer's text in
``edited_query`` / ``edited_problem``; the original is never silently
overwritten.  Only ``ACCEPTED`` and ``EDITED`` items appear in exports.
``REJECTED`` items are retained in internal state so statistics remain
accurate; they are excluded from all Gold Dataset exports.

# Versioning

``export_gold_dataset(version="retrieval_v2", ...)`` raises
``VersionAlreadyExportedError`` if that version string has already been
exported in this session.  Callers must use a new version tag for each
export — no silent overwriting.

# Statistics

``AuthoringStats`` is computed on-demand from current state (no separate
accumulator that could diverge from truth):

- total_generated, accepted, edited, rejected, pending
- acceptance_rate = (accepted + edited) / total_generated
- edit_rate       = edited / max(1, accepted + edited)
- mean_queries_per_incident (across all incidents seen so far)

# Forbidden imports

This module must never import: IncidentSearchService, evaluation harness,
Planner, Judge, Benchmark, Regression.  It depends only on Gold Dataset
schemas, ReasoningGoldDataset schema, standard library, and the
``AuthorLLMClient`` Protocol (defined here — no OpenAI SDK required).

# Risks discovered

- LLM JSON parsing uses a best-effort field extraction; a model that
  produces free-text instead of JSON will raise ``AuthorResponseError``,
  which the caller must handle — there is no automatic retry.
- ``acceptance_rate`` counts both ACCEPTED and EDITED as "accepted" in
  the numerator; a batch of all-edited candidates still shows a 100%
  acceptance rate even if every item was substantially changed.
- Version collision is tracked in-process only: a fresh ``LLMDatasetAuthor``
  instance will allow re-exporting any version string.
- ``mean_queries_per_incident`` divides by distinct ``incident_id``s seen
  in the query pool only (not scenario pool), so a session that generates
  only scenarios will report ``mean_queries_per_incident = 0.0``.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol

from app.evaluation.gold_dataset import (
    GoldDataset,
    GoldQuery,
    ExpectedIncident,
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
)
from app.evaluation.reasoning_dataset import (
    ReasoningGoldDataset,
    InvestigationScenario,
    VALID_STRATEGIES,
)

# ── Public constants ────────────────────────────────────────────────────────────

DEFAULT_N_QUERIES = 5

#: Methods the LLM is asked to use for diversity.  The prompt maps each one
#: to a short instruction; reviewers can filter by method in the queue.
GENERATION_METHODS = (
    "exact_keyword",
    "paraphrase",
    "symptom_description",
    "novice_wording",
    "multi_concept",
)

#: Map from generation_method to the Gold Dataset v2 category it produces.
#: symptom_description / novice_wording are paraphrase-style queries; they
#: do not introduce a new category — they diversify the paraphrase bucket.
_METHOD_TO_CATEGORY: dict[str, str] = {
    "exact_keyword": "lexical-overlap",
    "paraphrase": "paraphrase",
    "symptom_description": "paraphrase",
    "novice_wording": "paraphrase",
    "multi_concept": "multi-concept",
}

#: Difficulty assigned per generation method.  Exact-keyword queries are
#: easiest (the retrieval system should surface them via BM25); multi-concept
#: queries are hardest (require semantic understanding of multiple signals).
_METHOD_TO_DIFFICULTY: dict[str, str] = {
    "exact_keyword": "easy",
    "paraphrase": "medium",
    "symptom_description": "medium",
    "novice_wording": "medium",
    "multi_concept": "hard",
}


# ── LLM client protocol ─────────────────────────────────────────────────────────


class AuthorLLMClient(Protocol):
    """Minimal protocol for the LLM used by ``LLMDatasetAuthor``.

    Decouples the authoring framework from any specific SDK (OpenAI, Anthropic,
    etc.) so tests can inject a fake without touching environment variables.
    Only one method is required: send a prompt string, get a string back.
    The implementation is responsible for all authentication, retry, and
    model-selection concerns.
    """

    def complete(self, prompt: str) -> str:  # pragma: no cover
        ...


# ── Exceptions ──────────────────────────────────────────────────────────────────


class AuthorResponseError(ValueError):
    """Raised when the LLM response cannot be parsed into the expected shape."""


class VersionAlreadyExportedError(ValueError):
    """Raised when ``export_gold_dataset`` is called with a version that has
    already been exported from this authoring session.
    """


class CandidateNotFoundError(KeyError):
    """Raised when ``review`` is called with an unknown candidate ID."""


# ── Minimal incident summary ────────────────────────────────────────────────────


@dataclass(frozen=True)
class IncidentSummary:
    """Plain-data view of an incident used as generation input.

    Accepts any dict-like incident representation; deliberately avoids
    importing the SQLAlchemy ``Incident`` model so this module stays
    schema-only.  Callers build one from whatever incident object they hold.
    ``source_type`` and ``source_external_id`` flow through into
    ``ExpectedIncident`` when a reviewer provides expected identities on
    export.
    """

    incident_id: str
    title: str
    description: str
    source_type: str = ""
    source_external_id: str = ""

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.incident_id:
            problems.append("incident_id must be non-empty")
        if not self.title:
            problems.append("title must be non-empty")
        if not self.description:
            problems.append("description must be non-empty")
        return problems

    def is_valid(self) -> bool:
        return not self.issues()


# ── Review decision ─────────────────────────────────────────────────────────────


class ReviewDecision(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


# ── Candidate dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateQuery:
    """One AI-generated retrieval query candidate awaiting human review.

    ``query`` always holds the original LLM output.  If a reviewer edits
    it, the corrected text is stored in ``edited_query``; the original is
    never overwritten.  The field that feeds the Gold Dataset on export is
    ``effective_query``: ``edited_query`` when present, else ``query``.
    """

    id: str
    incident_id: str
    query: str
    category: str
    difficulty: str
    rationale: str
    generation_method: str
    status: ReviewDecision = ReviewDecision.PENDING
    edited_query: str | None = None

    @property
    def effective_query(self) -> str:
        return self.edited_query if self.edited_query is not None else self.query

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.id:
            problems.append("candidate.id must be non-empty")
        if not self.incident_id:
            problems.append(f"candidate {self.id!r}: incident_id must be non-empty")
        if not self.query:
            problems.append(f"candidate {self.id!r}: query must be non-empty")
        if self.category not in VALID_CATEGORIES:
            problems.append(
                f"candidate {self.id!r}: category {self.category!r} "
                f"not in {sorted(VALID_CATEGORIES)}"
            )
        if self.difficulty not in VALID_DIFFICULTIES:
            problems.append(
                f"candidate {self.id!r}: difficulty {self.difficulty!r} "
                f"not in {sorted(VALID_DIFFICULTIES)}"
            )
        if self.generation_method not in GENERATION_METHODS:
            problems.append(
                f"candidate {self.id!r}: generation_method {self.generation_method!r} "
                f"not in {GENERATION_METHODS}"
            )
        if self.status == ReviewDecision.EDITED and not self.edited_query:
            problems.append(
                f"candidate {self.id!r}: status EDITED requires a non-empty edited_query"
            )
        return problems

    def is_valid(self) -> bool:
        return not self.issues()


@dataclass(frozen=True)
class CandidateScenario:
    """One AI-generated investigation scenario candidate awaiting human review.

    ``problem`` always holds the original LLM output.  Edits are stored in
    ``edited_problem``; the original is preserved.  ``effective_problem`` is
    the field that feeds the ReasoningGoldDataset on export.
    """

    id: str
    incident_id: str
    problem: str
    expected_root_causes: tuple[str, ...]
    suggested_strategy: str
    rationale: str
    status: ReviewDecision = ReviewDecision.PENDING
    edited_problem: str | None = None

    @property
    def effective_problem(self) -> str:
        return self.edited_problem if self.edited_problem is not None else self.problem

    def issues(self) -> list[str]:
        problems: list[str] = []
        if not self.id:
            problems.append("scenario.id must be non-empty")
        if not self.incident_id:
            problems.append(f"scenario {self.id!r}: incident_id must be non-empty")
        if not self.problem:
            problems.append(f"scenario {self.id!r}: problem must be non-empty")
        if self.suggested_strategy not in VALID_STRATEGIES:
            problems.append(
                f"scenario {self.id!r}: suggested_strategy {self.suggested_strategy!r} "
                f"not in {sorted(VALID_STRATEGIES)}"
            )
        if self.status == ReviewDecision.EDITED and not self.edited_problem:
            problems.append(
                f"scenario {self.id!r}: status EDITED requires a non-empty edited_problem"
            )
        return problems

    def is_valid(self) -> bool:
        return not self.issues()


# ── Statistics ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthoringStats:
    """Read-only snapshot of the current authoring session state.

    Computed on-demand from the live candidate pool — never accumulated
    separately, so it can never diverge from truth.
    """

    total_generated: int
    accepted: int
    edited: int
    rejected: int
    pending: int
    acceptance_rate: float
    edit_rate: float
    mean_queries_per_incident: float
    total_scenarios: int
    accepted_scenarios: int
    edited_scenarios: int
    rejected_scenarios: int
    pending_scenarios: int


# ── DatasetAuthor ABC ───────────────────────────────────────────────────────────


class DatasetAuthor(ABC):
    """Abstract base for human-in-the-loop dataset authoring agents.

    Each implementation is responsible for: generating diverse candidates
    from an ``IncidentSummary``, maintaining an internal review queue,
    applying human review decisions, and exporting accepted items into the
    existing Gold Dataset schemas.

    State is intentionally mutable within the author instance (candidates
    accumulate across multiple ``generate_*`` calls before a batch review).
    The individual candidate objects are frozen dataclasses.
    """

    @abstractmethod
    def generate_queries(
        self,
        incident: IncidentSummary,
        *,
        n: int = DEFAULT_N_QUERIES,
    ) -> tuple[CandidateQuery, ...]:
        """Generate ``n`` diverse retrieval query candidates for ``incident``.

        All generated candidates start with ``status=PENDING``.  Returns the
        newly-created candidates; they are also added to the internal queue.
        """

    @abstractmethod
    def generate_investigation(
        self,
        incident: IncidentSummary,
    ) -> CandidateScenario:
        """Generate one investigation scenario candidate for ``incident``.

        The candidate starts with ``status=PENDING``.
        """

    @abstractmethod
    def review(
        self,
        candidate_id: str,
        decision: ReviewDecision,
        *,
        edited_text: str | None = None,
    ) -> CandidateQuery | CandidateScenario:
        """Apply a human review decision to a candidate.

        ``edited_text`` is required when ``decision=EDITED`` and is stored
        in ``edited_query`` / ``edited_problem`` respectively; the original
        AI-generated text is never overwritten.

        Returns the updated (new) candidate object.
        Raises ``CandidateNotFoundError`` for unknown IDs.
        Raises ``ValueError`` when ``decision=EDITED`` and ``edited_text``
        is absent or blank.
        """

    @abstractmethod
    def export_gold_dataset(
        self,
        *,
        version: str,
        description: str,
        author: str | None = None,
    ) -> GoldDataset:
        """Export accepted retrieval candidates as a versioned ``GoldDataset``.

        Only ``ACCEPTED`` and ``EDITED`` candidates are exported.  Raises
        ``VersionAlreadyExportedError`` if ``version`` has already been
        exported from this session.  Raises ``ValueError`` if there are no
        exportable candidates.
        """

    @abstractmethod
    def export_reasoning_dataset(
        self,
        *,
        version: str,
        description: str,
        author: str | None = None,
    ) -> ReasoningGoldDataset:
        """Export accepted scenario candidates as a ``ReasoningGoldDataset``.

        Only ``ACCEPTED`` and ``EDITED`` scenario candidates are exported.
        Raises ``VersionAlreadyExportedError`` if ``version`` was already used.
        Raises ``ValueError`` if there are no exportable scenarios.
        """

    @abstractmethod
    def stats(self) -> AuthoringStats:
        """Return a fresh statistics snapshot from the current candidate pool."""

    @abstractmethod
    def pending_queries(self) -> tuple[CandidateQuery, ...]:
        """Return all query candidates currently in PENDING state."""

    @abstractmethod
    def pending_scenarios(self) -> tuple[CandidateScenario, ...]:
        """Return all scenario candidates currently in PENDING state."""


# ── Prompt builders ─────────────────────────────────────────────────────────────

_QUERY_RESPONSE_TEMPLATE = """\
{
  "candidates": [
    {
      "generation_method": "<one of: exact_keyword, paraphrase, \
symptom_description, novice_wording, multi_concept>",
      "query": "<the natural-language retrieval query>",
      "rationale": "<one sentence explaining the diversity value of this query>"
    }
  ]
}"""


def _build_query_prompt(incident: IncidentSummary, n: int) -> str:
    methods_desc = (
        "- exact_keyword: verbatim terms from the incident title/description\n"
        "- paraphrase: same meaning, completely different phrasing\n"
        "- symptom_description: what an on-call engineer would type while "
        "observing the failure\n"
        "- novice_wording: plain everyday English, no technical jargon\n"
        "- multi_concept: combines TWO or more distinct aspects of the incident"
    )
    return (
        "You are an expert at designing information retrieval benchmarks for "
        "incident management systems.\n\n"
        f"INCIDENT TITLE: {incident.title}\n"
        f"INCIDENT DESCRIPTION:\n{incident.description}\n\n"
        f"Generate exactly {n} diverse retrieval queries that a user might type "
        "to find this incident in a search system.\n\n"
        "Use DIFFERENT generation methods to maximise diversity:\n"
        f"{methods_desc}\n\n"
        "Return ONLY valid JSON matching this exact structure:\n"
        f"{_QUERY_RESPONSE_TEMPLATE}\n\n"
        'Return nothing outside the JSON object. No markdown. No "```json".'
    )


_SCENARIO_RESPONSE_TEMPLATE = """\
{
  "problem": "<the incident investigation problem statement>",
  "expected_root_causes": ["<root cause phrase>", "<another root cause phrase>"],
  "suggested_strategy": "<one of: infrastructure_failure, configuration, \
authentication, network, application_failure, unknown>",
  "rationale": "<one sentence explaining why this is the right strategy>"
}"""


def _build_scenario_prompt(incident: IncidentSummary) -> str:
    return (
        "You are an expert at designing incident investigation evaluation datasets.\n\n"
        f"INCIDENT TITLE: {incident.title}\n"
        f"INCIDENT DESCRIPTION:\n{incident.description}\n\n"
        "Create ONE investigation scenario from this incident:\n"
        "1. A problem statement an on-call investigator would start with\n"
        "2. The expected root cause phrases (keywords the investigation should surface)\n"
        "3. The investigation strategy category that best fits this incident\n"
        "4. A brief rationale for the strategy choice\n\n"
        "Return ONLY valid JSON matching this exact structure:\n"
        f"{_SCENARIO_RESPONSE_TEMPLATE}\n\n"
        'Return nothing outside the JSON object. No markdown. No "```json".'
    )


# ── LLM response parsing ────────────────────────────────────────────────────────


def _parse_query_response(
    raw: str,
    incident_id: str,
) -> list[dict[str, str]]:
    """Parse and validate the LLM's JSON response for query generation.

    Returns a list of dicts with keys: generation_method, query, rationale.
    Raises ``AuthorResponseError`` on any parse or validation failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthorResponseError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "candidates" not in data:
        raise AuthorResponseError(
            f"LLM response missing 'candidates' key; got keys: {list(data)}"
        )
    candidates = data["candidates"]
    if not isinstance(candidates, list):
        raise AuthorResponseError("'candidates' must be a JSON array")
    parsed: list[dict[str, str]] = []
    for i, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise AuthorResponseError(f"candidates[{i}] is not a JSON object")
        for key in ("generation_method", "query", "rationale"):
            if key not in item:
                raise AuthorResponseError(
                    f"candidates[{i}] missing required field '{key}'"
                )
            if not isinstance(item[key], str) or not item[key].strip():
                raise AuthorResponseError(
                    f"candidates[{i}].{key!r} must be a non-empty string"
                )
        method = item["generation_method"]
        if method not in GENERATION_METHODS:
            raise AuthorResponseError(
                f"candidates[{i}].generation_method {method!r} "
                f"not in {GENERATION_METHODS}"
            )
        parsed.append({
            "generation_method": method,
            "query": item["query"].strip(),
            "rationale": item["rationale"].strip(),
        })
    return parsed


def _parse_scenario_response(raw: str) -> dict[str, object]:
    """Parse and validate the LLM's JSON response for scenario generation.

    Returns a dict with keys: problem, expected_root_causes, suggested_strategy,
    rationale.  Raises ``AuthorResponseError`` on any failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthorResponseError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AuthorResponseError("LLM scenario response must be a JSON object")
    for key in ("problem", "suggested_strategy", "rationale"):
        if key not in data:
            raise AuthorResponseError(f"Scenario response missing required field '{key}'")
        if not isinstance(data[key], str) or not str(data[key]).strip():
            raise AuthorResponseError(f"Scenario response field {key!r} must be non-empty string")
    if "expected_root_causes" not in data:
        raise AuthorResponseError("Scenario response missing 'expected_root_causes'")
    causes = data["expected_root_causes"]
    if not isinstance(causes, list):
        raise AuthorResponseError("'expected_root_causes' must be a JSON array")
    strategy = str(data["suggested_strategy"])
    if strategy not in VALID_STRATEGIES:
        raise AuthorResponseError(
            f"suggested_strategy {strategy!r} not in {sorted(VALID_STRATEGIES)}"
        )
    return {
        "problem": str(data["problem"]).strip(),
        "expected_root_causes": tuple(str(c) for c in causes if str(c).strip()),
        "suggested_strategy": strategy,
        "rationale": str(data["rationale"]).strip(),
    }


# ── LLMDatasetAuthor ────────────────────────────────────────────────────────────


class LLMDatasetAuthor(DatasetAuthor):
    """Concrete ``DatasetAuthor`` that uses an ``AuthorLLMClient`` to generate
    candidate queries and investigation scenarios.

    State model:
    - ``_queries``: dict[id → CandidateQuery]  (insertion order preserved)
    - ``_scenarios``: dict[id → CandidateScenario]
    - ``_exported_query_versions``: set of version strings already exported
    - ``_exported_scenario_versions``: set of version strings already exported

    All candidates are immutable frozen dataclasses; reviews create new
    objects (replacing the entry in the dict) rather than mutating in place.
    """

    def __init__(self, llm_client: AuthorLLMClient) -> None:
        self._llm = llm_client
        self._queries: dict[str, CandidateQuery] = {}
        self._scenarios: dict[str, CandidateScenario] = {}
        self._exported_query_versions: set[str] = set()
        self._exported_scenario_versions: set[str] = set()

    # ── Generation ──────────────────────────────────────────────────────────────

    def generate_queries(
        self,
        incident: IncidentSummary,
        *,
        n: int = DEFAULT_N_QUERIES,
    ) -> tuple[CandidateQuery, ...]:
        prompt = _build_query_prompt(incident, n)
        raw = self._llm.complete(prompt)
        parsed = _parse_query_response(raw, incident.incident_id)
        results: list[CandidateQuery] = []
        for item in parsed:
            method = item["generation_method"]
            candidate = CandidateQuery(
                id=str(uuid.uuid4()),
                incident_id=incident.incident_id,
                query=item["query"],
                category=_METHOD_TO_CATEGORY[method],
                difficulty=_METHOD_TO_DIFFICULTY[method],
                rationale=item["rationale"],
                generation_method=method,
                status=ReviewDecision.PENDING,
            )
            self._queries[candidate.id] = candidate
            results.append(candidate)
        return tuple(results)

    def generate_investigation(
        self,
        incident: IncidentSummary,
    ) -> CandidateScenario:
        prompt = _build_scenario_prompt(incident)
        raw = self._llm.complete(prompt)
        parsed = _parse_scenario_response(raw)
        scenario = CandidateScenario(
            id=str(uuid.uuid4()),
            incident_id=incident.incident_id,
            problem=str(parsed["problem"]),
            expected_root_causes=parsed["expected_root_causes"],  # type: ignore[arg-type]
            suggested_strategy=str(parsed["suggested_strategy"]),
            rationale=str(parsed["rationale"]),
            status=ReviewDecision.PENDING,
        )
        self._scenarios[scenario.id] = scenario
        return scenario

    # ── Review ──────────────────────────────────────────────────────────────────

    def review(
        self,
        candidate_id: str,
        decision: ReviewDecision,
        *,
        edited_text: str | None = None,
    ) -> CandidateQuery | CandidateScenario:
        if candidate_id in self._queries:
            return self._review_query(candidate_id, decision, edited_text)
        if candidate_id in self._scenarios:
            return self._review_scenario(candidate_id, decision, edited_text)
        raise CandidateNotFoundError(
            f"No candidate found with id {candidate_id!r}"
        )

    def _review_query(
        self,
        candidate_id: str,
        decision: ReviewDecision,
        edited_text: str | None,
    ) -> CandidateQuery:
        original = self._queries[candidate_id]
        if decision == ReviewDecision.EDITED:
            if not edited_text or not edited_text.strip():
                raise ValueError(
                    "edited_text must be non-empty when decision is EDITED"
                )
            updated = CandidateQuery(
                id=original.id,
                incident_id=original.incident_id,
                query=original.query,
                category=original.category,
                difficulty=original.difficulty,
                rationale=original.rationale,
                generation_method=original.generation_method,
                status=ReviewDecision.EDITED,
                edited_query=edited_text.strip(),
            )
        else:
            updated = CandidateQuery(
                id=original.id,
                incident_id=original.incident_id,
                query=original.query,
                category=original.category,
                difficulty=original.difficulty,
                rationale=original.rationale,
                generation_method=original.generation_method,
                status=decision,
                edited_query=original.edited_query,
            )
        self._queries[candidate_id] = updated
        return updated

    def _review_scenario(
        self,
        candidate_id: str,
        decision: ReviewDecision,
        edited_text: str | None,
    ) -> CandidateScenario:
        original = self._scenarios[candidate_id]
        if decision == ReviewDecision.EDITED:
            if not edited_text or not edited_text.strip():
                raise ValueError(
                    "edited_text must be non-empty when decision is EDITED"
                )
            updated = CandidateScenario(
                id=original.id,
                incident_id=original.incident_id,
                problem=original.problem,
                expected_root_causes=original.expected_root_causes,
                suggested_strategy=original.suggested_strategy,
                rationale=original.rationale,
                status=ReviewDecision.EDITED,
                edited_problem=edited_text.strip(),
            )
        else:
            updated = CandidateScenario(
                id=original.id,
                incident_id=original.incident_id,
                problem=original.problem,
                expected_root_causes=original.expected_root_causes,
                suggested_strategy=original.suggested_strategy,
                rationale=original.rationale,
                status=decision,
                edited_problem=original.edited_problem,
            )
        self._scenarios[candidate_id] = updated
        return updated

    # ── Export ──────────────────────────────────────────────────────────────────

    def export_gold_dataset(
        self,
        *,
        version: str,
        description: str,
        author: str | None = None,
    ) -> GoldDataset:
        if version in self._exported_query_versions:
            raise VersionAlreadyExportedError(
                f"Version {version!r} has already been exported from this session. "
                "Use a new version string (e.g. retrieval_v2, retrieval_v3)."
            )
        exportable = [
            q for q in self._queries.values()
            if q.status in (ReviewDecision.ACCEPTED, ReviewDecision.EDITED)
        ]
        if not exportable:
            raise ValueError(
                "No accepted candidates to export. Accept or edit at least one query first."
            )
        gold_queries: list[GoldQuery] = []
        for candidate in exportable:
            gold_queries.append(GoldQuery(
                id=candidate.id,
                query=candidate.effective_query,
                category=candidate.category,
                difficulty=candidate.difficulty,
                expected_incidents=(),
            ))
        dataset = GoldDataset(
            version=version,
            description=description,
            created_at=datetime.now(UTC).isoformat(),
            queries=tuple(gold_queries),
            author=author,
        )
        self._exported_query_versions.add(version)
        return dataset

    def export_reasoning_dataset(
        self,
        *,
        version: str,
        description: str,
        author: str | None = None,
    ) -> ReasoningGoldDataset:
        if version in self._exported_scenario_versions:
            raise VersionAlreadyExportedError(
                f"Version {version!r} has already been exported from this session. "
                "Use a new version string."
            )
        exportable = [
            s for s in self._scenarios.values()
            if s.status in (ReviewDecision.ACCEPTED, ReviewDecision.EDITED)
        ]
        if not exportable:
            raise ValueError(
                "No accepted scenarios to export. "
                "Accept or edit at least one scenario first."
            )
        scenarios: list[InvestigationScenario] = []
        for candidate in exportable:
            scenarios.append(InvestigationScenario(
                id=candidate.id,
                problem=candidate.effective_problem,
                expected_strategy=candidate.suggested_strategy,
                expected_root_causes=candidate.expected_root_causes,
                expected_verdict="inconclusive",
                expected_stopping_reason="max_iterations",
                notes=candidate.rationale,
            ))
        dataset = ReasoningGoldDataset(
            version=version,
            description=description,
            created_at=datetime.now(UTC).isoformat(),
            scenarios=tuple(scenarios),
            author=author,
        )
        self._exported_scenario_versions.add(version)
        return dataset

    # ── Statistics ──────────────────────────────────────────────────────────────

    def stats(self) -> AuthoringStats:
        queries = list(self._queries.values())
        scenarios = list(self._scenarios.values())

        total = len(queries)
        accepted = sum(1 for q in queries if q.status == ReviewDecision.ACCEPTED)
        edited = sum(1 for q in queries if q.status == ReviewDecision.EDITED)
        rejected = sum(1 for q in queries if q.status == ReviewDecision.REJECTED)
        pending = sum(1 for q in queries if q.status == ReviewDecision.PENDING)

        reviewable = accepted + edited
        acceptance_rate = reviewable / total if total > 0 else 0.0
        edit_rate = edited / reviewable if reviewable > 0 else 0.0

        incident_ids = {q.incident_id for q in queries}
        mean_queries = total / len(incident_ids) if incident_ids else 0.0

        total_s = len(scenarios)
        accepted_s = sum(1 for s in scenarios if s.status == ReviewDecision.ACCEPTED)
        edited_s = sum(1 for s in scenarios if s.status == ReviewDecision.EDITED)
        rejected_s = sum(1 for s in scenarios if s.status == ReviewDecision.REJECTED)
        pending_s = sum(1 for s in scenarios if s.status == ReviewDecision.PENDING)

        return AuthoringStats(
            total_generated=total,
            accepted=accepted,
            edited=edited,
            rejected=rejected,
            pending=pending,
            acceptance_rate=acceptance_rate,
            edit_rate=edit_rate,
            mean_queries_per_incident=mean_queries,
            total_scenarios=total_s,
            accepted_scenarios=accepted_s,
            edited_scenarios=edited_s,
            rejected_scenarios=rejected_s,
            pending_scenarios=pending_s,
        )

    # ── Queue inspection ────────────────────────────────────────────────────────

    def pending_queries(self) -> tuple[CandidateQuery, ...]:
        return tuple(q for q in self._queries.values() if q.status == ReviewDecision.PENDING)

    def pending_scenarios(self) -> tuple[CandidateScenario, ...]:
        return tuple(s for s in self._scenarios.values() if s.status == ReviewDecision.PENDING)

    def all_queries(self) -> tuple[CandidateQuery, ...]:
        return tuple(self._queries.values())

    def all_scenarios(self) -> tuple[CandidateScenario, ...]:
        return tuple(self._scenarios.values())
