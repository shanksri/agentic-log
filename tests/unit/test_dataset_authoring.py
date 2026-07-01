"""Tests for Phase 21C: Gold Dataset Authoring Framework."""
from __future__ import annotations

import json

import pytest

from app.evaluation.dataset_authoring import (
    AuthorResponseError,
    AuthoringStats,
    CandidateNotFoundError,
    CandidateQuery,
    CandidateScenario,
    DatasetAuthor,
    IncidentSummary,
    LLMDatasetAuthor,
    ReviewDecision,
    VersionAlreadyExportedError,
    _build_query_prompt,
    _build_scenario_prompt,
    _parse_query_response,
    _parse_scenario_response,
)
from app.evaluation.gold_dataset import GoldDataset
from app.evaluation.reasoning_dataset import ReasoningGoldDataset


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _incident(**overrides) -> IncidentSummary:
    defaults = dict(
        incident_id="INC-001",
        title="API gateway 502 errors",
        description="The API gateway started returning 502 Bad Gateway for all /checkout routes.",
        source_type="pagerduty",
        source_external_id="PD-123",
    )
    defaults.update(overrides)
    return IncidentSummary(**defaults)


def _query_json(*methods: str) -> str:
    candidates = []
    for method in methods:
        candidates.append({
            "generation_method": method,
            "query": f"query for {method}",
            "rationale": f"rationale for {method}",
        })
    return json.dumps({"candidates": candidates})


def _scenario_json(
    *,
    problem: str = "API gateway returning 502 errors for all checkout routes",
    causes: list[str] | None = None,
    strategy: str = "infrastructure_failure",
    rationale: str = "gateway sits in the infrastructure layer",
) -> str:
    return json.dumps({
        "problem": problem,
        "expected_root_causes": causes or ["502 gateway error", "backend pool unhealthy"],
        "suggested_strategy": strategy,
        "rationale": rationale,
    })


class FakeLLM:
    """Minimal stub: cycles through a list of pre-baked responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self._calls.append(prompt)
        return self._responses.pop(0)


def _author(*responses: str) -> LLMDatasetAuthor:
    return LLMDatasetAuthor(FakeLLM(list(responses)))


# ── IncidentSummary ──────────────────────────────────────────────────────────────


def test_incident_summary_valid() -> None:
    assert _incident().is_valid()


def test_incident_summary_empty_fields_are_invalid() -> None:
    assert not IncidentSummary("", "title", "desc").is_valid()
    assert not IncidentSummary("id", "", "desc").is_valid()
    assert not IncidentSummary("id", "title", "").is_valid()


# ── generate_queries ─────────────────────────────────────────────────────────────


def test_generate_queries_returns_candidates_for_each_method() -> None:
    methods = ["exact_keyword", "paraphrase", "symptom_description"]
    author = _author(_query_json(*methods))
    candidates = author.generate_queries(_incident(), n=3)
    assert len(candidates) == 3
    assert all(isinstance(c, CandidateQuery) for c in candidates)
    assert {c.generation_method for c in candidates} == set(methods)


def test_generate_queries_all_start_as_pending() -> None:
    author = _author(_query_json("exact_keyword", "paraphrase"))
    candidates = author.generate_queries(_incident())
    assert all(c.status == ReviewDecision.PENDING for c in candidates)


def test_generate_queries_maps_method_to_category() -> None:
    author = _author(_query_json("exact_keyword", "multi_concept"))
    candidates = author.generate_queries(_incident())
    by_method = {c.generation_method: c for c in candidates}
    assert by_method["exact_keyword"].category == "lexical-overlap"
    assert by_method["multi_concept"].category == "multi-concept"


def test_generate_queries_maps_method_to_difficulty() -> None:
    author = _author(_query_json("exact_keyword", "multi_concept"))
    candidates = author.generate_queries(_incident())
    by_method = {c.generation_method: c for c in candidates}
    assert by_method["exact_keyword"].difficulty == "easy"
    assert by_method["multi_concept"].difficulty == "hard"


def test_generate_queries_assigns_incident_id() -> None:
    author = _author(_query_json("paraphrase"))
    candidates = author.generate_queries(_incident(incident_id="INC-999"))
    assert all(c.incident_id == "INC-999" for c in candidates)


def test_generate_queries_candidates_have_unique_ids() -> None:
    author = _author(_query_json("exact_keyword", "paraphrase", "novice_wording"))
    candidates = author.generate_queries(_incident())
    ids = [c.id for c in candidates]
    assert len(ids) == len(set(ids))


def test_generate_queries_accumulates_across_calls() -> None:
    author = _author(
        _query_json("exact_keyword"),
        _query_json("paraphrase"),
    )
    author.generate_queries(_incident(incident_id="A"))
    author.generate_queries(_incident(incident_id="B"))
    assert len(author.all_queries()) == 2


# ── generate_investigation ───────────────────────────────────────────────────────


def test_generate_investigation_returns_scenario() -> None:
    author = _author(_scenario_json())
    scenario = author.generate_investigation(_incident())
    assert isinstance(scenario, CandidateScenario)
    assert scenario.status == ReviewDecision.PENDING
    assert scenario.suggested_strategy == "infrastructure_failure"


def test_generate_investigation_preserves_root_causes() -> None:
    author = _author(_scenario_json(causes=["upstream timeout", "connection pool exhausted"]))
    scenario = author.generate_investigation(_incident())
    assert "upstream timeout" in scenario.expected_root_causes
    assert "connection pool exhausted" in scenario.expected_root_causes


def test_generate_investigation_assigns_incident_id() -> None:
    author = _author(_scenario_json())
    scenario = author.generate_investigation(_incident(incident_id="INC-042"))
    assert scenario.incident_id == "INC-042"


# ── review — queries ─────────────────────────────────────────────────────────────


def test_review_accept_query() -> None:
    author = _author(_query_json("paraphrase"))
    (candidate,) = author.generate_queries(_incident())
    updated = author.review(candidate.id, ReviewDecision.ACCEPTED)
    assert updated.status == ReviewDecision.ACCEPTED
    assert isinstance(updated, CandidateQuery)


def test_review_reject_query() -> None:
    author = _author(_query_json("exact_keyword"))
    (candidate,) = author.generate_queries(_incident())
    updated = author.review(candidate.id, ReviewDecision.REJECTED)
    assert updated.status == ReviewDecision.REJECTED


def test_review_edit_query_stores_edited_text_and_preserves_original() -> None:
    author = _author(_query_json("novice_wording"))
    (candidate,) = author.generate_queries(_incident())
    original_query = candidate.query
    updated = author.review(candidate.id, ReviewDecision.EDITED, edited_text="better query text")
    assert updated.status == ReviewDecision.EDITED
    assert updated.edited_query == "better query text"
    assert updated.query == original_query  # original preserved
    assert updated.effective_query == "better query text"


def test_review_edit_query_requires_edited_text() -> None:
    author = _author(_query_json("paraphrase"))
    (candidate,) = author.generate_queries(_incident())
    with pytest.raises(ValueError):
        author.review(candidate.id, ReviewDecision.EDITED)


def test_review_edit_query_rejects_blank_text() -> None:
    author = _author(_query_json("paraphrase"))
    (candidate,) = author.generate_queries(_incident())
    with pytest.raises(ValueError):
        author.review(candidate.id, ReviewDecision.EDITED, edited_text="   ")


def test_review_unknown_candidate_raises() -> None:
    author = _author(_query_json("paraphrase"))
    author.generate_queries(_incident())
    with pytest.raises(CandidateNotFoundError):
        author.review("nonexistent-id", ReviewDecision.ACCEPTED)


# ── review — scenarios ───────────────────────────────────────────────────────────


def test_review_accept_scenario() -> None:
    author = _author(_scenario_json())
    scenario = author.generate_investigation(_incident())
    updated = author.review(scenario.id, ReviewDecision.ACCEPTED)
    assert updated.status == ReviewDecision.ACCEPTED
    assert isinstance(updated, CandidateScenario)


def test_review_edit_scenario_preserves_original_problem() -> None:
    author = _author(_scenario_json(problem="original problem text"))
    scenario = author.generate_investigation(_incident())
    updated = author.review(
        scenario.id, ReviewDecision.EDITED, edited_text="refined problem statement"
    )
    assert isinstance(updated, CandidateScenario)
    assert updated.problem == "original problem text"
    assert updated.edited_problem == "refined problem statement"
    assert updated.effective_problem == "refined problem statement"


# ── export_gold_dataset ──────────────────────────────────────────────────────────


def test_export_gold_dataset_includes_only_accepted_and_edited() -> None:
    author = _author(_query_json("exact_keyword", "paraphrase", "novice_wording"))
    candidates = author.generate_queries(_incident())
    author.review(candidates[0].id, ReviewDecision.ACCEPTED)
    author.review(candidates[1].id, ReviewDecision.EDITED, edited_text="edited query")
    author.review(candidates[2].id, ReviewDecision.REJECTED)
    dataset = author.export_gold_dataset(version="retrieval_v1", description="test")
    assert isinstance(dataset, GoldDataset)
    assert len(dataset.queries) == 2


def test_export_gold_dataset_uses_effective_query() -> None:
    author = _author(_query_json("paraphrase"))
    (candidate,) = author.generate_queries(_incident())
    author.review(candidate.id, ReviewDecision.EDITED, edited_text="my edited query")
    dataset = author.export_gold_dataset(version="retrieval_v1", description="d")
    assert dataset.queries[0].query == "my edited query"


def test_export_gold_dataset_version_collision_raises() -> None:
    author = _author(_query_json("exact_keyword"), _query_json("paraphrase"))
    (c,) = author.generate_queries(_incident())
    author.review(c.id, ReviewDecision.ACCEPTED)
    author.export_gold_dataset(version="retrieval_v1", description="first")
    (c2,) = author.generate_queries(_incident())
    author.review(c2.id, ReviewDecision.ACCEPTED)
    with pytest.raises(VersionAlreadyExportedError):
        author.export_gold_dataset(version="retrieval_v1", description="second attempt")


def test_export_gold_dataset_different_versions_succeed() -> None:
    author = _author(_query_json("exact_keyword"), _query_json("paraphrase"))
    (c1,) = author.generate_queries(_incident())
    author.review(c1.id, ReviewDecision.ACCEPTED)
    d1 = author.export_gold_dataset(version="retrieval_v1", description="v1")
    (c2,) = author.generate_queries(_incident())
    author.review(c2.id, ReviewDecision.ACCEPTED)
    d2 = author.export_gold_dataset(version="retrieval_v2", description="v2")
    assert d1.version == "retrieval_v1"
    assert d2.version == "retrieval_v2"


def test_export_gold_dataset_no_accepted_raises() -> None:
    author = _author(_query_json("paraphrase"))
    (candidate,) = author.generate_queries(_incident())
    author.review(candidate.id, ReviewDecision.REJECTED)
    with pytest.raises(ValueError):
        author.export_gold_dataset(version="retrieval_v1", description="d")


def test_export_gold_dataset_pending_items_excluded() -> None:
    author = _author(_query_json("exact_keyword", "paraphrase"))
    candidates = author.generate_queries(_incident())
    author.review(candidates[0].id, ReviewDecision.ACCEPTED)
    # candidates[1] stays PENDING
    dataset = author.export_gold_dataset(version="retrieval_v1", description="d")
    assert len(dataset.queries) == 1


# ── export_reasoning_dataset ─────────────────────────────────────────────────────


def test_export_reasoning_dataset_accepted_scenarios() -> None:
    author = _author(_scenario_json(), _scenario_json())
    s1 = author.generate_investigation(_incident())
    s2 = author.generate_investigation(_incident())
    author.review(s1.id, ReviewDecision.ACCEPTED)
    author.review(s2.id, ReviewDecision.REJECTED)
    dataset = author.export_reasoning_dataset(version="reasoning_v1", description="d")
    assert isinstance(dataset, ReasoningGoldDataset)
    assert len(dataset.scenarios) == 1


def test_export_reasoning_dataset_uses_effective_problem() -> None:
    author = _author(_scenario_json(problem="original"))
    s = author.generate_investigation(_incident())
    author.review(s.id, ReviewDecision.EDITED, edited_text="clearer problem")
    dataset = author.export_reasoning_dataset(version="reasoning_v1", description="d")
    assert dataset.scenarios[0].problem == "clearer problem"


def test_export_reasoning_dataset_version_collision_raises() -> None:
    author = _author(_scenario_json(), _scenario_json())
    s1 = author.generate_investigation(_incident())
    author.review(s1.id, ReviewDecision.ACCEPTED)
    author.export_reasoning_dataset(version="reasoning_v1", description="d")
    s2 = author.generate_investigation(_incident())
    author.review(s2.id, ReviewDecision.ACCEPTED)
    with pytest.raises(VersionAlreadyExportedError):
        author.export_reasoning_dataset(version="reasoning_v1", description="d2")


def test_export_reasoning_dataset_no_accepted_raises() -> None:
    author = _author(_scenario_json())
    s = author.generate_investigation(_incident())
    author.review(s.id, ReviewDecision.REJECTED)
    with pytest.raises(ValueError):
        author.export_reasoning_dataset(version="reasoning_v1", description="d")


# ── Statistics ───────────────────────────────────────────────────────────────────


def test_stats_empty_session() -> None:
    author = LLMDatasetAuthor(FakeLLM([]))
    s = author.stats()
    assert s.total_generated == 0
    assert s.acceptance_rate == 0.0
    assert s.edit_rate == 0.0
    assert s.mean_queries_per_incident == 0.0


def test_stats_counts_all_statuses() -> None:
    author = _author(_query_json("exact_keyword", "paraphrase", "novice_wording"))
    candidates = author.generate_queries(_incident())
    author.review(candidates[0].id, ReviewDecision.ACCEPTED)
    author.review(candidates[1].id, ReviewDecision.EDITED, edited_text="edit")
    author.review(candidates[2].id, ReviewDecision.REJECTED)
    s = author.stats()
    assert s.total_generated == 3
    assert s.accepted == 1
    assert s.edited == 1
    assert s.rejected == 1
    assert s.pending == 0
    assert s.acceptance_rate == pytest.approx(2 / 3)
    assert s.edit_rate == pytest.approx(0.5)


def test_stats_pending_count() -> None:
    author = _author(_query_json("exact_keyword", "paraphrase"))
    candidates = author.generate_queries(_incident())
    author.review(candidates[0].id, ReviewDecision.ACCEPTED)
    s = author.stats()
    assert s.pending == 1


def test_stats_mean_queries_per_incident() -> None:
    author = _author(
        _query_json("exact_keyword", "paraphrase"),
        _query_json("exact_keyword"),
    )
    author.generate_queries(_incident(incident_id="A"))
    author.generate_queries(_incident(incident_id="B"))
    s = author.stats()
    assert s.mean_queries_per_incident == pytest.approx(1.5)


def test_stats_scenario_counts() -> None:
    author = _author(_scenario_json(), _scenario_json())
    s1 = author.generate_investigation(_incident())
    s2 = author.generate_investigation(_incident())
    author.review(s1.id, ReviewDecision.ACCEPTED)
    s = author.stats()
    assert s.total_scenarios == 2
    assert s.accepted_scenarios == 1
    assert s.pending_scenarios == 1


# ── Pending queue ────────────────────────────────────────────────────────────────


def test_pending_queries_shows_only_pending() -> None:
    author = _author(_query_json("exact_keyword", "paraphrase"))
    candidates = author.generate_queries(_incident())
    author.review(candidates[0].id, ReviewDecision.ACCEPTED)
    pending = author.pending_queries()
    assert len(pending) == 1
    assert pending[0].id == candidates[1].id


def test_pending_scenarios_shows_only_pending() -> None:
    author = _author(_scenario_json(), _scenario_json())
    s1 = author.generate_investigation(_incident())
    s2 = author.generate_investigation(_incident())
    author.review(s1.id, ReviewDecision.REJECTED)
    pending = author.pending_scenarios()
    assert len(pending) == 1
    assert pending[0].id == s2.id


# ── Immutability ─────────────────────────────────────────────────────────────────


def test_candidate_query_is_frozen() -> None:
    author = _author(_query_json("paraphrase"))
    (candidate,) = author.generate_queries(_incident())
    with pytest.raises(Exception):
        candidate.status = ReviewDecision.ACCEPTED  # type: ignore[misc]


def test_candidate_scenario_is_frozen() -> None:
    author = _author(_scenario_json())
    scenario = author.generate_investigation(_incident())
    with pytest.raises(Exception):
        scenario.status = ReviewDecision.ACCEPTED  # type: ignore[misc]


def test_authoring_stats_is_frozen() -> None:
    author = LLMDatasetAuthor(FakeLLM([]))
    s = author.stats()
    with pytest.raises(Exception):
        s.total_generated = 99  # type: ignore[misc]


# ── LLM response parsing edge cases ─────────────────────────────────────────────


def test_parse_query_response_rejects_non_json() -> None:
    with pytest.raises(AuthorResponseError, match="not valid JSON"):
        _parse_query_response("this is not json", "INC-1")


def test_parse_query_response_rejects_missing_candidates_key() -> None:
    with pytest.raises(AuthorResponseError, match="missing 'candidates'"):
        _parse_query_response(json.dumps({"results": []}), "INC-1")


def test_parse_query_response_rejects_invalid_generation_method() -> None:
    raw = json.dumps({"candidates": [{
        "generation_method": "magic_mode",
        "query": "q",
        "rationale": "r",
    }]})
    with pytest.raises(AuthorResponseError, match="generation_method"):
        _parse_query_response(raw, "INC-1")


def test_parse_query_response_rejects_missing_field() -> None:
    raw = json.dumps({"candidates": [{"generation_method": "paraphrase", "query": "q"}]})
    with pytest.raises(AuthorResponseError, match="rationale"):
        _parse_query_response(raw, "INC-1")


def test_parse_query_response_rejects_empty_query() -> None:
    raw = json.dumps({"candidates": [{
        "generation_method": "paraphrase", "query": "  ", "rationale": "r"
    }]})
    with pytest.raises(AuthorResponseError, match="non-empty string"):
        _parse_query_response(raw, "INC-1")


def test_parse_scenario_response_rejects_invalid_strategy() -> None:
    raw = json.dumps({
        "problem": "p", "expected_root_causes": ["x"], "suggested_strategy": "magic",
        "rationale": "r",
    })
    with pytest.raises(AuthorResponseError, match="suggested_strategy"):
        _parse_scenario_response(raw)


def test_parse_scenario_response_rejects_non_json() -> None:
    with pytest.raises(AuthorResponseError, match="not valid JSON"):
        _parse_scenario_response("not json")


def test_parse_scenario_response_rejects_missing_field() -> None:
    raw = json.dumps({"problem": "p", "expected_root_causes": []})
    with pytest.raises(AuthorResponseError):
        _parse_scenario_response(raw)


# ── Prompt content checks ────────────────────────────────────────────────────────


def test_query_prompt_includes_incident_title_and_description() -> None:
    inc = _incident(title="DB connection failure", description="The database is down.")
    prompt = _build_query_prompt(inc, 5)
    assert "DB connection failure" in prompt
    assert "The database is down." in prompt
    assert "exact_keyword" in prompt
    assert "multi_concept" in prompt


def test_scenario_prompt_includes_incident_details() -> None:
    inc = _incident(title="Memory leak in worker", description="Workers run out of heap.")
    prompt = _build_scenario_prompt(inc)
    assert "Memory leak in worker" in prompt
    assert "Workers run out of heap." in prompt
    assert "infrastructure_failure" in prompt


# ── DatasetAuthor is abstract ────────────────────────────────────────────────────


def test_dataset_author_is_abstract() -> None:
    with pytest.raises(TypeError):
        DatasetAuthor()  # type: ignore[abstract]
