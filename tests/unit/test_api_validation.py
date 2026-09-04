"""Tests for the shared API input validators.

These guard identifiers that get joined onto a filesystem path, so the
traversal cases below are the reason the module exists rather than edge cases.

The all-punctuation rule was added after ``GET /evaluation/runs/....`` returned
a 500: the allow-list permits "." and "-", so "..", "...", "." and "-" all
passed validation. ``history_root / ".."`` resolves one directory ABOVE the
intended root, and on Windows a segment of only dots is stripped entirely, so
"...." collapses onto the root itself. The 500 was the symptom -- reading
``metadata.json`` from a directory that is not a run -- but where that file
happens to exist the endpoint would have served data from outside the run
directory instead of erroring.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.api.validation import validate_safe_identifier, validate_uuid


def _reject(value: str) -> HTTPException:
    with pytest.raises(HTTPException) as excinfo:
        validate_safe_identifier(value, field_name="run_id")
    return excinfo.value


# ── Path traversal ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["..", "...", "....", ".", ".....", "../", ".."])
def test_dot_only_identifiers_are_rejected(value: str) -> None:
    assert _reject(value).status_code == 422


@pytest.mark.parametrize("value", ["-", "--", "_", "__", "-_-", "._-"])
def test_punctuation_only_identifiers_are_rejected(value: str) -> None:
    assert _reject(value).status_code == 422


@pytest.mark.parametrize("value", ["../etc/passwd", "a/b", "a\\b", "a b", "a;b", "a%2e%2e"])
def test_separators_and_disallowed_characters_are_rejected(value: str) -> None:
    assert _reject(value).status_code == 422


def test_traversal_rejection_names_the_field() -> None:
    assert "run_id" in str(_reject("..").detail)


# ── Values that must keep working ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "run-2024-01-15",
        "v1.2.3",
        "a_b-c",
        "experiment.2024.01",
        "0",
        "a",
        "phase22f_confidence_calibration",
        "..run",       # leading dots are a filename, not traversal
        "run..",       # as are trailing ones, given an alphanumeric is present
    ],
)
def test_realistic_identifiers_are_accepted(value: str) -> None:
    assert validate_safe_identifier(value, field_name="run_id") == value


# ── Empty and oversized ────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
def test_empty_or_whitespace_is_rejected(value: str) -> None:
    assert _reject(value).status_code == 422


def test_length_boundary_is_inclusive_at_200() -> None:
    assert validate_safe_identifier("a" * 200, field_name="run_id") == "a" * 200


def test_over_the_length_limit_is_rejected() -> None:
    assert _reject("a" * 201).status_code == 422


# ── UUID validator ─────────────────────────────────────────────────────────────


def test_valid_uuid_is_parsed() -> None:
    value = str(uuid.uuid4())
    assert validate_uuid(value, field_name="incident_id") == uuid.UUID(value)


@pytest.mark.parametrize("value", ["not-a-uuid", "", "..", "12345"])
def test_invalid_uuid_is_rejected(value: str) -> None:
    with pytest.raises(HTTPException) as excinfo:
        validate_uuid(value, field_name="incident_id")
    assert excinfo.value.status_code == 422
