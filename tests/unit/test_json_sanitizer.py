from __future__ import annotations

import copy

from app.services.deduplication import DeduplicationService
from app.utils.json_sanitizer import sanitize_json, sanitize_json_with_stats

NUL = chr(0)   # U+0000
SOH = chr(1)   # U+0001
STX = chr(2)   # U+0002


# ── Test 1 — simple string ───────────────────────────────────────────────────

def test_simple_string() -> None:
    assert sanitize_json("abc" + NUL + "def") == "abcdef"


# ── Test 2 — multiple control characters ─────────────────────────────────────

def test_multiple_control_chars() -> None:
    assert sanitize_json(NUL + "a" + SOH + "b" + STX + "c") == "abc"


# ── Test 3 — nested dictionary ───────────────────────────────────────────────

def test_nested_dict() -> None:
    assert sanitize_json({"a": {"b": "x" + NUL + "y"}}) == {"a": {"b": "xy"}}


# ── Test 4 — lists ───────────────────────────────────────────────────────────

def test_lists() -> None:
    assert sanitize_json(["a" + NUL, "b", {"c": "d" + SOH}]) == ["a", "b", {"c": "d"}]


# ── Test 5 — whitespace preservation (\t \n \r kept) ─────────────────────────

def test_whitespace_preserved() -> None:
    s = "line1\nline2\tvalue\r"
    assert sanitize_json(s) == s


# ── Test 6 — non-English text / unicode / emoji ──────────────────────────────

def test_non_english_text() -> None:
    assert sanitize_json("你好" + NUL + "世界") == "你好世界"


def test_emoji_preserved() -> None:
    assert sanitize_json("crash \U0001f4a5" + NUL + "dump") == "crash \U0001f4a5dump"


# ── Test 7 — original object unchanged + new object returned ──────────────────

def test_original_object_unchanged() -> None:
    payload = {"body": "panic" + NUL + "trace", "nested": ["x", {"y": "z"}]}
    original = copy.deepcopy(payload)

    stored = sanitize_json(payload)

    assert payload == original          # caller object untouched (NUL still present)
    assert NUL in payload["body"]
    assert payload is not stored        # fresh object returned
    assert payload["nested"] is not stored["nested"]
    assert stored == {"body": "panictrace", "nested": ["x", {"y": "z"}]}


# ── Scalars unchanged ────────────────────────────────────────────────────────

def test_scalars_unchanged() -> None:
    assert sanitize_json(42) == 42
    assert sanitize_json(3.14) == 3.14
    assert sanitize_json(True) is True
    assert sanitize_json(False) is False
    assert sanitize_json(None) is None


# ── Stats / observability ────────────────────────────────────────────────────

def test_stats_counts_removed_chars() -> None:
    sanitized, removed = sanitize_json_with_stats(
        {"a": "x" + NUL + SOH + "y", "b": ["z" + STX]}
    )
    assert removed == 3
    assert sanitized == {"a": "xy", "b": ["z"]}


def test_stats_zero_when_clean() -> None:
    _, removed = sanitize_json_with_stats({"a": "clean text\nwith\ttabs"})
    assert removed == 0


# ── Test 9 — hash consistency (stored payload == hashed payload) ──────────────

def test_hash_consistency_after_sanitization() -> None:
    dedup = DeduplicationService()
    raw = {"body": "panic" + NUL + "trace"}
    sanitized = sanitize_json(raw)

    # re-sanitizing the sanitized payload is a no-op → stable hash
    assert dedup.payload_hash(sanitized) == dedup.payload_hash(sanitize_json(sanitized))
    # and differs from the raw payload's hash — proving we must hash the
    # sanitized form to avoid perpetual false updates
    assert dedup.payload_hash(sanitized) != dedup.payload_hash(raw)
