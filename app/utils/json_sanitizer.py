"""Persistence-boundary JSON sanitization.

PostgreSQL ``jsonb`` cannot store certain Unicode control characters (notably
the NUL byte ``\\u0000``), which routinely appear in GitHub issue bodies that
embed panic dumps, stack traces, compiler output, or binary logs. This module
strips exactly those characters before persistence, recursively, without
mutating the caller's object and without source-specific logic.

It is the single sanitization layer: collectors, normalizers, adapters, and
search code must NOT sanitize — only the persistence boundary does.
"""

from __future__ import annotations

import re
from typing import Any

# PostgreSQL-invalid control characters in text/jsonb:
#   U+0000–U+0008, U+000B, U+000C, U+000E–U+001F
# Deliberately PRESERVED (valid in text/jsonb and semantically meaningful):
#   U+0009 \t, U+000A \n, U+000D \r
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize(value: Any, counter: list[int]) -> Any:
    if isinstance(value, str):
        cleaned, removed = _CONTROL_CHARS.subn("", value)
        counter[0] += removed
        return cleaned
    # NOTE: bool is a subclass of int but is not str/dict/list, so it falls
    # through to the scalar return unchanged — checked implicitly by ordering.
    if isinstance(value, dict):
        # New dict (no caller mutation); keys preserved verbatim per spec.
        return {key: _sanitize(item, counter) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, counter) for item in value]
    # Scalars (int, float, bool, None) and any other type: unchanged.
    return value


def sanitize_json_with_stats(value: Any) -> tuple[Any, int]:
    """Return ``(sanitized_value, removed_control_char_count)``.

    Builds a fresh structure; never mutates ``value``.
    """
    counter = [0]
    sanitized = _sanitize(value, counter)
    return sanitized, counter[0]


def sanitize_json(value: Any) -> Any:
    """Return a sanitized deep copy of ``value`` with PostgreSQL-invalid
    control characters removed from all nested strings."""
    return sanitize_json_with_stats(value)[0]
