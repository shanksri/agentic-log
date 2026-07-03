"""Shared pytest fixtures for tests/api/.

Phase 23C: rate limiting (app/api/rate_limit.py) uses a process-local,
in-memory backend shared by every request the running process serves —
including every TestClient request across the whole test session, since
they all import the same `app` object. Without a reset between tests,
a low-limit endpoint (e.g. /evaluation/full at 2 requests/minute) would
accumulate hits across dozens of unrelated test functions within the same
60-second wall-clock window and start failing them with 429s that have
nothing to do with what each test is actually checking.

This autouse fixture resets the counters before every test in this
directory, keeping rate limiting genuinely ENABLED (not bypassed) through
the whole suite — proving it doesn't interfere with normal request flows —
while giving each test a clean slate. tests/api/test_rate_limiting.py
relies on this same reset for its own test-to-test isolation.
"""
from __future__ import annotations

import pytest

from app.api.rate_limit import reset_rate_limits


@pytest.fixture(autouse=True)
def _reset_rate_limits() -> None:
    reset_rate_limits()
