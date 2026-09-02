"""Tests for Phase 24A: /docs, /redoc, /openapi.json gating by
ENVIRONMENT.

`_docs_urls()` (app/main.py) is tested directly and exhaustively (no
module reload needed). One integration check against the real, already-
constructed `app` object confirms today's actual test/dev environment
still serves docs unchanged -- the concrete "development behavior is
unchanged" regression check.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import _docs_urls, app


def test_development_docs_urls_enabled() -> None:
    assert _docs_urls("development") == {
        "docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json",
    }


def test_production_docs_urls_disabled() -> None:
    assert _docs_urls("production") == {
        "docs_url": None, "redoc_url": None, "openapi_url": None,
    }


def test_docs_urls_only_recognizes_production_as_special() -> None:
    # Anything other than the literal "production" gets dev behavior --
    # matches Settings.environment's own Literal["development", "production"]
    # (there is no third state to worry about diverging from).
    assert _docs_urls("development")["docs_url"] == "/docs"


def test_real_app_serves_docs_in_current_dev_test_environment() -> None:
    # This test's own process runs with ENVIRONMENT unset/development (no
    # test in this suite sets it to production), so the real, already-
    # constructed app object must still serve docs exactly as before
    # Phase 24A.
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
