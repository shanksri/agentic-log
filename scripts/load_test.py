"""Phase 23 Part 2: Load Testing.

Drives the real FastAPI ASGI app in-process (``httpx.ASGITransport`` — no
network socket, no separate server process) at 10/50/100 concurrent workers
and reports latency distribution, throughput, and error rate per endpoint.

# Why in-process, and what that does/doesn't measure

This sandbox has no live Postgres or OpenAI credentials, so the database and
LLM/embedding dependencies are overridden with fast in-memory fakes (the
same ``app.dependency_overrides`` / monkeypatch technique the test suite
uses) — see ``_install_fakes``. The numbers below therefore measure the
platform's own overhead — FastAPI routing, Pydantic validation, the global
exception handlers added in this phase, response serialization — NOT real
database query time or real OpenAI round-trip latency. That is a genuine
scope limit of this environment, not an attempt to make the numbers look
better than they are; it is called out again in the printed report and in
the Phase 23 deliverables' Performance Findings section. Running this same
script against a real deployment (real ``DATABASE_URL``/``OPENAI_API_KEY``)
would substitute real backend latency for the near-zero fake latency and is
the recommended follow-up before shipping.

Usage::

    .venv/Scripts/python.exe scripts/load_test.py

Writes ``.benchmarks/phase23_load_test/report.json`` and a human-readable
summary to stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx

logging.getLogger("httpx").setLevel(logging.WARNING)

from app.db.session import get_db
from app.main import app

REPORT_DIR = Path(".benchmarks/phase23_load_test")
CONCURRENCY_LEVELS = (10, 50, 100)
REQUESTS_PER_WORKER = 20


@dataclass
class EndpointSpec:
    name: str
    method: str
    path: str
    json_body: dict | None = None


@dataclass
class RunResult:
    endpoint: str
    concurrency: int
    total_requests: int
    successes: int
    errors: int
    latencies_ms: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.total_requests if self.total_requests else 0.0

    @property
    def throughput_rps(self) -> float:
        return self.total_requests / self.wall_seconds if self.wall_seconds else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[index]

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "successes": self.successes,
            "errors": self.errors,
            "error_rate": round(self.error_rate, 4),
            "throughput_rps": round(self.throughput_rps, 2),
            "latency_ms": {
                "mean": round(statistics.mean(self.latencies_ms), 2) if self.latencies_ms else 0.0,
                "p50": round(self.percentile(0.50), 2),
                "p95": round(self.percentile(0.95), 2),
                "p99": round(self.percentile(0.99), 2),
                "max": round(max(self.latencies_ms), 2) if self.latencies_ms else 0.0,
            },
        }


def _fake_investigation_session() -> SimpleNamespace:
    """A minimal stand-in for MultiAgentInvestigationOrchestrator.investigate()'s
    return value — just enough attribute shape for the /agent/investigate
    route (Phase 23A: the single canonical, orchestrator-backed endpoint) to
    build a response without touching a real LLM.
    """
    from app.services.critic_agent import CritiqueVerdict
    from app.services.investigation_orchestrator import StoppingReason

    investigation = SimpleNamespace(
        problem="load-test problem",
        selected_hypothesis=SimpleNamespace(root_cause="fake root cause"),
        confidence=0.8,
        confidence_level="HIGH",
        supporting_evidence=(),
        contradicting_evidence=(),
        remaining_uncertainty=(),
        is_uncertain=False,
        rejected_hypotheses=(),
    )
    critique = SimpleNamespace(
        verdict=CritiqueVerdict.APPROVED,
        confidence=0.8,
        explanation="approved",
        findings=(),
        unresolved_questions=(),
        missing_evidence=(),
        recommended_actions=(),
    )
    return SimpleNamespace(
        final_report=SimpleNamespace(investigation=investigation, critique=critique),
        total_iterations=1,
        stopping_reason=StoppingReason.CRITIC_APPROVED,
        stop_explanation="stopped after 1 iteration",
    )


def _install_fakes() -> None:
    """Override the DB dependency and monkeypatch the search/agent
    construction points with fast in-memory fakes, exactly like the test
    suite does — see module docstring for why.
    """
    import app.api.routes.agent as agent_mod
    import app.api.routes.search as search_mod
    from app.api.auth import require_api_key
    from app.core.config import settings

    app.dependency_overrides[get_db] = lambda: MagicMock()
    # Phase 23B: every business route now requires Bearer auth; this script
    # measures routing/serialization overhead, not authentication, so it
    # bypasses the check the same way the test suite does.
    app.dependency_overrides[require_api_key] = lambda: None
    # Phase 23C: this script deliberately sends far more than any endpoint's
    # per-minute limit (that's the point of a load test) — disable rate
    # limiting entirely rather than getting 429s mixed into the results.
    settings.rate_limit_enabled = False

    fake_search_service = MagicMock()
    fake_search_service.search.return_value = []
    fake_search_service.search_debug.return_value = []
    search_mod.build_routed_search_service = lambda db, **kw: fake_search_service

    fake_orchestrator = MagicMock()
    fake_orchestrator.investigate.return_value = _fake_investigation_session()
    agent_mod.MultiAgentInvestigationOrchestrator = lambda db: fake_orchestrator


ENDPOINTS = [
    EndpointSpec("GET /health", "GET", "/health"),
    EndpointSpec(
        "POST /search/incidents",
        "POST",
        "/search/incidents",
        {"query": "database connection pool exhausted"},
    ),
    EndpointSpec(
        "POST /agent/investigate",
        "POST",
        "/agent/investigate",
        {"problem": "users report intermittent 500 errors during checkout"},
    ),
]


async def _worker(
    client: httpx.AsyncClient, spec: EndpointSpec, n_requests: int, result: RunResult
) -> None:
    for _ in range(n_requests):
        start = time.perf_counter()
        try:
            if spec.method == "GET":
                resp = await client.get(spec.path)
            else:
                resp = await client.post(spec.path, json=spec.json_body)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed_ms)
            if resp.status_code < 500:
                result.successes += 1
            else:
                result.errors += 1
        except Exception:  # noqa: BLE001 — a transport-level failure counts as an error
            result.errors += 1
            result.latencies_ms.append((time.perf_counter() - start) * 1000)
        result.total_requests += 1


async def _run_one(spec: EndpointSpec, concurrency: int) -> RunResult:
    result = RunResult(endpoint=spec.name, concurrency=concurrency, total_requests=0, successes=0, errors=0)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://loadtest") as client:
        start = time.perf_counter()
        await asyncio.gather(
            *(_worker(client, spec, REQUESTS_PER_WORKER, result) for _ in range(concurrency))
        )
        result.wall_seconds = time.perf_counter() - start
    return result


async def main() -> None:
    _install_fakes()
    all_results: list[RunResult] = []
    for spec in ENDPOINTS:
        for concurrency in CONCURRENCY_LEVELS:
            result = await _run_one(spec, concurrency)
            all_results.append(result)
            print(
                f"{spec.name:28s} concurrency={concurrency:4d}  "
                f"reqs={result.total_requests:5d}  "
                f"errors={result.errors:3d} ({result.error_rate:.1%})  "
                f"throughput={result.throughput_rps:8.1f} req/s  "
                f"p50={result.percentile(0.50):7.2f}ms  "
                f"p95={result.percentile(0.95):7.2f}ms  "
                f"p99={result.percentile(0.99):7.2f}ms"
            )
    app.dependency_overrides.clear()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "scope_limitation": (
            "Measured against in-process fakes (no live DB/LLM in this "
            "environment) — see module docstring. Re-run against a real "
            "deployment before using these numbers for capacity planning."
        ),
        "concurrency_levels": list(CONCURRENCY_LEVELS),
        "requests_per_worker": REQUESTS_PER_WORKER,
        "results": [r.to_dict() for r in all_results],
    }
    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_DIR / 'report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
