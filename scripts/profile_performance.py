"""Phase 23 Part 5: Performance Profiling.

Profiles the platform's pure-computation hot paths — the parts that don't
require a live Postgres/OpenAI connection in this environment — and reports
latency distributions (mean/p50/p95/p99) over many repetitions:

  1. Retrieval  — BM25 indexing + query scoring (real ``BM25Retriever``)
  2. Routing    — rule-based strategy decision (real ``RoutingEngine``)
  3. Generation — BERTScore greedy-matching (real ``bert_score``, fake embedder)
  4. Evaluation — Recall/MRR/NDCG metric computation (real ``app.evaluation.metrics``)
  5. Orchestration — multi-agent loop control flow, with a fake instant LLM
     and fake search service standing in for the network calls

# What this does NOT measure

None of these numbers include real database query latency or real OpenAI
round-trip latency — this sandbox has neither. That is the actual, honest
scope of what can be measured here; it is not an attempt to make the
platform look faster than it is. In a real deployment, retrieval and
generation/orchestration latency is dominated by DB I/O and LLM round trips
respectively, not by the computation profiled below — re-run against a live
environment for real end-to-end numbers before using this for capacity
planning.

Usage::

    .venv/Scripts/python.exe scripts/profile_performance.py

Writes ``.benchmarks/phase23_performance/report.json``.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

REPORT_DIR = Path(".benchmarks/phase23_performance")
ITERATIONS = 200


def _timed(fn, iterations: int = ITERATIONS) -> dict:
    latencies_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - start) * 1000)
    ordered = sorted(latencies_ms)

    def pct(p: float) -> float:
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    return {
        "iterations": iterations,
        "mean_ms": round(statistics.mean(latencies_ms), 4),
        "p50_ms": round(pct(0.50), 4),
        "p95_ms": round(pct(0.95), 4),
        "p99_ms": round(pct(0.99), 4),
        "max_ms": round(max(latencies_ms), 4),
    }


# ── 1. Retrieval: BM25 indexing + scoring ────────────────────────────────────


def profile_retrieval() -> dict:
    from app.services.bm25_search import BM25Document, BM25Retriever

    corpus = [
        BM25Document(
            document_id=str(i),
            text=f"database connection pool exhausted incident {i} timeout retry backoff",
        )
        for i in range(500)
    ]

    index_result = _timed(lambda: BM25Retriever.from_documents(corpus), iterations=20)

    retriever = BM25Retriever.from_documents(corpus)
    query_result = _timed(lambda: retriever.retrieve("connection pool timeout", limit=10))

    return {"index_build_500_docs": index_result, "query_scoring": query_result}


# ── 2. Routing: rule-based decision ──────────────────────────────────────────


def profile_routing() -> dict:
    from app.services.routing import DefaultRuleBasedRoutingPolicy, RoutingEngine

    engine = RoutingEngine(DefaultRuleBasedRoutingPolicy())
    queries = [
        "db timeout",
        "NullPointerException at com.example.Service.process(Service.java:42)",
        'error code "E_CONN_RESET"',
        "users are reporting intermittent checkout failures across multiple regions during peak traffic hours",
    ]
    return _timed(lambda: [engine.route(q) for q in queries])


# ── 3. Generation: BERTScore ──────────────────────────────────────────────────


class _FakeTokenEmbedder:
    """Deterministic, instant stand-in for SentenceTransformerTokenEmbedder —
    isolates BERTScore's greedy-matching cost from real model inference.
    """

    def embed_tokens(self, text: str):
        tokens = text.split()
        return [[float((hash(t) % 997)) / 997.0, float(len(t)) / 20.0] for t in tokens]


def profile_generation() -> dict:
    from app.evaluation.generation_metrics import bert_score

    embedder = _FakeTokenEmbedder()
    candidate = "The incident was caused by database connection pool exhaustion under load."
    reference = "Root cause: the connection pool ran out of connections during a traffic spike."
    return _timed(lambda: bert_score(candidate, reference, token_embedder=embedder))


# ── 4. Evaluation: Recall / MRR / NDCG ───────────────────────────────────────


def profile_evaluation() -> dict:
    from app.evaluation.metrics import dcg_at_k, recall_at_k, reciprocal_rank

    retrieved = [uuid.uuid4() for _ in range(50)]
    relevant_ids = set(retrieved[3:8])
    relevance_by_id = {rid: 2 for rid in relevant_ids}

    def run_all():
        recall_at_k(retrieved, relevant_ids, 10)
        reciprocal_rank(retrieved, relevant_ids)
        dcg_at_k(retrieved, relevance_by_id, 10)

    return _timed(run_all)


# ── 5. Orchestration: multi-agent loop control flow (fake LLM/search) ───────


class _InstantFakeLLM:
    def generate_hypotheses(self, *, problem, context, n=2, existing_root_causes=None):
        return [
            {
                "root_cause": "expired auth token",
                "confidence_score": 0.9,
                "validation_keywords": ["token", "expired"],
                "rationale": "matches symptom timing",
            }
        ]


class _InstantFakeSearch:
    llm_service = None

    def retrieve(self, query, *, limit=10, expand=False, rerank=False, call_site=None):
        incident = SimpleNamespace(
            title="token expiry incident",
            symptoms=[SimpleNamespace(text="token"), SimpleNamespace(text="expired")],
        )
        from app.services.search import IncidentSearchResult

        return [IncidentSearchResult(incident=incident, distance=0.1)]

    def search(self, query, *, limit=10, call_site=None):
        return self.retrieve(query, limit=limit)


def profile_orchestration() -> dict:
    from app.services.investigation_orchestrator import MultiAgentInvestigationOrchestrator

    def run_one():
        orchestrator = MultiAgentInvestigationOrchestrator(
            db=None, search_service=_InstantFakeSearch(), llm_service=_InstantFakeLLM(),
        )
        orchestrator.investigate("users cannot log in", n_hypotheses=1)

    return _timed(run_one, iterations=50)


def main() -> None:
    results = {
        "retrieval_bm25": profile_retrieval(),
        "routing_decision": profile_routing(),
        "generation_bertscore": profile_generation(),
        "evaluation_metrics": profile_evaluation(),
        "orchestration_loop_fake_llm": profile_orchestration(),
    }
    for name, stats in results.items():
        if "iterations" in stats:
            print(
                f"{name:32s} n={stats['iterations']:4d}  "
                f"mean={stats['mean_ms']:8.4f}ms  p50={stats['p50_ms']:8.4f}ms  "
                f"p95={stats['p95_ms']:8.4f}ms  p99={stats['p99_ms']:8.4f}ms"
            )
        else:
            for sub_name, sub_stats in stats.items():
                print(
                    f"{name}.{sub_name:20s} n={sub_stats['iterations']:4d}  "
                    f"mean={sub_stats['mean_ms']:8.4f}ms  p50={sub_stats['p50_ms']:8.4f}ms  "
                    f"p95={sub_stats['p95_ms']:8.4f}ms  p99={sub_stats['p99_ms']:8.4f}ms"
                )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "scope_limitation": (
            "Pure-computation profiling only — no live DB or OpenAI round trip "
            "included (see module docstring). Real deployment latency for "
            "retrieval/generation/orchestration is dominated by DB I/O and LLM "
            "round trips, not the computation profiled here."
        ),
        "results": results,
    }
    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
