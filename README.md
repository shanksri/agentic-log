# Enterprise Incident Intelligence Platform

**This README describes the original Phase 1 scope only** (GitHub incident ingestion and semantic
incident search). The codebase has since grown well beyond it — see
[docs/README.md](docs/README.md) for the full, current engineering reference, including
[docs/architecture/18](docs/architecture/18_adaptive_routing_and_hybrid_confidence.md) through
[22](docs/architecture/22_evaluation_api.md) covering adaptive routing, hybrid retrieval, a
multi-agent investigation framework, reasoning evaluation, LLM-as-judge, and a 15-endpoint
evaluation REST API. Adaptive routing (doc 18) and the multi-agent investigation orchestrator
(doc 19) are now wired into the primary `/search` and `/agent` routes below (dense-only by default
until explicitly opted in); the reasoning-evaluation platform (docs 20–21) remains reachable only
through the REST API in doc 22 or the CLI scripts in `scripts/` — see the "Status" note in
[docs/README.md](docs/README.md#status-whats-built-vs-whats-wired).

## Included (Phase 1)

- FastAPI backend
- PostgreSQL schema with pgvector
- SQLAlchemy models
- Alembic migrations
- GitHub issue collector and normalizer
- Deduplication service
- SentenceTransformers embedding service
- Similarity search API
- Unit tests

## Excluded From Phase 1 (now built — see docs/README.md)

- LangGraph — still not used anywhere in this codebase.
- Agents — a single-shot investigation agent shipped early on; a separate four-agent framework
  (planner, hypothesis generation, critic, iterative orchestrator) has since been built (doc 19)
  and is wired in as `POST /agent/investigate-orchestrated`, the canonical investigation endpoint,
  alongside the original single-shot agents which remain available unmodified.
- Evaluation framework — built far beyond Phase 1's scope: retrieval evaluation (doc 15), reasoning
  evaluation and LLM-as-judge (doc 20), AI quality intelligence and judge validation (doc 21), and a
  REST API over all of it (doc 22).
- Feedback system — still not implemented.

## Local Run

```bash
docker compose up --build
```

API docs are available at `http://localhost:8000/docs`.

