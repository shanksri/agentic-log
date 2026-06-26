# Enterprise Incident Intelligence Platform

Phase 1 implements GitHub incident ingestion and semantic incident search.

## Included

- FastAPI backend
- PostgreSQL schema with pgvector
- SQLAlchemy models
- Alembic migrations
- GitHub issue collector and normalizer
- Deduplication service
- SentenceTransformers embedding service
- Similarity search API
- Unit tests

## Excluded From Phase 1

- LangGraph
- Agents
- Evaluation framework
- Feedback system

## Local Run

```bash
docker compose up --build
```

API docs are available at `http://localhost:8000/docs`.

