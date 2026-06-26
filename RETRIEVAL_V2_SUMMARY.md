# Retrieval Layer v2 Summary

## Goals

Retrieval v2 improves incident relevance before further agent work by adding:

- first-class source metadata filters
- query expansion
- candidate deduplication
- LLM reranking
- a debug endpoint for retrieval inspection

## Metadata

The `incidents` table now includes:

- `owner`
- `repo`
- `source`
- `state`

For GitHub ingestion these values come from the repository metadata and issue
state:

- `owner`: GitHub organization or user
- `repo`: repository name
- `source`: `github`
- `state`: original GitHub issue state such as `open` or `closed`

Existing records are backfilled by Alembic from the `environment`, `source_type`,
and `status` fields.

## SearchService Changes

The existing `search()` method is preserved and now supports optional filters:

- `owner`
- `repo`
- `source`
- `state`

Existing parameters remain supported:

- `limit`
- `source_type`
- `tags`

## Debug Retrieval Pipeline

The new `search_debug()` workflow:

1. Searches the original query.
2. Uses the LLM to generate 3 to 5 related search phrases.
3. Searches each expanded phrase.
4. Deduplicates incidents by incident ID.
5. Keeps the best vector distance for duplicated incidents.
6. Sends the final candidate set to the LLM for reranking.
7. Returns the best 5 reranked incidents.

If no LLM service is provided, the pipeline falls back to vector ranking.

## Endpoint

```text
POST /search/debug
```

Request:

```json
{
  "query": "scheduler timeout",
  "owner": "apache",
  "repo": "airflow",
  "source": "github",
  "state": "closed"
}
```

Response:

```json
{
  "query": "scheduler timeout",
  "filters": {
    "owner": "apache",
    "repo": "airflow",
    "source": "github",
    "state": "closed"
  },
  "results": [
    {
      "title": "Scheduler heartbeat missed",
      "repo": "airflow",
      "similarity_score": 0.91
    }
  ]
}
```

## Preserved Behavior

- Existing endpoints are preserved.
- Existing `/search/incidents` still returns `SearchResponse`.
- `SearchService.search()` still performs vector search.
- Ingestion behavior is unchanged except for storing the new metadata fields.
- Database schema changes are additive.

## Migration

New migration:

```text
alembic/versions/20260611_0002_incident_retrieval_metadata.py
```

Run:

```powershell
python -m alembic upgrade head
```

