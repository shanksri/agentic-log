# Cleanup Report

## Summary

Temporary ingestion troubleshooting code has been removed. The GitHub ingestion
endpoint and ingestion service flow have been restored without changing
business logic, database schema, or API contracts.

## Files Modified

### `app/api/routes/ingestion.py`

- Removed `raise Exception("ROUTE_MARKER_123")`.
- Restored the normal call to
  `IncidentIngestionService(db).ingest_github_repo(...)`.
- Restored response validation with
  `GitHubIngestResponse.model_validate(result)`.

### `app/main.py`

- Removed the temporary `print("MAIN_PY_LOADED_123")` startup marker.

### `app/services/incident_ingestion.py`

- Removed temporary `failed_payload.json` generation.
- Removed the temporary `_dump_failed_payload()` helper.
- Removed the temporary `_log_stage_exception()` helper.
- Removed stage-by-stage troubleshooting wrappers and verbose per-stage logs.
- Removed temporary `_current_payload` diagnostic state.
- Restored the original normalize, raw-document upsert, incident upsert, and
  embedding upsert flow.

### `transaction_state_analysis.md`

- Removed stale references to the temporary route marker and payload dump
  artifact so repository-wide troubleshooting-marker searches are clean.

## Logging Intentionally Retained

- The module-level logger in `app/services/incident_ingestion.py` remains in
  place as part of the application logging structure.
- Existing collector logging remains unchanged. In particular,
  `app/ingestion/collectors/github_collector.py` continues to log the number of
  collected GitHub issues and repository name.

## Repository Scan

The repository was searched for:

- `ROUTE_MARKER_123`
- `MAIN_PY_LOADED_123`
- `failed_payload.json`
- `DEBUG_`
- `print(`

No matches remain outside the excluded local virtual environment.

No generated `failed_payload.json` file exists in the repository.

## Verification

### Ruff

Passed:

```text
All checks passed!
```

### Python Syntax

Passed for:

- `app/api/routes/ingestion.py`
- `app/main.py`
- `app/services/incident_ingestion.py`

### Application Runtime

Runtime verification could not be completed because the local Docker daemon
became unresponsive during:

```text
docker compose up -d --build
```

The rebuild command timed out after approximately 20 minutes. Subsequent
`docker compose ps`, `docker compose logs`, and `docker version` probes also
timed out. Direct requests to:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

timed out because the local service was not responding.

The cleaned Python files compile and pass Ruff. Docker Desktop must recover or
be restarted before `/health` and `/docs` can be verified against a running
container.

