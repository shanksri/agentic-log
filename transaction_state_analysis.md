# Transaction State Analysis

## Scope

This analysis covers the current ingestion pipeline and FastAPI database-session lifecycle.

## Transaction Lifecycle

`app/db/session.py:10-11` creates a shared SQLAlchemy engine and a `SessionLocal`
factory with `autoflush=False`.

`app/db/session.py:14-19` creates one SQLAlchemy `Session` per FastAPI dependency
invocation:

1. `db = SessionLocal()` at line 15.
2. The session is yielded to the request at line 17.
3. `db.close()` is always called in `finally` at line 19.

The ingestion service receives that request-scoped session at
`app/services/incident_ingestion.py:39-47`.

The ingestion transaction is a single batch transaction:

1. Collect all requested GitHub payloads.
2. Create or retrieve the source.
3. Normalize and upsert each payload.
4. Flush intermediate writes.
5. Commit once after the complete payload loop.

The single batch commit occurs at `app/services/incident_ingestion.py:151`.

## commit() Usage

| File | Line | Operation | Notes |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 151 | `self.db.commit()` | Commits the entire ingestion batch after all payloads are processed. It is not wrapped in `try/except`, and no rollback follows a commit failure. |

No other `commit()` usage exists under `app`, `alembic`, or `tests`.

## flush() Usage

| File | Line | Operation | Notes |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 174 | `self.db.flush()` | Flushes a newly created `IncidentSource` in `_get_or_create_source()`. This call is outside the current stage wrappers. |
| `app/services/incident_ingestion.py` | 194 | `self.db.flush()` | Flushes updates to an existing `RawDocument`. It is inside the `_upsert_raw_document` stage wrapper. |
| `app/services/incident_ingestion.py` | 205 | `self.db.flush()` | Flushes a newly added `RawDocument`. It is inside the `_upsert_raw_document` stage wrapper. |
| `app/services/incident_ingestion.py` | 254 | `self.db.flush()` | Flushes incident and symptom changes before embedding upsert. It is inside the `_upsert_incident` stage wrapper. |

Because `SessionLocal` sets `autoflush=False` at `app/db/session.py:11`, these
explicit flushes are the primary points where queued database changes are sent
to PostgreSQL before the final commit.

## rollback() Usage

There are no calls to `rollback()` anywhere under `app`, `alembic`, or `tests`.

Rollback is therefore **not explicitly called** after any ingestion failure.

`db.close()` at `app/db/session.py:19` releases the request-scoped session and
its transactional resources. SQLAlchemy closes any remaining transaction as
part of session cleanup. However, this only happens when control exits the
FastAPI dependency scope.

## Exception Paths

### GitHub Collection

| File | Line | Path | Transaction Impact |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 63-74 | `collect_issues` wrapper logs and re-raises any exception. | Occurs before source creation and before ingestion writes. |
| `app/ingestion/collectors/github_collector.py` | 42-53 | GitHub issue request, `raise_for_status()`, and JSON parsing may fail. | No database write has occurred yet. |
| `app/ingestion/collectors/github_collector.py` | 62-63 | Comment collection may fail when comments are enabled. | No database write has occurred yet. |
| `app/ingestion/collectors/github_collector.py` | 79-81 | Comment request, `raise_for_status()`, and JSON parsing may fail. | No database write has occurred yet. |

### Source Upsert

| File | Line | Path | Transaction Impact |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 77 | `_get_or_create_source()` is called without a stage wrapper. | Query or flush failures propagate without a stage-specific log, payload dump, or rollback. |
| `app/services/incident_ingestion.py` | 162-174 | Source lookup, insert, and flush may fail. | A database exception during line 174 can leave the session transaction inactive until rollback or close. |

### Normalization

| File | Line | Path | Transaction Impact |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 98-102 | `normalize` wrapper logs, dumps the payload, and re-raises. | Python-level errors do not necessarily invalidate the transaction, but the batch transaction remains open until request cleanup. |
| `app/ingestion/normalizers/github_normalizer.py` | 259 | `datetime.fromisoformat(...)` may reject malformed timestamps. | No new SQL is emitted by normalization itself. |

### Raw Document Upsert

| File | Line | Path | Transaction Impact |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 115-119 | `_upsert_raw_document` wrapper logs, dumps the payload, and re-raises. | No explicit rollback. |
| `app/services/incident_ingestion.py` | 183-205 | Hashing, lookup, mutation, insert, or flush may fail. | A database exception during lines 194 or 205 can leave the session transaction inactive until rollback or close. |

### Incident Upsert

| File | Line | Path | Transaction Impact |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 133-137 | `_upsert_incident` wrapper logs, dumps the payload, and re-raises. | No explicit rollback. |
| `app/services/incident_ingestion.py` | 213-255 | Lookup, ORM mutation, relationship loading, symptom replacement, flush, or nested embedding upsert may fail. | A database exception during line 254 can leave the session transaction inactive until rollback or close. |

### Embedding Upsert

| File | Line | Path | Transaction Impact |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 273-301 | `_upsert_embedding` wrapper logs, dumps the payload, and re-raises. | No explicit rollback. |
| `app/services/incident_ingestion.py` | 274 | Embedding model loading or inference may fail. | A Python/runtime failure does not necessarily invalidate the database transaction, but prior batch writes remain uncommitted. |
| `app/services/incident_ingestion.py` | 275-298 | Embedding lookup, mutation, or add may fail. | `db.add()` does not force SQL immediately. Some embedding insert failures may surface later during the final commit. |

### Final Commit

| File | Line | Path | Transaction Impact |
|---|---:|---|---|
| `app/services/incident_ingestion.py` | 151 | Final batch `commit()` may fail. | A failed commit can leave the session inactive and requiring rollback. This failure is not wrapped, does not dump the current payload, and does not call rollback. |

## Can an Exception Leave the Session Invalid?

Yes.

A SQLAlchemy session can enter a failed or inactive transaction state when
PostgreSQL rejects SQL emitted by `flush()` or `commit()`. In this pipeline,
that can occur at:

- `app/services/incident_ingestion.py:174`
- `app/services/incident_ingestion.py:194`
- `app/services/incident_ingestion.py:205`
- `app/services/incident_ingestion.py:254`
- `app/services/incident_ingestion.py:151`

After such a failure, further use of the same `Session` can raise errors such as
`PendingRollbackError` until `Session.rollback()` is called.

The current code logs and re-raises some exceptions but never explicitly calls
`rollback()`.

## Is Rollback Always Called After Failures?

No.

There is no explicit `rollback()` call in the repository.

For normal FastAPI requests, `get_db()` creates a fresh request-scoped session
and closes it in `finally` at `app/db/session.py:14-19`. That cleanup should
prevent reuse of the same failed session by a later request.

However, there are still important risks:

1. Any code that catches an ingestion exception and continues using the same
   session before dependency cleanup will encounter an invalid transaction
   state after a database error.
2. Any non-FastAPI caller that constructs and reuses a session can carry the
   invalid state into subsequent ingestion attempts.
3. Final commit failures at `app/services/incident_ingestion.py:151` are not
   stage-logged and do not identify the payload that caused a deferred database
   error.
4. `_get_or_create_source()` failures at
   `app/services/incident_ingestion.py:77` and
   `app/services/incident_ingestion.py:174` bypass the payload-stage wrappers.

## Interpretation of the Observed Behavior

The current pipeline performs one transaction for the complete batch. Larger
ingestions execute more payload transformations and more flushes before one
final commit, so they have a higher chance of encountering one rejected row or
a deferred commit-time failure.

The absence of explicit rollback is a real transaction-handling gap.

The reported need to restart the application after a failed HTTP request is not
fully explained by request-scoped SQLAlchemy session reuse alone:
`get_db()` creates a new session for each request and closes it in `finally`.
Additional runtime evidence is needed to determine whether the persistent
failure comes from:

- an unobserved commit-time exception,
- a connection-pool or driver-level failure,
- a non-request session reuse path,
- a repeated deterministic bad payload,
- or another runtime integration issue outside the request-scoped session.

No code changes are recommended or applied in this analysis.
