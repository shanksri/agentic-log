# Database Recovery

This document explains how to restore the local PostgreSQL + pgvector
environment after the Windows reinstall.

## Current Finding

The application `.env` currently contains:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/incidents
```

That URL expects the hostname `db` to resolve to PostgreSQL.

After the reinstall, there is no PostgreSQL container and no Docker network, so
`db` cannot resolve. Any local command that uses `DATABASE_URL` will fail until
the database environment is recreated or the URL is changed.

Per the requested constraint, this document does **not** modify `.env` or
application code.

## Repository Analysis

### Files Searched

Searched for:

- `docker-compose.yml`
- `compose.yaml`
- `compose.yml`
- `Dockerfile`
- setup scripts
- requirements files
- README instructions
- Alembic configuration
- PostgreSQL/pgvector references

### Relevant Files Found

| File | Finding |
|---|---|
| `README.md` | Mentions `docker compose up --build`, so the project was originally documented as Docker Compose based. |
| `.env.example` | Uses `DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/incidents`, which is a Docker Compose service-hostname style URL. |
| `.env` | Also uses `db` as the PostgreSQL hostname. |
| `alembic.ini` | Contains fallback `postgresql+psycopg://postgres:postgres@localhost:5432/incidents`, but Alembic actually uses app settings from `.env`. |
| `app/core/config.py` | Defaults `database_url` to `localhost`, but `.env` overrides it. |
| `app/db/session.py` | Creates the SQLAlchemy engine from `settings.database_url`. |
| `alembic/env.py` | Sets Alembic's URL from `settings.database_url`, so migrations also use `.env`. |
| `alembic/versions/20260530_0001_initial_phase1.py` | Creates `vector` and `pg_trgm` extensions and the `embeddings` table with `VECTOR(384)`. |
| `app/db/models/embedding.py` | Uses `pgvector.sqlalchemy.Vector`. |

### Docker Files Found

No `docker-compose.yml`, `compose.yaml`, `compose.yml`, or `Dockerfile` is
present in the current repository checkout.

This means the checked-in repository currently references Docker Compose in
documentation and environment naming, but the Compose files themselves are not
available.

## Original Expected Database Architecture

The original intended architecture was almost certainly:

```text
FastAPI application
  |
  | DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/incidents
  v
PostgreSQL service named "db"
  |
  v
pgvector extension enabled
```

The hostname `db` is correct **inside a Docker Compose network** when there is a
Compose service named `db`.

The hostname `db` is **not automatically correct** when running the FastAPI app
directly from the Windows host virtual environment. The Windows host cannot
resolve Docker service names unless you add a hosts entry or change
`DATABASE_URL` to `localhost`.

## Does The Project Expect Docker Compose?

Yes, based on repository clues:

1. `README.md` says:

   ```powershell
   docker compose up --build
   ```

2. `.env.example` uses:

   ```env
   DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/incidents
   ```

The `db` hostname strongly implies a Docker Compose service named `db`.

However, the current checkout does not contain a Compose file, so Docker Compose
cannot currently recreate the environment directly.

## Is The Hostname `db` Correct?

It depends on where the application runs.

| Runtime | Is `db` correct? | Explanation |
|---|---:|---|
| FastAPI running inside Docker Compose | Yes | Compose service names resolve through the Compose network DNS. |
| FastAPI running directly from Windows `.venv` | No | Windows does not know Docker service aliases by default. |
| FastAPI running from Windows `.venv` with `db` mapped in hosts file | Yes | `db` can resolve to `127.0.0.1`, where Docker publishes PostgreSQL. |
| FastAPI running from Windows `.venv` with `.env` changed to `localhost` | Yes | Simplest local-host setup, but this changes `.env`. |

Because the current `.env` must not be modified, the recovery path below keeps
`db` as the hostname and makes Windows resolve it.

## Required Database Service

Use PostgreSQL 16 with pgvector:

```text
Image: pgvector/pgvector:pg16
Database: incidents
User: postgres
Password: postgres
Container hostname/alias: db
Host port: 5432
Vector dimension: 384
```

## Required Environment Variables

Current application-required database variables:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/incidents
EMBEDDING_DIMENSIONS=384
```

Other application variables may exist, but database recovery needs the two
above to be consistent with the migration.

The current migration creates:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

and:

```text
embeddings.embedding VECTOR(384)
```

Therefore `EMBEDDING_DIMENSIONS` should remain `384`.

## Recovery Option A: Recreate A Compose-Like Database Environment Without A Compose File

This is the best recovery option if you must keep:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/incidents
```

It creates:

- a Docker network
- a PostgreSQL + pgvector container
- a Docker network alias named `db`
- a Windows hosts entry so the local `.venv` can resolve `db`

### 1. Create Docker Network

```powershell
docker network create incident-intelligence-net
```

Verify:

```powershell
docker network ls
docker network inspect incident-intelligence-net
```

### 2. Create Persistent Volume

```powershell
docker volume create incident_pgdata
```

Verify:

```powershell
docker volume ls
```

### 3. Start PostgreSQL + pgvector

```powershell
docker run `
  --name incident-postgres `
  --hostname db `
  --network incident-intelligence-net `
  --network-alias db `
  -e POSTGRES_DB=incidents `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -p 5432:5432 `
  -v incident_pgdata:/var/lib/postgresql/data `
  -d pgvector/pgvector:pg16
```

Verify:

```powershell
docker ps --filter "name=incident-postgres"
docker logs incident-postgres --tail 50
docker exec incident-postgres pg_isready -U postgres -d incidents
```

### 4. Make `db` Resolve From Windows Host

Because the FastAPI app is currently run from the Windows virtual environment,
Windows must resolve `db`.

Open PowerShell as Administrator and run:

```powershell
Add-Content `
  -Path "C:\Windows\System32\drivers\etc\hosts" `
  -Value "`n127.0.0.1 db"
```

Verify from a normal PowerShell window:

```powershell
Resolve-DnsName db
Test-NetConnection db -Port 5432
```

Expected:

```text
Name: db
IPAddress: 127.0.0.1
TcpTestSucceeded: True
```

### 5. Verify Python Can Reach The Database URL

From the project root:

```powershell
cd "D:\projects\agentic log"
.\.venv\Scripts\Activate.ps1

python -c "from app.core.config import settings; print(settings.database_url)"
```

Expected:

```text
postgresql+psycopg://postgres:postgres@db:5432/incidents
```

Test a connection:

```powershell
python -c "from sqlalchemy import create_engine, text; from app.core.config import settings; e=create_engine(settings.database_url); print(e.connect().execute(text('select 1')).scalar())"
```

Expected:

```text
1
```

## Alembic Migration Commands

Run migrations after PostgreSQL is available:

```powershell
cd "D:\projects\agentic log"
.\.venv\Scripts\Activate.ps1
python -m alembic upgrade head
```

Verify current Alembic revision:

```powershell
python -m alembic current
```

Expected revision:

```text
20260530_0001
```

Verify tables:

```powershell
docker exec incident-postgres psql -U postgres -d incidents -c "\dt"
```

Expected tables:

```text
incident_sources
raw_documents
incidents
symptoms
embeddings
```

Verify extensions:

```powershell
docker exec incident-postgres psql -U postgres -d incidents -c "SELECT extname FROM pg_extension ORDER BY extname;"
```

Expected extensions include:

```text
pg_trgm
plpgsql
vector
```

Verify vector column:

```powershell
docker exec incident-postgres psql -U postgres -d incidents -c "\d embeddings"
```

Expected:

```text
embedding vector(384)
```

## Application Verification

Start the app from the virtual environment:

```powershell
cd "D:\projects\agentic log"
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8000/docs -UseBasicParsing
```

Expected health response:

```json
{
  "status": "ok"
}
```

## Database Verification Queries

```powershell
docker exec incident-postgres psql -U postgres -d incidents -c "SELECT COUNT(*) FROM incidents;"
docker exec incident-postgres psql -U postgres -d incidents -c "SELECT COUNT(*) FROM embeddings;"
docker exec incident-postgres psql -U postgres -d incidents -c "SELECT model_name, COUNT(*) FROM embeddings GROUP BY model_name;"
```

If this is a fresh database, counts may be `0` until ingestion runs.

## Ingestion Verification

With the app running:

```powershell
$body = @{
  owner = "pallets"
  repo = "flask"
  state = "closed"
  limit = 1
  include_comments = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/ingestion/github" `
  -ContentType "application/json" `
  -Body $body
```

Then verify rows:

```powershell
docker exec incident-postgres psql -U postgres -d incidents -c "SELECT COUNT(*) AS incidents FROM incidents;"
docker exec incident-postgres psql -U postgres -d incidents -c "SELECT COUNT(*) AS embeddings FROM embeddings;"
```

## Recovery Option B: Simplest Local Development Setup

The simplest local-development setup is:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/incidents
```

Then run the same PostgreSQL container with:

```powershell
docker run `
  --name incident-postgres `
  -e POSTGRES_DB=incidents `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -p 5432:5432 `
  -v incident_pgdata:/var/lib/postgresql/data `
  -d pgvector/pgvector:pg16
```

This avoids the Windows hosts-file entry.

However, this option changes `.env`, so it is not the exact current
configuration and is only a recommendation for future local development.

## Recovery Option C: Recreate The Missing Compose File Later

The original setup likely had a Compose file equivalent to:

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: incidents
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

If an `api` service is also added later, then
`DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/incidents` is
correct for that API container.

Do not add this file as part of this recovery document unless you intentionally
want to restore Docker Compose as a checked-in project artifact.

## Cleanup Commands

Only use these if you want to remove the recovered database completely.

```powershell
docker rm -f incident-postgres
docker volume rm incident_pgdata
docker network rm incident-intelligence-net
```

If you added a Windows hosts entry for `db`, remove this line from:

```text
C:\Windows\System32\drivers\etc\hosts
```

Line to remove:

```text
127.0.0.1 db
```

## Final Recommendation

To recover the project exactly as the current `.env` expects, use **Recovery
Option A**:

1. Create Docker network `incident-intelligence-net`.
2. Start `pgvector/pgvector:pg16` with hostname and network alias `db`.
3. Add `127.0.0.1 db` to the Windows hosts file.
4. Run `python -m alembic upgrade head`.
5. Start FastAPI from the virtual environment.

For future simplicity, consider changing local `.env` to use `localhost` or
restoring a checked-in `docker-compose.yml`. That is a configuration
recommendation only; no application code or `.env` changes were made here.

