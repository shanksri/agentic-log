# Environment Setup

This guide rebuilds the local development environment for the Enterprise
Incident Intelligence Platform on a fresh Windows installation.

The project is a FastAPI application using PostgreSQL, pgvector, SQLAlchemy,
Alembic, SentenceTransformers, and the official OpenAI Python SDK.

## Required Software

Install these tools in this order:

1. Windows Terminal
2. Git
3. Python 3.12
4. Docker Desktop
5. Optional but useful: Visual Studio Build Tools
6. Optional but useful: PostgreSQL command-line tools

The application requires Python `>=3.12`. Use Python 3.12 specifically for the
least surprising compatibility with the ML dependencies.

## Install Base Tools On Windows

Open PowerShell as your normal user.

```powershell
winget install --id Microsoft.WindowsTerminal -e
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id Docker.DockerDesktop -e
```

Optional:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
winget install --id PostgreSQL.PostgreSQL.16 -e
```

Restart PowerShell after installation so `git`, `python`, `py`, and `docker`
are available on `PATH`.

Verify:

```powershell
git --version
py -3.12 --version
docker version
```

Start Docker Desktop before running database commands.

## Repository Setup

Clone or copy the project, then enter the repository:

```powershell
cd D:\projects
git clone <your-repository-url> "agentic log"
cd "D:\projects\agentic log"
```

If the repository already exists:

```powershell
cd "D:\projects\agentic log"
```

## Python Virtual Environment

Create a local virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install the application and development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

## Python Dependencies

Primary dependencies are declared in `pyproject.toml`:

```text
alembic>=1.13.2
fastapi>=0.115.0
httpx>=0.27.0
openai>=1.57.0
pgvector>=0.3.2
psycopg[binary]>=3.2.1
pydantic-settings>=2.4.0
sentence-transformers>=3.0.1
sqlalchemy>=2.0.32
uvicorn[standard]>=0.30.6
```

Development dependencies:

```text
pytest>=8.3.2,<9
pytest-asyncio>=0.24.0
ruff>=0.6.3
```

The installed command-line tools include:

```text
alembic
uvicorn
pytest
ruff
```

Verify installed tools:

```powershell
python -m alembic --version
python -m uvicorn --version
python -m pytest --version
python -m ruff --version
python -c "import fastapi, sqlalchemy, pgvector, psycopg, openai; print('imports ok')"
```

## Environment Variables

Create a `.env` file in the repository root.

For a local FastAPI process talking to PostgreSQL running in Docker on port
`5432`, use:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/incidents
GITHUB_TOKEN=
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
API_KEY=
```

Notes:

- `DATABASE_URL` is required.
- `GITHUB_TOKEN` is optional for public repositories, but recommended to avoid
  GitHub API rate limits.
- `OPENAI_API_KEY` is required for `POST /agent/investigate`.
- `OPENAI_MODEL` defaults to `gpt-4o-mini`.
- `EMBEDDING_MODEL_NAME` defaults to
  `sentence-transformers/all-MiniLM-L6-v2`.
- `EMBEDDING_DIMENSIONS` must match the database vector dimension. The current
  migration creates `VECTOR(384)`.
- `API_KEY` is required (Phase 23B) for every `/ingestion`, `/search`,
  `/agent`, `/incidents`, and `/evaluation` request — generate one with
  e.g. `openssl rand -hex 32`. See the root README's Authentication section.
- Do not commit real API keys.

## Docker Requirements

Docker Desktop is required if you want the recommended local PostgreSQL setup.

Docker Compose status:

- The current repository does **not** include `docker-compose.yml` or
  `docker-compose.yaml`.
- Docker Compose is therefore not currently required by checked-in project
  files.
- PostgreSQL with pgvector is still required.

Required local service:

```text
PostgreSQL 16 with pgvector extension
```

Recommended container image:

```text
pgvector/pgvector:pg16
```

Required running container when using Docker directly:

```text
incident-postgres
```

## Start PostgreSQL With pgvector

Start the database container:

```powershell
docker volume create incident_pgdata

docker run `
  --name incident-postgres `
  -e POSTGRES_DB=incidents `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -p 5432:5432 `
  -v incident_pgdata:/var/lib/postgresql/data `
  -d pgvector/pgvector:pg16
```

Verify the container:

```powershell
docker ps --filter "name=incident-postgres"
docker logs incident-postgres --tail 50
```

Wait until PostgreSQL is ready:

```powershell
docker exec incident-postgres pg_isready -U postgres -d incidents
```

If you need to recreate the database from scratch:

```powershell
docker rm -f incident-postgres
docker volume rm incident_pgdata
docker volume create incident_pgdata

docker run `
  --name incident-postgres `
  -e POSTGRES_DB=incidents `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -p 5432:5432 `
  -v incident_pgdata:/var/lib/postgresql/data `
  -d pgvector/pgvector:pg16
```

## Initialize PostgreSQL

Alembic migrations create the required PostgreSQL extensions:

```sql
vector
pg_trgm
```

Run migrations:

```powershell
python -m alembic upgrade head
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

Verify tables:

```powershell
docker exec incident-postgres psql -U postgres -d incidents -c "\dt"
```

Expected application tables include:

```text
incident_sources
raw_documents
incidents
symptoms
embeddings
```

## Start The FastAPI Application

Activate the virtual environment:

```powershell
cd "D:\projects\agentic log"
.\.venv\Scripts\Activate.ps1
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

PowerShell verification:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8000/docs -UseBasicParsing
```

## Run Tests

Run all tests:

```powershell
python -m pytest
```

Run only unit tests:

```powershell
python -m pytest tests\unit
```

Run Ruff:

```powershell
python -m ruff check .
```

Format check only:

```powershell
python -m ruff format . --check
```

## Ingest GitHub Incidents

Every business endpoint requires an `Authorization: Bearer <API_KEY>` header (Phase 23B — see the
root README's Authentication section). Set `API_KEY` in `.env` first, then reuse this header
variable across all the requests below:

```powershell
$headers = @{ Authorization = "Bearer $env:API_KEY" }
```

With the API running, ingest public GitHub issues:

```powershell
$body = @{
  owner = "pallets"
  repo = "flask"
  state = "closed"
  limit = 5
  include_comments = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/ingestion/github" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

This will:

1. Fetch GitHub issues.
2. Normalize incidents.
3. Store raw payloads and normalized incidents.
4. Generate SentenceTransformer embeddings.
5. Store vectors in PostgreSQL using pgvector.

The first embedding call may download the SentenceTransformer model.

## Semantic Search

```powershell
$body = @{
  query = "database timeout during peak traffic"
  limit = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/search/incidents" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

## Investigation Agent

Set `OPENAI_API_KEY` in `.env`, then call:

```powershell
$body = @{
  problem = "database timeout during peak traffic"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/agent/investigate" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

The agent:

1. Uses `IncidentSearchService` to retrieve the top 5 similar incidents.
2. Builds context from title, symptoms, severity, status, and resolution summary.
3. Sends the problem and context to OpenAI.
4. Returns root-cause analysis, confidence, evidence, and recommended actions.

## Fresh Machine Setup Verification

Run these commands after completing setup.

### 1. Verify Windows Tools

```powershell
git --version
py -3.12 --version
docker version
```

### 2. Verify Virtual Environment

```powershell
cd "D:\projects\agentic log"
.\.venv\Scripts\Activate.ps1
python --version
python -m pip --version
```

Expected Python version:

```text
Python 3.12.x
```

### 3. Verify Python Dependencies

```powershell
python -c "import fastapi, sqlalchemy, pgvector, psycopg, openai, sentence_transformers; print('dependencies ok')"
python -m alembic --version
python -m uvicorn --version
python -m pytest --version
python -m ruff --version
```

### 4. Verify PostgreSQL Container

```powershell
docker ps --filter "name=incident-postgres"
docker exec incident-postgres pg_isready -U postgres -d incidents
```

### 5. Verify Database URL

```powershell
python -c "from app.core.config import settings; print(settings.database_url)"
```

For local development with Docker DB, it should print:

```text
postgresql+psycopg://postgres:postgres@localhost:5432/incidents
```

### 6. Verify Migrations

```powershell
python -m alembic current
python -m alembic upgrade head
```

### 7. Verify PostgreSQL Extensions

```powershell
docker exec incident-postgres psql -U postgres -d incidents -c "SELECT extname FROM pg_extension ORDER BY extname;"
```

Confirm `vector` and `pg_trgm` are listed.

### 8. Verify API Starts

```powershell
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

### 9. Verify Tests And Lint

```powershell
python -m pytest tests\unit
python -m ruff check .
```

### 10. Verify Database Tables

```powershell
docker exec incident-postgres psql -U postgres -d incidents -c "\dt"
```

Confirm the incident and embedding tables exist.

## Troubleshooting

### PowerShell Cannot Activate `.venv`

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Docker Is Not Running

Start Docker Desktop manually from the Start Menu, then retry:

```powershell
docker version
```

### Port 5432 Is Already In Use

Find the process:

```powershell
netstat -ano | findstr :5432
```

Either stop the existing PostgreSQL service or map Docker to another host port
and update `DATABASE_URL`.

### Alembic Cannot Create Extensions

Use the `pgvector/pgvector:pg16` image and connect as the `postgres` superuser.
The migration needs permission to run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### OpenAI Calls Fail

Verify:

```powershell
python -c "from app.core.config import settings; print(bool(settings.openai_api_key), settings.openai_model)"
```

Make sure `.env` contains a valid `OPENAI_API_KEY`.

### GitHub Rate Limits

Set `GITHUB_TOKEN` in `.env`.

## Current Docker Compose Status

No Docker Compose file is currently present in the repository. If a future
`docker-compose.yml` is added, it should define at minimum:

```text
db: PostgreSQL 16 with pgvector
api: FastAPI application
```

Until then, run PostgreSQL with `docker run` as documented above and run the
FastAPI app directly from the Python virtual environment.

