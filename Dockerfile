# Phase 23: Production Validation & Hardening — this file did not exist
# before this phase despite the README documenting `docker compose up
# --build` as the local-run instructions (a deployment-readiness gap; see
# the Phase 23 report's Deployment Readiness section).

# ── Builder stage: resolve dependencies into a venv, nothing else ───────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ── Runtime stage: no build toolchain, no source cache, non-root user ───────
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Apply migrations, then serve. `alembic upgrade head` is idempotent — safe
# to run on every container start (see docs/architecture's deployment note).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
