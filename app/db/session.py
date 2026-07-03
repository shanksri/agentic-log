from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# No connection/pool timeout was previously configured, so a DB that accepts
# TCP connections but never responds (network partition, overloaded server)
# could hang a request indefinitely (Phase 23 hardening finding).
# ``connect_timeout`` is psycopg's libpq keyword for the initial TCP+auth
# handshake; ``pool_timeout`` bounds how long a request waits for a free
# connection out of the pool once it's exhausted.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_timeout=10,
    connect_args={"connect_timeout": 10},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
