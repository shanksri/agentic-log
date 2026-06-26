"""multi-source foundation: source_metadata, config, last_ingested_at

Revision ID: 20260619_0003
Revises: 20260611_0002
Create Date: 2026-06-19 00:03:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "20260619_0003"
down_revision: str | None = "20260611_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # incidents: generic source-specific metadata bag
    op.add_column(
        "incidents",
        sa.Column(
            "source_metadata",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    # incident_sources: connection/query config per source instance
    op.add_column(
        "incident_sources",
        sa.Column(
            "config",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    # incident_sources: watermark for incremental ingestion
    op.add_column(
        "incident_sources",
        sa.Column("last_ingested_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("incident_sources", "last_ingested_at")
    op.drop_column("incident_sources", "config")
    op.drop_column("incidents", "source_metadata")
