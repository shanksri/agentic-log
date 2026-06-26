"""add incident retrieval metadata

Revision ID: 20260611_0002
Revises: 20260530_0001
Create Date: 2026-06-11 00:02:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260611_0002"
down_revision: str | None = "20260530_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("incidents", sa.Column("owner", sa.String(length=255), nullable=True))
    op.add_column("incidents", sa.Column("repo", sa.String(length=255), nullable=True))
    op.add_column("incidents", sa.Column("source", sa.String(length=50), nullable=True))
    op.add_column("incidents", sa.Column("state", sa.String(length=50), nullable=True))

    op.execute(
        """
        UPDATE incidents
        SET
            owner = environment ->> 'repository_owner',
            repo = environment ->> 'repository_name',
            source = COALESCE(environment ->> 'source', source_type),
            state = CASE
                WHEN status = 'resolved' THEN 'closed'
                WHEN status = 'open' THEN 'open'
                ELSE status
            END
        """
    )

    op.create_index("ix_incidents_owner", "incidents", ["owner"])
    op.create_index("ix_incidents_repo", "incidents", ["repo"])
    op.create_index("ix_incidents_source", "incidents", ["source"])
    op.create_index("ix_incidents_state", "incidents", ["state"])


def downgrade() -> None:
    op.drop_index("ix_incidents_state", table_name="incidents")
    op.drop_index("ix_incidents_source", table_name="incidents")
    op.drop_index("ix_incidents_repo", table_name="incidents")
    op.drop_index("ix_incidents_owner", table_name="incidents")
    op.drop_column("incidents", "state")
    op.drop_column("incidents", "source")
    op.drop_column("incidents", "repo")
    op.drop_column("incidents", "owner")

