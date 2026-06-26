"""initial phase 1 schema

Revision ID: 20260530_0001
Revises:
Create Date: 2026-05-30 00:01:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260530_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "incident_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_incident_sources_source_type", "incident_sources", ["source_type"])

    op.create_table(
        "raw_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("payload_hash", sa.Text(), nullable=False),
        sa.Column(
            "ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["source_id"], ["incident_sources.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("payload_hash", name="uq_raw_documents_payload_hash"),
        sa.UniqueConstraint("source_id", "external_id", name="uq_raw_documents_source_external"),
    )
    op.create_index(
        "ix_raw_documents_payload_gin", "raw_documents", ["payload"], postgresql_using="gin"
    )

    op.create_table(
        "incidents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("raw_document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_external_id", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("incident_type", sa.String(length=50), nullable=False),
        sa.Column("environment", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("affected_components", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("root_cause_summary", sa.Text(), nullable=True),
        sa.Column("root_cause_category", sa.String(length=100), nullable=True),
        sa.Column("resolution_summary", sa.Text(), nullable=True),
        sa.Column("remediation_steps", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("prevention_steps", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("canonical_text", sa.Text(), nullable=False),
        sa.Column("confidence_score", sa.Numeric(4, 3), nullable=False),
        sa.Column("is_gold_labeled", sa.Boolean(), nullable=False),
        sa.Column("deduplication_key", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at_source", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at_source", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["raw_document_id"], ["raw_documents.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("deduplication_key", name="uq_incidents_deduplication_key"),
    )
    op.create_index("ix_incidents_source_type", "incidents", ["source_type"])
    op.create_index("ix_incidents_source_external_id", "incidents", ["source_external_id"])
    op.create_index("ix_incidents_severity", "incidents", ["severity"])
    op.create_index("ix_incidents_status", "incidents", ["status"])
    op.create_index("ix_incidents_incident_type", "incidents", ["incident_type"])
    op.create_index("ix_incidents_tags_gin", "incidents", ["tags"], postgresql_using="gin")
    op.create_index(
        "ix_incidents_affected_components_gin",
        "incidents",
        ["affected_components"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_incidents_environment_gin", "incidents", ["environment"], postgresql_using="gin"
    )
    op.create_index(
        "ix_incidents_full_text",
        "incidents",
        ["title", "description", "canonical_text"],
        postgresql_using="gin",
        postgresql_ops={
            "title": "gin_trgm_ops",
            "description": "gin_trgm_ops",
            "canonical_text": "gin_trgm_ops",
        },
    )

    op.create_table(
        "symptoms",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("incident_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("observed_value", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_symptoms_incident_id", "symptoms", ["incident_id"])

    op.create_table(
        "embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("incident_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("incident_id", "model_name", name="uq_embeddings_incident_model"),
    )
    op.create_index("ix_embeddings_incident_id", "embeddings", ["incident_id"])
    op.create_index("ix_embeddings_model_name", "embeddings", ["model_name"])
    op.create_index(
        "ix_embeddings_vector_hnsw",
        "embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_embeddings_vector_hnsw", table_name="embeddings")
    op.drop_index("ix_embeddings_model_name", table_name="embeddings")
    op.drop_index("ix_embeddings_incident_id", table_name="embeddings")
    op.drop_table("embeddings")
    op.drop_index("ix_symptoms_incident_id", table_name="symptoms")
    op.drop_table("symptoms")
    op.drop_index("ix_incidents_full_text", table_name="incidents")
    op.drop_index("ix_incidents_environment_gin", table_name="incidents")
    op.drop_index("ix_incidents_affected_components_gin", table_name="incidents")
    op.drop_index("ix_incidents_tags_gin", table_name="incidents")
    op.drop_index("ix_incidents_incident_type", table_name="incidents")
    op.drop_index("ix_incidents_status", table_name="incidents")
    op.drop_index("ix_incidents_severity", table_name="incidents")
    op.drop_index("ix_incidents_source_external_id", table_name="incidents")
    op.drop_index("ix_incidents_source_type", table_name="incidents")
    op.drop_table("incidents")
    op.drop_index("ix_raw_documents_payload_gin", table_name="raw_documents")
    op.drop_table("raw_documents")
    op.drop_index("ix_incident_sources_source_type", table_name="incident_sources")
    op.drop_table("incident_sources")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS vector")
