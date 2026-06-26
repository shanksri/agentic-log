from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Incident(Base):
    __tablename__ = "incidents"
    __table_args__ = (
        UniqueConstraint("deduplication_key", name="uq_incidents_deduplication_key"),
        Index("ix_incidents_tags_gin", "tags", postgresql_using="gin"),
        Index(
            "ix_incidents_affected_components_gin", "affected_components", postgresql_using="gin"
        ),
        Index("ix_incidents_environment_gin", "environment", postgresql_using="gin"),
        Index(
            "ix_incidents_full_text",
            "title",
            "description",
            "canonical_text",
            postgresql_using="gin",
            postgresql_ops={
                "title": "gin_trgm_ops",
                "description": "gin_trgm_ops",
                "canonical_text": "gin_trgm_ops",
            },
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("raw_documents.id", ondelete="SET NULL")
    )
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    source_external_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_url: Mapped[str | None] = mapped_column(Text)
    owner: Mapped[str | None] = mapped_column(String(255), index=True)
    repo: Mapped[str | None] = mapped_column(String(255), index=True)
    source: Mapped[str | None] = mapped_column(String(50), index=True)
    state: Mapped[str | None] = mapped_column(String(50), index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[str] = mapped_column(String(30), nullable=False, default="unknown", index=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="unknown", index=True)
    incident_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="bug", index=True
    )
    environment: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    affected_components: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    root_cause_summary: Mapped[str | None] = mapped_column(Text)
    root_cause_category: Mapped[str | None] = mapped_column(String(100))
    resolution_summary: Mapped[str | None] = mapped_column(Text)
    remediation_steps: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    prevention_steps: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    canonical_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=0.5)
    is_gold_labeled: Mapped[bool] = mapped_column(nullable=False, default=False)
    deduplication_key: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at_source: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at_source: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    raw_document = relationship("RawDocument", back_populates="incident")
    symptoms = relationship("Symptom", back_populates="incident", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="incident", cascade="all, delete-orphan")
