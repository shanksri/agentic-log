"""Phase 3A backfill: re-normalize canonical_text from stored raw payloads and
re-embed.

Re-normalizing recomputes canonical_text using the updated GitHubNormalizer
(Phase 3A template). If the resulting text_hash differs from the stored
embedding's text_hash, the embedding is regenerated. Safe to re-run: rows
whose canonical_text/text_hash are already up to date are skipped.

Usage:
    python -m scripts.backfill_canonical_text
    python -m scripts.backfill_canonical_text --limit 50
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.db.models import Incident, RawDocument
from app.db.session import SessionLocal
from app.ingestion.normalizers.github_normalizer import GitHubNormalizer
from app.services.deduplication import DeduplicationService
from app.services.embedding_service import EmbeddingService

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill canonical_text + embeddings")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    normalizer = GitHubNormalizer()
    deduplication = DeduplicationService()
    embedding_service = EmbeddingService()

    db = SessionLocal()
    try:
        statement = (
            select(Incident)
            .options(joinedload(Incident.raw_document).joinedload(RawDocument.source))
            .where(Incident.source_type == "github")
        )
        if args.limit:
            statement = statement.limit(args.limit)
        incidents = db.execute(statement).unique().scalars().all()

        updated = 0
        skipped = 0
        for incident in incidents:
            if incident.raw_document is None:
                skipped += 1
                continue

            normalized = normalizer.normalize(incident.raw_document.payload)
            new_text_hash = deduplication.text_hash(normalized.canonical_text)

            already_fresh = any(
                embedding.model_name == embedding_service.model_name
                and embedding.text_hash == new_text_hash
                for embedding in incident.embeddings
            )
            if already_fresh:
                skipped += 1
                continue

            incident.canonical_text = normalized.canonical_text
            vector = embedding_service.embed_text(normalized.canonical_text)

            embedding = next(
                (e for e in incident.embeddings if e.model_name == embedding_service.model_name),
                None,
            )
            if embedding:
                embedding.embedding = vector
                embedding.text_hash = new_text_hash
            else:
                from app.db.models import Embedding

                db.add(
                    Embedding(
                        incident_id=incident.id,
                        model_name=embedding_service.model_name,
                        embedding=vector,
                        text_hash=new_text_hash,
                    )
                )

            updated += 1
            if updated % 50 == 0:
                db.commit()
                logger.info("Committed %d updates so far (%d skipped)", updated, skipped)

        db.commit()
        logger.info("Done. updated=%d skipped=%d total=%d", updated, skipped, len(incidents))
    finally:
        db.close()


if __name__ == "__main__":
    main()
