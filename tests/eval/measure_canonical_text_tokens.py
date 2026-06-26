"""Phase 3A: measure actual MiniLM tokenizer lengths of canonical_text.

Compares the OLD canonical_text (as currently stored in the database) against
the NEW canonical_text (produced by re-normalizing each incident's raw GitHub
payload with the updated GitHubNormalizer), using the real
sentence-transformers tokenizer.

Usage:
    python -m tests.eval.measure_canonical_text_tokens
    python -m tests.eval.measure_canonical_text_tokens --output tests/eval/results/token_lengths_v3a.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.db.models import Incident, RawDocument
from app.db.session import SessionLocal
from app.ingestion.normalizers.github_normalizer import GitHubNormalizer
from app.services.embedding_service import EmbeddingService

DEFAULT_OUTPUT = Path(__file__).parent / "results" / "token_lengths_v3a.json"


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def summarize(token_counts: list[int]) -> dict:
    return {
        "count": len(token_counts),
        "p50": percentile(token_counts, 0.50),
        "p95": percentile(token_counts, 0.95),
        "max": max(token_counts) if token_counts else 0,
        "mean": sum(token_counts) / len(token_counts) if token_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure canonical_text tokenizer lengths")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of incidents (default: all)",
    )
    args = parser.parse_args()

    embedding_service = EmbeddingService()
    tokenizer = embedding_service.model.tokenizer
    normalizer = GitHubNormalizer()

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

        old_lengths: list[int] = []
        new_lengths: list[int] = []
        examples: list[dict] = []

        for incident in incidents:
            if incident.raw_document is None:
                continue

            old_text = incident.canonical_text
            old_tokens = len(tokenizer.encode(old_text, add_special_tokens=True))
            old_lengths.append(old_tokens)

            normalized = normalizer.normalize(incident.raw_document.payload)
            new_text = normalized.canonical_text
            new_tokens = len(tokenizer.encode(new_text, add_special_tokens=True))
            new_lengths.append(new_tokens)

            examples.append(
                {
                    "incident_id": str(incident.id),
                    "title": incident.title[:80],
                    "old_tokens": old_tokens,
                    "new_tokens": new_tokens,
                    "old_canonical_text": old_text,
                    "new_canonical_text": new_text,
                }
            )
    finally:
        db.close()

    report = {
        "embedding_model_name": embedding_service.model_name,
        "old": summarize(old_lengths),
        "new": summarize(new_lengths),
        "examples": examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"embedding_model_name: {report['embedding_model_name']}")
    print("\nOLD canonical_text token lengths:")
    for key, value in report["old"].items():
        print(f"  {key}: {value}")
    print("\nNEW canonical_text token lengths:")
    for key, value in report["new"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
