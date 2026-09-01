"""Live verification of the Phase 22D low-confidence decline gate
(generation_harness.py) against the real corpus: re-runs generation eval on
phase22d_generation_v1.json (which includes negative-control query
v2-neg-01) and reports whether the negative control now declines instead of
fabricating, plus BERTScore/Faithfulness before/after for that query.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.db.session import SessionLocal
from app.evaluation.generation_harness import evaluate_generation
from app.evaluation.gold_loader import load_gold_dataset
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.evaluation.generation_harness import LLMServiceAnswerGenerator

GOLD_PATH = Path("tests/eval/gold/phase22d_generation_v1.json")


def main() -> None:
    dataset = load_gold_dataset(GOLD_PATH)
    db = SessionLocal()

    from app.evaluation.generation_metrics import SentenceTransformerTokenEmbedder
    from app.services.search import IncidentSearchService

    embedding_service = EmbeddingService()
    search_service = IncidentSearchService(db, embedding_service=embedding_service)
    llm_service = LLMService()
    answer_generator = LLMServiceAnswerGenerator(llm_service)

    report = evaluate_generation(
        dataset, search_service, answer_generator,
        token_embedder=SentenceTransformerTokenEmbedder(embedding_service),
        grounding_llm=llm_service,
    )

    print(f"answered={report.num_answered} skipped={report.num_skipped} failed={report.num_failed}")
    for result in report.results:
        marker = " <-- NEGATIVE CONTROL" if result.query_id == "v2-neg-01" else ""
        declined = any("answer declined" in n for n in result.notes)
        print(f"\n[{result.query_id}]{marker} declined={declined}")
        print(f"  query: {result.query[:80]}")
        print(f"  answer: {(result.generated_answer or '')[:200]}")
        if result.grounding:
            print(f"  faithfulness={result.grounding.faithfulness}")
        if result.generation:
            print(f"  bert_score_f1={result.generation.bert_score_f1}")
        print(f"  notes: {result.notes}")

    out = Path(".benchmarks/decline_gate_verification.json")
    out.write_text(
        json.dumps(
            [
                {
                    "query_id": r.query_id,
                    "declined": any("answer declined" in n for n in r.notes),
                    "answer": r.generated_answer,
                    "faithfulness": r.grounding.faithfulness if r.grounding else None,
                    "bert_score_f1": r.generation.bert_score_f1 if r.generation else None,
                    "notes": r.notes,
                }
                for r in report.results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
