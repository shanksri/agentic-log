"""Phase 5 / Phase 7: hypothesis generation evaluation.

Phase 7 adds parallel evaluation of two keyword strategies per hypothesis:
  A. LLM-generated validation_keywords (current / baseline)
  C. Evidence-oriented keywords: literal terms from the title of the retrieved
     incident most similar to the hypothesis root_cause (C_hyp strategy).

Both strategies are evaluated on the same LLM output in a single pass to
eliminate non-determinism as a confound.  The report carries separate
recall and confidence-correlation metrics for A and C, plus per-hypothesis
latency breakdown for the C derivation step.

For each case in hypothesis_gold.json:
  A. Retrieval correctness  - does dense search() return the expected
     incident in its top 5 (the same candidates that would seed
     `_build_incident_context`)?
  B. Root-cause Recall@1/@3 - does a generated hypothesis (top-1 / top-3)
     semantically match an acceptable root cause? Matching is done via
     cosine similarity between MiniLM embeddings of the hypothesis's
     `root_cause` and each acceptable root-cause string, thresholded by
     `root_cause_match_threshold` from hypothesis_gold.json.
  C. Root-cause MRR        - 1/rank of the first matching hypothesis.
  D. Validation-keyword quality - strategy A and strategy C evaluated in
     parallel for every hypothesis.
  E. Confidence correlation - across all generated hypotheses (all cases),
     compare composite confidence for correct vs. incorrect hypotheses,
     using strategy-C keyword recall in the composite score.

Each non-negative case is also classified into exactly one failure stage
(or "pass") using strategy C keyword recall as the ground truth:
  - retrieval_failure    : expected incident not in top-5 dense results
  - hypothesis_failure   : retrieval ok, but no top-3 hypothesis matches
                            an acceptable root cause
  - validation_keyword_failure : a correct hypothesis exists, but its
                            strategy-C keywords don't retrieve the expected
                            incident in top-5
  - pass                 : retrieval ok, root cause matched in top-3, and
                            strategy-C keywords retrieve the expected
                            incident

Usage:
    python -m tests.eval.run_hypothesis_eval
    python -m tests.eval.run_hypothesis_eval --output tests/eval/results/hypothesis_v7.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db.session import SessionLocal
from app.services.advanced_investigation_agent import AdvancedInvestigationAgent
from app.services.confidence import classify_confidence, composite_hypothesis_confidence
from app.services.embedding_service import EmbeddingService
from app.services.keyword_extraction import derive_evidence_keywords
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchResult, IncidentSearchService

GOLD_PATH = Path(__file__).parent / "hypothesis_gold.json"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "hypothesis_v7.json"

RECALL_K = 3
KEYWORD_RECALL_K = 5
RETRIEVAL_RECALL_K = 5


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_context(results: list[IncidentSearchResult]) -> str:
    """Minimal incident context, mirroring
    AdvancedInvestigationAgent._build_incident_context's per-incident format
    without the Phase 4 confidence header (kept out so this harness doesn't
    depend on confidence-calibration framing)."""
    if not results:
        return "No similar incidents were retrieved."

    sections = []
    for index, result in enumerate(results, start=1):
        incident = result.incident
        symptoms = "; ".join(symptom.text for symptom in incident.symptoms) or "Unknown"
        sections.append(
            "\n".join(
                [
                    f"Incident {index}",
                    f"Similarity score: {result.similarity_score:.3f}",
                    f"Title: {incident.title}",
                    f"Symptoms: {symptoms}",
                    f"Severity: {incident.severity}",
                    f"Status: {incident.status}",
                    f"Resolution summary: {incident.resolution_summary or 'Unknown'}",
                ]
            )
        )
    return "\n\n".join(sections)


def evaluate_case(
    case: dict[str, Any],
    *,
    search_service: IncidentSearchService,
    agent: AdvancedInvestigationAgent,
    embedding_service: EmbeddingService,
    match_threshold: float,
) -> dict[str, Any]:
    problem = case["problem"]
    expected_incident_ids = set(case["expected_incident_ids"])
    acceptable_root_causes = (
        case["expected_root_causes"] + case.get("acceptable_alternative_root_causes", [])
    )
    is_negative_case = not expected_incident_ids

    # --- A. Retrieval correctness (dense-only, same as initial retrieval seed) ---
    results = search_service.search(problem, limit=10)
    retrieved_ids = [str(result.incident.id) for result in results]
    retrieval_ok = (
        bool(expected_incident_ids & set(retrieved_ids[:RETRIEVAL_RECALL_K]))
        if expected_incident_ids
        else None
    )
    initial_top1_score, initial_confidence_level = IncidentSearchService.confidence_for(results)

    context = build_context(results[:5])

    # --- Generate hypotheses (real LLM call, prompts unmodified) ---
    raw_hypotheses = agent.llm_service.generate_hypotheses(problem=problem, context=context)
    hypotheses_a = [agent._normalize_hypothesis(item) for item in raw_hypotheses[:5]]

    # --- Strategy C: derive evidence-oriented keywords (timed) ---
    t0 = time.perf_counter()
    hypotheses_c = derive_evidence_keywords(hypotheses_a, results[:10], embedding_service)
    evidence_keyword_latency_s = time.perf_counter() - t0

    # --- B/C. Root-cause Recall@1/@3 and MRR ---
    root_cause_vectors = [
        embedding_service.embed_text(text) for text in acceptable_root_causes
    ]

    hypothesis_rows: list[dict[str, Any]] = []
    first_match_rank: int | None = None

    for rank, (hyp_a, hyp_c) in enumerate(zip(hypotheses_a, hypotheses_c), start=1):
        root_cause = hyp_a["root_cause"]
        if root_cause_vectors:
            hyp_vector = embedding_service.embed_text(root_cause)
            similarities = [cosine_similarity(hyp_vector, vec) for vec in root_cause_vectors]
            best_similarity = max(similarities)
        else:
            best_similarity = 0.0
        is_match = best_similarity >= match_threshold
        if is_match and first_match_rank is None:
            first_match_rank = rank

        # --- Strategy A keyword recall ---
        kws_a = hyp_a["validation_keywords"]
        query_a = " ".join(str(k) for k in kws_a if k)
        kw_recall_a: bool | None
        if not query_a or not expected_incident_ids:
            kw_recall_a = None
        else:
            res_a = search_service.search(query_a, limit=KEYWORD_RECALL_K)
            kw_recall_a = bool(expected_incident_ids & {str(r.incident.id) for r in res_a})

        # --- Strategy C keyword recall ---
        kws_c = hyp_c["validation_keywords"]
        query_c = " ".join(str(k) for k in kws_c if k)
        kw_recall_c: bool | None
        if not query_c or not expected_incident_ids:
            kw_recall_c = None
        else:
            res_c = search_service.search(query_c, limit=KEYWORD_RECALL_K)
            kw_recall_c = bool(expected_incident_ids & {str(r.incident.id) for r in res_c})

        raw_confidence = hyp_a["confidence_score"]

        # Composite confidence uses strategy-C keyword recall (the better signal)
        composite_confidence = composite_hypothesis_confidence(
            raw_confidence=raw_confidence,
            retrieval_confidence_level=initial_confidence_level,
            validation_keyword_recall_ok=kw_recall_c,
        )

        hypothesis_rows.append(
            {
                "rank": rank,
                "root_cause": root_cause,
                "raw_confidence_score": raw_confidence,
                # Strategy A
                "validation_keywords_a": kws_a,
                "validation_keyword_recall_ok_a": kw_recall_a,
                # Strategy C
                "validation_keywords_c": kws_c,
                "validation_keyword_recall_ok_c": kw_recall_c,
                # Shared
                "composite_confidence_score": composite_confidence,
                "best_root_cause_similarity": best_similarity,
                "is_match": is_match,
            }
        )

    recall_at_1 = (
        1.0 if hypothesis_rows and hypothesis_rows[0]["is_match"] else (0.0 if acceptable_root_causes else None)
    )
    recall_at_3 = (
        1.0
        if any(row["is_match"] for row in hypothesis_rows[:RECALL_K])
        else (0.0 if acceptable_root_causes else None)
    )
    mrr = (1.0 / first_match_rank) if first_match_rank else (0.0 if acceptable_root_causes else None)

    # --- D. Case-level validation-keyword eval (first matching hyp, or hyp #1) ---
    keyword_eval_a = keyword_eval_c = None
    if hypotheses_a:
        target_index = (first_match_rank - 1) if first_match_rank else 0

        for strat, hyp_list, key in [("a", hypotheses_a, "keyword_eval_a"), ("c", hypotheses_c, "keyword_eval_c")]:
            kws = hyp_list[target_index]["validation_keywords"]
            query = " ".join(str(k) for k in kws if k)
            if query:
                res = search_service.search(query, limit=KEYWORD_RECALL_K)
                recall = (
                    1.0 if expected_incident_ids & {str(r.incident.id) for r in res} else 0.0
                    if expected_incident_ids else None
                )
            else:
                recall = 0.0 if expected_incident_ids else None
            ev = {"source_hypothesis_rank": target_index + 1, "query": query, "recall_at_5": recall}
            if strat == "a":
                keyword_eval_a = ev
            else:
                keyword_eval_c = ev

    # --- Failure attribution (using strategy C as the primary signal) ---
    kw_c_case_recall = keyword_eval_c["recall_at_5"] if keyword_eval_c else None
    if is_negative_case:
        stage = "negative_case"
    elif retrieval_ok is False:
        stage = "retrieval_failure"
    elif recall_at_3 == 0.0:
        stage = "hypothesis_failure"
    elif kw_c_case_recall == 0.0:
        stage = "validation_keyword_failure"
    else:
        stage = "pass"

    return {
        "id": case["id"],
        "problem": problem,
        "source_query_id": case.get("source_query_id"),
        "is_negative_case": is_negative_case,
        "retrieval_ok": retrieval_ok,
        "initial_top1_score": initial_top1_score,
        "initial_confidence_level": initial_confidence_level,
        "retrieved_top5_ids": retrieved_ids[:5],
        "hypotheses": hypothesis_rows,
        "root_cause_recall_at_1": recall_at_1,
        "root_cause_recall_at_3": recall_at_3,
        "root_cause_mrr": mrr,
        "validation_keyword_eval_a": keyword_eval_a,
        "validation_keyword_eval_c": keyword_eval_c,
        "evidence_keyword_latency_s": evidence_keyword_latency_s,
        "failure_stage": stage,
    }


def _point_biserial(points: list[tuple[float, float]]) -> dict[str, Any]:
    correct = [c for c, m in points if m == 1.0]
    incorrect = [c for c, m in points if m == 0.0]

    mean_correct = sum(correct) / len(correct) if correct else None
    mean_incorrect = sum(incorrect) / len(incorrect) if incorrect else None

    correlation = None
    if len(points) > 1:
        confidences = [c for c, _ in points]
        matches = [m for _, m in points]
        mean_c = sum(confidences) / len(confidences)
        mean_m = sum(matches) / len(matches)
        cov = sum((c - mean_c) * (m - mean_m) for c, m in points)
        var_c = sum((c - mean_c) ** 2 for c in confidences)
        var_m = sum((m - mean_m) ** 2 for m in matches)
        if var_c > 0 and var_m > 0:
            correlation = cov / math.sqrt(var_c * var_m)

    return {
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "mean_confidence_correct": mean_correct,
        "mean_confidence_incorrect": mean_incorrect,
        "point_biserial_correlation": correlation,
    }


def confidence_correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_points: list[tuple[float, float]] = []
    composite_points: list[tuple[float, float]] = []
    for row in rows:
        if row["is_negative_case"]:
            continue
        for hyp in row["hypotheses"]:
            match = 1.0 if hyp["is_match"] else 0.0
            raw_points.append((hyp["raw_confidence_score"], match))
            composite_points.append((hyp["composite_confidence_score"], match))

    return {
        "n_hypotheses": len(raw_points),
        "raw_confidence": _point_biserial(raw_points),
        "composite_confidence": _point_biserial(composite_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hypothesis generation evaluation")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    match_threshold = gold["root_cause_match_threshold"]

    db = SessionLocal()
    try:
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        search_service = IncidentSearchService(
            db, embedding_service=embedding_service, llm_service=llm_service
        )
        agent = AdvancedInvestigationAgent(
            db=db,
            search_service=search_service,
            llm_service=llm_service,
            embedding_service=embedding_service,
        )

        rows = [
            evaluate_case(
                case,
                search_service=search_service,
                agent=agent,
                embedding_service=embedding_service,
                match_threshold=match_threshold,
            )
            for case in gold["cases"]
        ]
    finally:
        db.close()

    positive_rows = [row for row in rows if not row["is_negative_case"]]

    def mean(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    # Keyword recall for each strategy
    kw_recalls_a = [
        row["validation_keyword_eval_a"]["recall_at_5"]
        for row in positive_rows
        if row["validation_keyword_eval_a"] is not None
    ]
    kw_recalls_c = [
        row["validation_keyword_eval_c"]["recall_at_5"]
        for row in positive_rows
        if row["validation_keyword_eval_c"] is not None
    ]

    # Per-hypothesis recall counts
    hyp_kw_a = [h["validation_keyword_recall_ok_a"] for row in positive_rows for h in row["hypotheses"]]
    hyp_kw_c = [h["validation_keyword_recall_ok_c"] for row in positive_rows for h in row["hypotheses"]]
    hyp_kw_a_rate = mean([float(v) for v in hyp_kw_a if v is not None])
    hyp_kw_c_rate = mean([float(v) for v in hyp_kw_c if v is not None])

    # Latency
    latencies = [row["evidence_keyword_latency_s"] for row in rows]
    mean_latency = sum(latencies) / len(latencies)
    total_latency = sum(latencies)

    failure_breakdown: dict[str, int] = {}
    for row in rows:
        failure_breakdown[row["failure_stage"]] = failure_breakdown.get(row["failure_stage"], 0) + 1

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "embedding_model_name": embedding_service.model_name,
        "llm_model_name": llm_service.model,
        "root_cause_match_threshold": match_threshold,
        "overall": {
            "retrieval_recall_at_5": mean([row["retrieval_ok"] for row in positive_rows]),
            "root_cause_recall_at_1": mean([row["root_cause_recall_at_1"] for row in positive_rows]),
            "root_cause_recall_at_3": mean([row["root_cause_recall_at_3"] for row in positive_rows]),
            "root_cause_mrr": mean([row["root_cause_mrr"] for row in positive_rows]),
            "keyword_strategy_a": {
                "case_level_recall_at_5": mean(kw_recalls_a),
                "per_hypothesis_recall": hyp_kw_a_rate,
            },
            "keyword_strategy_c": {
                "case_level_recall_at_5": mean(kw_recalls_c),
                "per_hypothesis_recall": hyp_kw_c_rate,
            },
        },
        "latency": {
            "evidence_keyword_derivation_mean_s": mean_latency,
            "evidence_keyword_derivation_total_s": total_latency,
            "per_case_s": {row["id"]: row["evidence_keyword_latency_s"] for row in rows},
        },
        "confidence_correlation": confidence_correlation(rows),
        "failure_breakdown": failure_breakdown,
        "cases": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print("\nOverall:")
    for key, value in report["overall"].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k2, v2 in value.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {value}")
    correlation = report["confidence_correlation"]
    print(f"\nConfidence correlation (n_hypotheses={correlation['n_hypotheses']}):")
    for label in ("raw_confidence", "composite_confidence"):
        sub = correlation[label]
        print(f"  {label}: r={sub['point_biserial_correlation']:.3f}  "
              f"mean_correct={sub['mean_confidence_correct']:.3f}  "
              f"mean_incorrect={sub['mean_confidence_incorrect']:.3f}")
    print("\nLatency (evidence keyword derivation):")
    print(f"  mean per case: {mean_latency:.3f}s")
    print(f"  total: {total_latency:.3f}s")
    print("\nFailure breakdown:")
    for key, value in failure_breakdown.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
