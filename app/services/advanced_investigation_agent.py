from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.embedding_service import EmbeddingService

from sqlalchemy.orm import Session

from app.services.confidence import (
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    composite_hypothesis_confidence,
    classify_confidence,
)
from app.services.keyword_extraction import USE_EVIDENCE_KEYWORDS, derive_evidence_keywords
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchResult, IncidentSearchService

logger = logging.getLogger(__name__)

# Policy B: generate this many hypotheses in the baseline pass.
_POLICY_B_BASELINE_N = 2
# Policy B: generate this many additional hypotheses when escalation is triggered.
_POLICY_B_ESCALATION_EXTRA_N = 2
# Escalation fires when MEDIUM retrieval confidence AND no top-2 hypothesis exceeds
# this composite floor.  Derivation: a well-grounded hypothesis under MEDIUM retrieval
# (weight=0.85) with successful keyword evidence (weight=1.0) should reach at least
# raw×0.85 ≥ 0.60, i.e. raw ≥ ~0.71.  If neither top-2 hypothesis clears this bar,
# neither is confidently backed by retrieved evidence.
_ESCALATION_COMPOSITE_FLOOR = 0.60


class AdvancedInvestigationAgent:
    def __init__(
        self,
        db: Session,
        *,
        search_service: IncidentSearchService | None = None,
        llm_service: LLMService | None = None,
        embedding_service: Any | None = None,
    ) -> None:
        self.search_service = search_service or IncidentSearchService(db)
        self.llm_service = llm_service or LLMService()
        self._embedding_service = embedding_service

    def investigate(self, problem: str) -> dict[str, Any]:
        t_start = time.perf_counter()

        # ── Step 1: retrieval ────────────────────────────────────────────────
        initial_incidents = self.search_service.retrieve(
            problem,
            limit=10,
            expand=True,
            rerank=True,
            call_site="advanced_investigation_agent.investigate",
        )
        top1_score, confidence_level = IncidentSearchService.confidence_for(initial_incidents)
        initial_context = self._build_incident_context(initial_incidents, confidence_level, top1_score)

        # ── Step 2: baseline — top-2 hypotheses ─────────────────────────────
        hypotheses = self._generate_hypotheses(
            problem, initial_context, initial_incidents, n=_POLICY_B_BASELINE_N
        )
        evidence = self._collect_evidence(hypotheses)

        # ── Step 3: escalation check ─────────────────────────────────────────
        escalation_triggered = self._should_escalate(confidence_level, hypotheses, evidence)

        if escalation_triggered:
            existing_root_causes = [h["root_cause"] for h in hypotheses]
            extra_hypotheses = self._generate_hypotheses(
                problem,
                initial_context,
                initial_incidents,
                n=_POLICY_B_ESCALATION_EXTRA_N,
                existing_root_causes=existing_root_causes,
            )
            extra_evidence = self._collect_evidence(extra_hypotheses)
            hypotheses.extend(extra_hypotheses)
            evidence.extend(extra_evidence)

        hypothesis_count = len(hypotheses)
        t_hypotheses_done = time.perf_counter()

        # ── Step 4: final report ─────────────────────────────────────────────
        evidence_context = self._build_evidence_context(evidence)
        report = self._assemble_report(problem, initial_context, evidence_context, confidence_level)

        t_end = time.perf_counter()
        latency_s = t_end - t_start

        policy_metadata = {
            "policy_used": "B",
            "retrieval_confidence": confidence_level,
            "escalation_triggered": escalation_triggered,
            "hypothesis_count_generated": hypothesis_count,
            "latency_s": round(latency_s, 3),
        }

        logger.info(
            "investigation_complete",
            extra={
                "policy_used": "B",
                "retrieval_confidence": confidence_level,
                "escalation_triggered": escalation_triggered,
                "hypothesis_count_generated": hypothesis_count,
                "latency_s": round(latency_s, 3),
            },
        )

        return {
            "problem": problem,
            "retrieval_confidence": {
                "level": confidence_level,
                "top1_score": top1_score,
            },
            "initial_incidents": [
                self._incident_to_payload(result) for result in initial_incidents
            ],
            "hypotheses": hypotheses,
            "evidence": evidence,
            "report": report,
            "policy_metadata": policy_metadata,
        }

    # ── Policy B escalation ──────────────────────────────────────────────────

    def _should_escalate(
        self,
        retrieval_confidence: str,
        hypotheses: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> bool:
        """Return True when Policy B's escalation condition is met.

        Condition: retrieval confidence is MEDIUM AND no top-2 hypothesis
        reaches _ESCALATION_COMPOSITE_FLOOR composite confidence.  The
        composite is computed using the evidence-search confidence as a
        keyword-quality proxy: MEDIUM/HIGH evidence → keyword recall assumed
        successful; LOW evidence → keyword recall assumed failed.
        """
        if retrieval_confidence != CONFIDENCE_MEDIUM:
            return False
        for hyp, ev in zip(hypotheses, evidence):
            keyword_ok = ev.get("confidence_level") != CONFIDENCE_LOW
            composite = composite_hypothesis_confidence(
                raw_confidence=hyp["confidence_score"],
                retrieval_confidence_level=retrieval_confidence,
                validation_keyword_recall_ok=keyword_ok,
            )
            if composite >= _ESCALATION_COMPOSITE_FLOOR:
                return False
        return True

    # ── Hypothesis generation ────────────────────────────────────────────────

    def _generate_hypotheses(
        self,
        problem: str,
        context: str,
        initial_incidents: list[IncidentSearchResult] | None = None,
        *,
        n: int = _POLICY_B_BASELINE_N,
        existing_root_causes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        raw = self.llm_service.generate_hypotheses(
            problem=problem,
            context=context,
            n=n,
            existing_root_causes=existing_root_causes,
        )
        hypotheses = [self._normalize_hypothesis(item) for item in raw[:n]]
        if USE_EVIDENCE_KEYWORDS and initial_incidents and self._embedding_service:
            hypotheses = derive_evidence_keywords(
                hypotheses, initial_incidents, self._embedding_service
            )
        return hypotheses

    # ── Evidence collection ──────────────────────────────────────────────────

    def _collect_evidence(self, hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for hypothesis in hypotheses:
            keywords = hypothesis.get("validation_keywords", [])
            query = " ".join(str(keyword) for keyword in keywords if keyword)
            if not query:
                query = str(hypothesis.get("root_cause", ""))

            supporting_results = (
                self.search_service.search(
                    query,
                    limit=5,
                    call_site="advanced_investigation_agent.collect_evidence",
                )
                if query
                else []
            )
            evidence_top1_score, evidence_confidence = IncidentSearchService.confidence_for(
                supporting_results
            )
            evidence.append(
                {
                    "hypothesis": hypothesis,
                    "query": query,
                    "confidence_level": evidence_confidence,
                    "top1_score": evidence_top1_score,
                    "supporting_incidents": [
                        self._incident_to_payload(result) for result in supporting_results
                    ],
                }
            )
        return evidence

    # ── Report assembly ──────────────────────────────────────────────────────

    def _assemble_report(
        self,
        problem: str,
        initial_context: str,
        evidence_context: str,
        confidence_level: str,
    ) -> dict[str, Any]:
        report = self.llm_service.evaluate_investigation_evidence(
            problem=problem,
            initial_context=initial_context,
            evidence_context=evidence_context,
        )
        confidence_assessment = str(report.get("confidence_assessment", ""))
        if confidence_level == CONFIDENCE_LOW:
            confidence_assessment = (
                "LOW RETRIEVAL CONFIDENCE: no strong historical match was found "
                "for this problem. The assessment below relies primarily on "
                "general reasoning rather than retrieved incident evidence. "
                f"{confidence_assessment}"
            ).strip()
        return {
            "executive_summary": str(report.get("executive_summary", "")),
            "ranked_hypotheses": self._list_of_strings(report.get("ranked_hypotheses")),
            "supporting_evidence": self._list_of_strings(report.get("supporting_evidence")),
            "recommended_actions": self._list_of_strings(report.get("recommended_actions")),
            "confidence_assessment": confidence_assessment,
        }

    # ── Context builders ─────────────────────────────────────────────────────

    def _build_incident_context(
        self,
        results: list[IncidentSearchResult],
        confidence_level: str,
        top1_score: float | None,
    ) -> str:
        if not results:
            return (
                "Retrieval confidence: LOW (no similar incidents were retrieved).\n"
                "No historical evidence is available for the initial retrieval - "
                "rely on general reasoning and state this explicitly."
            )

        header = f"Retrieval confidence: {confidence_level}"
        header += f" (top1_score={top1_score:.3f})" if top1_score is not None else ""
        if confidence_level == CONFIDENCE_LOW:
            header += (
                "\nNo strong historical match was found. The incidents below are "
                "the closest available but may not be directly relevant. Clearly "
                "separate evidence drawn from these incidents from general "
                "reasoning, and flag conclusions as lower confidence."
            )

        body = "\n\n".join(
            self._format_incident_context(index, result)
            for index, result in enumerate(results, start=1)
        )
        return f"{header}\n\n{body}"

    def _build_evidence_context(self, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return "No hypothesis-specific supporting incidents were retrieved."

        sections: list[str] = []
        for index, item in enumerate(evidence, start=1):
            hypothesis = item["hypothesis"]
            incidents = item["supporting_incidents"]
            incident_lines = [
                (
                    f"- {incident['title']} | symptoms={'; '.join(incident['symptoms'])} | "
                    f"severity={incident['severity']} | status={incident['status']} | "
                    f"resolution={incident['resolution_summary']}"
                )
                for incident in incidents
            ]
            if not incident_lines:
                incident_lines = ["- No supporting incidents found."]
            evidence_confidence = item.get("confidence_level", CONFIDENCE_LOW)
            lines = [
                f"Hypothesis {index}: {hypothesis['root_cause']}",
                f"Confidence: {hypothesis['confidence_score']:.2f}",
                f"Validation query: {item['query']}",
                f"Retrieval confidence for supporting incidents: {evidence_confidence}",
            ]
            if evidence_confidence == CONFIDENCE_LOW:
                lines.append(
                    "No strong supporting evidence was found for this hypothesis; "
                    "treat it as based on reasoning rather than precedent."
                )
            lines.append("Supporting incidents:")
            lines.extend(incident_lines)
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def _format_incident_context(self, index: int, result: IncidentSearchResult) -> str:
        incident = result.incident
        payload = self._incident_to_payload(result)
        return "\n".join(
            [
                f"Incident {index}",
                f"Similarity score: {payload['similarity_score']:.3f}",
                f"Title: {incident.title}",
                f"Symptoms: {'; '.join(payload['symptoms']) or 'Unknown'}",
                f"Severity: {incident.severity}",
                f"Status: {incident.status}",
                f"Resolution summary: {payload['resolution_summary']}",
            ]
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _incident_to_payload(self, result: IncidentSearchResult) -> dict[str, Any]:
        incident = result.incident
        return {
            "title": incident.title,
            "symptoms": [symptom.text for symptom in incident.symptoms],
            "severity": incident.severity,
            "status": incident.status,
            "resolution_summary": incident.resolution_summary or "Unknown",
            "similarity_score": result.similarity_score,
        }

    def _normalize_hypothesis(self, item: dict[str, Any]) -> dict[str, Any]:
        keywords = item.get("validation_keywords", [])
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        return {
            "root_cause": str(item.get("root_cause", "")),
            "confidence_score": self._coerce_confidence(item.get("confidence_score", 0.0)),
            "validation_keywords": [str(keyword) for keyword in keywords if str(keyword).strip()],
            "rationale": str(item.get("rationale", "")),
        }

    def _coerce_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _list_of_strings(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value:
            return [str(value)]
        return []
