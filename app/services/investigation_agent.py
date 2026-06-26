from __future__ import annotations

from sqlalchemy.orm import Session

from app.services.confidence import CONFIDENCE_LOW, classify_confidence
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchResult, IncidentSearchService


class InvestigationAgent:
    def __init__(
        self,
        db: Session,
        *,
        search_service: IncidentSearchService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.search_service = search_service or IncidentSearchService(db)
        self.llm_service = llm_service or LLMService()

    def investigate(self, problem: str) -> str:
        similar_incidents = self.search_service.retrieve(
            problem,
            limit=5,
            expand=True,
            rerank=True,
            call_site="investigation_agent.investigate",
        )
        context = self._build_context(similar_incidents)
        return self.llm_service.generate_investigation(problem=problem, context=context)

    def _build_context(self, results: list[IncidentSearchResult]) -> str:
        top1_score = results[0].similarity_score if results else None
        confidence_level = classify_confidence(top1_score)

        if not results:
            return (
                "Retrieval confidence: LOW (no similar incidents were retrieved).\n"
                "No historical evidence is available. Any analysis below must be "
                "based on general reasoning, not retrieved incidents - state this "
                "explicitly."
            )

        sections: list[str] = []
        for index, result in enumerate(results, start=1):
            incident = result.incident
            symptoms = "; ".join(symptom.text for symptom in incident.symptoms) or "Unknown"
            resolution = incident.resolution_summary or "Unknown"
            sections.append(
                "\n".join(
                    [
                        f"Incident {index}",
                        f"Similarity score: {result.similarity_score:.3f}",
                        f"Title: {incident.title}",
                        f"Symptoms: {symptoms}",
                        f"Severity: {incident.severity}",
                        f"Status: {incident.status}",
                        f"Resolution summary: {resolution}",
                    ]
                )
            )

        header = f"Retrieval confidence: {confidence_level} (top1_score={top1_score:.3f})"
        if confidence_level == CONFIDENCE_LOW:
            header += (
                "\nNo strong historical match was found. The incidents below are "
                "the closest available but may not be directly relevant. State "
                "explicitly that no strong historical match was found, and clearly "
                "separate evidence drawn from these incidents from your own general "
                "reasoning."
            )

        return header + "\n\n" + "\n\n".join(sections)

