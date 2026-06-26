from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.core.config import settings


class LLMService:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for incident investigation")
        self.client = OpenAI(api_key=self.api_key)

    def generate_investigation(self, *, problem: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior incident investigator. Use the provided similar "
                        "incident context as evidence. Be explicit about uncertainty. "
                        "Return concise sections for probable root causes, confidence "
                        "assessment, supporting evidence, and recommended actions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current problem statement:\n{problem}\n\n"
                        f"Similar incident context:\n{context}\n\n"
                        "Analyze the current problem using only defensible inferences from "
                        "the problem and retrieved incidents."
                    ),
                },
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content or ""

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI response was not a JSON object")
        return parsed

    def generate_hypotheses(
        self,
        *,
        problem: str,
        context: str,
        n: int = 2,
        existing_root_causes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        exclusion = ""
        if existing_root_causes:
            formatted = "; ".join(f'"{rc}"' for rc in existing_root_causes)
            exclusion = (
                f"\n\nThe following root causes have already been proposed — "
                f"do NOT repeat them; generate distinct alternatives: {formatted}."
            )
        payload = self.generate_json(
            system_prompt=(
                "You are a senior incident investigator. Return only valid JSON with a "
                "'hypotheses' array. Each hypothesis must include root_cause, "
                "confidence_score from 0 to 1, validation_keywords as an array of strings, "
                "and rationale."
            ),
            user_prompt=(
                f"Problem statement:\n{problem}\n\n"
                f"Retrieved incident context:\n{context}\n\n"
                f"Generate exactly {n} possible root-cause hypotheses.{exclusion}"
            ),
        )
        hypotheses = payload.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            return []
        return [item for item in hypotheses if isinstance(item, dict)]

    def evaluate_investigation_evidence(
        self,
        *,
        problem: str,
        initial_context: str,
        evidence_context: str,
    ) -> dict[str, Any]:
        return self.generate_json(
            system_prompt=(
                "You are a senior incident commander. Return only valid JSON with these "
                "keys: executive_summary, ranked_hypotheses, supporting_evidence, "
                "recommended_actions, confidence_assessment. ranked_hypotheses, "
                "supporting_evidence, and recommended_actions must be arrays."
            ),
            user_prompt=(
                f"Problem statement:\n{problem}\n\n"
                f"Initial retrieved incidents:\n{initial_context}\n\n"
                f"Hypothesis validation evidence:\n{evidence_context}\n\n"
                "Evaluate all evidence, rank hypotheses from most likely to least likely, "
                "and produce a structured investigation report."
            ),
        )

    def expand_search_query(self, query: str) -> list[str]:
        payload = self.generate_json(
            system_prompt=(
                "You improve incident retrieval queries. Return only valid JSON with a "
                "'queries' array containing 3 to 5 concise related search phrases."
            ),
            user_prompt=(
                f"Original incident search query:\n{query}\n\n"
                "Generate related search phrases that could find similar incidents."
            ),
        )
        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            return []
        return [str(item) for item in queries if str(item).strip()][:5]

    def rerank_incident_search_results(
        self,
        *,
        query: str,
        candidates: list[dict[str, object]],
        limit: int = 5,
    ) -> list[str]:
        payload = self.generate_json(
            system_prompt=(
                "You rerank incident search results for operational relevance. Return only "
                "valid JSON with a 'selected_ids' array containing candidate_id values for "
                "the most relevant incidents in best-to-worst order."
            ),
            user_prompt=(
                f"Original query:\n{query}\n\n"
                f"Candidate incidents:\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
                f"Select the best {limit} incidents."
            ),
        )
        selected_ids = payload.get("selected_ids", [])
        if not isinstance(selected_ids, list):
            return []
        return [str(item) for item in selected_ids if str(item).strip()][:limit]
