"""Normalizes raw Jira issues into the generic NormalizedIncident.

Unlike GitHubNormalizer (which predates the adapter layer and emits a
GitHub-specific dataclass), JiraNormalizer emits NormalizedIncident directly —
the target pattern for all post-Phase-11A sources.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.ingestion.adapters.base import NormalizedIncident

_RESOLVED_STATUSES = {"resolved", "closed", "done", "fixed"}
_OPEN_STATUSES = {"open", "in progress", "to do", "reopened", "in review"}

_SEVERITY_LABELS = {
    "blocker": "critical",
    "critical": "critical",
    "sev0": "critical",
    "sev1": "critical",
    "major": "high",
    "high": "high",
    "sev2": "high",
    "minor": "medium",
    "medium": "medium",
    "sev3": "medium",
    "trivial": "low",
    "low": "low",
}
_TYPE_LABELS = {
    "bug": "bug",
    "incident": "outage",
    "outage": "outage",
    "performance": "performance",
    "regression": "deployment",
    "deployment": "deployment",
    "configuration": "configuration",
}
_RESOLUTION_KEYWORDS = ("fixed", "resolved", "closed by", "workaround", "rollback", "patched")

# Jira priority → generic severity. Covers the classic priority scheme
# (Blocker/Critical/Major/Minor/Trivial) plus the default Cloud scheme
# (Highest/High/Medium/Low/Lowest).
_PRIORITY_SEVERITY = {
    "blocker": "critical",
    "critical": "critical",
    "highest": "critical",
    "major": "high",
    "high": "high",
    "minor": "medium",
    "medium": "medium",
    "trivial": "low",
    "low": "low",
    "lowest": "low",
}

# Structured Jira resolution values that count as a genuine fix (gold-worthy).
# Non-fixes like "Won't Fix", "Duplicate", "Cannot Reproduce", "Incomplete"
# are deliberately excluded.
_POSITIVE_RESOLUTIONS = {
    "fixed",
    "done",
    "resolved",
    "completed",
    "implemented",
    "delivered",
}


class JiraNormalizer:
    def normalize(self, issue: dict[str, Any]) -> NormalizedIncident:
        fields = issue.get("fields", {}) or {}
        key = str(issue.get("key") or issue.get("id") or "UNKNOWN")
        project_key = str((fields.get("project") or {}).get("key") or key.split("-")[0])
        base_url = str(issue.get("_base_url") or "").rstrip("/")

        summary = self._clean_text(fields.get("summary") or "Untitled Jira issue")
        body = self._clean_text(fields.get("description") or "")
        comments = [
            self._clean_text(comment.get("body") or "")
            for comment in (fields.get("comment") or {}).get("comments", [])
        ]
        description = self._join_non_empty([body, *comments[:3]])

        jira_status = str((fields.get("status") or {}).get("name") or "unknown")
        status = self._status(jira_status)
        labels = self._normalize_labels(fields.get("labels") or [])
        priority_name = str((fields.get("priority") or {}).get("name") or "") or None
        jira_resolution = str((fields.get("resolution") or {}).get("name") or "") or None
        components = self._component_names(fields.get("components") or [])

        severity = self._severity(priority_name, labels)
        incident_type = self._incident_type(labels, summary, description)
        symptoms = self._extract_symptoms(summary, description)
        resolution_summary, is_gold = self._resolve(status, jira_resolution, comments)
        confidence = self._confidence(body, labels, resolution_summary)

        environment = {
            "source": "jira",
            "project": project_key,
            "base_url": base_url or None,
        }
        affected_components = components or ([project_key] if project_key else [])
        created_at_source = self._parse_datetime(fields.get("created"))
        updated_at_source = self._parse_datetime(fields.get("updated"))

        canonical_text = self._canonical_text(
            title=summary,
            description=description,
            symptoms=symptoms,
            severity=severity,
            status=status,
            incident_type=incident_type,
            project_key=project_key,
            resolution_summary=resolution_summary,
        )

        source_url = f"{base_url}/browse/{key}" if base_url else None

        source_metadata: dict[str, Any] = {
            "project_key": project_key,
            "issue_key": key,
            "jira_status": jira_status,
            "priority": priority_name,
            "resolution": jira_resolution,
            "components": components,
            "labels": labels,
            "incident_type": incident_type,
        }

        return NormalizedIncident(
            source_type="jira",
            source_external_id=key,
            source_url=source_url,
            title=summary,
            description=description,
            severity=severity,
            status=status,
            incident_type=incident_type,
            environment=environment,
            affected_components=affected_components,
            tags=labels,
            symptoms=symptoms,
            root_cause_summary=None,
            resolution_summary=resolution_summary,
            canonical_text=canonical_text,
            confidence_score=confidence,
            is_gold_labeled=is_gold,
            created_at_source=created_at_source,
            updated_at_source=updated_at_source,
            source_metadata=source_metadata,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _status(self, jira_status: str) -> str:
        lowered = jira_status.strip().lower()
        if lowered in _RESOLVED_STATUSES:
            return "resolved"
        if lowered in _OPEN_STATUSES:
            return "open"
        return "unknown"

    def _normalize_labels(self, labels: list[Any]) -> list[str]:
        return sorted({str(label).strip().lower() for label in labels if str(label).strip()})

    def _severity(self, priority_name: str | None, labels: list[str]) -> str:
        # Priority is the authoritative Jira severity signal; labels are a
        # secondary fallback for projects that encode severity as a label.
        if priority_name:
            mapped = _PRIORITY_SEVERITY.get(priority_name.strip().lower())
            if mapped:
                return mapped
        for label in labels:
            if label in _SEVERITY_LABELS:
                return _SEVERITY_LABELS[label]
        return "unknown"

    def _component_names(self, components_field: list[Any]) -> list[str]:
        names = [
            str(component.get("name")).strip()
            for component in components_field
            if isinstance(component, dict) and component.get("name")
        ]
        return sorted({name for name in names if name})

    def _incident_type(self, labels: list[str], title: str, description: str) -> str:
        for label in labels:
            if label in _TYPE_LABELS:
                return _TYPE_LABELS[label]
        content = f"{title} {description}".lower()
        if any(token in content for token in ["timeout", "latency", "slow", "performance"]):
            return "performance"
        if any(token in content for token in ["deploy", "release", "rollback", "regression"]):
            return "deployment"
        if any(token in content for token in ["outage", "down", "unavailable"]):
            return "outage"
        return "bug"

    def _extract_symptoms(self, title: str, description: str) -> list[str]:
        candidates = [title]
        for line in description.splitlines():
            line = line.strip(" -*\t")
            if not line:
                continue
            lowered = line.lower()
            if any(
                keyword in lowered
                for keyword in [
                    "error",
                    "exception",
                    "timeout",
                    "failed",
                    "failure",
                    "latency",
                    "crash",
                ]
            ):
                candidates.append(line)
        return list(dict.fromkeys(candidates))[:8]

    def _resolve(
        self,
        status: str,
        jira_resolution: str | None,
        comments: list[str],
    ) -> tuple[str | None, bool]:
        """Return (resolution_summary, is_gold).

        Primary signal is the structured Jira ``resolution`` field. Comment
        extraction is used only as a fallback when that field is absent. A
        non-fix resolution (Won't Fix, Duplicate, ...) yields a summary but is
        NOT gold-labeled.
        """
        if jira_resolution:
            positive = jira_resolution.strip().lower() in _POSITIVE_RESOLUTIONS
            return jira_resolution, (status == "resolved" and positive)

        # Structured field missing → fall back to comment extraction.
        if status != "resolved":
            return None, False
        for comment in reversed(comments):
            lowered = comment.lower()
            if any(keyword in lowered for keyword in _RESOLUTION_KEYWORDS):
                return comment[:2000], True
        return None, False

    def _canonical_text(
        self,
        *,
        title: str,
        description: str,
        symptoms: list[str],
        severity: str,
        status: str,
        incident_type: str,
        project_key: str,
        resolution_summary: str | None,
    ) -> str:
        sections = [title, ""]
        repo_label = project_key or "UNKNOWN"
        sections.append(f"{repo_label} | {incident_type} | severity {severity}")

        symptom_line = self._format_symptoms(symptoms)
        if symptom_line:
            sections.append("")
            sections.append(f"Symptoms: {symptom_line}")

        excerpt = self._description_excerpt(description)
        if excerpt:
            sections.append("")
            sections.append(f"What happened: {excerpt}")

        if status == "resolved" and resolution_summary:
            resolution_excerpt = self._truncate(self._strip_markup_noise(resolution_summary), 150)
            if resolution_excerpt:
                sections.append("")
                sections.append(f"Resolution: {resolution_excerpt}")

        return "\n".join(sections).strip()

    def _format_symptoms(self, symptoms: list[str]) -> str:
        items = [
            self._truncate(self._strip_markup_noise(symptom), 60) for symptom in symptoms[:2]
        ]
        return self._truncate("; ".join(item for item in items if item), 120)

    def _description_excerpt(self, description: str) -> str:
        for paragraph in description.split("\n\n"):
            cleaned = self._strip_markup_noise(paragraph)
            if len(cleaned) > 20:
                return self._truncate(cleaned, 280)
        return self._truncate(self._strip_markup_noise(description), 280)

    def _strip_markup_noise(self, text: str) -> str:
        # Strip common Jira wiki markup so embeddings see prose, not syntax.
        text = re.sub(r"\{code(:[^}]*)?\}.*?\{code\}", " ", text, flags=re.DOTALL)
        text = re.sub(r"\{noformat\}.*?\{noformat\}", " ", text, flags=re.DOTALL)
        text = re.sub(r"\{[^}]*\}", " ", text)  # remaining {panel}, {quote}, etc.
        text = re.sub(r"h[1-6]\.\s*", "", text)  # headings: h2. Title
        text = re.sub(r"\[([^|\]]+)\|[^\]]+\]", r"\1", text)  # [text|url] -> text
        text = re.sub(r"[*_+]", "", text)  # bold/italic/inserted markers
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _truncate(self, text: str, max_len: int) -> str:
        text = text.strip()
        if len(text) <= max_len:
            return text
        return text[:max_len].rstrip() + "…"

    def _confidence(self, body: str, labels: list[str], resolution_summary: str | None) -> float:
        score = 0.45
        if len(body) > 100:
            score += 0.2
        if labels:
            score += 0.1
        if resolution_summary:
            score += 0.2
        return min(score, 0.95)

    def _clean_text(self, value: str) -> str:
        value = re.sub(r"\r\n?", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    def _join_non_empty(self, values: list[str]) -> str:
        return "\n\n".join(value for value in values if value)

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        # Jira returns e.g. "2026-05-01T10:00:00.000+0000"; normalize the tz.
        cleaned = value.replace("Z", "+00:00")
        cleaned = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", cleaned)
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None
