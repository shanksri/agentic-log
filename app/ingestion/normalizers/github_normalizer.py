from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class NormalizedGitHubIncident:
    source_type: str
    source_external_id: str
    source_url: str | None
    owner: str | None
    repo: str | None
    source: str
    state: str
    title: str
    description: str
    severity: str
    status: str
    incident_type: str
    environment: dict[str, Any]
    affected_components: list[str]
    tags: list[str]
    symptoms: list[str]
    root_cause_summary: str | None
    resolution_summary: str | None
    canonical_text: str
    confidence_score: float
    is_gold_labeled: bool
    created_at_source: datetime | None
    updated_at_source: datetime | None


class GitHubNormalizer:
    severity_labels = {
        "sev0": "critical",
        "sev1": "critical",
        "severity: critical": "critical",
        "critical": "critical",
        "sev2": "high",
        "severity: high": "high",
        "high": "high",
        "sev3": "medium",
        "severity: medium": "medium",
        "medium": "medium",
        "low": "low",
        "severity: low": "low",
    }
    type_labels = {
        "bug": "bug",
        "incident": "outage",
        "outage": "outage",
        "performance": "performance",
        "regression": "deployment",
        "deployment": "deployment",
        "configuration": "configuration",
    }

    def normalize(self, issue: dict[str, Any]) -> NormalizedGitHubIncident:
        repo = issue.get("repository", {})
        repo_full_name = str(repo.get("full_name") or "")
        repo_owner = repo.get("owner")
        repo_name = repo.get("name")
        labels = self._labels(issue)
        body = self._clean_text(issue.get("body") or "")
        comments = [
            self._clean_text(comment.get("body") or "")
            for comment in issue.get("comments_payload", [])
        ]
        description = self._join_non_empty([body, *comments[:3]])
        tags = self._normalize_tags(labels)
        title = self._clean_text(issue.get("title") or "Untitled GitHub issue")
        status = self._status(issue)
        severity = self._severity(tags)
        incident_type = self._incident_type(tags, title, description)
        symptoms = self._extract_symptoms(title, description)
        resolution_summary = self._extract_resolution(status, comments)
        confidence = self._confidence(body, labels, resolution_summary)

        environment = {
            "source": "github",
            "repository": repo_full_name,
            "repository_owner": repo_owner,
            "repository_name": repo_name,
        }
        affected_components = self._affected_components(repo_full_name, tags)
        created_at_source = self._parse_datetime(issue.get("created_at"))
        updated_at_source = self._parse_datetime(issue.get("updated_at"))

        canonical_text = self._canonical_text(
            title=title,
            description=description,
            symptoms=symptoms,
            severity=severity,
            status=status,
            incident_type=incident_type,
            repo_full_name=repo_full_name,
            resolution_summary=resolution_summary,
        )

        source_external_id = (
            f"{repo_full_name}#{issue['number']}" if repo_full_name else str(issue["id"])
        )
        return NormalizedGitHubIncident(
            source_type="github",
            source_external_id=source_external_id,
            source_url=issue.get("html_url"),
            owner=str(repo_owner) if repo_owner else None,
            repo=str(repo_name) if repo_name else None,
            source="github",
            state=str(issue.get("state") or "unknown").lower(),
            title=title,
            description=description,
            severity=severity,
            status=status,
            incident_type=incident_type,
            environment=environment,
            affected_components=affected_components,
            tags=tags,
            symptoms=symptoms,
            root_cause_summary=None,
            resolution_summary=resolution_summary,
            canonical_text=canonical_text,
            confidence_score=confidence,
            is_gold_labeled=bool(resolution_summary and status == "resolved"),
            created_at_source=created_at_source,
            updated_at_source=updated_at_source,
        )

    def _labels(self, issue: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for label in issue.get("labels", []):
            if isinstance(label, dict):
                values.append(str(label.get("name", "")))
            else:
                values.append(str(label))
        return values

    def _normalize_tags(self, labels: list[str]) -> list[str]:
        return sorted({label.strip().lower() for label in labels if label.strip()})

    def _status(self, issue: dict[str, Any]) -> str:
        state = str(issue.get("state") or "unknown").lower()
        if state == "closed":
            return "resolved"
        if state == "open":
            return "open"
        return "unknown"

    def _severity(self, tags: list[str]) -> str:
        for tag in tags:
            if tag in self.severity_labels:
                return self.severity_labels[tag]
        return "unknown"

    def _incident_type(self, tags: list[str], title: str, description: str) -> str:
        for tag in tags:
            if tag in self.type_labels:
                return self.type_labels[tag]
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

    def _extract_resolution(self, status: str, comments: list[str]) -> str | None:
        if status != "resolved":
            return None
        resolution_keywords = (
            "fixed",
            "resolved",
            "closed by",
            "workaround",
            "rollback",
            "patched",
        )
        for comment in reversed(comments):
            lowered = comment.lower()
            if any(keyword in lowered for keyword in resolution_keywords):
                return comment[:2000]
        return None

    def _affected_components(self, repo_full_name: str, tags: list[str]) -> list[str]:
        components = [repo_full_name] if repo_full_name else []
        for tag in tags:
            if tag.startswith(("component:", "area:", "service:")):
                components.append(tag.split(":", 1)[1].strip())
        return sorted({component for component in components if component})

    def _canonical_text(
        self,
        *,
        title: str,
        description: str,
        symptoms: list[str],
        severity: str,
        status: str,
        incident_type: str,
        repo_full_name: str,
        resolution_summary: str | None,
    ) -> str:
        sections = [title, ""]

        repo_label = repo_full_name or "unknown/unknown"
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
            resolution_excerpt = self._truncate(self._strip_markdown_noise(resolution_summary), 150)
            if resolution_excerpt:
                sections.append("")
                sections.append(f"Resolution: {resolution_excerpt}")

        return "\n".join(sections).strip()

    def _format_symptoms(self, symptoms: list[str]) -> str:
        items = [self._truncate(symptom, 60) for symptom in symptoms[:2]]
        return self._truncate("; ".join(item for item in items if item), 120)

    def _description_excerpt(self, description: str) -> str:
        for paragraph in description.split("\n\n"):
            cleaned = self._strip_markdown_noise(paragraph)
            if len(cleaned) > 20:
                return self._truncate(cleaned, 280)
        return self._truncate(self._strip_markdown_noise(description), 280)

    def _strip_markdown_noise(self, text: str) -> str:
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
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
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
