# Phase 3 Summary

## Overview

Phase 3 adds a multi-step incident investigation workflow without introducing
LangGraph, CrewAI, AutoGen, or another orchestration framework.

The existing `InvestigationAgent` remains unchanged. The new advanced workflow
lives in `app/services/advanced_investigation_agent.py` and is exposed through:

```text
POST /agent/investigate-advanced
```

## Architecture

The advanced agent uses a retrieval-augmented reasoning loop:

1. Search with the original problem statement and retrieve the top 10 incidents.
2. Ask the LLM for 3 to 5 root-cause hypotheses, confidence scores, and
   validation keywords.
3. Search again for each hypothesis using the suggested keywords.
4. Ask the LLM to evaluate all gathered evidence.
5. Return a structured JSON investigation report.

## Main Components

### `AdvancedInvestigationAgent`

Coordinates the full workflow:

- initial retrieval
- hypothesis generation
- hypothesis-specific evidence collection
- final report assembly

### `LLMService`

Keeps the OpenAI integration centralized. Phase 3 adds JSON-oriented helper
methods for hypothesis generation and evidence evaluation while preserving the
existing single-call investigation behavior.

### Pydantic Schemas

The advanced endpoint returns structured JSON through:

- `AdvancedInvestigationRequest`
- `AdvancedInvestigationResponse`
- `AdvancedHypothesis`
- `AdvancedEvidenceIncident`
- `AdvancedHypothesisEvidence`
- `AdvancedInvestigationReport`

## Endpoint

Request:

```json
{
  "problem": "database timeout during peak traffic"
}
```

Response shape:

```json
{
  "problem": "...",
  "initial_incidents": [],
  "hypotheses": [],
  "evidence": [],
  "report": {
    "executive_summary": "...",
    "ranked_hypotheses": [],
    "supporting_evidence": [],
    "recommended_actions": [],
    "confidence_assessment": "..."
  }
}
```

## Constraints Preserved

- `SearchService` is unchanged.
- Existing `InvestigationAgent` is unchanged.
- Ingestion functionality is unchanged.
- Database schema is unchanged.

