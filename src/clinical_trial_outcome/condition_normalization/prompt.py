"""Strict Qwen prompt and response parser for SNOMED candidate selection."""

from __future__ import annotations

import json
from typing import Iterable

from .contract import SnomedCandidate

QWEN_CONDITION_SYSTEM_PROMPT = (
    "You map clinical-trial condition text to SNOMED CT. Select exactly one "
    "candidate supplied by the retrieval system. Do not invent a concept."
)


def build_condition_selection_prompt(
    condition_text: str, candidates: Iterable[SnomedCandidate]
) -> str:
    candidate_list = tuple(candidates)
    if not condition_text.strip():
        raise ValueError("condition_text must not be empty")
    if not candidate_list:
        raise ValueError("at least one SNOMED candidate is required")
    payload = [candidate.to_dict() for candidate in candidate_list]
    return (
        "Select the best SNOMED CT mapping for the condition below.\n\n"
        f"CONDITION: {condition_text.strip()}\n\n"
        "CANDIDATES (retrieved by FAISS over L2-normalized NV-Embed-v2 vectors):\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only with this exact shape: "
        '{"selected_concept_id":"<one supplied concept_id>"}'
    )


def parse_condition_selection(
    response_text: str, candidates: Iterable[SnomedCandidate]
) -> SnomedCandidate:
    candidate_list = tuple(candidates)
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen condition response must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"selected_concept_id"}:
        raise ValueError("Qwen condition response has unexpected fields")
    selected_id = str(payload["selected_concept_id"])
    matches = [c for c in candidate_list if c.concept_id == selected_id]
    if len(matches) != 1:
        raise ValueError("Qwen must select one supplied SNOMED candidate")
    return matches[0]
