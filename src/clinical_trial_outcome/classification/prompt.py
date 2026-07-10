"""Constrained Qwen classification prompt and strict JSON parser."""

from __future__ import annotations

import json
from typing import Any

from .taxonomy import (
    COA_TYPES,
    OUTCOME_CATEGORIES,
    validate_coa_type,
    validate_outcome_category,
)

QWEN_CLASSIFICATION_SYSTEM_PROMPT = (
    "You classify normalized clinical-trial outcomes into locked paper "
    "taxonomies. Select labels exactly as supplied and return JSON only."
)


def build_classification_prompt(normalized_outcome: str) -> str:
    if not normalized_outcome.strip():
        raise ValueError("normalized_outcome must not be empty")
    return (
        "Classify the normalized clinical-trial outcome below.\n\n"
        f"NORMALIZED_OUTCOME: {normalized_outcome.strip()}\n\n"
        f"OUTCOME_CATEGORIES: {json.dumps(OUTCOME_CATEGORIES)}\n\n"
        f"FDA_COA_TYPES: {json.dumps(COA_TYPES)}\n\n"
        "Return JSON only with exactly these keys: "
        '{"outcome_category":"<one supplied category>",'
        '"coa_type":"<one supplied COA type>"}'
    )


def parse_classification_response(response_text: str) -> dict[str, str]:
    try:
        payload: Any = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen classification response must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"outcome_category", "coa_type"}:
        raise ValueError("Qwen classification response has unexpected fields")
    outcome_category = validate_outcome_category(str(payload["outcome_category"]))
    coa_type = validate_coa_type(str(payload["coa_type"]))
    return {"outcome_category": outcome_category, "coa_type": coa_type}
