"""Single-source prompt contracts used by preparation and inference."""

from __future__ import annotations

import json
from typing import Any


OUTCOME_SYSTEM = (
    "You normalize clinical-trial outcome objects. Select the candidate group that "
    "best preserves the source meaning and granularity. Do not infer unsupported "
    "rate, instrument, population, temporal, or objective-response qualifiers. "
    "Return one short explanation followed by exactly one GROUP: line."
)

CONDITION_SYSTEM = (
    "You normalize clinical-trial condition strings to retrieved terminology "
    "candidates. Select exactly one candidate_id only when it is supported by the "
    "source. Otherwise abstain. Never create or modify a terminology identifier. "
    "Return JSON only."
)


def _scored_lines(items: list[dict[str, Any]], label_field: str) -> str:
    return "\n".join(
        f"- {item[label_field]} (similarity={float(item['similarity']):.4f})" for item in items
    )


def outcome_user(record: dict[str, Any]) -> str:
    context = record.get("context", "")
    return (
        f"Source outcome object: {record['source_object']}\n"
        f"Context retained from structuring: {context or '[none supplied]'}\n\n"
        "Nearest reference terms:\n"
        f"{_scored_lines(record['candidate_terms'], 'term')}\n\n"
        "Nearest normalized groups:\n"
        f"{_scored_lines(record['candidate_groups'], 'group')}\n\n"
        "Choose one supplied normalized group or UNMAPPED_OUTCOME. If no supplied "
        "group is suitable and a simple accurate label is warranted, return the "
        "review state shown below; the proposed label cannot enter aggregation "
        "until adjudicated.\n"
        "Return exactly:\nEXPLANATION: <one or two sentences>\nGROUP: <one supplied label, "
        "UNMAPPED_OUTCOME, or PROPOSED_NEW_GROUP>\n"
        "If GROUP is PROPOSED_NEW_GROUP, add:\nPROPOSED_NEW_GROUP: <simple label>"
    )


def condition_user(record: dict[str, Any]) -> str:
    candidates = [
        {
            "candidate_id": item["candidate_id"],
            "concept_id": item["concept_id"],
            "preferred_term": item["preferred_term"],
            "similarity": item["similarity"],
        }
        for item in record["candidates"]
    ]
    return (
        f"Raw condition: {record['raw_condition']}\n"
        f"Retrieved candidates: {json.dumps(candidates, ensure_ascii=False)}\n"
        "Return exactly one object: "
        '{"no_match":false,"selected_candidate_id":"<candidate_id>"} or '
        '{"no_match":true,"selected_candidate_id":null}'
    )


def outcome_assistant(gold_group: str, explanation: str | None = None) -> str:
    reason = explanation or "This label best preserves the source meaning and available granularity."
    return f"EXPLANATION: {reason}\nGROUP: {gold_group}"


def condition_assistant(gold: dict[str, Any]) -> str:
    return json.dumps(gold, ensure_ascii=False, separators=(",", ":"))
