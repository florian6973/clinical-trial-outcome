"""Contract validation and constrained selector parsers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import SCHEMA_DIR, read_json


class ContractError(ValueError):
    """Raised when an input or prediction violates a task contract."""


def validate_schema(instance: Any, schema_name: str) -> None:
    schema = read_json(SCHEMA_DIR / schema_name)
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        _minimal_validate(instance, schema_name)
        return
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda err: list(err.path))
    if errors:
        rendered = "; ".join(f"{list(error.path)}: {error.message}" for error in errors[:5])
        raise ContractError(f"{schema_name} validation failed: {rendered}")


def _minimal_validate(instance: Any, schema_name: str) -> None:
    if not isinstance(instance, dict):
        raise ContractError(f"{schema_name}: expected an object")
    required_by_schema = {
        "outcome_annotation.schema.json": {"record_id", "split", "source_object", "candidate_terms", "candidate_groups", "gold_group"},
        "condition_annotation.schema.json": {"record_id", "split", "raw_condition", "candidates", "gold"},
        "canonical_paper_metrics.schema.json": {"status", "source_document", "metrics"},
        "vocabulary.schema.json": {"vocabulary_id"},
        "raw_outcome.schema.json": {"record_id", "nct_id", "outcome_id", "title", "description", "outcome_type", "time_frame"},
        "raw_condition.schema.json": {"record_id", "nct_id", "raw_condition"},
        "source_manifest.schema.json": {"manifest_version", "status", "aact", "snomed_ct", "filters", "outputs"},
    }
    missing = required_by_schema.get(schema_name, set()) - set(instance)
    if missing:
        raise ContractError(f"{schema_name}: missing {sorted(missing)}")
    if schema_name == "vocabulary.schema.json":
        outcome = {"term", "normalized_group"}.issubset(instance)
        condition = {"concept_id", "preferred_term"}.issubset(instance)
        if not (outcome or condition):
            raise ContractError("vocabulary item is neither an outcome term nor a condition concept")


def validate_annotation(task: str, record: dict[str, Any]) -> None:
    schema_name = f"{task}_annotation.schema.json"
    validate_schema(record, schema_name)
    if task == "outcome":
        groups = {item["group"] for item in record["candidate_groups"]}
        if record["gold_group"] not in groups and record["gold_group"] != "UNMAPPED_OUTCOME":
            raise ContractError("outcome gold_group must be a supplied candidate group or UNMAPPED_OUTCOME")
    else:
        selected = record["gold"]["selected_candidate_id"]
        candidate_ids = {item["candidate_id"] for item in record["candidates"]}
        if selected is not None and selected not in candidate_ids:
            raise ContractError("condition gold selection must refer to a supplied candidate_id")


def parse_outcome_prediction(raw_text: str, allowed_groups: set[str]) -> dict[str, Any]:
    matches = re.findall(r"(?m)^GROUP:\s*(.+?)\s*$", raw_text.strip())
    if len(matches) != 1:
        raise ContractError(f"expected exactly one GROUP line; found {len(matches)}")
    group = matches[0].strip()
    review_labels = {"UNMAPPED_OUTCOME", "PROPOSED_NEW_GROUP"}
    if group not in allowed_groups and group not in review_labels:
        raise ContractError(f"out-of-vocabulary group: {group}")
    explanation_match = re.search(r"(?m)^EXPLANATION:\s*(.+?)\s*$", raw_text.strip())
    proposal_matches = re.findall(r"(?m)^PROPOSED_NEW_GROUP:\s*(.+?)\s*$", raw_text.strip())
    if group == "PROPOSED_NEW_GROUP":
        if len(proposal_matches) != 1 or not proposal_matches[0].strip():
            raise ContractError("PROPOSED_NEW_GROUP requires exactly one non-empty proposed label")
        proposed_group = proposal_matches[0].strip()
        if proposed_group in allowed_groups or proposed_group in review_labels:
            raise ContractError("proposed label must be a genuinely new non-reserved label")
        status = "proposed_new_group"
    else:
        if proposal_matches:
            raise ContractError("PROPOSED_NEW_GROUP line is only allowed in the proposal review state")
        proposed_group = None
        status = "abstain" if group == "UNMAPPED_OUTCOME" else "ok"
    return {
        "explanation": explanation_match.group(1).strip() if explanation_match else None,
        "selected_group": group if status in {"ok", "abstain"} else None,
        "proposed_group": proposed_group,
        "eligible_for_aggregation": status == "ok",
        "status": status,
    }


def parse_condition_prediction(raw_text: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"condition selector returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"no_match", "selected_candidate_id"}:
        raise ContractError("condition output must contain only no_match and selected_candidate_id")
    no_match = value["no_match"]
    selected = value["selected_candidate_id"]
    if not isinstance(no_match, bool):
        raise ContractError("no_match must be boolean")
    if no_match and selected is not None:
        raise ContractError("abstention must have a null selected_candidate_id")
    by_id = {item["candidate_id"]: item for item in candidates}
    if not no_match and selected not in by_id:
        raise ContractError("selection is not one of the retrieved candidate_ids")
    if no_match:
        return {"no_match": True, "selected_candidate_id": None, "status": "abstain"}
    candidate = by_id[selected]
    return {
        "no_match": False,
        "selected_candidate_id": selected,
        "selected_concept_id": candidate["concept_id"],
        "selected_preferred_term": candidate["preferred_term"],
        "status": "ok",
    }


def validate_file(path: str | Path, schema_name: str) -> int:
    from .io import read_jsonl

    count = 0
    for record in read_jsonl(path):
        validate_schema(record, schema_name)
        count += 1
    return count
