"""Prompt and schema helpers for decomposing free-text outcome titles.

This module is intentionally dependency-free.  It prepares and validates model
I/O, but it never loads a model or starts inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Mapping


OUTCOME_FIELDS = (
    "Quantity or Object of Interest",
    "Quantity Measure",
    "Time",
    "Quantity Unit",
    "Quantity Range",
    "Additional Constraints",
)


class OutcomeStructureError(ValueError):
    """Raised when a structured outcome response violates the schema."""


@dataclass(frozen=True)
class StructuredOutcome:
    """Validated decomposition of a single outcome title."""

    source_title: str
    objects: tuple[str, ...]
    measures: tuple[str, ...] = field(default_factory=tuple)
    times: tuple[str, ...] = field(default_factory=tuple)
    units: tuple[str, ...] = field(default_factory=tuple)
    ranges: tuple[str, ...] = field(default_factory=tuple)
    constraints: tuple[str, ...] = field(default_factory=tuple)

    def to_model_mapping(self) -> dict[str, list[str]]:
        return {
            "Quantity or Object of Interest": list(self.objects),
            "Quantity Measure": list(self.measures),
            "Time": list(self.times),
            "Quantity Unit": list(self.units),
            "Quantity Range": list(self.ranges),
            "Additional Constraints": list(self.constraints),
        }


def build_outcome_structuring_prompt(title: str) -> str:
    """Build the paper-aligned extraction prompt for one outcome title."""

    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Outcome title must not be empty")
    return f'''Extract the clinical outcome title into the JSON fields below.

Rules:
- "Quantity or Object of Interest" contains the highest-level medical,
  scientific, or measurable concept. Composite endpoints may contain more than
  one object.
- Keep count/change modifiers in "Quantity Measure", timing expressions in
  "Time", units in "Quantity Unit", thresholds in "Quantity Range", and other
  qualifiers in "Additional Constraints".
- Use only words or concepts supported by the source title. Do not add an
  explanation and do not duplicate an item across fields.
- Return every field as a JSON list of strings; use an empty list when absent.

Required JSON schema:
{{
  "Quantity or Object of Interest": [],
  "Quantity Measure": [],
  "Time": [],
  "Quantity Unit": [],
  "Quantity Range": [],
  "Additional Constraints": []
}}

Outcome title: {clean_title}'''


def _extract_json_object(text: str) -> Mapping[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise OutcomeStructureError("Response does not contain a JSON object")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise OutcomeStructureError(f"Invalid JSON response: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise OutcomeStructureError("Structured response must be a JSON object")
    return parsed


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable) or isinstance(value, (bytes, Mapping)):
        raise OutcomeStructureError(f"{field_name!r} must be a list of strings")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise OutcomeStructureError(f"{field_name!r} must contain only strings")
        item = " ".join(item.split())
        if item and item not in cleaned:
            cleaned.append(item)
    return tuple(cleaned)


def parse_structured_outcome(response: str, *, source_title: str) -> StructuredOutcome:
    """Parse model text into a strict :class:`StructuredOutcome`."""

    payload = _extract_json_object(response)
    unknown = set(payload).difference(OUTCOME_FIELDS)
    if unknown:
        raise OutcomeStructureError(f"Unexpected outcome fields: {sorted(unknown)}")
    values = {name: _string_tuple(payload.get(name, []), name) for name in OUTCOME_FIELDS}
    objects = values["Quantity or Object of Interest"]
    if not objects:
        raise OutcomeStructureError("At least one outcome object is required")
    return StructuredOutcome(
        source_title=source_title,
        objects=objects,
        measures=values["Quantity Measure"],
        times=values["Time"],
        units=values["Quantity Unit"],
        ranges=values["Quantity Range"],
        constraints=values["Additional Constraints"],
    )
