"""Serializable classification dry-run and output contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .taxonomy import validate_coa_type, validate_outcome_category


@dataclass(frozen=True)
class ClassificationPlan:
    mode: str
    normalized_outcome: str
    stages: tuple[str, ...]
    qwen_model: str
    outcome_category_count: int = 21
    coa_type_count: int = 4
    execution_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = list(self.stages)
        return payload


@dataclass(frozen=True)
class OutcomeClassificationResult:
    normalized_outcome: str
    outcome_category: str
    coa_type: str
    qwen_model: str
    classification_method: str = "qwen_constrained_taxonomy_selection"

    def __post_init__(self) -> None:
        if not self.normalized_outcome.strip():
            raise ValueError("normalized_outcome must not be empty")
        validate_outcome_category(self.outcome_category)
        validate_coa_type(self.coa_type)
        if self.classification_method != "qwen_constrained_taxonomy_selection":
            raise ValueError("unexpected classification method")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
