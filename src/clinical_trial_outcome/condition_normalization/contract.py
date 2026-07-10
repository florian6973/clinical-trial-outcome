"""Serializable input/output contracts for condition normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .disease_areas import validate_disease_area


@dataclass(frozen=True)
class SnomedCandidate:
    """One SNOMED CT candidate returned by FAISS retrieval."""

    concept_id: str
    term: str
    disease_area: str
    similarity: float
    rank: int

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise ValueError("concept_id must not be empty")
        if not self.term.strip():
            raise ValueError("term must not be empty")
        validate_disease_area(self.disease_area)
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if not -1.0 <= float(self.similarity) <= 1.0:
            raise ValueError("similarity must be between -1 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionNormalizationPlan:
    """Dry-run plan; it never imports FAISS or a model runtime."""

    mode: str
    condition_text: str
    stages: tuple[str, ...]
    retrieval_top_k: int
    qwen_model: str
    disease_area_count: int = 24
    execution_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = list(self.stages)
        return payload


@dataclass(frozen=True)
class ConditionNormalizationResult:
    """Selected SNOMED concept after FAISS retrieval and Qwen selection."""

    condition_text: str
    snomed_concept_id: str
    snomed_term: str
    disease_area: str
    similarity: float
    candidate_rank: int
    qwen_model: str
    selection_method: str
    candidates: tuple[SnomedCandidate, ...]

    def __post_init__(self) -> None:
        validate_disease_area(self.disease_area)
        if self.selection_method != "faiss_candidates_then_qwen_selection":
            raise ValueError("selection_method must preserve the paper method order")
        if not self.candidates:
            raise ValueError("at least one retrieved candidate is required")
        selected = [
            candidate
            for candidate in self.candidates
            if candidate.concept_id == self.snomed_concept_id
        ]
        if len(selected) != 1:
            raise ValueError("Qwen must select exactly one of the retrieved candidates")
        candidate = selected[0]
        if (
            candidate.term != self.snomed_term
            or candidate.disease_area != self.disease_area
            or candidate.rank != self.candidate_rank
        ):
            raise ValueError("selected fields must match the retrieved candidate")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload
