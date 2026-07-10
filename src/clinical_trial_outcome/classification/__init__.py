"""Paper-aligned outcome-category and FDA COA classification contracts."""

from .contract import ClassificationPlan, OutcomeClassificationResult
from .pipeline import OutcomeClassificationPipeline
from .prompt import build_classification_prompt, parse_classification_response
from .taxonomy import COA_TYPES, OUTCOME_CATEGORIES

__all__ = [
    "COA_TYPES",
    "OUTCOME_CATEGORIES",
    "ClassificationPlan",
    "OutcomeClassificationPipeline",
    "OutcomeClassificationResult",
    "build_classification_prompt",
    "parse_classification_response",
]
