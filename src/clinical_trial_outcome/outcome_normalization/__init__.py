"""Retrieval-augmented Qwen outcome normalization."""

from .config import QwenLoRAConfig, RetrievalConfig
from .data import (
    OutcomeAnnotation,
    OutcomeSplitManifest,
    SupervisedRecord,
    build_split_manifest,
    split_annotated_examples,
)
from .parser import GroupParseError, parse_group_response
from .prompt import build_group_prompt
from .retrieval import Candidate, RetrievalContext, retrieve_context, top_k_cosine
from .inference import (
    InferencePlan,
    NormalizedOutcome,
    build_inference_plan,
    normalize_outcome_objects,
)
from .training import (
    TrainingPlan,
    TrainingRunSummary,
    build_training_plan,
    train_qwen_lora,
)

__all__ = [
    "Candidate",
    "GroupParseError",
    "InferencePlan",
    "NormalizedOutcome",
    "OutcomeAnnotation",
    "OutcomeSplitManifest",
    "QwenLoRAConfig",
    "RetrievalConfig",
    "RetrievalContext",
    "SupervisedRecord",
    "TrainingPlan",
    "TrainingRunSummary",
    "build_group_prompt",
    "build_inference_plan",
    "build_split_manifest",
    "build_training_plan",
    "normalize_outcome_objects",
    "parse_group_response",
    "retrieve_context",
    "split_annotated_examples",
    "top_k_cosine",
    "train_qwen_lora",
]
