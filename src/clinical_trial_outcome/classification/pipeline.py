"""Dry-run-first Qwen classifier for the locked analytic taxonomies."""

from __future__ import annotations

from typing import Any, Callable

from .contract import ClassificationPlan, OutcomeClassificationResult
from .prompt import (
    QWEN_CLASSIFICATION_SYSTEM_PROMPT,
    build_classification_prompt,
    parse_classification_response,
)
from .taxonomy import COA_TYPES, OUTCOME_CATEGORIES

QwenGenerator = Callable[[str, str], str]


class OutcomeClassificationPipeline:
    def __init__(
        self,
        qwen_generator: QwenGenerator | None = None,
        qwen_model: str = "Qwen/Qwen2.5-32B-Instruct",
    ) -> None:
        self.qwen_generator = qwen_generator
        self.qwen_model = qwen_model

    def classify(
        self,
        normalized_outcome: str,
        *,
        execute: bool = False,
        confirm_execute: bool = False,
    ) -> dict[str, Any]:
        if not normalized_outcome.strip():
            raise ValueError("normalized_outcome must not be empty")
        if not execute:
            return ClassificationPlan(
                mode="dry_run",
                normalized_outcome=normalized_outcome,
                stages=(
                    "render_locked_21_category_and_4_coa_prompt",
                    "select_labels_with_qwen",
                    "validate_exact_taxonomy_membership",
                ),
                qwen_model=self.qwen_model,
                outcome_category_count=len(OUTCOME_CATEGORIES),
                coa_type_count=len(COA_TYPES),
            ).to_dict()
        if not confirm_execute:
            raise PermissionError(
                "Classification inference requires execute=True and confirm_execute=True"
            )
        if self.qwen_generator is None:
            raise RuntimeError("--execute requires an injected Qwen generator")
        prompt = build_classification_prompt(normalized_outcome)
        parsed = parse_classification_response(
            self.qwen_generator(QWEN_CLASSIFICATION_SYSTEM_PROMPT, prompt)
        )
        return OutcomeClassificationResult(
            normalized_outcome=normalized_outcome,
            outcome_category=parsed["outcome_category"],
            coa_type=parsed["coa_type"],
            qwen_model=self.qwen_model,
        ).to_dict()
