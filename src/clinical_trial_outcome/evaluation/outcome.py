"""Exact-match and macro metrics for held-out outcome normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PaperOutcomeResults:
    source: str
    status: str
    distinct_input_expressions: int
    distinct_normalized_concepts: int
    held_out_examples: int
    exact_match_accuracy: float


PAPER_OUTCOME_RESULTS = PaperOutcomeResults(
    source="paper/expected_results.json",
    status="validation_expectation_only",
    distinct_input_expressions=480_273,
    distinct_normalized_concepts=237_820,
    held_out_examples=50,
    exact_match_accuracy=0.88,
)


@dataclass(frozen=True)
class OutcomePrediction:
    reference_group: str
    predicted_group: str


@dataclass(frozen=True)
class OutcomeMetrics:
    number_of_examples: int
    exact_matches: int
    exact_match_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def normalize_group_label(label: str) -> str:
    value = " ".join(label.strip().split())
    if value.casefold().startswith("group:"):
        value = value.split(":", 1)[1].strip()
    return value.casefold()


def evaluate_outcome_predictions(predictions: Iterable[OutcomePrediction]) -> OutcomeMetrics:
    rows = list(predictions)
    if not rows:
        raise ValueError("At least one outcome prediction is required")
    normalized = [
        (normalize_group_label(row.reference_group), normalize_group_label(row.predicted_group))
        for row in rows
    ]
    labels = sorted({label for pair in normalized for label in pair})
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(
            reference == label and predicted == label for reference, predicted in normalized
        )
        false_positive = sum(
            reference != label and predicted == label for reference, predicted in normalized
        )
        false_negative = sum(
            reference == label and predicted != label for reference, predicted in normalized
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    matches = sum(reference == predicted for reference, predicted in normalized)
    return OutcomeMetrics(
        number_of_examples=len(rows),
        exact_matches=matches,
        exact_match_accuracy=matches / len(rows),
        macro_precision=sum(precision_values) / len(labels),
        macro_recall=sum(recall_values) / len(labels),
        macro_f1=sum(f1_values) / len(labels),
    )
