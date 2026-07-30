"""Transparent item-level exact-match evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_jsonl, sha256_file
from .validation import validate_annotation


def evaluate(task: str, prediction_path: str | Path, gold_path: str | Path) -> dict[str, Any]:
    predictions = {row["record_id"]: row for row in read_jsonl(prediction_path)}
    gold = list(read_jsonl(gold_path))
    if not gold:
        raise ValueError("gold file is empty")
    rows = []
    status_counts: Counter[str] = Counter()
    for item in gold:
        validate_annotation(task, item)
        record_id = item["record_id"]
        prediction = predictions.get(record_id)
        if prediction is None:
            rows.append({"record_id": record_id, "correct": False, "error_category": "missing_prediction"})
            status_counts["missing_prediction"] += 1
            continue
        status = prediction.get("status", "unknown")
        status_counts[status] += 1
        if task == "outcome":
            predicted = prediction.get("selected_group")
            expected = item["gold_group"]
        else:
            predicted = prediction.get("selected_candidate_id")
            expected = item["gold"]["selected_candidate_id"]
        correct = status in {"ok", "abstain"} and predicted == expected
        rows.append(
            {
                "record_id": record_id,
                "expected": expected,
                "predicted": predicted,
                "status": status,
                "correct": correct,
                "error_category": None if correct else classify_error(prediction),
            }
        )
    correct_count = sum(bool(row["correct"]) for row in rows)
    return {
        "status": "newly_executed_not_canonical_until_author_verified",
        "task": task,
        "metric": "exact_match_accuracy",
        "numerator": correct_count,
        "denominator": len(rows),
        "value": correct_count / len(rows),
        "prediction_sha256": sha256_file(prediction_path),
        "gold_sha256": sha256_file(gold_path),
        "prediction_status_counts": dict(sorted(status_counts.items())),
        "items": rows,
    }


def classify_error(prediction: dict[str, Any]) -> str:
    status = prediction.get("status")
    if status == "retrieval_error":
        return "candidate_retrieval"
    if status == "format_error":
        return "output_format"
    if status in {"selector_error", "crosswalk_error"}:
        return status
    if status == "proposed_new_group":
        return "requires_adjudication"
    return "selector_disagreement"


def evaluate_to_file(task: str, prediction_path: str | Path, gold_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    result = evaluate(task, prediction_path, gold_path)
    atomic_write_json(output_path, result)
    return result
