"""Convert fixed-split annotations into Qwen chat records."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Iterator

from .prompts import (
    CONDITION_SYSTEM,
    OUTCOME_SYSTEM,
    condition_assistant,
    condition_user,
    outcome_assistant,
    outcome_user,
)
from .validation import validate_annotation


def prepare_record(task: str, record: dict[str, Any]) -> dict[str, Any]:
    validate_annotation(task, record)
    if task == "outcome":
        messages = [
            {"role": "system", "content": OUTCOME_SYSTEM},
            {"role": "user", "content": outcome_user(record)},
            {"role": "assistant", "content": outcome_assistant(record["gold_group"])},
        ]
    elif task == "condition":
        messages = [
            {"role": "system", "content": CONDITION_SYSTEM},
            {"role": "user", "content": condition_user(record)},
            {"role": "assistant", "content": condition_assistant(record["gold"])},
        ]
    else:
        raise ValueError(f"unsupported task: {task}")
    return {
        "record_id": record["record_id"],
        "split": record["split"],
        "task": task,
        "messages": messages,
    }


def prepare_records(task: str, records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for record in records:
        record_id = record.get("record_id")
        if record_id in seen:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        yield prepare_record(task, record)


def summarize_splits(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["split"] for record in records)
    return {split: counts.get(split, 0) for split in ("train", "validation", "test")}
