"""Dry-run-first model-size and learning-curve experiment planner."""

from __future__ import annotations

from typing import Any

from .config import load_configs


def experiment_plan() -> dict[str, Any]:
    config = load_configs()["experiments"]
    comparison = [
        {
            "experiment_id": f"model-{model['label']}",
            "family": "model_comparison",
            "model_label": model["label"],
            "model_id": model["model_id"],
            "training_size": 200,
            "heldout_split": config["heldout_split"],
            "metric": config["metric"],
        }
        for model in config["model_comparison"]
    ]
    selected = config["reported_conclusions_only"]["selected_model_label"]
    selected_entry = next(model for model in config["model_comparison"] if model["label"] == selected)
    learning_curve = [
        {
            "experiment_id": f"learning-{size:03d}",
            "family": "learning_curve",
            "model_label": selected,
            "model_id": selected_entry["model_id"],
            "training_size": size,
            "heldout_split": config["heldout_split"],
            "metric": config["metric"],
        }
        for size in config["learning_curve_training_sizes"]
    ]
    cells = comparison + learning_curve
    return {
        "status": "planned_not_executed",
        "cells": cells,
        "cell_count": len(cells),
        "missing_model_ids": sorted({cell["model_label"] for cell in cells if cell["model_id"] is None}),
        "required_before_execution": config["required_before_execution"],
        "reported_conclusions_only": config["reported_conclusions_only"],
    }


def require_executable_plan(plan: dict[str, Any]) -> None:
    if plan["missing_model_ids"]:
        labels = ", ".join(plan["missing_model_ids"])
        raise RuntimeError(
            "experiment execution is blocked because the manuscript labels do not "
            f"establish runnable checkpoint identifiers for: {labels}"
        )
    raise RuntimeError(
        "This command is intentionally a dry-run planner. Execute individual cells "
        "only after binding immutable split/training artifacts and a run manifest."
    )
