"""Dry-run-first orchestration for the paper-aligned pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = (
    "cohort",
    "outcome_structuring",
    "outcome_normalization",
    "condition_normalization",
    "classification",
    "evaluation",
    "analysis",
)


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Pipeline configuration must be a YAML mapping")
    return loaded


def build_plan(config_path: str | Path, *, stage: str = "all") -> dict[str, Any]:
    config = load_config(config_path)
    selected = list(STAGES) if stage == "all" else [stage]
    unknown = [name for name in selected if name not in STAGES]
    if unknown:
        raise ValueError(f"Unknown pipeline stage: {unknown[0]}")
    return {
        "mode": "dry-run",
        "config": str(Path(config_path)),
        "stages": selected,
        "model": config.get("models", {}).get("qwen", {}).get("name"),
        "expected_outputs": config.get("outputs", {}),
        "writes_performed": False,
        "notice": "Published values are validation targets; this plan does not recompute them.",
    }


def run_pipeline(
    config_path: str | Path,
    *,
    stage: str = "all",
    execute: bool = False,
    confirm_full_run: bool = False,
) -> dict[str, Any]:
    plan = build_plan(config_path, stage=stage)
    if not execute:
        return plan
    if not confirm_full_run:
        raise RuntimeError("Execution requires both --execute and --confirm-full-run")
    raise NotImplementedError(
        "Full orchestration is intentionally gated. Run the documented stage commands "
        "after supplying licensed inputs and local compute."
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", default="configs/paper_pipeline.yaml")
    command.add_argument("--stage", choices=("all", *STAGES), default="all")
    command.add_argument("--execute", action="store_true")
    command.add_argument("--confirm-full-run", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run_pipeline(
        args.config,
        stage=args.stage,
        execute=args.execute,
        confirm_full_run=args.confirm_full_run,
    )
    print(json.dumps(result, indent=2))
    return 0

