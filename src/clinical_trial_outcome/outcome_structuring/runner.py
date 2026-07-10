"""Dry-run-first execution wrapper for outcome-title structuring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from .schema import StructuredOutcome, build_outcome_structuring_prompt, parse_structured_outcome


@dataclass(frozen=True)
class StructuringPlan:
    mode: str
    operation: str
    stages: tuple[str, ...]
    model_name: str
    number_of_titles: int
    max_length: int
    writes_performed: bool
    execute: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_outcome_structuring(
    titles: Sequence[str],
    *,
    model_name: str = "Qwen/Qwen2.5-32B-Instruct",
    max_length: int = 512,
) -> StructuringPlan:
    if not titles:
        raise ValueError("At least one outcome title is required")
    if any(not title.strip() for title in titles):
        raise ValueError("Outcome titles must not be empty")
    return StructuringPlan(
        mode="dry_run",
        operation="outcome_structuring",
        stages=("build_structuring_prompt", "generate_json", "validate_outcome_schema"),
        model_name=model_name,
        number_of_titles=len(titles),
        max_length=max_length,
        writes_performed=False,
    )


def run_outcome_structuring(
    titles: Sequence[str],
    *,
    execute: bool = False,
    confirm_execute: bool = False,
    generator: Callable[[str], str] | None = None,
) -> dict[str, object] | list[StructuredOutcome]:
    """Plan by default; execute only with two explicit flags and a generator.

    The caller owns model loading.  This keeps imports safe and prevents an
    accidental Qwen inference run merely by importing or calling this package.
    """

    plan = plan_outcome_structuring(titles)
    if not execute:
        return plan.to_dict()
    if not confirm_execute:
        raise PermissionError("Execution requires confirm_execute=True")
    if generator is None:
        raise ValueError("An explicit model generator callable is required")
    results: list[StructuredOutcome] = []
    for title in titles:
        response = generator(build_outcome_structuring_prompt(title))
        results.append(parse_structured_outcome(response, source_title=title))
    return results
