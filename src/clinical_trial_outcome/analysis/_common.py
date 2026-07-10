"""Shared helpers for analysis specifications, not reproduced paper evidence.

EVIDENCE BOUNDARY: ``--execute`` authorizes computation on caller-supplied
data, but it does not authenticate lineage or reproduce manuscript results.
Dry-run plans contain schemas and parameters only; they must not contain
reported constants, metrics, predictions, or claims of reproduction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar


@dataclass(frozen=True)
class OutputSchema:
    """A planned artifact and its machine-readable column contract."""

    filename: str
    columns: tuple[str, ...]
    description: str
    artifact_type: str = "csv"


@dataclass(frozen=True)
class AnalysisPlan:
    """A side-effect-free specification, distinct from a paper result artifact."""

    analysis: str
    input_path: Path
    output_dir: Path
    outputs: tuple[OutputSchema, ...]
    required_columns: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    execute_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


T = TypeVar("T")


def dispatch(plan: AnalysisPlan, *, execute: bool, computation: Callable[[], T]) -> AnalysisPlan | T:
    """Return a plan unless computation is authorized; authorization is not evidence."""

    if not execute:
        return plan
    return computation()


def load_table(path: Path, columns: Sequence[str] | None = None):
    """Load CSV or Parquet lazily after the execute gate has been crossed."""

    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=list(columns) if columns else None)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    raise ValueError(f"Unsupported input type {suffix!r}; expected CSV or Parquet")


def require_columns(frame, required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame, path: Path) -> Path:
    """Write one computed table after output authorization."""

    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def print_plan(plan: AnalysisPlan) -> None:
    print(plan.to_json())
