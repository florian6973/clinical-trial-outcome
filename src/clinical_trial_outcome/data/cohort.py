"""Cohort specification corresponding to the manuscript."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .aact import missing_tables, resolve_aact_tables


@dataclass(frozen=True)
class CohortSpec:
    registered_from: str = "2000-01-01"
    registered_through: str = "2024-12-31"
    study_type: str = "INTERVENTIONAL"
    require_results: bool = True


def plan_cohort(aact_dir: str | Path, output_path: str | Path, spec: CohortSpec | None = None) -> dict[str, Any]:
    tables = resolve_aact_tables(aact_dir)
    return {
        "mode": "dry-run",
        "inputs": {name: str(path) for name, path in tables.items()},
        "missing_inputs": missing_tables(tables),
        "output": str(output_path),
        "filters": asdict(spec or CohortSpec()),
        "writes_performed": False,
    }


def build_cohort(
    aact_dir: str | Path,
    output_path: str | Path,
    *,
    spec: CohortSpec | None = None,
    execute: bool = False,
):
    """Build the cohort only after explicit execution authorization."""

    plan = plan_cohort(aact_dir, output_path, spec)
    if not execute:
        return plan
    if plan["missing_inputs"]:
        raise FileNotFoundError(f"Missing AACT tables: {', '.join(plan['missing_inputs'])}")

    from .aact import read_aact_table

    current = spec or CohortSpec()
    tables = {name: Path(path) for name, path in plan["inputs"].items()}
    studies = read_aact_table(tables["studies"])
    outcomes = read_aact_table(tables["outcomes"])
    conditions = read_aact_table(tables["conditions"])

    study_type = studies.get("study_type", "").astype(str).str.upper()
    mask = study_type.eq(current.study_type)
    if current.require_results and "results_first_posted_date" in studies:
        mask &= studies["results_first_posted_date"].notna()
    cohort_studies = studies.loc[mask].copy()
    ids = set(cohort_studies["nct_id"])
    result = {
        "studies": cohort_studies,
        "outcomes": outcomes[outcomes["nct_id"].isin(ids)].copy(),
        "conditions": conditions[conditions["nct_id"].isin(ids)].copy(),
    }
    destination = Path(output_path)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    return {**plan, "mode": "execute", "writes_performed": True}

