"""Specify annual and pre-specified-period category prevalence.

EVIDENCE BOUNDARY: ``--execute`` would compute new descriptive trends from a
caller-supplied year field.  It is not reproduced paper evidence, and the
pre-specified periods prevent data-driven window selection but do not prove
cohort lineage or reproduce manuscript findings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import AnalysisPlan, OutputSchema, dispatch, load_table, prepare_output_dir, print_plan, require_columns, write_csv


def output_schemas() -> tuple[OutputSchema, ...]:
    return (
        OutputSchema(
            "annual_category_prevalence.csv",
            ("year", "outcome_category", "n_trials_with_category", "total_trials_in_year", "pct_trials_with_category"),
            "Annual trial-level outcome-category prevalence.",
        ),
        OutputSchema(
            "period_category_comparison.csv",
            ("outcome_category", "early_start", "early_end", "early_n_trials", "early_total_trials", "early_pct", "recent_start", "recent_end", "recent_n_trials", "recent_total_trials", "recent_pct", "delta_percentage_points", "relative_change_pct"),
            "Comparison of caller-specified early and recent periods.",
        ),
    )


def build_plan(input_path: Path, output_dir: Path, *, trial_column: str = "nct_id", category_column: str = "outcome_category", year_column: str = "completion_year", early_period: tuple[int, int] = (2005, 2010), recent_period: tuple[int, int] = (2020, 2025)) -> AnalysisPlan:
    if early_period[0] > early_period[1] or recent_period[0] > recent_period[1]:
        raise ValueError("Period start must not exceed period end")
    return AnalysisPlan(
        analysis="temporal_trends",
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        outputs=output_schemas(),
        required_columns=(trial_column, category_column, year_column),
        parameters={"trial_column": trial_column, "category_column": category_column, "year_column": year_column, "early_period": list(early_period), "recent_period": list(recent_period)},
        notes=("Each trial is counted at most once per year-category combination.", "Periods are parameters, never inferred from observed changes."),
    )


def _execute(plan: AnalysisPlan) -> tuple[Path, ...]:
    import numpy as np
    import pandas as pd

    trial = str(plan.parameters["trial_column"])
    category = str(plan.parameters["category_column"])
    year = str(plan.parameters["year_column"])
    early = tuple(plan.parameters["early_period"])
    recent = tuple(plan.parameters["recent_period"])
    frame = load_table(plan.input_path, plan.required_columns)
    require_columns(frame, plan.required_columns)
    frame[year] = pd.to_numeric(frame[year], errors="coerce").astype("Int64")
    trial_year = frame[[trial, year]].dropna().drop_duplicates()
    if trial_year[trial].duplicated().any():
        raise ValueError("A trial maps to multiple years; resolve the temporal field before analysis")
    totals = trial_year.groupby(year)[trial].nunique().rename("total_trials_in_year")
    presence = frame[[trial, year, category]].dropna().drop_duplicates()
    annual = presence.groupby([year, category])[trial].nunique().rename("n_trials_with_category").reset_index()
    annual = annual.merge(totals, on=year, validate="many_to_one")
    annual["pct_trials_with_category"] = 100 * annual["n_trials_with_category"] / annual["total_trials_in_year"]
    annual = annual.rename(columns={year: "year", category: "outcome_category"}).sort_values(["year", "outcome_category"])

    categories = sorted(presence[category].astype(str).unique())
    rows = []
    for value in categories:
        row = {"outcome_category": value}
        for label, period in (("early", early), ("recent", recent)):
            trial_ids = set(trial_year.loc[trial_year[year].between(period[0], period[1]), trial])
            count = presence.loc[(presence[trial].isin(trial_ids)) & presence[category].astype(str).eq(value), trial].nunique()
            row.update({f"{label}_start": period[0], f"{label}_end": period[1], f"{label}_n_trials": int(count), f"{label}_total_trials": len(trial_ids), f"{label}_pct": 100 * count / len(trial_ids) if trial_ids else np.nan})
        row["delta_percentage_points"] = row["recent_pct"] - row["early_pct"]
        row["relative_change_pct"] = 100 * row["delta_percentage_points"] / row["early_pct"] if row["early_pct"] else np.nan
        rows.append(row)
    period = pd.DataFrame(rows).sort_values(["delta_percentage_points", "outcome_category"], ascending=[False, True])
    out = prepare_output_dir(plan.output_dir)
    return (
        write_csv(annual, out / plan.outputs[0].filename),
        write_csv(period, out / plan.outputs[1].filename),
    )


def run(input_path: Path, output_dir: Path, *, execute: bool = False, trial_column: str = "nct_id", category_column: str = "outcome_category", year_column: str = "completion_year", early_period: tuple[int, int] = (2005, 2010), recent_period: tuple[int, int] = (2020, 2025)):
    plan = build_plan(input_path, output_dir, trial_column=trial_column, category_column=category_column, year_column=year_column, early_period=early_period, recent_period=recent_period)
    return dispatch(plan, execute=execute, computation=lambda: _execute(plan))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-column", default="nct_id")
    parser.add_argument("--category-column", default="outcome_category")
    parser.add_argument("--year-column", default="completion_year")
    parser.add_argument("--early-period", nargs=2, type=int, default=(2005, 2010), metavar=("START", "END"))
    parser.add_argument("--recent-period", nargs=2, type=int, default=(2020, 2025), metavar=("START", "END"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.input, args.output_dir, execute=args.execute, trial_column=args.trial_column, category_column=args.category_column, year_column=args.year_column, early_period=tuple(args.early_period), recent_period=tuple(args.recent_period))
    if isinstance(result, AnalysisPlan):
        print_plan(result)
    else:
        print("\n".join(str(path) for path in result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
