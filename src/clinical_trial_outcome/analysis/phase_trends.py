"""Specify trial-level outcome-category prevalence by phase.

EVIDENCE BOUNDARY: ``--execute`` would compute new descriptive values from the
caller's phase field.  Those values are not reproduced paper evidence and must
not be cited as manuscript findings without the locked cohort, metadata, and
lineage-verified archived analysis inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import AnalysisPlan, OutputSchema, dispatch, load_table, prepare_output_dir, print_plan, require_columns, write_csv


def output_schemas() -> tuple[OutputSchema, ...]:
    return (
        OutputSchema(
            "phase_category_prevalence.csv",
            ("phase", "outcome_category", "n_trials_with_category", "total_trials_in_phase", "pct_trials_with_category", "rank_within_phase"),
            "Trial-level category prevalence within each phase.",
        ),
    )


def build_plan(input_path: Path, output_dir: Path, *, trial_column: str = "nct_id", category_column: str = "outcome_category", phase_column: str = "phase_clean") -> AnalysisPlan:
    return AnalysisPlan(
        analysis="phase_trends",
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        outputs=output_schemas(),
        required_columns=(trial_column, category_column, phase_column),
        parameters={"trial_column": trial_column, "category_column": category_column, "phase_column": phase_column},
        notes=("Each trial is counted at most once per phase-category combination.",),
    )


def _execute(plan: AnalysisPlan) -> tuple[Path, ...]:
    trial = str(plan.parameters["trial_column"])
    category = str(plan.parameters["category_column"])
    phase = str(plan.parameters["phase_column"])
    frame = load_table(plan.input_path, plan.required_columns)
    require_columns(frame, plan.required_columns)
    trial_phase = frame[[trial, phase]].dropna().drop_duplicates()
    if trial_phase[trial].duplicated().any():
        raise ValueError("A trial maps to multiple phase values; resolve phase before analysis")
    totals = trial_phase.groupby(phase)[trial].nunique().rename("total_trials_in_phase")
    presence = frame[[trial, phase, category]].dropna().drop_duplicates()
    result = presence.groupby([phase, category])[trial].nunique().rename("n_trials_with_category").reset_index()
    result = result.merge(totals, on=phase, validate="many_to_one")
    result["pct_trials_with_category"] = 100 * result["n_trials_with_category"] / result["total_trials_in_phase"]
    result["rank_within_phase"] = result.groupby(phase)["pct_trials_with_category"].rank(method="min", ascending=False).astype(int)
    result = result.rename(columns={phase: "phase", category: "outcome_category"}).sort_values(["phase", "rank_within_phase", "outcome_category"])
    out = prepare_output_dir(plan.output_dir)
    return (write_csv(result, out / plan.outputs[0].filename),)


def run(input_path: Path, output_dir: Path, *, execute: bool = False, trial_column: str = "nct_id", category_column: str = "outcome_category", phase_column: str = "phase_clean"):
    plan = build_plan(input_path, output_dir, trial_column=trial_column, category_column=category_column, phase_column=phase_column)
    return dispatch(plan, execute=execute, computation=lambda: _execute(plan))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-column", default="nct_id")
    parser.add_argument("--category-column", default="outcome_category")
    parser.add_argument("--phase-column", default="phase_clean")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.input, args.output_dir, execute=args.execute, trial_column=args.trial_column, category_column=args.category_column, phase_column=args.phase_column)
    if isinstance(result, AnalysisPlan):
        print_plan(result)
    else:
        print("\n".join(str(path) for path in result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
