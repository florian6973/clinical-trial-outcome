"""Specify category-frequency auditing; dry-run is the default.

EVIDENCE BOUNDARY: ``--execute`` would count categories in the caller's input,
and its output is not reproduced paper evidence.  The optional expected-count
file must be an existing locked artifact; this module never derives or writes
expectations to ``paper/expected_results.json``.  An exact match is validation,
not proof of reproduction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import AnalysisPlan, OutputSchema, dispatch, load_table, prepare_output_dir, print_plan, require_columns, write_csv


EXPECTED_CATEGORY_COUNT = 21


def output_schemas(include_validation: bool = False) -> tuple[OutputSchema, ...]:
    schemas = [
        OutputSchema(
            "category_frequency.csv",
            ("rank", "outcome_category", "n_records", "pct_records"),
            "All observed categories ranked by input-record frequency.",
        )
    ]
    if include_validation:
        schemas.append(
            OutputSchema(
                "category_frequency_validation.csv",
                ("outcome_category", "expected_n_records", "observed_n_records", "difference", "exact_match"),
                "Comparison against caller-supplied expected counts; does not define new expected findings.",
            )
        )
    return tuple(schemas)


def build_plan(input_path: Path, output_dir: Path, *, category_column: str = "outcome_category", expected_counts_path: Path | None = None) -> AnalysisPlan:
    return AnalysisPlan(
        analysis="category_frequency",
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        outputs=output_schemas(expected_counts_path is not None),
        required_columns=(category_column,),
        parameters={"category_column": category_column, "expected_counts_path": str(expected_counts_path) if expected_counts_path else None, "expected_category_count": EXPECTED_CATEGORY_COUNT},
        notes=(
            "The expected value of 21 validates category-system completeness, not category frequencies.",
            "Expected counts, when used, must be supplied by the caller from an existing locked artifact.",
        ),
    )


def _execute(plan: AnalysisPlan) -> tuple[Path, ...]:
    import pandas as pd

    category_column = str(plan.parameters["category_column"])
    frame = load_table(plan.input_path, [category_column])
    require_columns(frame, [category_column])
    counts = frame[category_column].dropna().astype(str).value_counts().rename_axis("outcome_category").reset_index(name="n_records")
    if len(counts) != EXPECTED_CATEGORY_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_CATEGORY_COUNT} categories, observed {len(counts)}")
    counts["pct_records"] = 100 * counts["n_records"] / counts["n_records"].sum()
    counts = counts.sort_values(["n_records", "outcome_category"], ascending=[False, True]).reset_index(drop=True)
    counts.insert(0, "rank", range(1, len(counts) + 1))
    out = prepare_output_dir(plan.output_dir)
    paths = [write_csv(counts, out / plan.outputs[0].filename)]
    expected_path = plan.parameters.get("expected_counts_path")
    if expected_path:
        expected = load_table(Path(str(expected_path)))
        require_columns(expected, ["outcome_category", "n_records"])
        expected = expected[["outcome_category", "n_records"]].rename(columns={"n_records": "expected_n_records"})
        validation = expected.merge(counts[["outcome_category", "n_records"]], on="outcome_category", how="outer").rename(columns={"n_records": "observed_n_records"})
        validation[["expected_n_records", "observed_n_records"]] = validation[["expected_n_records", "observed_n_records"]].fillna(0).astype(int)
        validation["difference"] = validation["observed_n_records"] - validation["expected_n_records"]
        validation["exact_match"] = validation["difference"].eq(0)
        paths.append(write_csv(validation.sort_values("outcome_category"), out / plan.outputs[1].filename))
    return tuple(paths)


def run(input_path: Path, output_dir: Path, *, execute: bool = False, category_column: str = "outcome_category", expected_counts_path: Path | None = None):
    plan = build_plan(input_path, output_dir, category_column=category_column, expected_counts_path=expected_counts_path)
    return dispatch(plan, execute=execute, computation=lambda: _execute(plan))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category-column", default="outcome_category")
    parser.add_argument("--expected-counts", type=Path, help="Optional existing 21-category count table")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.input, args.output_dir, execute=args.execute, category_column=args.category_column, expected_counts_path=args.expected_counts)
    if isinstance(result, AnalysisPlan):
        print_plan(result)
    else:
        print("\n".join(str(path) for path in result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
