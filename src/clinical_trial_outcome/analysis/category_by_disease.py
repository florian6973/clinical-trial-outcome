"""Specify a category-by-disease descriptive association analysis.

EVIDENCE BOUNDARY: ``--execute`` would calculate a new descriptive association
from caller-supplied rows.  It is not reproduced paper evidence, does not
establish the locked 21-by-24 manuscript result, and cannot replace archived
analysis outputs with verified record-unit and lineage contracts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ._common import AnalysisPlan, OutputSchema, dispatch, load_table, prepare_output_dir, print_plan, require_columns, write_csv


def output_schemas() -> tuple[OutputSchema, ...]:
    return (
        OutputSchema("category_by_disease_counts.csv", ("disease_domain", "<one column per outcome category>"), "Observed contingency counts."),
        OutputSchema("category_by_disease_within_domain_pct.csv", ("disease_domain", "<one column per outcome category>"), "Within-domain row percentages."),
        OutputSchema("category_by_disease_standardized_residuals.csv", ("disease_domain", "<one column per outcome category>"), "Pearson standardized residuals."),
        OutputSchema(
            "category_by_disease_association_summary.csv",
            ("analysis_unit", "n_records", "n_disease_domains", "n_outcome_categories", "chi_square", "degrees_of_freedom", "p_value", "cramers_v", "inference_caution"),
            "Global descriptive association statistics.",
        ),
    )


def build_plan(input_path: Path, output_dir: Path, *, category_column: str = "outcome_category", disease_column: str = "snomed_condition_category_description", record_columns: tuple[str, ...] = ()) -> AnalysisPlan:
    return AnalysisPlan(
        analysis="category_by_disease",
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        outputs=output_schemas(),
        required_columns=tuple(dict.fromkeys((*record_columns, category_column, disease_column))),
        parameters={"category_column": category_column, "disease_column": disease_column, "record_columns": list(record_columns)},
        notes=(
            "Caller defines the record unit through record_columns; duplicates on that unit are removed.",
            "Chi-square p-values are descriptive unless independence/cluster assumptions are justified externally.",
        ),
    )


def _regularized_gamma_q(shape: float, value: float) -> float:
    """Regularized upper incomplete gamma, used for the chi-square survival function."""

    if value < 0 or shape <= 0:
        raise ValueError("Gamma arguments must be positive")
    if value == 0:
        return 1.0
    epsilon = 3.0e-14
    tiny = 1.0e-300
    iterations = 10_000
    log_term = -value + shape * math.log(value) - math.lgamma(shape)
    if value < shape + 1.0:
        total = 1.0 / shape
        term = total
        parameter = shape
        for _ in range(iterations):
            parameter += 1.0
            term *= value / parameter
            total += term
            if abs(term) < abs(total) * epsilon:
                return max(0.0, min(1.0, 1.0 - total * math.exp(log_term)))
        raise ArithmeticError("Incomplete-gamma series failed to converge")
    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for index in range(1, iterations + 1):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        d = tiny if abs(d) < tiny else d
        c = b + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) < epsilon:
            return max(0.0, min(1.0, fraction * math.exp(log_term)))
    raise ArithmeticError("Incomplete-gamma continued fraction failed to converge")


def _execute(plan: AnalysisPlan) -> tuple[Path, ...]:
    import numpy as np
    import pandas as pd
    category = str(plan.parameters["category_column"])
    disease = str(plan.parameters["disease_column"])
    record_columns = list(plan.parameters["record_columns"])
    frame = load_table(plan.input_path, plan.required_columns)
    require_columns(frame, plan.required_columns)
    records = frame[[*record_columns, category, disease]].dropna(subset=[category, disease]).drop_duplicates()
    table = pd.crosstab(records[disease], records[category]).sort_index(axis=0).sort_index(axis=1)
    observed = table.to_numpy(dtype=float)
    n = int(observed.sum())
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi2 = float((np.square(observed - expected) / expected).sum())
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    p_value = _regularized_gamma_q(dof / 2.0, chi2 / 2.0)
    cramers_v = math.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1)))
    proportions = table.div(table.sum(axis=1), axis=0) * 100
    residuals = pd.DataFrame((table.to_numpy() - expected) / np.sqrt(expected), index=table.index, columns=table.columns)
    summary = pd.DataFrame(
        [
            {
                "analysis_unit": "+".join(record_columns) if record_columns else "input row",
                "n_records": n,
                "n_disease_domains": table.shape[0],
                "n_outcome_categories": table.shape[1],
                "chi_square": chi2,
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "cramers_v": cramers_v,
                "inference_caution": "Descriptive unless record independence and clustering assumptions are justified.",
            }
        ]
    )
    out = prepare_output_dir(plan.output_dir)
    return (
        write_csv(table.reset_index(), out / plan.outputs[0].filename),
        write_csv(proportions.reset_index(), out / plan.outputs[1].filename),
        write_csv(residuals.reset_index(), out / plan.outputs[2].filename),
        write_csv(summary, out / plan.outputs[3].filename),
    )


def run(input_path: Path, output_dir: Path, *, execute: bool = False, category_column: str = "outcome_category", disease_column: str = "snomed_condition_category_description", record_columns: tuple[str, ...] = ()):
    plan = build_plan(input_path, output_dir, category_column=category_column, disease_column=disease_column, record_columns=record_columns)
    return dispatch(plan, execute=execute, computation=lambda: _execute(plan))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category-column", default="outcome_category")
    parser.add_argument("--disease-column", default="snomed_condition_category_description")
    parser.add_argument("--record-column", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.input, args.output_dir, execute=args.execute, category_column=args.category_column, disease_column=args.disease_column, record_columns=tuple(args.record_column))
    if isinstance(result, AnalysisPlan):
        print_plan(result)
    else:
        print("\n".join(str(path) for path in result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
