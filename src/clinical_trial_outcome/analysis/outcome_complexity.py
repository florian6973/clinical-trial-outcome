"""Specify an outcome-title complexity analysis; dry-run is the default.

EVIDENCE BOUNDARY: ``--execute`` would compute a new descriptive table from a
caller-supplied input.  That table is not reproduced paper evidence and must
not be used to populate reported manuscript constants without the archived,
lineage-verified analysis source required by the paper contract.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ._common import AnalysisPlan, OutputSchema, dispatch, load_table, prepare_output_dir, print_plan, require_columns, write_csv


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def output_schemas() -> tuple[OutputSchema, ...]:
    return (
        OutputSchema(
            "outcome_complexity_records.csv",
            ("outcome_id", "outcome_title", "word_count", "character_count", "has_temporal_expression", "has_measurement_operator"),
            "Record-level lexical complexity measures; no model-quality claim.",
        ),
        OutputSchema(
            "outcome_complexity_summary.csv",
            ("measure", "n", "mean", "sd", "min", "p05", "p25", "median", "p75", "p95", "p99", "max"),
            "Distribution summary for words and characters per title.",
        ),
    )


def build_plan(input_path: Path, output_dir: Path, *, title_column: str = "aact_outcome_title", id_column: str | None = None) -> AnalysisPlan:
    return AnalysisPlan(
        analysis="outcome_complexity",
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        outputs=output_schemas(),
        required_columns=tuple(c for c in (id_column, title_column) if c),
        parameters={"title_column": title_column, "id_column": id_column},
        notes=(
            "Lexical counts are descriptive and do not constitute normalization validation.",
            "Dry-run does not inspect the input or create the output directory.",
        ),
    )


def _summary(series, measure: str) -> dict[str, float | int | str]:
    return {
        "measure": measure,
        "n": int(series.notna().sum()),
        "mean": float(series.mean()),
        "sd": float(series.std(ddof=1)),
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def _execute(plan: AnalysisPlan) -> tuple[Path, ...]:
    import pandas as pd

    title_column = str(plan.parameters["title_column"])
    id_column = plan.parameters.get("id_column")
    frame = load_table(plan.input_path, plan.required_columns)
    require_columns(frame, plan.required_columns)
    text = frame[title_column].fillna("").astype(str)
    records = pd.DataFrame(
        {
            "outcome_id": frame[id_column].astype(str) if id_column else frame.index.astype(str),
            "outcome_title": text,
            "word_count": text.map(lambda value: len(WORD_RE.findall(value))),
            "character_count": text.str.len(),
            "has_temporal_expression": text.str.contains(r"\b(?:day|week|month|year|hour)s?\b|\btime to\b", case=False, regex=True),
            "has_measurement_operator": text.str.contains(r"\b(?:change|number|percentage|rate|mean|score|concentration|level)\b", case=False, regex=True),
        }
    )
    summary = pd.DataFrame(
        [
            _summary(records["word_count"], "words_per_outcome_title"),
            _summary(records["character_count"], "characters_per_outcome_title"),
        ]
    )
    out = prepare_output_dir(plan.output_dir)
    return (
        write_csv(records, out / plan.outputs[0].filename),
        write_csv(summary, out / plan.outputs[1].filename),
    )


def run(input_path: Path, output_dir: Path, *, execute: bool = False, title_column: str = "aact_outcome_title", id_column: str | None = None):
    plan = build_plan(input_path, output_dir, title_column=title_column, id_column=id_column)
    return dispatch(plan, execute=execute, computation=lambda: _execute(plan))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-column", default="aact_outcome_title")
    parser.add_argument("--id-column")
    parser.add_argument("--execute", action="store_true", help="Read data and write computed outputs; absent means dry-run")
    args = parser.parse_args(argv)
    result = run(args.input, args.output_dir, execute=args.execute, title_column=args.title_column, id_column=args.id_column)
    if isinstance(result, AnalysisPlan):
        print_plan(result)
    else:
        print("\n".join(str(path) for path in result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
