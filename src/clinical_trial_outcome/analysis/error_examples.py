"""Specify a qualitative review template, separate from formal validation.

EVIDENCE BOUNDARY: ``--execute`` only copies caller-supplied examples into a
blank manual-review template.  It does not infer errors, compute an error rate,
and its output is not reproduced paper evidence.  Formal validation still
requires frozen predictions and independent gold labels from the archived
validation set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import AnalysisPlan, OutputSchema, dispatch, load_table, prepare_output_dir, print_plan, require_columns, write_csv


def output_schemas() -> tuple[OutputSchema, ...]:
    return (
        OutputSchema(
            "qualitative_error_review_template.csv",
            ("example_id", "source_text", "predicted_label", "gold_label", "review_status", "error_type", "reviewer_note", "formal_validation_eligible"),
            "Blank manual-review fields appended to caller-provided examples; no error is inferred automatically.",
        ),
    )


def build_plan(input_path: Path, output_dir: Path, *, id_column: str, text_column: str, prediction_column: str | None = None, gold_column: str | None = None, max_examples: int = 100) -> AnalysisPlan:
    required = tuple(c for c in (id_column, text_column, prediction_column, gold_column) if c)
    return AnalysisPlan(
        analysis="qualitative_error_examples",
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        outputs=output_schemas(),
        required_columns=required,
        parameters={"id_column": id_column, "text_column": text_column, "prediction_column": prediction_column, "gold_column": gold_column, "max_examples": max_examples},
        notes=(
            "No row is labeled an error automatically.",
            "Formal validation eligibility is false unless both frozen predictions and independent gold labels are supplied.",
        ),
    )


def _execute(plan: AnalysisPlan) -> tuple[Path, ...]:
    import pandas as pd

    frame = load_table(plan.input_path, plan.required_columns)
    require_columns(frame, plan.required_columns)
    p = plan.parameters
    selected = frame[list(plan.required_columns)].sort_values([str(p["id_column"]), str(p["text_column"])]).head(int(p["max_examples"])).copy()
    result = pd.DataFrame(
        {
            "example_id": selected[str(p["id_column"])].astype(str),
            "source_text": selected[str(p["text_column"])].astype(str),
            "predicted_label": selected[str(p["prediction_column"])].astype(str) if p.get("prediction_column") else "",
            "gold_label": selected[str(p["gold_column"])].astype(str) if p.get("gold_column") else "",
            "review_status": "unreviewed",
            "error_type": "",
            "reviewer_note": "",
            "formal_validation_eligible": bool(p.get("prediction_column") and p.get("gold_column")),
        }
    )
    out = prepare_output_dir(plan.output_dir)
    return (write_csv(result, out / plan.outputs[0].filename),)


def run(input_path: Path, output_dir: Path, *, id_column: str, text_column: str, execute: bool = False, prediction_column: str | None = None, gold_column: str | None = None, max_examples: int = 100):
    plan = build_plan(input_path, output_dir, id_column=id_column, text_column=text_column, prediction_column=prediction_column, gold_column=gold_column, max_examples=max_examples)
    return dispatch(plan, execute=execute, computation=lambda: _execute(plan))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--prediction-column")
    parser.add_argument("--gold-column")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.input, args.output_dir, id_column=args.id_column, text_column=args.text_column, execute=args.execute, prediction_column=args.prediction_column, gold_column=args.gold_column, max_examples=args.max_examples)
    if isinstance(result, AnalysisPlan):
        print_plan(result)
    else:
        print("\n".join(str(path) for path in result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
