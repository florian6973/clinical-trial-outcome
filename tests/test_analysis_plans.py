from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clinical_trial_outcome.analysis import AnalysisPlan  # noqa: E402
from clinical_trial_outcome.analysis import (  # noqa: E402
    category_by_disease,
    category_frequency,
    error_examples,
    outcome_complexity,
    phase_trends,
    temporal_trends,
)


@pytest.mark.parametrize(
    ("runner", "kwargs", "expected_analysis"),
    [
        (outcome_complexity.run, {}, "outcome_complexity"),
        (category_frequency.run, {}, "category_frequency"),
        (category_by_disease.run, {}, "category_by_disease"),
        (phase_trends.run, {}, "phase_trends"),
        (temporal_trends.run, {}, "temporal_trends"),
        (
            error_examples.run,
            {"id_column": "example_id", "text_column": "outcome_title"},
            "qualitative_error_examples",
        ),
    ],
)
def test_dry_run_does_not_read_input_or_create_outputs(tmp_path, runner, kwargs, expected_analysis):
    missing_input = tmp_path / "missing.parquet"
    output_dir = tmp_path / "not-created"

    result = runner(missing_input, output_dir, **kwargs)

    assert isinstance(result, AnalysisPlan)
    assert result.analysis == expected_analysis
    assert result.execute_required is True
    assert result.input_path == missing_input
    assert result.output_dir == output_dir
    assert result.outputs
    assert not output_dir.exists()


def test_all_planned_csv_outputs_have_columns(tmp_path):
    plans = [
        outcome_complexity.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        category_frequency.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        category_by_disease.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        phase_trends.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        temporal_trends.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        error_examples.build_plan(tmp_path / "in.csv", tmp_path / "out", id_column="id", text_column="text"),
    ]
    for plan in plans:
        for schema in plan.outputs:
            assert schema.filename.endswith(".csv")
            assert schema.columns
            assert schema.description


def test_plan_json_is_machine_readable(tmp_path):
    plan = category_by_disease.build_plan(
        tmp_path / "mapping.csv",
        tmp_path / "out",
        record_columns=("nct_id", "aact_outcome_title"),
    )
    payload = json.loads(plan.to_json())
    assert payload["analysis"] == "category_by_disease"
    assert payload["parameters"]["record_columns"] == ["nct_id", "aact_outcome_title"]
    assert payload["execute_required"] is True


def test_category_frequency_validation_is_caller_supplied(tmp_path):
    without_expected = category_frequency.build_plan(tmp_path / "input.csv", tmp_path / "out")
    with_expected = category_frequency.build_plan(
        tmp_path / "input.csv",
        tmp_path / "out",
        expected_counts_path=tmp_path / "locked_expected_counts.csv",
    )
    assert len(without_expected.outputs) == 1
    assert len(with_expected.outputs) == 2
    assert with_expected.parameters["expected_category_count"] == 21
    assert with_expected.parameters["expected_counts_path"].endswith("locked_expected_counts.csv")


def test_temporal_periods_are_prespecified_and_validated(tmp_path):
    plan = temporal_trends.build_plan(
        tmp_path / "input.csv",
        tmp_path / "out",
        early_period=(2005, 2010),
        recent_period=(2020, 2025),
    )
    assert plan.parameters["early_period"] == [2005, 2010]
    assert plan.parameters["recent_period"] == [2020, 2025]
    with pytest.raises(ValueError, match="Period start"):
        temporal_trends.build_plan(tmp_path / "input.csv", tmp_path / "out", early_period=(2010, 2005))


def test_error_examples_never_infer_errors_in_plan(tmp_path):
    plan = error_examples.build_plan(
        tmp_path / "examples.csv",
        tmp_path / "out",
        id_column="id",
        text_column="text",
    )
    assert "error_type" in plan.outputs[0].columns
    assert any("No row is labeled an error automatically" in note for note in plan.notes)


@pytest.mark.parametrize(
    "module",
    [
        outcome_complexity,
        category_frequency,
        category_by_disease,
        phase_trends,
        temporal_trends,
        error_examples,
    ],
)
def test_every_analysis_cli_declares_the_evidence_boundary(module):
    docstring = module.__doc__ or ""
    assert "EVIDENCE BOUNDARY" in docstring
    assert "not reproduced paper evidence" in docstring


def test_dry_run_plans_do_not_embed_reported_performance_values(tmp_path):
    plans = [
        outcome_complexity.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        category_frequency.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        category_by_disease.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        phase_trends.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        temporal_trends.build_plan(tmp_path / "in.csv", tmp_path / "out"),
        error_examples.build_plan(
            tmp_path / "in.csv",
            tmp_path / "out",
            id_column="id",
            text_column="text",
        ),
    ]
    for plan in plans:
        payload = plan.to_json()
        assert '"writes_performed": true' not in payload
        assert "0.88" not in payload
        assert "0.92" not in payload
        assert "0.95" not in payload
