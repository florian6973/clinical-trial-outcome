from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clinical_trial_outcome.analysis import category_frequency, outcome_complexity  # noqa: E402


def test_cli_defaults_to_dry_run_for_missing_input(tmp_path, capsys):
    output_dir = tmp_path / "out"
    exit_code = outcome_complexity.main(
        [
            "--input",
            str(tmp_path / "does-not-exist.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["analysis"] == "outcome_complexity"
    assert payload["execute_required"] is True
    assert not output_dir.exists()


def test_category_cli_reports_expected_counts_as_input_not_result(tmp_path, capsys):
    expected = tmp_path / "locked_counts.csv"
    category_frequency.main(
        [
            "--input",
            str(tmp_path / "mapping.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--expected-counts",
            str(expected),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["parameters"]["expected_counts_path"] == str(expected)
    assert payload["parameters"]["expected_category_count"] == 21
    assert not (tmp_path / "out").exists()


def test_cli_help_retains_the_explicit_execute_gate(capsys):
    with pytest.raises(SystemExit) as exc_info:
        outcome_complexity.main(["--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--execute" in help_text
    assert "dry-run" in help_text
