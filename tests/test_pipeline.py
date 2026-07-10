from pathlib import Path

import pytest

from clinical_trial_outcome.data.cohort import plan_cohort
from clinical_trial_outcome.data.schemas import PaperExpectedCounts
from clinical_trial_outcome.pipeline import build_plan, run_pipeline


ROOT = Path(__file__).parents[1]


def test_pipeline_defaults_to_dry_run():
    plan = build_plan(ROOT / "configs/paper_pipeline.yaml")
    assert plan["mode"] == "dry-run"
    assert plan["writes_performed"] is False
    assert plan["model"] == "Qwen/Qwen2.5-32B-Instruct"


def test_execute_requires_second_confirmation():
    with pytest.raises(RuntimeError, match="--confirm-full-run"):
        run_pipeline(ROOT / "configs/paper_pipeline.yaml", execute=True)


def test_expected_counts_are_validator_targets():
    expected = PaperExpectedCounts()
    assert expected.validate({"registered_trials": 563_077})


def test_cohort_plan_never_writes(tmp_path):
    plan = plan_cohort(tmp_path / "aact", tmp_path / "derived")
    assert plan["mode"] == "dry-run"
    assert plan["writes_performed"] is False
    assert len(plan["missing_inputs"]) == 3

