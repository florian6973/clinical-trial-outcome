import json

import pytest

from clinical_trial_outcome.outcome_structuring import (
    OutcomeStructureError,
    build_outcome_structuring_prompt,
    parse_structured_outcome,
    run_outcome_structuring,
)


def test_outcome_structuring_prompt_has_complete_schema() -> None:
    prompt = build_outcome_structuring_prompt(
        "Percentage of participants with adverse events through week 12"
    )
    assert "Quantity or Object of Interest" in prompt
    assert "Quantity Measure" in prompt
    assert "Time" in prompt
    assert "adverse events through week 12" in prompt


def test_outcome_structuring_parser_validates_and_deduplicates() -> None:
    response = json.dumps(
        {
            "Quantity or Object of Interest": ["Adverse Events", "Adverse Events"],
            "Quantity Measure": ["Percentage of Participants"],
            "Time": ["Week 12"],
            "Quantity Unit": [],
            "Quantity Range": [],
            "Additional Constraints": [],
        }
    )
    parsed = parse_structured_outcome(response, source_title="source")
    assert parsed.objects == ("Adverse Events",)
    assert parsed.measures == ("Percentage of Participants",)
    assert parsed.times == ("Week 12",)


def test_outcome_structuring_requires_an_object() -> None:
    response = json.dumps({"Quantity or Object of Interest": []})
    with pytest.raises(OutcomeStructureError, match="At least one outcome object"):
        parse_structured_outcome(response, source_title="source")


def test_outcome_structuring_is_dry_run_by_default() -> None:
    called = False

    def generator(_: str) -> str:
        nonlocal called
        called = True
        raise AssertionError("generator must not run in dry-run mode")

    result = run_outcome_structuring(["Overall survival"], generator=generator)
    assert result["mode"] == "dry_run"
    assert result["execute"] is False
    assert result["writes_performed"] is False
    assert result["model_name"] == "Qwen/Qwen2.5-32B-Instruct"
    assert called is False


def test_outcome_structuring_execution_requires_confirmation() -> None:
    with pytest.raises(PermissionError, match="confirm_execute"):
        run_outcome_structuring(["Overall survival"], execute=True)
