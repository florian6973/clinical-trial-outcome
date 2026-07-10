import json
from pathlib import Path

import pytest

from clinical_trial_outcome.classification import (
    COA_TYPES,
    OUTCOME_CATEGORIES,
    OutcomeClassificationPipeline,
    build_classification_prompt,
    parse_classification_response,
)


def test_classification_taxonomies_are_locked_to_21_and_4() -> None:
    assert len(OUTCOME_CATEGORIES) == 21
    assert len(set(OUTCOME_CATEGORIES)) == 21
    assert len(COA_TYPES) == 4
    assert len(set(COA_TYPES)) == 4


def test_classification_pipeline_is_dry_run_by_default() -> None:
    plan = OutcomeClassificationPipeline().classify("quality of life")

    assert plan["mode"] == "dry_run"
    assert plan["execution_required"] is True
    assert plan["outcome_category_count"] == 21
    assert plan["coa_type_count"] == 4


def test_classification_execute_constrains_both_labels() -> None:
    def qwen(system: str, prompt: str) -> str:
        assert "Adverse Events/Safety" in prompt
        assert "Patient-Reported Outcome (PRO)" in prompt
        return json.dumps(
            {
                "outcome_category": "Quality of Life/PRO",
                "coa_type": "Patient-Reported Outcome (PRO)",
            }
        )

    result = OutcomeClassificationPipeline(qwen_generator=qwen).classify(
        "health-related quality of life", execute=True, confirm_execute=True
    )

    assert result["outcome_category"] == "Quality of Life/PRO"
    assert result["coa_type"] == "Patient-Reported Outcome (PRO)"
    assert result["classification_method"] == "qwen_constrained_taxonomy_selection"


def test_classification_rejects_unlocked_label() -> None:
    with pytest.raises(ValueError, match="Unknown outcome category"):
        parse_classification_response(
            json.dumps(
                {
                    "outcome_category": "Invented category",
                    "coa_type": "Patient-Reported Outcome (PRO)",
                }
            )
        )


def test_classification_prompt_contains_all_labels() -> None:
    prompt = build_classification_prompt("overall survival")

    assert all(category in prompt for category in OUTCOME_CATEGORIES)
    assert all(coa_type in prompt for coa_type in COA_TYPES)


def test_classification_expected_results_are_validation_only() -> None:
    payload = json.loads(Path("paper/expected_results.json").read_text())

    assert payload["validation_expectations_only"] is True
    assert payload["recompute"] is False
    assert payload["annotation_contract"] == {"train_size": 200, "validation_size": 50}
    assert payload["model_comparison"]["checkpoint_count"] == 6
    assert payload["model_comparison"]["learning_sizes"] == [0, 50, 100, 150, 200]


def test_classification_model_comparison_config_has_required_grid() -> None:
    text = Path("configs/model_comparison.yaml").read_text()

    for size in ("7B", "14B", "32B", "8B", "15B", "34B"):
        assert f"size: {size}" in text
    assert "learning_sizes: [0, 50, 100, 150, 200]" in text
    assert "validation_size: 50" in text
    assert "recompute_reported_constants: false" in text
