import pytest

from clinical_trial_outcome.condition_normalization import (
    ConditionAnnotation,
    DISEASE_AREAS,
    SnomedCandidate,
    train_condition_qwen_lora,
)


def annotations() -> list[ConditionAnnotation]:
    return [
        ConditionAnnotation(
            condition_text=f"condition {index}",
            candidates=(
                SnomedCandidate(
                    concept_id=f"{1_000_000 + index}",
                    term=f"Condition {index}",
                    disease_area=DISEASE_AREAS[0],
                    similarity=0.9,
                    rank=1,
                ),
            ),
            selected_concept_id=f"{1_000_000 + index}",
        )
        for index in range(250)
    ]


def test_condition_training_defaults_to_no_write_plan(tmp_path):
    result = train_condition_qwen_lora(annotations(), output_dir=tmp_path)
    assert result["mode"] == "dry_run"
    assert result["train_examples"] == 200
    assert result["validation_examples"] == 50
    assert result["writes_performed"] is False
    assert not any(tmp_path.iterdir())


def test_condition_training_requires_second_gate(tmp_path):
    with pytest.raises(PermissionError, match="confirm"):
        train_condition_qwen_lora(
            annotations(), output_dir=tmp_path, execute=True
        )
    assert not any(tmp_path.iterdir())


def test_condition_training_execution_can_use_authorized_executor(tmp_path):
    calls: list[tuple[int, int]] = []

    def executor(train, validation, output_dir, config):
        calls.append((len(train), len(validation)))
        assert output_dir == tmp_path
        assert config.model_name == "Qwen/Qwen2.5-32B-Instruct"
        return {"mode": "executed", "writes_performed": True}

    result = train_condition_qwen_lora(
        annotations(),
        output_dir=tmp_path,
        execute=True,
        confirm_execute=True,
        executor=executor,
    )
    assert calls == [(200, 50)]
    assert result == {"mode": "executed", "writes_performed": True}
    assert not any(tmp_path.iterdir())
