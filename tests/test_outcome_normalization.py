import pytest

from clinical_trial_outcome.outcome_normalization.config import QwenLoRAConfig, RetrievalConfig
from clinical_trial_outcome.outcome_normalization.data import (
    OutcomeAnnotation,
    build_split_manifest,
    split_annotated_examples,
)
from clinical_trial_outcome.outcome_normalization.inference import (
    build_inference_plan,
    normalize_outcome_objects,
)
from clinical_trial_outcome.outcome_normalization.parser import (
    GroupParseError,
    parse_group_response,
)
from clinical_trial_outcome.outcome_normalization.prompt import build_group_prompt
from clinical_trial_outcome.outcome_normalization.retrieval import retrieve_context
from clinical_trial_outcome.outcome_normalization import training as outcome_training
from clinical_trial_outcome.outcome_normalization.training import train_qwen_lora


def annotations() -> list[OutcomeAnnotation]:
    return [
        OutcomeAnnotation(
            term=f"outcome {index}",
            group=f"group {index % 10}",
            prompt=f"prompt {index}",
        )
        for index in range(250)
    ]


def retrieval_context():
    labels = [f"candidate {index}" for index in range(8)]
    vectors = [[1.0, index / 10] for index in range(8)]
    return retrieve_context(
        [1.0, 0.4],
        term_labels=labels,
        term_vectors=vectors,
        group_labels=[f"group {index}" for index in range(8)],
        group_vectors=vectors,
    )


def test_outcome_qwen_configuration_matches_paper() -> None:
    config = QwenLoRAConfig()
    assert config.model_name == "Qwen/Qwen2.5-32B-Instruct"
    assert config.load_in_8bit is True
    assert config.max_length == 512
    assert (config.lora_r, config.lora_alpha, config.lora_dropout) == (8, 32, 0.1)
    assert (config.optimizer, config.learning_rate) == ("AdamW", 1e-5)
    assert (config.per_device_batch_size, config.num_train_epochs) == (1, 3)
    assert (config.train_size, config.validation_size) == (200, 50)


def test_outcome_split_is_exact_disjoint_and_deterministic() -> None:
    rows = annotations()
    train_a, validation_a = split_annotated_examples(rows)
    train_b, validation_b = split_annotated_examples(rows)
    assert (len(train_a), len(validation_a)) == (200, 50)
    assert train_a == train_b
    assert validation_a == validation_b
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a).union(validation_a) == set(rows)


def test_outcome_split_rejects_wrong_annotation_count() -> None:
    with pytest.raises(ValueError, match="exactly 250"):
        split_annotated_examples(annotations()[:-1])


def test_outcome_split_manifest_freezes_membership_and_hash() -> None:
    rows = annotations()
    manifest_a = build_split_manifest(rows)
    manifest_b = build_split_manifest(rows)
    assert len(manifest_a.training_ids) == 200
    assert len(manifest_a.validation_ids) == 50
    assert manifest_a.manifest_sha256 == manifest_b.manifest_sha256
    assert set(manifest_a.training_ids).isdisjoint(manifest_a.validation_ids)
    train, validation = split_annotated_examples(rows, manifest=manifest_a)
    assert (len(train), len(validation)) == (200, 50)


def test_outcome_retrieval_returns_top_five_terms_and_groups() -> None:
    context = retrieval_context()
    assert len(context.terms) == RetrievalConfig().top_k_terms == 5
    assert len(context.groups) == RetrievalConfig().top_k_groups == 5
    assert list(context.terms) == sorted(context.terms, key=lambda item: item.score, reverse=True)


def test_outcome_prompt_and_group_parser_contract() -> None:
    prompt = build_group_prompt("response", retrieval_context())
    assert "Five similar terms" in prompt
    assert "Five candidate groups" in prompt
    assert "GROUP: [group name]" in prompt
    assert parse_group_response("GROUP: Clinical Response</s>") == "Clinical Response"
    with pytest.raises(GroupParseError):
        parse_group_response("Clinical Response")
    with pytest.raises(GroupParseError):
        parse_group_response("GROUP: Response\nGROUP: Clinical Response")


def test_outcome_training_is_dry_run_by_default() -> None:
    result = train_qwen_lora(annotations(), output_dir="outputs/qwen_adapter")
    assert result["mode"] == "dry_run"
    assert result["train_examples"] == 200
    assert result["validation_examples"] == 50
    assert result["validation_policy"].startswith("held_out_evaluation_only")
    assert len(result["split_manifest_sha256"]) == 64
    assert result["writes_performed"] is False
    assert result["execute"] is False


def test_outcome_training_execute_requires_second_confirmation() -> None:
    with pytest.raises(PermissionError, match="confirm_execute"):
        train_qwen_lora(annotations(), output_dir="outputs/qwen_adapter", execute=True)


def test_outcome_training_confirmed_branch_uses_lazy_executor(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = outcome_training.TrainingRunSummary(
        output_dir="outputs/qwen_adapter/qwen_finetuned_32b",
        adapter_saved=True,
        train_examples=200,
        validation_examples=50,
        global_step=600,
        training_loss=0.1,
        evaluation_metrics={"eval_loss": 0.2},
        split_manifest_sha256="a" * 64,
    )

    def fake_execute(train_records, validation_records, **kwargs):
        captured["train_records"] = train_records
        captured["validation_records"] = validation_records
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(outcome_training, "_execute_training", fake_execute)
    result = train_qwen_lora(
        annotations(),
        output_dir="outputs/qwen_adapter",
        execute=True,
        confirm_execute=True,
    )
    assert result is sentinel
    assert len(captured["train_records"]) == 200
    assert len(captured["validation_records"]) == 50
    assert captured["config"].to_dict()["model_name"] == "Qwen/Qwen2.5-32B-Instruct"


def test_outcome_inference_is_dry_run_by_default() -> None:
    plan = normalize_outcome_objects(
        ["response"],
        [[1.0, 0.0]],
        term_labels=[f"term {index}" for index in range(5)],
        term_vectors=[[1.0, index / 10] for index in range(5)],
        group_labels=[f"group {index}" for index in range(5)],
        group_vectors=[[1.0, index / 10] for index in range(5)],
    )
    assert plan["mode"] == "dry_run"
    assert plan["execute"] is False
    assert plan["writes_performed"] is False
    assert plan["retrieval"]["top_k_terms"] == 5
    assert plan["retrieval"]["top_k_groups"] == 5


def test_outcome_inference_execute_requires_second_confirmation() -> None:
    with pytest.raises(PermissionError, match="confirm_execute"):
        normalize_outcome_objects(
            ["response"],
            [[1.0, 0.0]],
            term_labels=[],
            term_vectors=[],
            group_labels=[],
            group_vectors=[],
            execute=True,
        )


def test_outcome_inference_requires_injected_generator() -> None:
    with pytest.raises(RuntimeError, match="injected Qwen generator"):
        normalize_outcome_objects(
            ["response"],
            [[1.0, 0.0]],
            term_labels=[f"term {index}" for index in range(5)],
            term_vectors=[[1.0, index / 10] for index in range(5)],
            group_labels=[f"group {index}" for index in range(5)],
            group_vectors=[[1.0, index / 10] for index in range(5)],
            adapter_path="models/qwen_adapter",
            execute=True,
            confirm_execute=True,
        )


def test_outcome_inference_accepts_only_injected_generator() -> None:
    calls: list[tuple[str, str]] = []

    def generator(system: str, prompt: str) -> str:
        calls.append((system, prompt))
        return "GROUP: Clinical Response"

    result = normalize_outcome_objects(
        ["response"],
        [[1.0, 0.0]],
        term_labels=[f"term {index}" for index in range(5)],
        term_vectors=[[1.0, index / 10] for index in range(5)],
        group_labels=[f"group {index}" for index in range(5)],
        group_vectors=[[1.0, index / 10] for index in range(5)],
        adapter_path="models/qwen_adapter",
        qwen_generator=generator,
        execute=True,
        confirm_execute=True,
    )
    assert len(calls) == 1
    assert result[0].group == "Clinical Response"
    assert len(result[0].similar_terms) == 5
    assert len(result[0].candidate_groups) == 5


def test_outcome_inference_plan_serializes_without_secrets() -> None:
    plan = build_inference_plan(["pain"], adapter_path="models/qwen_adapter")
    payload = plan.to_dict()
    assert payload["adapter_path"] == "models/qwen_adapter"
    rendered = repr(payload).casefold()
    assert "access_token" not in rendered
    assert "hf_token" not in rendered
    assert "api_key" not in rendered
