"""Guarded LoRA training for the paper's separate condition Qwen selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Sequence

from clinical_trial_outcome.outcome_normalization.config import QwenLoRAConfig

from .contract import SnomedCandidate
from .prompt import build_condition_selection_prompt


@dataclass(frozen=True)
class ConditionAnnotation:
    """One frozen condition-selection annotation."""

    condition_text: str
    candidates: tuple[SnomedCandidate, ...]
    selected_concept_id: str

    def __post_init__(self) -> None:
        if not self.condition_text.strip():
            raise ValueError("condition_text must not be empty")
        matches = [
            candidate
            for candidate in self.candidates
            if candidate.concept_id == self.selected_concept_id
        ]
        if len(matches) != 1:
            raise ValueError("selected_concept_id must identify one supplied candidate")


@dataclass(frozen=True)
class ConditionSplitManifest:
    seed: int
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionTrainingPlan:
    mode: str
    operation: str
    stages: tuple[str, ...]
    output_dir: str
    train_examples: int
    validation_examples: int
    split_manifest: dict[str, Any]
    configuration: dict[str, object]
    lazy_dependencies: tuple[str, ...]
    writes_performed: bool
    execute: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def annotation_id(annotation: ConditionAnnotation) -> str:
    payload = {
        "condition_text": annotation.condition_text,
        "candidates": [candidate.to_dict() for candidate in annotation.candidates],
        "selected_concept_id": annotation.selected_concept_id,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_condition_split_manifest(
    annotations: Sequence[ConditionAnnotation],
    *,
    config: QwenLoRAConfig | None = None,
) -> ConditionSplitManifest:
    config = config or QwenLoRAConfig()
    expected = config.train_size + config.validation_size
    if len(annotations) != expected:
        raise ValueError(f"Expected exactly {expected} condition annotations")
    identifiers = [annotation_id(annotation) for annotation in annotations]
    if len(set(identifiers)) != expected:
        raise ValueError("Condition annotations must be unique")
    random.Random(config.split_seed).shuffle(identifiers)
    training_ids = tuple(identifiers[: config.train_size])
    validation_ids = tuple(identifiers[config.train_size :])
    content = json.dumps(
        {
            "seed": config.split_seed,
            "training_ids": training_ids,
            "validation_ids": validation_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ConditionSplitManifest(
        seed=config.split_seed,
        training_ids=training_ids,
        validation_ids=validation_ids,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
    )


def split_condition_annotations(
    annotations: Sequence[ConditionAnnotation],
    *,
    config: QwenLoRAConfig | None = None,
    manifest: ConditionSplitManifest | None = None,
) -> tuple[list[ConditionAnnotation], list[ConditionAnnotation]]:
    config = config or QwenLoRAConfig()
    manifest = manifest or build_condition_split_manifest(annotations, config=config)
    if len(manifest.training_ids) != 200 or len(manifest.validation_ids) != 50:
        raise ValueError("Condition split must contain 200 training and 50 validation IDs")
    by_id = {annotation_id(annotation): annotation for annotation in annotations}
    expected_ids = set(manifest.training_ids).union(manifest.validation_ids)
    if set(by_id) != expected_ids:
        raise ValueError("Condition split manifest does not match supplied annotations")
    return (
        [by_id[identifier] for identifier in manifest.training_ids],
        [by_id[identifier] for identifier in manifest.validation_ids],
    )


def build_condition_training_plan(
    annotations: Sequence[ConditionAnnotation],
    *,
    output_dir: str | Path,
    config: QwenLoRAConfig | None = None,
) -> ConditionTrainingPlan:
    config = config or QwenLoRAConfig()
    manifest = build_condition_split_manifest(annotations, config=config)
    train, validation = split_condition_annotations(
        annotations, config=config, manifest=manifest
    )
    return ConditionTrainingPlan(
        mode="dry_run",
        operation="qwen_condition_selector_lora_training",
        stages=(
            "freeze_200_50_condition_split",
            "render_faiss_candidate_selection_prompts",
            "tokenize_response_only",
            "fit_condition_lora_adapter",
        ),
        output_dir=str(Path(output_dir)),
        train_examples=len(train),
        validation_examples=len(validation),
        split_manifest=manifest.to_dict(),
        configuration=config.to_dict(),
        lazy_dependencies=("transformers", "peft", "trl", "datasets"),
        writes_performed=False,
    )


TrainingExecutor = Callable[
    [Sequence[ConditionAnnotation], Sequence[ConditionAnnotation], Path, QwenLoRAConfig],
    dict[str, Any],
]


def _training_row(annotation: ConditionAnnotation) -> dict[str, str]:
    prompt = build_condition_selection_prompt(
        annotation.condition_text, annotation.candidates
    )
    response = json.dumps(
        {"selected_concept_id": annotation.selected_concept_id}, separators=(",", ":")
    )
    return {"text": f"{prompt}\n\nRESPONSE_JSON:\n{response}"}


def _execute_condition_training(
    train: Sequence[ConditionAnnotation],
    validation: Sequence[ConditionAnnotation],
    output_dir: Path,
    config: QwenLoRAConfig,
) -> dict[str, Any]:
    """Load ML dependencies lazily and run the explicitly authorized training job."""

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "seed": config.split_seed,
        "training_ids": [annotation_id(row) for row in train],
        "validation_ids": [annotation_id(row) for row in validation],
    }
    split_bytes = json.dumps(
        split_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    split_payload["manifest_sha256"] = hashlib.sha256(split_bytes).hexdigest()
    (output_dir / "condition_split_manifest.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_8bit=config.load_in_8bit,
        llm_int8_threshold=config.llm_int8_threshold,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization,
        device_map=config.device_map,
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    training_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        learning_rate=config.learning_rate,
        max_seq_length=config.max_length,
        dataset_text_field="text",
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template="RESPONSE_JSON:\n", tokenizer=tokenizer
    )
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=Dataset.from_list([_training_row(row) for row in train]),
        eval_dataset=Dataset.from_list([_training_row(row) for row in validation]),
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    result = {
        "mode": "executed",
        "operation": "qwen_condition_selector_lora_training",
        "output_dir": str(output_dir),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "qwen_model": config.model_name,
        "split_manifest_sha256": split_payload["manifest_sha256"],
        "writes_performed": True,
    }
    (output_dir / "condition_training_run_manifest.json").write_text(
        json.dumps(
            {**result, "configuration": config.to_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def train_condition_qwen_lora(
    annotations: Sequence[ConditionAnnotation],
    *,
    output_dir: str | Path,
    config: QwenLoRAConfig | None = None,
    execute: bool = False,
    confirm_execute: bool = False,
    executor: TrainingExecutor | None = None,
) -> dict[str, Any]:
    """Plan by default; train only after both explicit execution gates."""

    config = config or QwenLoRAConfig()
    plan = build_condition_training_plan(
        annotations, output_dir=output_dir, config=config
    )
    if not execute:
        return plan.to_dict()
    if not confirm_execute:
        raise PermissionError("Condition training requires --execute and --confirm-execute")
    manifest = build_condition_split_manifest(annotations, config=config)
    train, validation = split_condition_annotations(
        annotations, config=config, manifest=manifest
    )
    runner = executor or _execute_condition_training
    return runner(train, validation, Path(output_dir), config)
