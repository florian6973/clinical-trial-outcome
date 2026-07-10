"""Dry-run-first LoRA training for paper-aligned Qwen outcome normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
from typing import Sequence

from .config import QwenLoRAConfig
from .data import (
    OutcomeAnnotation,
    SupervisedRecord,
    build_split_manifest,
    split_annotated_examples,
)


@dataclass(frozen=True)
class TrainingPlan:
    mode: str
    operation: str
    stages: tuple[str, ...]
    output_dir: str
    train_examples: int
    validation_examples: int
    validation_policy: str
    split_manifest_sha256: str
    split_manifest: dict[str, object]
    configuration: dict[str, object]
    writes_performed: bool
    execute: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunSummary:
    output_dir: str
    adapter_saved: bool
    train_examples: int
    validation_examples: int
    global_step: int
    training_loss: float | None
    evaluation_metrics: dict[str, float]
    split_manifest_sha256: str


def build_training_plan(
    annotations: Sequence[OutcomeAnnotation],
    *,
    output_dir: str | Path,
    config: QwenLoRAConfig | None = None,
) -> TrainingPlan:
    config = config or QwenLoRAConfig()
    manifest = build_split_manifest(annotations, config=config)
    train, validation = split_annotated_examples(
        annotations,
        config=config,
        manifest=manifest,
    )
    return TrainingPlan(
        mode="dry_run",
        operation="qwen_outcome_lora_training",
        stages=("freeze_200_50_split", "tokenize_response_only", "fit_lora_adapter"),
        output_dir=str(Path(output_dir)),
        train_examples=len(train),
        validation_examples=len(validation),
        validation_policy=(
            "held_out_evaluation_only; excluded from prompt development, retrieval tuning, "
            "model selection, and learning-curve fitting"
        ),
        split_manifest_sha256=manifest.manifest_sha256,
        split_manifest=manifest.to_dict(),
        configuration=config.to_dict(),
        writes_performed=False,
    )


def _require_records(annotations: Sequence[OutcomeAnnotation]) -> list[SupervisedRecord]:
    records: list[SupervisedRecord] = []
    for annotation in annotations:
        if not annotation.prompt or not annotation.prompt.strip():
            raise ValueError(
                "Execution requires a complete retrieval-augmented prompt for every annotation"
            )
        records.append(
            SupervisedRecord(
                prompt=annotation.prompt.strip(),
                response=f"GROUP: {annotation.group.strip()}",
            )
        )
    return records


def train_qwen_lora(
    annotations: Sequence[OutcomeAnnotation],
    *,
    output_dir: str | Path,
    config: QwenLoRAConfig | None = None,
    execute: bool = False,
    confirm_execute: bool = False,
) -> dict[str, object] | TrainingRunSummary:
    """Plan training by default or run it after double confirmation.

    The confirmed branch is intended for a separately authorized compute
    environment with the ``ml`` dependencies, model access, and local inputs.
    Imports of Transformers, PEFT, TRL, torch, and datasets occur only inside
    that branch. Calling this function without both flags never loads a model,
    writes an artifact, or imports an optional ML dependency.
    """

    config = config or QwenLoRAConfig()
    plan = build_training_plan(annotations, output_dir=output_dir, config=config)
    if not execute:
        return plan.to_dict()
    if not confirm_execute:
        raise PermissionError("Training requires execute=True and confirm_execute=True")

    manifest = build_split_manifest(annotations, config=config)
    train_annotations, validation_annotations = split_annotated_examples(
        annotations,
        config=config,
        manifest=manifest,
    )
    return _execute_training(
        _require_records(train_annotations),
        _require_records(validation_annotations),
        output_dir=Path(output_dir),
        config=config,
        split_manifest=manifest.to_dict(),
    )


def _execute_training(
    train_records: Sequence[SupervisedRecord],
    validation_records: Sequence[SupervisedRecord],
    *,
    output_dir: Path,
    config: QwenLoRAConfig,
    split_manifest: dict[str, object],
) -> TrainingRunSummary:
    """Run the locked training recipe; called only after explicit authorization."""

    # Optional/heavy imports are intentionally inside the confirmed execution
    # function so dry-runs and unit tests remain CPU- and network-safe.
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from trl import DataCollatorForCompletionOnlyLM

    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = output_dir / "outcome_split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def serialize(record: SupervisedRecord) -> dict[str, str]:
        return {"text": f"{record.prompt}\n{record.response}{tokenizer.eos_token}"}

    train_dataset = Dataset.from_list([serialize(record) for record in train_records])
    validation_dataset = Dataset.from_list(
        [serialize(record) for record in validation_records]
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
        remove_columns=validation_dataset.column_names,
    )

    quantization = BitsAndBytesConfig(
        load_in_8bit=config.load_in_8bit,
        llm_int8_threshold=config.llm_int8_threshold,
        llm_int8_skip_modules=None,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization,
        device_map=config.device_map,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        ),
    )

    # Transformers renamed `evaluation_strategy` to `eval_strategy`; support
    # both without weakening the locked epoch-level validation behavior.
    arguments_parameters = inspect.signature(TrainingArguments.__init__).parameters
    evaluation_key = (
        "eval_strategy" if "eval_strategy" in arguments_parameters else "evaluation_strategy"
    )
    arguments_kwargs: dict[str, object] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "learning_rate": config.learning_rate,
        "optim": "adamw_torch",
        evaluation_key: "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": config.split_seed,
        "data_seed": config.split_seed,
    }
    training_arguments = TrainingArguments(**arguments_kwargs)
    collator = DataCollatorForCompletionOnlyLM(
        response_template="GROUP:",
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
    )

    training_result = trainer.train()
    evaluation_metrics = {
        key: float(value)
        for key, value in trainer.evaluate().items()
        if isinstance(value, (int, float))
    }
    adapter_dir = output_dir / "qwen_finetuned_32b"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    run_manifest = {
        "model_name": config.model_name,
        "adapter_dir": str(adapter_dir),
        "training_configuration": config.to_dict(),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "global_step": int(trainer.state.global_step),
        "training_loss": training_result.metrics.get("train_loss"),
        "evaluation_metrics": evaluation_metrics,
    }
    (output_dir / "outcome_training_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    training_loss = training_result.metrics.get("train_loss")
    return TrainingRunSummary(
        output_dir=str(adapter_dir),
        adapter_saved=True,
        train_examples=len(train_records),
        validation_examples=len(validation_records),
        global_step=int(trainer.state.global_step),
        training_loss=float(training_loss) if training_loss is not None else None,
        evaluation_metrics=evaluation_metrics,
        split_manifest_sha256=str(split_manifest["manifest_sha256"]),
    )
