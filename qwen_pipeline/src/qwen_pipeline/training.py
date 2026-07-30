"""Original-style multi-GPU Qwen2.5 LoRA training entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .config import config_digest, task_config
from .io import atomic_write_json, read_jsonl, sha256_file


def inspect_training_file(
    task: str,
    path: str | Path,
    *,
    allowed_splits: Iterable[str] = ("train",),
) -> dict[str, Any]:
    records = list(read_jsonl(path))
    if not records:
        raise ValueError("training file is empty")
    allowed = set(allowed_splits)
    invalid = [record.get("record_id") for record in records if record.get("task") != task]
    wrong_split = [
        record.get("record_id") for record in records if record.get("split") not in allowed
    ]
    if invalid:
        raise ValueError(f"training file includes records for another task: {invalid[:3]}")
    if wrong_split:
        raise ValueError(
            f"training file includes records outside {sorted(allowed)}: {wrong_split[:3]}"
        )
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise ValueError(f"invalid chat messages for {record.get('record_id')}")
    return {"task": task, "records": len(records), "sha256": sha256_file(path)}


def _prompt_and_response(record: dict[str, Any], tokenizer: Any) -> tuple[str, str]:
    prompt = tokenizer.apply_chat_template(
        record["messages"][:2],
        tokenize=False,
        add_generation_prompt=True,
    )
    response = record["messages"][2]["content"] + tokenizer.eos_token
    return prompt, response


def _encode_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    """Mask prompt tokens exactly as in the historical GPU training script."""

    import torch

    encoded: list[dict[str, Any]] = []
    for record in records:
        prompt, response = _prompt_and_response(record, tokenizer)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        input_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
        attention_mask = [1] * len(input_ids)
        padding = max_length - len(input_ids)
        input_ids.extend([tokenizer.pad_token_id] * padding)
        labels.extend([-100] * padding)
        attention_mask.extend([0] * padding)
        encoded.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )
    return encoded


def train_lora(
    task: str,
    train_jsonl: str | Path,
    output_dir: str | Path,
    *,
    validation_jsonl: str | Path | None = None,
    model_id: str | None = None,
    max_train_records: int | None = None,
) -> dict[str, Any]:
    """Run the historical response-masked Accelerate/PEFT training pattern."""

    train_summary = inspect_training_file(task, train_jsonl)
    validation_summary = None
    if validation_jsonl is not None:
        validation_summary = inspect_training_file(
            task,
            validation_jsonl,
            allowed_splits=("validation", "test"),
        )
    config = task_config(task)
    pipeline = config["pipeline"]
    selector = pipeline["selector"]
    lora = pipeline["lora"]
    selected_model = model_id or selector["model_id"]

    try:
        import torch
        from accelerate import Accelerator
        from peft import LoraConfig, get_peft_model
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            default_data_collator,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("full GPU training dependencies are not installed") from exc

    train_records = list(read_jsonl(train_jsonl))
    if max_train_records is not None:
        if max_train_records <= 0:
            raise ValueError("max_train_records must be positive for LoRA training")
        if max_train_records > len(train_records):
            raise ValueError("max_train_records exceeds the reviewed training file")
        train_records = train_records[:max_train_records]
    validation_records = (
        list(read_jsonl(validation_jsonl)) if validation_jsonl is not None else []
    )

    set_seed(int(lora["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(selected_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    max_length = int(selector["max_sequence_length"])
    train_dataset = _encode_records(train_records, tokenizer, max_length)
    validation_dataset = _encode_records(validation_records, tokenizer, max_length)

    quantization = BitsAndBytesConfig(
        load_in_8bit=bool(selector["load_in_8bit"]),
        llm_int8_threshold=float(selector["int8_threshold"]),
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        selected_model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_kwargs: dict[str, Any] = {
        "r": int(lora["rank"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora["dropout"]),
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
    }
    if lora.get("target_modules"):
        lora_kwargs["target_modules"] = lora["target_modules"]
    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    optimizer = AdamW(model.parameters(), lr=float(lora["learning_rate"]))
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=int(lora["per_device_train_batch_size"]),
        collate_fn=default_data_collator,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            shuffle=False,
            batch_size=int(lora["per_device_train_batch_size"]),
            collate_fn=default_data_collator,
        )
        if validation_dataset
        else None
    )

    accelerator = Accelerator()
    if validation_loader is None:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
    else:
        model, optimizer, train_loader, validation_loader = accelerator.prepare(
            model, optimizer, train_loader, validation_loader
        )

    epoch_metrics: list[dict[str, float | int | None]] = []
    for epoch in range(int(lora["epochs"])):
        model.train()
        training_loss = 0.0
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            training_loss += float(loss.detach().item())

        validation_loss: float | None = None
        if validation_loader is not None:
            model.eval()
            total = 0.0
            steps = 0
            with torch.no_grad():
                for batch in validation_loader:
                    total += float(model(**batch).loss.detach().item())
                    steps += 1
            validation_loss = total / steps
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "training_loss": training_loss / len(train_loader),
                "validation_loss": validation_loss,
            }
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(destination, save_function=accelerator.save)
    tokenizer.save_pretrained(destination)
    manifest = {
        **train_summary,
        "status": "executed",
        "training_records_used": len(train_records),
        "validation": validation_summary,
        "base_model_id": selected_model,
        "epoch_metrics": epoch_metrics,
        "configuration_sha256": config_digest(config),
        "training_pattern": (
            "Historical Accelerate/PEFT loop with 8-bit loading, separately "
            "tokenized prompt and response, and prompt labels masked to -100."
        ),
        "note": "This is a newly executed run and is not automatically a manuscript result.",
    }
    atomic_write_json(destination / "training_manifest.json", manifest)
    return manifest
