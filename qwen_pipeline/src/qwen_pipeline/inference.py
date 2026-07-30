"""Candidate retrieval, Qwen selection, and item-level audit records."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable

from .config import config_digest, task_config
from .io import atomic_write_json, read_jsonl, sha256_file, write_jsonl
from .prompts import CONDITION_SYSTEM, OUTCOME_SYSTEM, condition_user, outcome_user
from .retrieval import Retriever
from .validation import ContractError, parse_condition_prediction, parse_outcome_prediction, validate_schema


class QwenSelector:
    def __init__(self, adapter_path: str | Path):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("full inference dependencies are not installed") from exc
        config = task_config("outcome")["pipeline"]
        selector = config["selector"]
        quantization = BitsAndBytesConfig(load_in_8bit=bool(selector["load_in_8bit"]))
        base = AutoModelForCausalLM.from_pretrained(
            selector["model_id"],
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        self.selector_config = selector

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.selector_config["max_new_tokens"],
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_crosswalk(path: str | Path) -> dict[str, dict[str, Any]]:
    crosswalk: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        validate_schema(row, "disease_area_crosswalk.schema.json")
        if row["qa_status"] != "verified":
            continue
        concept_id = row["concept_id"]
        if concept_id in crosswalk and crosswalk[concept_id]["disease_area"] != row["disease_area"]:
            raise ValueError(f"verified concept maps to multiple disease areas: {concept_id}")
        crosswalk[concept_id] = row
    if not crosswalk:
        raise ValueError("crosswalk has no verified rows")
    return crosswalk


def _outcome_record(source: dict[str, Any], retriever: Retriever, selector: QwenSelector, run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    record_id = str(source["record_id"])
    source_object = str(source["source_object"])
    terms = retriever.search(source_object, config["candidate_terms_k"])
    groups = retriever.search_groups(source_object, config["candidate_groups_k"])
    prompt_record = {
        "source_object": source_object,
        "context": str(source.get("context", "")),
        "candidate_terms": [{"term": item["term"], "group": item["normalized_group"], "similarity": item["similarity"]} for item in terms],
        "candidate_groups": groups,
    }
    audit = {
        "record_id": record_id,
        **prompt_record,
        "raw_model_text": None,
        "selected_group": None,
        "proposed_group": None,
        "eligible_for_aggregation": False,
        "status": "selector_error",
        "run_id": run_id,
    }
    try:
        raw = selector.generate(OUTCOME_SYSTEM, outcome_user(prompt_record))
        audit["raw_model_text"] = raw
        audit.update(parse_outcome_prediction(raw, {item["group"] for item in groups}))
        audit["error_detail"] = None
    except ContractError as exc:
        audit.update({"status": "format_error", "error_detail": str(exc)})
    except Exception as exc:
        audit.update({"status": "selector_error", "error_detail": str(exc)})
    return audit


def _condition_record(
    source: dict[str, Any],
    retriever: Retriever,
    selector: QwenSelector,
    run_id: str,
    config: dict[str, Any],
    crosswalk: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record_id = str(source["record_id"])
    raw_condition = str(source["raw_condition"])
    retrieved = retriever.search(raw_condition, config["candidate_concepts_k"])
    candidates = [
        {
            "candidate_id": item["vocabulary_id"],
            "concept_id": item["concept_id"],
            "preferred_term": item["preferred_term"],
            "similarity": item["similarity"],
        }
        for item in retrieved
    ]
    prompt_record = {"raw_condition": raw_condition, "candidates": candidates}
    audit = {
        "record_id": record_id,
        **prompt_record,
        "raw_model_text": None,
        "selected_candidate_id": None,
        "selected_concept_id": None,
        "selected_preferred_term": None,
        "no_match": None,
        "disease_area": None,
        "status": "selector_error",
        "error_detail": None,
        "run_id": run_id,
    }
    try:
        raw = selector.generate(CONDITION_SYSTEM, condition_user(prompt_record))
        audit["raw_model_text"] = raw
        parsed = parse_condition_prediction(raw, candidates)
        audit.update(parsed)
        audit["error_detail"] = None
        if parsed["status"] == "ok":
            row = crosswalk.get(parsed["selected_concept_id"])
            if row is None:
                audit.update({"status": "crosswalk_error", "disease_area": None, "error_detail": "selected concept absent from verified crosswalk"})
            else:
                audit["disease_area"] = row["disease_area"]
        else:
            audit["disease_area"] = None
    except ContractError as exc:
        audit.update({"status": "format_error", "error_detail": str(exc)})
    except Exception as exc:
        audit.update({"status": "selector_error", "error_detail": str(exc)})
    return audit


def run_inference(
    task: str,
    input_path: str | Path,
    index_dir: str | Path,
    adapter_path: str | Path,
    output_path: str | Path,
    crosswalk_path: str | Path | None = None,
) -> dict[str, Any]:
    config = task_config(task)
    records = list(read_jsonl(input_path))
    if not records:
        raise ValueError("inference input is empty")
    if task == "condition" and crosswalk_path is None:
        raise ValueError("condition inference requires an author-verified --crosswalk")
    crosswalk = load_crosswalk(crosswalk_path) if crosswalk_path else {}
    retriever = Retriever(task, index_dir)
    selector = QwenSelector(adapter_path)
    run_id = str(uuid.uuid4())
    task_settings = config["task"]
    outputs = []
    for record in records:
        if task == "outcome":
            outputs.append(_outcome_record(record, retriever, selector, run_id, task_settings))
        else:
            outputs.append(_condition_record(record, retriever, selector, run_id, task_settings, crosswalk))
    write_jsonl(output_path, outputs)
    manifest = {
        "status": "executed",
        "run_id": run_id,
        "task": task,
        "input_records": len(records),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "configuration_sha256": config_digest(config),
        "adapter_path": str(adapter_path),
        "index_dir": str(index_dir),
        "note": "New run; not automatically evidence for the manuscript's canonical results.",
    }
    atomic_write_json(Path(output_path).with_suffix(".manifest.json"), manifest)
    return manifest
