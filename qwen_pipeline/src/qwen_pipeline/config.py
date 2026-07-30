"""Configuration loading and validation with no heavy runtime dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PACKAGE_ROOT / "config"
SCHEMA_DIR = PACKAGE_ROOT / "schemas"


class ConfigurationError(ValueError):
    """Raised when a paper-aligned configuration is incomplete or unsafe."""


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_configs() -> dict[str, Any]:
    return {
        "pipeline": read_json(CONFIG_DIR / "pipeline.json"),
        "outcome": read_json(CONFIG_DIR / "outcome.json"),
        "condition": read_json(CONFIG_DIR / "condition.json"),
        "experiments": read_json(CONFIG_DIR / "experiments.json"),
    }


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_configs(configs: dict[str, Any] | None = None) -> list[str]:
    configs = configs or load_configs()
    errors: list[str] = []
    pipeline = configs["pipeline"]
    embedding = pipeline.get("embedding", {})
    selector = pipeline.get("selector", {})
    lora = pipeline.get("lora", {})

    if embedding.get("model_id") != "nvidia/NV-Embed-v2":
        errors.append("embedding model must match the manuscript's NV-Embed-v2 implementation")
    if not embedding.get("l2_normalize"):
        errors.append("embedding vectors must be L2-normalized before inner-product retrieval")
    if selector.get("model_id") != "Qwen/Qwen2.5-32B-Instruct":
        errors.append("primary selector must match the manuscript's Qwen2.5-32B implementation")
    if selector.get("do_sample") is not False:
        errors.append("selector decoding must be deterministic")
    expected_lora = {"rank": 8, "alpha": 32, "dropout": 0.1, "epochs": 3}
    for key, expected in expected_lora.items():
        if lora.get(key) != expected:
            errors.append(f"LoRA {key} must be {expected!r}")
    if configs["condition"].get("selector_may_invent_concept_id") is not False:
        errors.append("condition selector must not invent terminology identifiers")
    if not configs["outcome"].get("split_artifact_required"):
        errors.append("outcome training requires an explicit split artifact")
    return errors


def task_config(task: str) -> dict[str, Any]:
    if task not in {"outcome", "condition"}:
        raise ConfigurationError(f"unsupported task: {task}")
    configs = load_configs()
    return {"pipeline": configs["pipeline"], "task": configs[task]}
