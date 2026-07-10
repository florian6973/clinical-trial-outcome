"""Deterministic preparation of the 200/50 annotated outcome split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Sequence

from .config import QwenLoRAConfig


@dataclass(frozen=True)
class OutcomeAnnotation:
    term: str
    group: str
    prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.term.strip() or not self.group.strip():
            raise ValueError("Annotation term and group must not be empty")


@dataclass(frozen=True)
class SupervisedRecord:
    prompt: str
    response: str


@dataclass(frozen=True)
class OutcomeSplitManifest:
    """Frozen membership for the paper's 200/50 split."""

    schema_version: int
    seed: int
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def annotation_id(annotation: OutcomeAnnotation) -> str:
    payload = json.dumps(
        {
            "term": annotation.term,
            "group": annotation.group,
            "prompt": annotation.prompt,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_split_manifest(
    examples: Sequence[OutcomeAnnotation],
    *,
    config: QwenLoRAConfig | None = None,
) -> OutcomeSplitManifest:
    """Freeze and hash split membership before any model fitting."""

    config = config or QwenLoRAConfig()
    expected = config.train_size + config.validation_size
    if len(examples) != expected:
        raise ValueError(f"Expected exactly {expected} annotations; received {len(examples)}")
    identifiers = [annotation_id(example) for example in examples]
    if len(set(identifiers)) != expected:
        raise ValueError("Annotations must be unique before a split manifest is created")
    random.Random(config.split_seed).shuffle(identifiers)
    training_ids = tuple(identifiers[: config.train_size])
    validation_ids = tuple(identifiers[config.train_size :])
    content = json.dumps(
        {
            "schema_version": 1,
            "seed": config.split_seed,
            "training_ids": training_ids,
            "validation_ids": validation_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return OutcomeSplitManifest(
        schema_version=1,
        seed=config.split_seed,
        training_ids=training_ids,
        validation_ids=validation_ids,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
    )


def split_annotated_examples(
    examples: Sequence[OutcomeAnnotation],
    *,
    config: QwenLoRAConfig | None = None,
    manifest: OutcomeSplitManifest | None = None,
) -> tuple[list[OutcomeAnnotation], list[OutcomeAnnotation]]:
    """Return the exact paper split with deterministic randomized membership."""

    config = config or QwenLoRAConfig()
    expected = config.train_size + config.validation_size
    if len(examples) != expected:
        raise ValueError(f"Expected exactly {expected} annotations; received {len(examples)}")
    manifest = manifest or build_split_manifest(examples, config=config)
    if len(manifest.training_ids) != config.train_size:
        raise ValueError("Split manifest does not contain 200 training identifiers")
    if len(manifest.validation_ids) != config.validation_size:
        raise ValueError("Split manifest does not contain 50 validation identifiers")
    by_id = {annotation_id(example): example for example in examples}
    manifest_ids = set(manifest.training_ids).union(manifest.validation_ids)
    if set(by_id) != manifest_ids:
        raise ValueError("Split manifest membership does not match the supplied annotations")
    train = [by_id[identifier] for identifier in manifest.training_ids]
    validation = [by_id[identifier] for identifier in manifest.validation_ids]
    return train, validation


def as_supervised_record(annotation: OutcomeAnnotation, *, prompt: str) -> SupervisedRecord:
    """Format one annotation for response-only causal-LM supervision."""

    final_prompt = annotation.prompt.strip() if annotation.prompt else prompt.strip()
    if not final_prompt:
        raise ValueError("A non-empty prompt is required")
    return SupervisedRecord(prompt=final_prompt, response=f"GROUP: {annotation.group.strip()}")
