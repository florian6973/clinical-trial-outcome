"""Guarded retrieval-augmented Qwen inference for outcome objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import QwenLoRAConfig, RetrievalConfig
from .parser import parse_group_response
from .prompt import SYSTEM_MESSAGE, build_group_prompt
from .retrieval import Candidate, Vector, retrieve_context


QwenGenerator = Callable[[str, str], str]


@dataclass(frozen=True)
class InferencePlan:
    mode: str
    operation: str
    stages: tuple[str, ...]
    number_of_objects: int
    adapter_path: str | None
    qwen: dict[str, object]
    retrieval: dict[str, object]
    writes_performed: bool
    execute: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedOutcome:
    object_text: str
    group: str
    raw_response: str
    prompt: str
    similar_terms: tuple[Candidate, ...]
    candidate_groups: tuple[Candidate, ...]


def build_inference_plan(
    objects: Sequence[str],
    *,
    adapter_path: str | Path | None = None,
    qwen_config: QwenLoRAConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
) -> InferencePlan:
    if not objects or any(not value.strip() for value in objects):
        raise ValueError("At least one non-empty outcome object is required")
    qwen_config = qwen_config or QwenLoRAConfig()
    retrieval_config = retrieval_config or RetrievalConfig()
    return InferencePlan(
        mode="dry_run",
        operation="retrieval_augmented_qwen_outcome_normalization",
        stages=(
            "retrieve_top_5_terms",
            "retrieve_top_5_groups",
            "select_or_create_group_with_qwen",
            "validate_group_prefix",
        ),
        number_of_objects=len(objects),
        adapter_path=str(adapter_path) if adapter_path is not None else None,
        qwen=qwen_config.to_dict(),
        retrieval=retrieval_config.to_dict(),
        writes_performed=False,
    )


def normalize_outcome_objects(
    objects: Sequence[str],
    query_vectors: Sequence[Vector],
    *,
    term_labels: Sequence[str],
    term_vectors: Sequence[Vector],
    group_labels: Sequence[str],
    group_vectors: Sequence[Vector],
    adapter_path: str | Path | None = None,
    qwen_config: QwenLoRAConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    qwen_generator: QwenGenerator | None = None,
    execute: bool = False,
    confirm_execute: bool = False,
) -> dict[str, object] | list[NormalizedOutcome]:
    """Plan by default; use only an explicitly injected generator on execution."""

    if len(objects) != len(query_vectors):
        raise ValueError("Each outcome object requires one precomputed query vector")
    qwen_config = qwen_config or QwenLoRAConfig()
    retrieval_config = retrieval_config or RetrievalConfig()
    plan = build_inference_plan(
        objects,
        adapter_path=adapter_path,
        qwen_config=qwen_config,
        retrieval_config=retrieval_config,
    )
    if not execute:
        return plan.to_dict()
    if not confirm_execute:
        raise PermissionError("Inference requires execute=True and confirm_execute=True")
    if adapter_path is None:
        raise ValueError("Execution requires an explicit LoRA adapter path")
    if qwen_generator is None:
        raise RuntimeError("--execute requires an injected Qwen generator")
    return _execute_inference(
        objects,
        query_vectors,
        term_labels=term_labels,
        term_vectors=term_vectors,
        group_labels=group_labels,
        group_vectors=group_vectors,
        retrieval_config=retrieval_config,
        qwen_generator=qwen_generator,
    )


def _execute_inference(
    objects: Sequence[str],
    query_vectors: Sequence[Vector],
    *,
    term_labels: Sequence[str],
    term_vectors: Sequence[Vector],
    group_labels: Sequence[str],
    group_vectors: Sequence[Vector],
    retrieval_config: RetrievalConfig,
    qwen_generator: QwenGenerator,
) -> list[NormalizedOutcome]:
    """Normalize with an injected, already configured Qwen-LoRA generator."""

    normalized: list[NormalizedOutcome] = []
    for object_text, query_vector in zip(objects, query_vectors):
        context = retrieve_context(
            query_vector,
            term_labels=term_labels,
            term_vectors=term_vectors,
            group_labels=group_labels,
            group_vectors=group_vectors,
            config=retrieval_config,
        )
        prompt = build_group_prompt(object_text, context)
        response = qwen_generator(SYSTEM_MESSAGE, prompt).strip()
        normalized.append(
            NormalizedOutcome(
                object_text=object_text,
                group=parse_group_response(response),
                raw_response=response,
                prompt=prompt,
                similar_terms=context.terms,
                candidate_groups=context.groups,
            )
        )
    return normalized
