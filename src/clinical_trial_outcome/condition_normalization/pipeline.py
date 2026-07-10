"""Orchestration contract: FAISS retrieval must precede Qwen selection."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .contract import ConditionNormalizationPlan, ConditionNormalizationResult, SnomedCandidate
from .disease_areas import DISEASE_AREAS
from .prompt import build_condition_selection_prompt, parse_condition_selection


class CandidateRetriever(Protocol):
    config: Any

    def retrieve(
        self, condition_text: str, query_embedding: Any
    ) -> tuple[SnomedCandidate, ...]: ...


QwenGenerator = Callable[[str, str], str]


class ConditionNormalizationPipeline:
    """Condition normalizer that is dry-run-only unless ``execute=True``."""

    def __init__(
        self,
        retriever: CandidateRetriever | None = None,
        qwen_generator: QwenGenerator | None = None,
        qwen_model: str = "Qwen/Qwen2.5-32B-Instruct",
        retrieval_top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.qwen_generator = qwen_generator
        self.qwen_model = qwen_model
        self.retrieval_top_k = retrieval_top_k

    def normalize(
        self,
        condition_text: str,
        query_embedding: Any = None,
        *,
        execute: bool = False,
        confirm_execute: bool = False,
    ) -> dict[str, Any]:
        if not condition_text.strip():
            raise ValueError("condition_text must not be empty")
        if not execute:
            return ConditionNormalizationPlan(
                mode="dry_run",
                condition_text=condition_text,
                stages=(
                    "embed_condition_with_nv_embed_v2",
                    "retrieve_snomed_candidates_with_faiss",
                    "select_one_retrieved_candidate_with_qwen",
                    "validate_24_disease_area_contract",
                ),
                retrieval_top_k=self.retrieval_top_k,
                qwen_model=self.qwen_model,
                disease_area_count=len(DISEASE_AREAS),
            ).to_dict()

        if not confirm_execute:
            raise PermissionError(
                "Condition inference requires execute=True and confirm_execute=True"
            )

        if self.retriever is None or self.qwen_generator is None:
            raise RuntimeError(
                "--execute requires an initialized FAISS retriever and Qwen generator"
            )
        if query_embedding is None:
            raise ValueError("--execute requires a precomputed L2-normalized query embedding")

        candidates = self.retriever.retrieve(condition_text, query_embedding)
        if not candidates:
            raise RuntimeError("Qwen selection cannot run before candidate retrieval")
        prompt = build_condition_selection_prompt(condition_text, candidates)
        response = self.qwen_generator(
            "You map clinical-trial condition text to SNOMED CT. Select exactly "
            "one supplied candidate and return JSON only.",
            prompt,
        )
        selected = parse_condition_selection(response, candidates)
        result = ConditionNormalizationResult(
            condition_text=condition_text,
            snomed_concept_id=selected.concept_id,
            snomed_term=selected.term,
            disease_area=selected.disease_area,
            similarity=selected.similarity,
            candidate_rank=selected.rank,
            qwen_model=self.qwen_model,
            selection_method="faiss_candidates_then_qwen_selection",
            candidates=candidates,
        )
        return result.to_dict()
