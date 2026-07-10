import json

import pytest

from clinical_trial_outcome.condition_normalization import (
    DISEASE_AREAS,
    ConditionNormalizationPipeline,
    SnomedCandidate,
    build_condition_selection_prompt,
    parse_condition_selection,
)


def candidates() -> tuple[SnomedCandidate, ...]:
    return (
        SnomedCandidate(
            concept_id="44054006",
            term="Diabetes mellitus type 2 (disorder)",
            disease_area="Disorder of endocrine system",
            similarity=0.91,
            rank=1,
        ),
        SnomedCandidate(
            concept_id="46635009",
            term="Diabetes mellitus type 1 (disorder)",
            disease_area="Disorder of endocrine system",
            similarity=0.84,
            rank=2,
        ),
    )


def test_condition_taxonomy_has_exactly_24_unique_disease_areas() -> None:
    assert len(DISEASE_AREAS) == 24
    assert len(set(DISEASE_AREAS)) == 24


def test_condition_pipeline_is_dry_run_by_default() -> None:
    plan = ConditionNormalizationPipeline().normalize("type 2 diabetes")

    assert plan["mode"] == "dry_run"
    assert plan["execution_required"] is True
    assert plan["disease_area_count"] == 24
    assert plan["stages"][1:3] == [
        "retrieve_snomed_candidates_with_faiss",
        "select_one_retrieved_candidate_with_qwen",
    ]


def test_condition_execute_retrieves_before_qwen_selection() -> None:
    calls: list[str] = []

    class Retriever:
        config = object()

        def retrieve(self, condition_text: str, query_embedding: object):
            assert condition_text == "type 2 diabetes"
            assert query_embedding == [1.0, 0.0]
            calls.append("faiss")
            return candidates()

    def qwen(system: str, prompt: str) -> str:
        assert calls == ["faiss"]
        assert "44054006" in prompt
        calls.append("qwen")
        return json.dumps({"selected_concept_id": "44054006"})

    result = ConditionNormalizationPipeline(
        retriever=Retriever(), qwen_generator=qwen
    ).normalize(
        "type 2 diabetes",
        [1.0, 0.0],
        execute=True,
        confirm_execute=True,
    )

    assert calls == ["faiss", "qwen"]
    assert result["snomed_concept_id"] == "44054006"
    assert result["disease_area"] == "Disorder of endocrine system"
    assert result["selection_method"] == "faiss_candidates_then_qwen_selection"
    assert len(result["candidates"]) == 2


def test_condition_qwen_cannot_invent_a_snomed_concept() -> None:
    with pytest.raises(ValueError, match="supplied SNOMED candidate"):
        parse_condition_selection(
            '{"selected_concept_id":"999999"}', candidates()
        )


def test_condition_prompt_preserves_candidate_audit_fields() -> None:
    prompt = build_condition_selection_prompt("type 2 diabetes", candidates())

    assert "FAISS" in prompt
    assert "Diabetes mellitus type 2" in prompt
    assert '"rank": 1' in prompt
    assert '"similarity": 0.91' in prompt


def test_condition_execute_fails_closed_without_runtime_dependencies() -> None:
    with pytest.raises(RuntimeError, match="FAISS retriever and Qwen generator"):
        ConditionNormalizationPipeline().normalize(
            "type 2 diabetes",
            [1.0, 0.0],
            execute=True,
            confirm_execute=True,
        )
