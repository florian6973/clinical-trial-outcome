from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_pipeline.config import load_configs, validate_configs  # noqa: E402
from qwen_pipeline.io import read_jsonl  # noqa: E402
from qwen_pipeline.preparation import prepare_records  # noqa: E402
from qwen_pipeline.validation import (  # noqa: E402
    ContractError,
    parse_condition_prediction,
    parse_outcome_prediction,
    validate_annotation,
    validate_schema,
)


class ContractTests(unittest.TestCase):
    def test_paper_aligned_config_is_valid(self) -> None:
        self.assertEqual(validate_configs(load_configs()), [])

    def test_fixture_annotations_validate_and_prepare(self) -> None:
        for task in ("outcome", "condition"):
            source = list(read_jsonl(ROOT / "fixtures" / f"{task}_annotations.jsonl"))
            for record in source:
                validate_annotation(task, record)
            prepared = list(prepare_records(task, source))
            self.assertEqual(len(prepared), 2)
            self.assertEqual(prepared[0]["messages"][0]["role"], "system")
            self.assertEqual(prepared[0]["messages"][-1]["role"], "assistant")

    def test_outcome_parser_requires_one_allowed_group(self) -> None:
        parsed = parse_outcome_prediction(
            "EXPLANATION: It preserves the broad wording.\nGROUP: Clinical Response",
            {"Clinical Response", "ORR/OR"},
        )
        self.assertEqual(parsed["status"], "ok")
        self.assertTrue(parsed["eligible_for_aggregation"])
        with self.assertRaises(ContractError):
            parse_outcome_prediction("GROUP: Invented Label", {"Clinical Response"})

    def test_new_outcome_group_enters_review_not_aggregation(self) -> None:
        parsed = parse_outcome_prediction(
            "EXPLANATION: No candidate preserves the specific construct.\n"
            "GROUP: PROPOSED_NEW_GROUP\n"
            "PROPOSED_NEW_GROUP: Device Acceptability",
            {"Clinical Response", "ORR/OR"},
        )
        self.assertEqual(parsed["status"], "proposed_new_group")
        self.assertEqual(parsed["proposed_group"], "Device Acceptability")
        self.assertIsNone(parsed["selected_group"])
        self.assertFalse(parsed["eligible_for_aggregation"])

    def test_condition_parser_is_candidate_constrained(self) -> None:
        candidates = [
            {
                "candidate_id": "candidate-1",
                "concept_id": "example-concept-1",
                "preferred_term": "Example disorder",
                "similarity": 0.9,
            }
        ]
        parsed = parse_condition_prediction(
            '{"no_match":false,"selected_candidate_id":"candidate-1"}', candidates
        )
        self.assertEqual(parsed["selected_concept_id"], "example-concept-1")
        with self.assertRaises(ContractError):
            parse_condition_prediction(
                '{"no_match":false,"selected_candidate_id":"invented"}', candidates
            )

    def test_canonical_metric_fixture_validates(self) -> None:
        import json

        metrics = json.loads(
            (ROOT / "fixtures" / "canonical_paper_metrics.example.json").read_text(encoding="utf-8")
        )
        validate_schema(metrics, "canonical_paper_metrics.schema.json")

    def test_complete_prediction_audit_schemas(self) -> None:
        outcome = {
            "record_id": "o-1",
            "source_object": "response",
            "candidate_terms": [],
            "candidate_groups": [],
            "raw_model_text": None,
            "explanation": None,
            "selected_group": None,
            "proposed_group": None,
            "eligible_for_aggregation": False,
            "status": "retrieval_error",
            "error_detail": "illustrative",
            "run_id": "run-1",
        }
        condition = {
            "record_id": "c-1",
            "raw_condition": "example",
            "candidates": [],
            "raw_model_text": None,
            "selected_candidate_id": None,
            "selected_concept_id": None,
            "selected_preferred_term": None,
            "no_match": None,
            "disease_area": None,
            "status": "retrieval_error",
            "error_detail": "illustrative",
            "run_id": "run-1",
        }
        validate_schema(outcome, "outcome_prediction.schema.json")
        validate_schema(condition, "condition_prediction.schema.json")


if __name__ == "__main__":
    unittest.main()
