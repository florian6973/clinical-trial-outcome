from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_pipeline.evaluation import evaluate  # noqa: E402
from qwen_pipeline.experiments import experiment_plan  # noqa: E402
from qwen_pipeline.io import read_jsonl, write_jsonl  # noqa: E402
from qwen_pipeline.paper_outputs import ingest_metrics  # noqa: E402
from qwen_pipeline.retrieval import inspect_vocabulary  # noqa: E402


class DryRunTests(unittest.TestCase):
    def test_vocabulary_inspection_does_not_load_models(self) -> None:
        for task in ("outcome", "condition"):
            result = inspect_vocabulary(task, ROOT / "fixtures" / f"{task}_vocabulary.jsonl")
            self.assertGreater(result["records"], 0)
            self.assertEqual(len(result["sha256"]), 64)

    def test_experiment_plan_has_no_fabricated_scores(self) -> None:
        plan = experiment_plan()
        self.assertEqual(plan["status"], "planned_not_executed")
        self.assertEqual(plan["cell_count"], 11)
        self.assertIn("Llama3-15B", plan["missing_model_ids"])
        self.assertTrue(all("score" not in cell for cell in plan["cells"]))

    def test_canonical_ingestion_is_explicitly_not_recomputed(self) -> None:
        result = ingest_metrics(ROOT / "fixtures" / "canonical_paper_metrics.example.json")
        self.assertEqual(result["status"], "supplied_canonical_not_recomputed")
        self.assertEqual(result["warnings"], [])
        self.assertIn("not recomputed", result["note"])

    def test_exact_match_evaluation_keeps_item_ledger(self) -> None:
        gold = list(read_jsonl(ROOT / "fixtures" / "outcome_annotations.jsonl"))
        predictions = [
            {
                "record_id": row["record_id"],
                "selected_group": row["gold_group"],
                "status": "ok",
            }
            for row in gold
        ]
        with tempfile.TemporaryDirectory() as temporary:
            prediction_path = Path(temporary) / "predictions.jsonl"
            write_jsonl(prediction_path, predictions)
            result = evaluate(
                "outcome", prediction_path, ROOT / "fixtures" / "outcome_annotations.jsonl"
            )
        self.assertEqual(result["numerator"], 2)
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(len(result["items"]), 2)


if __name__ == "__main__":
    unittest.main()
