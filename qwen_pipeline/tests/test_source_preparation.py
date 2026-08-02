from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_pipeline.cli import main  # noqa: E402
from qwen_pipeline.io import read_jsonl  # noqa: E402
from qwen_pipeline.sources import (  # noqa: E402
    CONDITION_VOCABULARY_FILE,
    CONDITIONS_FILE,
    MANIFEST_FILE,
    OUTCOMES_FILE,
    SourceContractError,
    prepare_sources,
    write_prepared_sources,
)
from qwen_pipeline.validation import validate_schema  # noqa: E402


FIXTURES = ROOT / "fixtures" / "source_data"
AACT = FIXTURES / "aact_20251231"
SNOMED = FIXTURES / "snomed_international_20250701"


class SourcePreparationTests(unittest.TestCase):
    def test_prepares_aact_records_without_collapsing_distinct_outcome_ids(self) -> None:
        prepared = prepare_sources(AACT, SNOMED)
        self.assertEqual(len(prepared.outcomes), 2)
        self.assertEqual([row["outcome_id"] for row in prepared.outcomes], ["O1", "O2"])
        self.assertEqual([row["title"] for row in prepared.outcomes], ["Shared endpoint", "Shared endpoint"])
        self.assertEqual(prepared.outcomes[0]["record_id"], "outcome:NCT00000001:O1")
        self.assertEqual(len(prepared.conditions), 2)
        self.assertEqual(
            [row["raw_condition"] for row in prepared.conditions],
            ["Example disorder A", "Example disorder B"],
        )

    def test_preserves_conditions_that_differ_only_by_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            aact = Path(temporary) / "aact"
            shutil.copytree(AACT, aact)
            with (aact / "conditions.txt").open("a", encoding="utf-8") as handle:
                handle.write("NCT00000001|example disorder a\n")
            prepared = prepare_sources(aact, SNOMED)
            self.assertEqual(len(prepared.conditions), 3)
            self.assertEqual(len({row["record_id"] for row in prepared.conditions}), 3)

    def test_builds_active_disorder_vocabulary_with_preferred_term_fallback(self) -> None:
        prepared = prepare_sources(AACT, SNOMED)
        by_concept = {row["concept_id"]: row for row in prepared.condition_vocabulary}
        self.assertEqual(set(by_concept), {"100001", "100002"})
        self.assertEqual(by_concept["100001"]["preferred_term"], "Preferred example A")
        self.assertEqual(by_concept["100002"]["preferred_term"], "Example disorder B")
        self.assertNotIn("100003", by_concept)
        self.assertNotIn("200001", by_concept)
        self.assertEqual(
            prepared.manifest_base["snomed_ct"]["counts"]["preferred_term_fallbacks"], 1
        )

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "prepare-sources",
                        "--aact-dir",
                        str(AACT),
                        "--snomed-rf2-dir",
                        str(SNOMED),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "dry_run_not_written")

    def test_execute_writes_schema_valid_outputs_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "prepare-sources",
                        "--aact-dir",
                        str(AACT),
                        "--snomed-rf2-dir",
                        str(SNOMED),
                        "--aact-snapshot-id",
                        "AACT-2025-12-31",
                        "--snomed-edition",
                        "SNOMED CT International Edition",
                        "--output-dir",
                        str(output),
                        "--execute",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "written")
            for filename in (OUTCOMES_FILE, CONDITIONS_FILE, CONDITION_VOCABULARY_FILE, MANIFEST_FILE):
                self.assertTrue((output / filename).is_file())
            for row in read_jsonl(output / OUTCOMES_FILE):
                validate_schema(row, "raw_outcome.schema.json")
            for row in read_jsonl(output / CONDITIONS_FILE):
                validate_schema(row, "raw_condition.schema.json")
            for row in read_jsonl(output / CONDITION_VOCABULARY_FILE):
                validate_schema(row, "vocabulary.schema.json")
            manifest = json.loads((output / MANIFEST_FILE).read_text(encoding="utf-8"))
            validate_schema(manifest, "source_manifest.schema.json")
            self.assertEqual(manifest["aact"]["snapshot_id"], "AACT-2025-12-31")
            self.assertEqual(manifest["aact"]["snapshot_date"], "2025-12-31")
            self.assertEqual(manifest["snomed_ct"]["version_date"], "2025-07-01")
            self.assertEqual(len(manifest["outputs"]), 3)
            self.assertEqual(manifest["aact"]["counts"]["outcomes_with_measurements"], 2)

    def test_manifest_passes_full_jsonschema_when_dependency_is_available(self) -> None:
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("jsonschema is not installed in the lightweight test environment")
        prepared = prepare_sources(AACT, SNOMED)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_prepared_sources(prepared, temporary)
        schema = json.loads(
            (ROOT / "schemas" / "source_manifest.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(manifest)), [])

    def test_outputs_and_manifest_are_deterministic(self) -> None:
        prepared = prepare_sources(AACT, SNOMED)
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            manifest_one = write_prepared_sources(prepared, first)
            manifest_two = write_prepared_sources(prepared, second)
            self.assertEqual(manifest_one, manifest_two)
            self.assertEqual(
                (first / MANIFEST_FILE).read_bytes(), (second / MANIFEST_FILE).read_bytes()
            )
            for filename in (OUTCOMES_FILE, CONDITIONS_FILE, CONDITION_VOCABULARY_FILE):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_missing_source_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            aact = Path(temporary) / "aact"
            shutil.copytree(AACT, aact)
            (aact / "conditions.txt").write_text("nct_id|wrong\nNCT1|value\n", encoding="utf-8")
            with self.assertRaisesRegex(SourceContractError, "missing required columns"):
                prepare_sources(aact, SNOMED)

    def test_inferred_relationship_snapshot_takes_precedence_over_stated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snomed = Path(temporary) / "snomed"
            shutil.copytree(SNOMED, snomed)
            stated = (
                snomed
                / "Snapshot"
                / "Terminology"
                / "sct2_StatedRelationship_Snapshot_INT_20250701.txt"
            )
            stated.write_text(
                "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\t"
                "relationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId\n"
                "800001\t20250701\t1\t900000000000207008\t200001\t64572001\t0\t"
                "116680003\t900000000000010007\t900000000000451002\n",
                encoding="utf-8",
            )
            prepared = prepare_sources(AACT, snomed)
            self.assertEqual(
                {row["concept_id"] for row in prepared.condition_vocabulary},
                {"100001", "100002"},
            )

    def test_infers_snomed_edition_from_snapshot_parent(self) -> None:
        prepared = prepare_sources(AACT, SNOMED / "Snapshot")
        self.assertEqual(
            prepared.manifest_base["snomed_ct"]["edition_identifier"],
            "snomed_international_20250701",
        )


if __name__ == "__main__":
    unittest.main()
