"""Fast regression test for the generated publication-data release."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PUBLICATION_DATA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLICATION_DATA_DIR))

from validate_publication_data import validate_release  # noqa: E402
from validate_outcome_release import validate_release as validate_outcome_release  # noqa: E402


class PublicationDataReleaseTest(unittest.TestCase):
    def test_release_contract(self) -> None:
        observed = validate_release(deep=False)
        self.assertEqual(observed["cohort_trials"], 73_427)
        self.assertEqual(observed["mapped_diseases"], 274)
        self.assertEqual(observed["disease_areas"], 24)
        self.assertEqual(observed["diseases_over_100_trials"], 213)
        self.assertEqual(observed["dated_trials"], 73_410)
        self.assertEqual(observed["dated_outcome_records"], 467_865)

    def test_outcome_release_contract(self) -> None:
        observed = validate_outcome_release(deep=False)
        self.assertEqual(observed["linked_outcome_records"], 467_903)
        self.assertEqual(observed["records_with_multiple_normalizations"], 947)
        self.assertEqual(observed["records_with_multiple_categories"], 124)


if __name__ == "__main__":
    unittest.main()
