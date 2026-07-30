#!/usr/bin/env python3
"""Validate the identifier-free publication-data release."""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from build_publication_data import (
    CHECKSUM_FILE,
    DISEASE_FIELDS,
    DISEASE_FILE,
    DISEASE_YEAR_FIELDS,
    DISEASE_YEAR_FILE,
    DISEASE_YEAR_OUTCOME_FIELDS,
    DISEASE_YEAR_OUTCOME_FILE,
    EXPECTED,
    MANIFEST_FILE,
    RELEASE_DIR,
    validate_identifier_free,
)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_checksums() -> None:
    lines = [
        line
        for line in (RELEASE_DIR / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(lines) != 4:
        raise AssertionError(f"Expected 4 SHA256 entries; found {len(lines)}")
    for line in lines:
        expected, filename = line.split("  ", 1)
        path = RELEASE_DIR / filename
        if not path.is_file() or _sha256(path) != expected:
            raise AssertionError(f"Checksum validation failed for {filename}")


def validate_release(*, deep: bool = False) -> dict[str, int]:
    manifest = json.loads((RELEASE_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    _validate_checksums()

    expected_columns = {
        DISEASE_FILE: DISEASE_FIELDS,
        DISEASE_YEAR_FILE: DISEASE_YEAR_FIELDS,
        DISEASE_YEAR_OUTCOME_FILE: DISEASE_YEAR_OUTCOME_FIELDS,
    }
    frames = {
        filename: pd.read_csv(RELEASE_DIR / filename)
        for filename in expected_columns
    }
    for filename, frame in frames.items():
        if list(frame.columns) != expected_columns[filename]:
            raise AssertionError(f"Unexpected columns in {filename}: {list(frame.columns)}")
        validate_identifier_free(frame, filename)

    disease = frames[DISEASE_FILE]
    disease_year = frames[DISEASE_YEAR_FILE]
    disease_year_outcome = frames[DISEASE_YEAR_OUTCOME_FILE]
    observed = {
        "cohort_trials": int(disease["trial_count"].sum()),
        "mapped_diseases": int(disease["disease"].nunique()),
        "disease_areas": int(disease["disease_area"].nunique()),
        "diseases_over_100_trials": int((disease["trial_count"] > 100).sum()),
        "dated_trials": int(disease_year["trial_count"].sum()),
        "undated_trials": int(disease["trial_count"].sum() - disease_year["trial_count"].sum()),
        "dated_outcome_records": int(disease_year_outcome["outcome_record_count"].sum()),
        "outcome_categories": int(disease_year_outcome["outcome_category"].nunique()),
    }
    if observed != EXPECTED:
        raise AssertionError(f"Release counts changed: {observed} != {EXPECTED}")
    if manifest["observed_counts"] != EXPECTED:
        raise AssertionError("Manifest counts do not match the locked release contract")

    assets = {item["file"]: item for item in manifest["assets"]}
    for filename, frame in frames.items():
        if assets[filename]["rows_excluding_header"] != len(frame):
            raise AssertionError(f"Manifest row count mismatch for {filename}")
        if assets[filename]["sha256"] != _sha256(RELEASE_DIR / filename):
            raise AssertionError(f"Manifest checksum mismatch for {filename}")

    if deep:
        disease_groups = disease.set_index("disease")["disease_area"].to_dict()
        for frame_name, frame in (
            (DISEASE_YEAR_FILE, disease_year),
            (DISEASE_YEAR_OUTCOME_FILE, disease_year_outcome),
        ):
            observed_groups = frame.set_index("disease")["disease_area"].to_dict()
            if any(disease_groups.get(name) != area for name, area in observed_groups.items()):
                raise AssertionError(f"Disease-area labels disagree in {frame_name}")
        if set(disease_year["disease"]) - set(disease["disease"]):
            raise AssertionError("Annual disease file contains an unknown disease")
        if set(disease_year_outcome["disease"]) - set(disease["disease"]):
            raise AssertionError("Annual outcome file contains an unknown disease")
        if not disease_year["start_year"].between(1900, 2100).all():
            raise AssertionError("Annual disease file contains an implausible year")
        if not disease_year_outcome["start_year"].between(1900, 2100).all():
            raise AssertionError("Annual outcome file contains an implausible year")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_release(deep=args.deep), indent=2, sort_keys=True))
    print("Identifier-free publication-data validation passed.")


if __name__ == "__main__":
    main()
