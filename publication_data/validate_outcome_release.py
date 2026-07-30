#!/usr/bin/env python3
"""Validate the shareable outcome-only publication extract."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from build_outcome_release import (
    CHECKSUM_FILE,
    EXPECTED,
    MANIFEST_FILE,
    OUTCOME_FILE,
    OUTPUT_COLUMNS,
    RELEASE_DIR,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release(*, deep: bool = False) -> dict[str, int]:
    checksum_lines = (RELEASE_DIR / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines()
    if len(checksum_lines) != 2:
        raise AssertionError(f"Expected 2 outcome checksum entries; found {len(checksum_lines)}")
    for line in checksum_lines:
        expected_hash, filename = line.split("  ", 1)
        path = RELEASE_DIR / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise AssertionError(f"Checksum failure for {filename}")

    manifest = json.loads((RELEASE_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    outcome_path = RELEASE_DIR / OUTCOME_FILE
    if manifest["asset"]["sha256"] != sha256(outcome_path):
        raise AssertionError("Outcome asset hash does not match its manifest")

    rows = 0
    normalized_conflicts = 0
    category_conflicts = 0
    raw_titles: set[str] | None = set() if deep else None
    previous_key: tuple[str, str] | None = None
    with gzip.open(outcome_path, mode="rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise AssertionError(f"Unexpected outcome columns: {reader.fieldnames}")
        for row in reader:
            rows += 1
            key = (row["nct_id"], row["aact_outcome_title"])
            if not all(key):
                raise AssertionError(f"Blank release key at row {rows + 1}")
            if previous_key is not None and key <= previous_key:
                raise AssertionError("Outcome release is not strictly sorted by its unique key")
            previous_key = key
            normalized_conflicts += int(row["n_normalized_variants"]) > 1
            category_conflicts += int(row["n_category_variants"]) > 1
            if raw_titles is not None:
                raw_titles.add(row["aact_outcome_title"])

    observed = {
        "linked_outcome_records": rows,
        "records_with_multiple_normalizations": normalized_conflicts,
        "records_with_multiple_categories": category_conflicts,
    }
    for key, value in observed.items():
        if value != EXPECTED[key]:
            raise AssertionError(f"Expected {EXPECTED[key]} for {key}; found {value}")
    if manifest["asset"]["rows_excluding_header"] != rows:
        raise AssertionError("Outcome manifest row count does not match the asset")
    if manifest["observed_counts"] != EXPECTED:
        raise AssertionError("Outcome manifest counts do not match the release contract")
    if deep:
        if len(raw_titles or ()) != EXPECTED["distinct_raw_title_strings"]:
            raise AssertionError("Distinct raw-title count changed")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_release(deep=args.deep), indent=2, sort_keys=True))
    print("Outcome release validation passed.")


if __name__ == "__main__":
    main()
