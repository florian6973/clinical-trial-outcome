#!/usr/bin/env python3
"""Package the outcome mapping as a shareable, outcome-only release.

This script does not train a model, generate predictions, or recompute a study
result. It performs a deterministic packaging transformation on the locked
outcome-by-condition mapping already used for the paper. Because that source is
expanded across conditions, the release contains one row per unique
``(nct_id, aact_outcome_title)`` pair. Where the source contains more
than one value for a field, the modal nonblank value is selected with an
alphabetical tie-break and the number of observed variants is retained.

The resulting 467,903-row file is deliberately distinguished from the paper's
480,273 row-level AACT outcome-record denominator.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_DATA_DIR = Path(__file__).resolve().parent
RELEASE_DIR = PUBLICATION_DATA_DIR / "release"
DEFAULT_SOURCE = PROJECT_ROOT / "data/output/research/outcome_and_condition_aact_mapping.csv"

OUTCOME_FILE = "linked_outcome_records.csv.gz"
MANIFEST_FILE = "outcome_release_manifest.json"
CHECKSUM_FILE = "OUTCOME_SHA256SUMS"

SOURCE_COLUMNS = [
    "nct_id",
    "aact_outcome_title",
    "aact_outcome_description",
    "outcome_normalized",
    "outcome_category",
    "outcome_coa_type",
]
KEY_COLUMNS = ["nct_id", "aact_outcome_title"]
VALUE_COLUMNS = [
    "aact_outcome_description",
    "outcome_normalized",
    "outcome_category",
    "outcome_coa_type",
]
VARIANT_COLUMNS = {
    "aact_outcome_description": "n_description_variants",
    "outcome_normalized": "n_normalized_variants",
    "outcome_category": "n_category_variants",
    "outcome_coa_type": "n_coa_variants",
}
OUTPUT_COLUMNS = KEY_COLUMNS + VALUE_COLUMNS + list(VARIANT_COLUMNS.values())

EXPECTED = {
    "linked_outcome_records": 467_903,
    "distinct_raw_title_strings": 390_228,
    "source_normalized_strings": 243_916,
    "records_with_multiple_normalizations": 947,
    "records_with_multiple_categories": 124,
}
PAPER_OUTCOME_RECORDS = 480_273


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def modal_by_keys(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Select the modal nonblank value with an alphabetical tie-break."""

    work = frame[KEY_COLUMNS + [value]].dropna(subset=[value]).copy()
    counts = (
        work.groupby(KEY_COLUMNS + [value], dropna=False, sort=False)
        .size()
        .rename("_n")
        .reset_index()
    )
    counts = counts.sort_values(
        KEY_COLUMNS + ["_n", value],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    return counts.drop_duplicates(KEY_COLUMNS, keep="first")[KEY_COLUMNS + [value]]


def load_source(source: Path) -> pd.DataFrame:
    if not source.is_file():
        raise FileNotFoundError(f"Missing locked mapping file: {source}")
    chunks = pd.read_csv(
        source,
        usecols=SOURCE_COLUMNS,
        dtype="string",
        chunksize=100_000,
        keep_default_na=False,
    )
    frame = pd.concat(chunks, ignore_index=True)
    for column in SOURCE_COLUMNS:
        frame[column] = frame[column].str.strip().mask(lambda values: values.eq(""))
    if frame["nct_id"].isna().any() or frame["aact_outcome_title"].isna().any():
        raise ValueError("The locked source contains a blank outcome-release key")
    return frame


def build_endpoint_rows(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(KEY_COLUMNS, dropna=False, sort=True)
    endpoint = grouped.agg(
        **{
            output_name: (source_name, "nunique")
            for source_name, output_name in VARIANT_COLUMNS.items()
        }
    ).reset_index()
    for value in VALUE_COLUMNS:
        endpoint = endpoint.merge(
            modal_by_keys(frame, value),
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    endpoint = endpoint[OUTPUT_COLUMNS].sort_values(
        KEY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    return endpoint


def write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            text_handle = io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="")
            frame.to_csv(text_handle, index=False, lineterminator="\n", na_rep="")
            text_handle.flush()
            text_handle.detach()


def build_release(source: Path = DEFAULT_SOURCE) -> dict[str, object]:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_source(source)
    endpoint = build_endpoint_rows(frame)

    observed = {
        "linked_outcome_records": len(endpoint),
        "distinct_raw_title_strings": frame["aact_outcome_title"].nunique(),
        "source_normalized_strings": frame["outcome_normalized"].nunique(),
        "records_with_multiple_normalizations": int(
            endpoint["n_normalized_variants"].gt(1).sum()
        ),
        "records_with_multiple_categories": int(
            endpoint["n_category_variants"].gt(1).sum()
        ),
    }
    if observed != EXPECTED:
        raise ValueError(f"Outcome-release counts changed: {observed} != {EXPECTED}")

    outcome_path = RELEASE_DIR / OUTCOME_FILE
    write_deterministic_gzip(endpoint, outcome_path)
    manifest = {
        "release_format_version": "1.0",
        "release_status": "shareable_outcome_only_extract",
        "source": {
            "source_path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        },
        "analysis_unit": "one unique NCT ID plus raw outcome title",
        "transformation": (
            "Removed condition and terminology columns; deduplicated the condition-expanded "
            "source by NCT ID and raw outcome title; selected modal field values with "
            "alphabetical tie-breaks; retained per-field variant counts."
        ),
        "observed_counts": observed,
        "paper_denominator_reconciliation": {
            "paper_mapped_outcome_records": PAPER_OUTCOME_RECORDS,
            "released_unique_trial_title_records": len(endpoint),
            "explanation": (
                "The paper count refers to row-level AACT outcome records. The source "
                "mapping lacks an AACT outcome-row identifier, so the public extract uses "
                "the reproducible NCT ID plus raw title key and must not replace the paper "
                "denominator."
            ),
        },
        "asset": {
            "file": OUTCOME_FILE,
            "format": "UTF-8 CSV compressed with deterministic gzip",
            "rows_excluding_header": len(endpoint),
            "columns": OUTPUT_COLUMNS,
            "bytes": outcome_path.stat().st_size,
            "sha256": sha256(outcome_path),
        },
    }
    manifest_path = RELEASE_DIR / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [outcome_path, manifest_path]
    (RELEASE_DIR / CHECKSUM_FILE).write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n" for path in sorted(checksum_paths)
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Locked condition-expanded outcome mapping used to build the release.",
    )
    args = parser.parse_args()
    manifest = build_release(args.source)
    print(json.dumps(manifest["observed_counts"], indent=2, sort_keys=True))
    print(f"Wrote {RELEASE_DIR / OUTCOME_FILE}")


if __name__ == "__main__":
    main()
