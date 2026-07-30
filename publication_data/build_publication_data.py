#!/usr/bin/env python3
"""Build the identifier-free condition and longitudinal dashboard release.

The builder reads the locked local analytic assets but exports only project
labels and aggregate counts. It never writes SNOMED CT identifiers, terminology
descriptions, hierarchy relationships, or reference-set content.
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

INTERNAL_LABEL_SOURCE = (
    PROJECT_ROOT
    / "1_data/1_condition_mapping/9_snomed_ai_mapping/final_snomed_mapping.csv"
)
TRIAL_SOURCE = PROJECT_ROOT / "data/output/research/research_trials.parquet"
OUTCOME_SOURCE = RELEASE_DIR / "linked_outcome_records.csv.gz"

DISEASE_FILE = "disease_crosswalk.csv.gz"
DISEASE_YEAR_FILE = "disease_year_counts.csv.gz"
DISEASE_YEAR_OUTCOME_FILE = "disease_year_outcome_counts.csv.gz"
MANIFEST_FILE = "release_manifest.json"
CHECKSUM_FILE = "SHA256SUMS"

DISEASE_FIELDS = ["disease_area", "disease", "trial_count"]
DISEASE_YEAR_FIELDS = ["disease_area", "disease", "start_year", "trial_count"]
DISEASE_YEAR_OUTCOME_FIELDS = [
    "disease_area",
    "disease",
    "start_year",
    "outcome_category",
    "outcome_record_count",
    "trial_count",
]

EXPECTED = {
    "cohort_trials": 73_427,
    "mapped_diseases": 274,
    "disease_areas": 24,
    "diseases_over_100_trials": 213,
    "dated_trials": 73_410,
    "undated_trials": 17,
    "dated_outcome_records": 467_865,
    "outcome_categories": 21,
}

FORBIDDEN_COLUMN_TOKENS = (
    "snomed",
    "sctid",
    "concept_id",
    "condition_code",
    "disease_area_code",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_identifier_free(frame: pd.DataFrame, name: str) -> None:
    """Reject terminology columns and code-like values in text fields."""

    prohibited = [
        column
        for column in frame.columns
        if any(token in column.casefold() for token in FORBIDDEN_COLUMN_TOKENS)
    ]
    if prohibited:
        raise ValueError(f"{name} contains prohibited terminology columns: {prohibited}")
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna().astype("string").str.strip()
        if values.str.fullmatch(r"[0-9]{6,18}").any():
            raise ValueError(f"{name}.{column} contains a code-like numeric identifier")


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
            frame.to_csv(text_handle, index=False, lineterminator="\n")
            text_handle.flush()
            text_handle.detach()


def build_frames(
    internal_label_source: Path = INTERNAL_LABEL_SOURCE,
    trial_source: Path = TRIAL_SOURCE,
    outcome_source: Path = OUTCOME_SOURCE,
) -> dict[str, pd.DataFrame]:
    for source in (internal_label_source, trial_source, outcome_source):
        if not source.is_file():
            raise FileNotFoundError(source)

    labels = pd.read_csv(
        internal_label_source,
        usecols=["mapped_indication", "disease_area"],
        dtype="string",
    ).rename(columns={"mapped_indication": "disease"})
    if (
        len(labels) != EXPECTED["mapped_diseases"]
        or labels["disease"].nunique() != EXPECTED["mapped_diseases"]
        or labels["disease_area"].nunique() != EXPECTED["disease_areas"]
    ):
        raise ValueError("Internal label crosswalk dimensions changed")

    outcomes = pd.read_csv(
        outcome_source,
        usecols=["nct_id", "outcome_category"],
        dtype="string",
    )
    cohort_ids = set(outcomes["nct_id"].dropna().unique())
    if len(cohort_ids) != EXPECTED["cohort_trials"]:
        raise ValueError("Outcome release no longer defines the locked trial cohort")

    trials = pd.read_parquet(
        trial_source,
        columns=["nct_id", "mapped_indication", "start_year"],
    ).rename(columns={"mapped_indication": "disease"})
    trials["nct_id"] = trials["nct_id"].astype("string")
    trials["disease"] = trials["disease"].astype("string")
    trials = trials.loc[trials["nct_id"].isin(cohort_ids)].copy()
    if len(trials) != EXPECTED["cohort_trials"] or trials["nct_id"].nunique() != len(trials):
        raise ValueError("Trial lookup is not one-to-one with the locked cohort")
    trials = trials.merge(labels, on="disease", how="left", validate="many_to_one")
    if trials["disease_area"].isna().any():
        raise ValueError("At least one mapped disease lacks a disease-area label")

    disease = (
        trials.groupby(["disease_area", "disease"], observed=True)["nct_id"]
        .nunique()
        .rename("trial_count")
        .reset_index()
        .sort_values(["trial_count", "disease_area", "disease"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    if (
        len(disease) != EXPECTED["mapped_diseases"]
        or int(disease["trial_count"].sum()) != EXPECTED["cohort_trials"]
        or int((disease["trial_count"] > 100).sum()) != EXPECTED["diseases_over_100_trials"]
    ):
        raise ValueError("Disease-level unique-trial counts changed")

    dated_trials = trials.dropna(subset=["start_year"]).copy()
    dated_trials["start_year"] = pd.to_numeric(
        dated_trials["start_year"], errors="raise"
    ).astype(int)
    disease_year = (
        dated_trials.groupby(
            ["disease_area", "disease", "start_year"], observed=True
        )["nct_id"]
        .nunique()
        .rename("trial_count")
        .reset_index()
        .sort_values(["start_year", "disease_area", "disease"])
        .reset_index(drop=True)
    )
    if int(disease_year["trial_count"].sum()) != EXPECTED["dated_trials"]:
        raise ValueError("Dated disease-trial counts changed")

    outcome_year = outcomes.merge(
        dated_trials[["nct_id", "disease", "disease_area", "start_year"]],
        on="nct_id",
        how="inner",
        validate="many_to_one",
    )
    disease_year_outcome = (
        outcome_year.groupby(
            ["disease_area", "disease", "start_year", "outcome_category"],
            observed=True,
        )
        .agg(outcome_record_count=("nct_id", "size"), trial_count=("nct_id", "nunique"))
        .reset_index()
        .sort_values(["start_year", "disease_area", "disease", "outcome_category"])
        .reset_index(drop=True)
    )
    if (
        int(disease_year_outcome["outcome_record_count"].sum())
        != EXPECTED["dated_outcome_records"]
        or disease_year_outcome["outcome_category"].nunique()
        != EXPECTED["outcome_categories"]
    ):
        raise ValueError("Dated disease-outcome counts changed")

    frames = {
        DISEASE_FILE: disease[DISEASE_FIELDS],
        DISEASE_YEAR_FILE: disease_year[DISEASE_YEAR_FIELDS],
        DISEASE_YEAR_OUTCOME_FILE: disease_year_outcome[DISEASE_YEAR_OUTCOME_FIELDS],
    }
    for filename, frame in frames.items():
        validate_identifier_free(frame, filename)
    return frames


def build_release(
    internal_label_source: Path = INTERNAL_LABEL_SOURCE,
    trial_source: Path = TRIAL_SOURCE,
    outcome_source: Path = OUTCOME_SOURCE,
) -> dict[str, object]:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    frames = build_frames(internal_label_source, trial_source, outcome_source)
    assets = []
    for filename, frame in frames.items():
        path = RELEASE_DIR / filename
        write_deterministic_gzip(frame, path)
        assets.append(
            {
                "file": filename,
                "rows_excluding_header": len(frame),
                "columns": list(frame.columns),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "release_format_version": "2.0",
        "release_status": "public_label_only_no_SNOMED_CT_identifiers",
        "analysis_unit": {
            DISEASE_FILE: "one mapped disease",
            DISEASE_YEAR_FILE: "one mapped disease and trial start year",
            DISEASE_YEAR_OUTCOME_FILE: (
                "one mapped disease, trial start year, and outcome category"
            ),
        },
        "public_fields": (
            "Project disease labels, analytic disease-area labels, years, outcome "
            "categories, and aggregate counts only"
        ),
        "excluded_content": (
            "SNOMED CT identifiers, terminology descriptions, hierarchy relationships, "
            "reference-set content, and raw condition strings"
        ),
        "source_files": [
            {
                "source_path": str(trial_source),
                "bytes": trial_source.stat().st_size,
                "sha256": sha256(trial_source),
            },
            {
                "source_path": str(outcome_source),
                "bytes": outcome_source.stat().st_size,
                "sha256": sha256(outcome_source),
            },
        ],
        "observed_counts": EXPECTED,
        "assets": assets,
    }
    manifest_path = RELEASE_DIR / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [RELEASE_DIR / filename for filename in frames] + [manifest_path]
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
        "--condition-crosswalk",
        type=Path,
        default=INTERNAL_LABEL_SOURCE,
        help="Reviewed condition-to-disease-area label file.",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=TRIAL_SOURCE,
        help="Trial-level file with nct_id, mapped_indication, and start_year.",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=OUTCOME_SOURCE,
        help="Outcome-only release used to define the cohort and categories.",
    )
    args = parser.parse_args()
    manifest = build_release(args.condition_crosswalk, args.trials, args.outcomes)
    print(json.dumps(manifest["observed_counts"], indent=2, sort_keys=True))
    print(f"Wrote identifier-free release to {RELEASE_DIR}")


if __name__ == "__main__":
    main()
