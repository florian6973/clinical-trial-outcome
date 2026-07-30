#!/usr/bin/env python3
"""Build deterministic analyses requested during advisor review.

This script is intentionally read-only against the locked research inputs.  It
uses ``outcome_and_condition_aact_mapping.csv`` as the cohort authority and
joins trial phase/completion metadata from ``research_trials.parquet`` only by
``nct_id``.  All derived files are replaced atomically under
``output/advisor_revision``.

Analysis units are explicit:

* endpoint record: one unique (NCT ID, raw outcome title), with rare category
  conflicts resolved by modal value and alphabetical tie-break;
* endpoint-domain record: one unique (NCT ID, raw outcome title, disease
  domain), used for the descriptive 21 x 24 association analysis;
* trial-category presence: one unique (NCT ID, outcome category), used for
  phase and time prevalence so trials with many endpoints are not overweighted;
* distinct title string: one unique raw title, used for lexical complexity.

The mapping-difficulty casebook is descriptive triage, not a formal error
analysis or performance estimate.

The default invocation is a dry run that prints the input/output contract and
does not read the large inputs or write results. Execution must be requested
explicitly; it was not performed as part of the manuscript revision.

Usage (plan only):
    python publication_data/scripts/descriptive_analyses.py

Usage:
    python publication_data/scripts/descriptive_analyses.py \
        --mapping /path/to/outcome_and_condition_aact_mapping.csv \
        --trials /path/to/research_trials.parquet \
        --output-dir /path/to/output \
        --execute
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = REPO_ROOT / "data/output/research/outcome_and_condition_aact_mapping.csv"
DEFAULT_TRIALS = REPO_ROOT / "data/output/research/research_trials.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "publication_data/advisor_requested_descriptive_analyses"

MAPPING_COLUMNS = [
    "nct_id",
    "aact_condition_name",
    "aact_outcome_title",
    "aact_outcome_description",
    "outcome_normalized",
    "outcome_category",
    "outcome_coa_type",
    "snomed_condition_code",
    "snomed_condition_description",
    "condition_category_code",
    "snomed_condition_category_description",
]

EARLY_YEARS = (2005, 2010)
RECENT_YEARS = (2020, 2025)
ANALYSIS_YEARS = tuple(range(2000, 2026))
PHASE_ORDER = ["Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Unknown"]

# These are lexical indicators, not a clinical NLP gold standard.  Patterns
# are deliberately conservative and documented in the generated definitions.
ASPECT_DEFINITIONS = OrderedDict(
    [
        (
            "temporal_expression",
            (
                r"\b(?:at|by|during|over|through|until|within|after|before|follow[- ]?up)\b|"
                r"\b(?:minute|hour|day|week|month|year)s?\b|\btime\s+to\b",
                "Explicit timing/window language, calendar units, follow-up, or time-to phrasing.",
            ),
        ),
        (
            "measurement_operator",
            (
                r"\b(?:change|difference|number|count|percentage|percent|proportion|rate|ratio|"
                r"mean|median|incidence|prevalence|duration|frequency|concentration|level|score|"
                r"area under|maximum|min(?:imum)?|time to)\b",
                "Quantification/operator language such as change, number, rate, score, or concentration.",
            ),
        ),
        (
            "baseline_or_comparator",
            (
                r"\b(?:baseline|placebo|control|comparator|versus|vs\.?|compared with|relative to|"
                r"from baseline)\b",
                "A baseline, treatment comparator, or explicit contrast is named.",
            ),
        ),
        (
            "instrument_or_scale",
            (
                r"\b(?:scale|questionnaire|index|inventory|survey|assessment|instrument|score|"
                r"eq-5d|sf-36|vas|eortc|fact-[a-z0-9]+|promis)\b",
                "A named or generic scale, questionnaire, index, survey, or assessment is present.",
            ),
        ),
        (
            "explicit_unit",
            (
                r"(?:\bmg(?:/dl|/l)?\b|\bmmol/l\b|\bmmhg\b|\bkg\b|\bcm\b|\bml\b|\bl/min\b|"
                r"\bbpm\b|\bseconds?\b|\bminutes?\b|\bhours?\b|\bdays?\b|\bweeks?\b|"
                r"\bmonths?\b|\byears?\b|%)",
                "An explicit physical, laboratory, percentage, or time unit is present.",
            ),
        ),
        (
            "composite_or_multiple_objects",
            (
                r"\b(?:composite|combined endpoint|co-primary|multiple endpoints?)\b|\band/or\b|"
                r"\b(?:death|mortality)\s*,?\s+(?:mi|myocardial infarction|stroke)\b",
                "Language strongly suggesting a composite or multiple outcome objects.",
            ),
        ),
        (
            "participant_population",
            (
                r"\b(?:participant|patient|subject|caregiver|observer)s?\b",
                "The assessed participant, patient, subject, caregiver, or observer is explicit.",
            ),
        ),
    ]
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--casebook-per-stratum", type=int, default=8)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the analyses and write outputs. Without this flag, print the plan only.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n", **kwargs)
    os.replace(tmp, path)


def atomic_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def savefig(fig: plt.Figure, path: Path) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    os.replace(tmp, path)


def modal_by_keys(frame: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Resolve a modal string value with an alphabetical tie-break, vectorized."""
    work = frame[keys + [value]].dropna(subset=[value]).copy()
    counts = work.groupby(keys + [value], dropna=False, sort=False).size().rename("_n").reset_index()
    counts = counts.sort_values(keys + ["_n", value], ascending=[True] * len(keys) + [False, True])
    return counts.drop_duplicates(keys, keep="first")[keys + [value]]


def load_inputs(mapping_path: Path, trials_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    if not trials_path.exists():
        raise FileNotFoundError(trials_path)
    # Chunked ingestion bounds parser memory and makes progress independent of
    # the 404 MB CSV's physical size.  The concatenated frame retains only the
    # eleven documented analysis columns.
    chunks = pd.read_csv(
        mapping_path,
        usecols=MAPPING_COLUMNS,
        dtype="string",
        chunksize=100_000,
    )
    mapping = pd.concat(chunks, ignore_index=True)
    for column in MAPPING_COLUMNS:
        mapping[column] = mapping[column].str.strip()
        mapping[column] = mapping[column].mask(mapping[column].eq(""))
    trials = pd.read_parquet(
        trials_path,
        columns=["nct_id", "phase_clean", "start_year", "completion_year"],
    )
    trials["nct_id"] = trials["nct_id"].astype("string").str.strip()
    trials = trials.sort_values("nct_id").drop_duplicates("nct_id", keep="first")
    trials["phase_clean"] = trials["phase_clean"].fillna("Unknown").astype(str)
    trials["completion_year"] = pd.to_numeric(trials["completion_year"], errors="coerce")
    trials["start_year"] = pd.to_numeric(trials["start_year"], errors="coerce")
    return mapping, trials


def build_endpoint_records(mapping: pd.DataFrame) -> pd.DataFrame:
    keys = ["nct_id", "aact_outcome_title"]
    endpoint = mapping.groupby(keys, dropna=False, sort=True).agg(
        n_normalized_values=("outcome_normalized", "nunique"),
        n_category_values=("outcome_category", "nunique"),
        n_disease_domains=("snomed_condition_category_description", "nunique"),
    ).reset_index()
    for value in ["outcome_normalized", "outcome_category", "outcome_coa_type"]:
        endpoint = endpoint.merge(modal_by_keys(mapping, keys, value), on=keys, how="left", validate="one_to_one")
    return endpoint


def build_manifest(mapping_path: Path, trials_path: Path, out: Path) -> None:
    records = []
    for role, path in [("cohort_authority", mapping_path), ("trial_metadata_join", trials_path)]:
        stat = path.stat()
        records.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "sha256": sha256(path),
                "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
            }
        )
    atomic_csv(pd.DataFrame(records), out / "source_manifest.csv")


def cohort_audit(mapping: pd.DataFrame, trials: pd.DataFrame, endpoint: pd.DataFrame, out: Path) -> pd.DataFrame:
    cohort_ncts = set(mapping["nct_id"].dropna())
    trial_subset = trials[trials["nct_id"].isin(cohort_ncts)]
    metrics = [
        ("mapping_rows", len(mapping), "Rows in locked mapping CSV before de-duplication"),
        ("exact_duplicate_rows", int(mapping.duplicated().sum()), "Exact duplicates across all 11 source columns"),
        ("unique_trials", mapping["nct_id"].nunique(), "Distinct NCT IDs in locked mapping cohort"),
        ("unique_raw_condition_strings", mapping["aact_condition_name"].nunique(), "Distinct original AACT condition strings"),
        ("unique_raw_outcome_title_strings", mapping["aact_outcome_title"].nunique(), "Distinct title text, ignoring trial"),
        ("unique_trial_raw_title_records", len(endpoint), "Distinct (NCT ID, raw outcome title) endpoint records"),
        ("unique_normalized_outcome_strings", mapping["outcome_normalized"].nunique(), "Distinct normalized outcome strings"),
        ("unique_trial_normalized_outcomes", mapping[["nct_id", "outcome_normalized"]].drop_duplicates().shape[0], "Distinct (NCT ID, normalized outcome) pairs"),
        ("outcome_categories", mapping["outcome_category"].nunique(), "Distinct outcome categories"),
        ("coa_types", mapping["outcome_coa_type"].nunique(), "Distinct FDA COA types"),
        ("snomed_condition_codes", mapping["snomed_condition_code"].nunique(), "Distinct SNOMED condition codes"),
        ("snomed_condition_descriptions", mapping["snomed_condition_description"].nunique(), "Distinct SNOMED condition descriptions"),
        ("disease_domain_codes", mapping["condition_category_code"].nunique(), "Distinct disease-domain codes"),
        ("disease_domain_descriptions", mapping["snomed_condition_category_description"].nunique(), "Distinct disease-domain descriptions"),
        ("endpoint_records_with_multiple_normalizations", int((endpoint["n_normalized_values"] > 1).sum()), "Endpoint records with >1 normalized value across source rows"),
        ("endpoint_records_with_multiple_categories", int((endpoint["n_category_values"] > 1).sum()), "Endpoint records with >1 outcome category across source rows"),
        ("endpoint_records_spanning_multiple_domains", int((endpoint["n_disease_domains"] > 1).sum()), "Endpoint records linked to >1 disease domain"),
        ("trial_metadata_rows_in_cohort", len(trial_subset), "Cohort NCT IDs found in locked trial metadata"),
        ("cohort_trials_missing_trial_metadata", len(cohort_ncts - set(trial_subset["nct_id"])), "Cohort NCT IDs absent from metadata join"),
        ("cohort_trials_unknown_phase", int(trial_subset["phase_clean"].eq("Unknown").sum()), "Cohort trials with unknown phase"),
        ("cohort_trials_missing_completion_year", int(trial_subset["completion_year"].isna().sum()), "Cohort trials missing completion year"),
    ]
    audit = pd.DataFrame(metrics, columns=["metric", "value", "definition"])
    atomic_csv(audit, out / "cohort_count_audit.csv")

    missing = pd.DataFrame(
        {
            "column": MAPPING_COLUMNS,
            "missing_n": [int(mapping[c].isna().sum()) for c in MAPPING_COLUMNS],
            "missing_pct": [float(mapping[c].isna().mean() * 100) for c in MAPPING_COLUMNS],
            "n_unique_nonmissing": [int(mapping[c].nunique(dropna=True)) for c in MAPPING_COLUMNS],
        }
    )
    atomic_csv(missing, out / "cohort_field_completeness.csv", float_format="%.6f")
    return audit


def lexical_features(titles: pd.Series) -> pd.DataFrame:
    text = titles.fillna("").astype(str)
    lower = text.str.lower()
    features = pd.DataFrame(index=titles.index)
    features["character_count"] = text.str.len().astype(int)
    features["word_count"] = text.map(lambda value: len(WORD_RE.findall(value))).astype(int)
    for aspect, (pattern, _) in ASPECT_DEFINITIONS.items():
        features[aspect] = lower.str.contains(pattern, regex=True, na=False)
    aspects = list(ASPECT_DEFINITIONS)
    features["semantic_aspect_count"] = features[aspects].sum(axis=1).astype(int)

    def signature(row: pd.Series) -> str:
        present = [name for name in aspects if bool(row[name])]
        return "object_only" if not present else "object+" + "+".join(present)

    features["semantic_pattern"] = features.apply(signature, axis=1)
    return features


def summarize_numeric(values: pd.Series, name: str) -> dict[str, float | str]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "measure": name,
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def complexity_and_patterns(endpoint: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    title_frequency = endpoint["aact_outcome_title"].value_counts().rename_axis("aact_outcome_title").reset_index(name="endpoint_record_count")
    features = lexical_features(title_frequency["aact_outcome_title"])
    titles = pd.concat([title_frequency.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    summary = pd.DataFrame(
        [
            summarize_numeric(titles["word_count"], "words_per_distinct_title"),
            summarize_numeric(titles["character_count"], "characters_per_distinct_title"),
            summarize_numeric(titles["semantic_aspect_count"], "detected_semantic_aspects_per_distinct_title"),
        ]
    )
    atomic_csv(summary, out / "outcome_complexity_summary.csv", float_format="%.6f")

    bins = [-1, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, np.inf]
    labels = ["0-1", "2", "3", "4", "5", "6-10", "11-15", "16-20", "21-30", "31-50", "51+"]
    titles["word_count_bin"] = pd.cut(titles["word_count"], bins=bins, labels=labels)
    word_bins = titles.groupby("word_count_bin", observed=False).agg(
        n_distinct_titles=("aact_outcome_title", "size"),
        n_endpoint_records=("endpoint_record_count", "sum"),
    ).reset_index()
    word_bins["pct_distinct_titles"] = 100 * word_bins["n_distinct_titles"] / len(titles)
    word_bins["pct_endpoint_records"] = 100 * word_bins["n_endpoint_records"] / titles["endpoint_record_count"].sum()
    atomic_csv(word_bins, out / "outcome_complexity_word_bins.csv", float_format="%.6f")

    definitions = pd.DataFrame(
        [
            {
                "aspect": name,
                "definition": definition,
                "case_insensitive_regex": pattern,
                "interpretation_limit": "Deterministic lexical flag; not a validated clinical concept extractor.",
            }
            for name, (pattern, definition) in ASPECT_DEFINITIONS.items()
        ]
    )
    atomic_csv(definitions, out / "semantic_pattern_definitions.csv")

    aspect_rows = []
    for aspect in ASPECT_DEFINITIONS:
        mask = titles[aspect]
        aspect_rows.append(
            {
                "aspect": aspect,
                "n_distinct_titles": int(mask.sum()),
                "pct_distinct_titles": float(mask.mean() * 100),
                "n_endpoint_records": int(titles.loc[mask, "endpoint_record_count"].sum()),
                "pct_endpoint_records": float(100 * titles.loc[mask, "endpoint_record_count"].sum() / titles["endpoint_record_count"].sum()),
            }
        )
    atomic_csv(pd.DataFrame(aspect_rows), out / "semantic_aspect_frequencies.csv", float_format="%.6f")

    patterns = titles.groupby("semantic_pattern", sort=False).agg(
        n_distinct_titles=("aact_outcome_title", "size"),
        n_endpoint_records=("endpoint_record_count", "sum"),
    ).reset_index()
    patterns["pct_distinct_titles"] = 100 * patterns["n_distinct_titles"] / len(titles)
    patterns["pct_endpoint_records"] = 100 * patterns["n_endpoint_records"] / titles["endpoint_record_count"].sum()
    patterns = patterns.sort_values(["n_endpoint_records", "semantic_pattern"], ascending=[False, True]).reset_index(drop=True)
    patterns.insert(0, "rank", np.arange(1, len(patterns) + 1))
    atomic_csv(patterns, out / "semantic_pattern_frequencies.csv", float_format="%.6f")

    examples = []
    for pattern, group in titles.groupby("semantic_pattern", sort=False):
        group = group.sort_values(["endpoint_record_count", "aact_outcome_title"], ascending=[False, True]).head(3)
        for rank, row in enumerate(group.itertuples(index=False), 1):
            examples.append(
                {
                    "semantic_pattern": pattern,
                    "example_rank": rank,
                    "aact_outcome_title": row.aact_outcome_title,
                    "endpoint_record_count": row.endpoint_record_count,
                    "word_count": row.word_count,
                }
            )
    atomic_csv(pd.DataFrame(examples), out / "semantic_pattern_examples.csv")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(titles["word_count"].clip(upper=50), bins=50, color="#2C7FB8")
    axes[0, 0].set(title="Words per distinct outcome title (values >50 clipped)", xlabel="Words", ylabel="Distinct titles")
    axes[0, 1].hist(titles["character_count"].clip(upper=300), bins=60, color="#41B6C4")
    axes[0, 1].set(title="Characters per distinct title (values >300 clipped)", xlabel="Characters", ylabel="Distinct titles")
    aspect_plot = pd.DataFrame(aspect_rows).sort_values("pct_distinct_titles")
    axes[1, 0].barh(aspect_plot["aspect"], aspect_plot["pct_distinct_titles"], color="#7FCDBB")
    axes[1, 0].set(title="Lexical aspects", xlabel="Percent of distinct titles", ylabel="")
    component_counts = titles["semantic_aspect_count"].value_counts().sort_index()
    axes[1, 1].bar(component_counts.index, component_counts.values, color="#253494")
    axes[1, 1].set(title="Number of detected lexical aspects", xlabel="Aspect count", ylabel="Distinct titles")
    fig.suptitle("Outcome-title complexity in the locked 73,427-trial cohort", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out / "outcome_complexity_distributions.png")
    return titles, patterns


def category_ranking(endpoint: pd.DataFrame, n_trials: int, out: Path) -> pd.DataFrame:
    ranking = endpoint.groupby("outcome_category", dropna=False).agg(
        n_endpoint_records=("aact_outcome_title", "size"),
        n_trials=("nct_id", "nunique"),
        n_unique_raw_title_strings=("aact_outcome_title", "nunique"),
        n_unique_normalized_concepts=("outcome_normalized", "nunique"),
    ).reset_index()
    ranking["pct_endpoint_records"] = 100 * ranking["n_endpoint_records"] / len(endpoint)
    ranking["pct_trials_with_category"] = 100 * ranking["n_trials"] / n_trials
    ranking = ranking.sort_values(["n_endpoint_records", "outcome_category"], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    if len(ranking) != 21:
        raise AssertionError(f"Expected 21 outcome categories, found {len(ranking)}")
    atomic_csv(ranking, out / "all_21_outcome_categories_ranked.csv", float_format="%.6f")
    return ranking


def association_analysis(mapping: pd.DataFrame, ranking: pd.DataFrame, out: Path) -> pd.DataFrame:
    cols = ["nct_id", "aact_outcome_title", "snomed_condition_category_description"]
    grouped = modal_by_keys(mapping, cols, "outcome_category")
    grouped = grouped.rename(columns={"snomed_condition_category_description": "disease_domain"})
    counts = pd.crosstab(grouped["disease_domain"], grouped["outcome_category"])
    category_order = ranking["outcome_category"].tolist()
    domain_order = counts.sum(axis=1).sort_values(ascending=False).index.tolist()
    counts = counts.reindex(index=domain_order, columns=category_order, fill_value=0)
    if counts.shape != (24, 21):
        raise AssertionError(f"Expected 24 x 21 association table, found {counts.shape}")
    proportions = counts.div(counts.sum(axis=1), axis=0) * 100
    chi2, p_value, dof, expected = chi2_contingency(counts.to_numpy(), correction=False)
    n = int(counts.to_numpy().sum())
    cramers_v = math.sqrt(chi2 / (n * min(counts.shape[0] - 1, counts.shape[1] - 1)))
    residuals = (counts.to_numpy() - expected) / np.sqrt(expected)
    residual_frame = pd.DataFrame(residuals, index=counts.index, columns=counts.columns)

    atomic_csv(counts.reset_index(), out / "category_by_disease_domain_counts.csv")
    atomic_csv(proportions.reset_index(), out / "category_by_disease_domain_within_domain_pct.csv", float_format="%.6f")
    atomic_csv(residual_frame.reset_index(), out / "category_by_disease_domain_standardized_residuals.csv", float_format="%.6f")

    long_rows = []
    for i, domain in enumerate(counts.index):
        for j, category in enumerate(counts.columns):
            residual = float(residuals[i, j])
            long_rows.append(
                {
                    "disease_domain": domain,
                    "outcome_category": category,
                    "observed_endpoint_domain_records": int(counts.iloc[i, j]),
                    "expected_under_independence": float(expected[i, j]),
                    "within_domain_pct": float(proportions.iloc[i, j]),
                    "standardized_residual": residual,
                    "residual_flag": "overrepresented" if residual >= 1.96 else ("underrepresented" if residual <= -1.96 else "not_large"),
                }
            )
    long_frame = pd.DataFrame(long_rows).sort_values(["standardized_residual", "disease_domain"], ascending=[False, True])
    atomic_csv(long_frame, out / "category_by_disease_domain_association_long.csv", float_format="%.6f")
    summary = pd.DataFrame(
        [
            {
                "analysis_unit": "unique NCT ID + raw outcome title + disease domain",
                "n_records": n,
                "rows_disease_domains": counts.shape[0],
                "columns_outcome_categories": counts.shape[1],
                "chi_square": chi2,
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "cramers_v": cramers_v,
                "inference_caution": "Descriptive: trials can contribute multiple outcomes/domains, so records are clustered and the chi-square p-value is not a cluster-robust inferential test.",
            }
        ]
    )
    atomic_csv(summary, out / "category_by_disease_domain_association_summary.csv", float_format="%.12g")

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(24, 16))
    sns.heatmap(proportions, cmap="YlGnBu", linewidths=0.25, linecolor="white", ax=ax, cbar_kws={"label": "Within-domain percent of endpoint-domain records"})
    ax.set(title="Outcome-category composition within 24 disease domains", xlabel="Outcome category", ylabel="Disease domain")
    ax.tick_params(axis="x", labelrotation=55, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    savefig(fig, out / "category_by_disease_domain_within_domain_heatmap.png")

    fig, ax = plt.subplots(figsize=(24, 16))
    limit = max(3.0, float(np.nanquantile(np.abs(residuals), 0.98)))
    sns.heatmap(residual_frame.clip(-limit, limit), cmap="RdBu_r", center=0, vmin=-limit, vmax=limit, linewidths=0.25, linecolor="white", ax=ax, cbar_kws={"label": "Pearson standardized residual (clipped at 98th percentile)"})
    ax.set(title="Category-domain departures from independence", xlabel="Outcome category", ylabel="Disease domain")
    ax.tick_params(axis="x", labelrotation=55, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    savefig(fig, out / "category_by_disease_domain_residual_heatmap.png")
    return long_frame


def phase_time_trends(endpoint: pd.DataFrame, trials: pd.DataFrame, ranking: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_ncts = set(endpoint["nct_id"])
    cohort_trials = trials[trials["nct_id"].isin(cohort_ncts)].copy()
    trial_category = endpoint[["nct_id", "outcome_category"]].drop_duplicates()
    trial_category = trial_category.merge(cohort_trials, on="nct_id", how="left", validate="many_to_one")
    categories = ranking["outcome_category"].tolist()

    phase_totals = cohort_trials.groupby("phase_clean")["nct_id"].nunique()
    phase_counts = trial_category.groupby(["phase_clean", "outcome_category"])["nct_id"].nunique()
    phase_rows = []
    for phase in PHASE_ORDER:
        total = int(phase_totals.get(phase, 0))
        for category in categories:
            count = int(phase_counts.get((phase, category), 0))
            phase_rows.append({"phase": phase, "outcome_category": category, "n_trials": count, "total_trials_in_phase": total, "pct_trials_with_category": 100 * count / total if total else np.nan})
    phase = pd.DataFrame(phase_rows)
    phase["rank_within_phase"] = phase.groupby("phase")["pct_trials_with_category"].rank(method="min", ascending=False).astype("Int64")
    atomic_csv(phase, out / "phase_category_prevalence.csv", float_format="%.6f")

    year_trials = cohort_trials[cohort_trials["completion_year"].isin(ANALYSIS_YEARS)].copy()
    year_trials["year"] = year_trials["completion_year"].astype(int)
    year_totals = year_trials.groupby("year")["nct_id"].nunique()
    year_presence = trial_category[trial_category["completion_year"].isin(ANALYSIS_YEARS)].copy()
    year_presence["year"] = year_presence["completion_year"].astype(int)
    year_counts = year_presence.groupby(["year", "outcome_category"])["nct_id"].nunique()
    annual_rows = []
    for year in ANALYSIS_YEARS:
        total = int(year_totals.get(year, 0))
        for category in categories:
            count = int(year_counts.get((year, category), 0))
            annual_rows.append({"year": year, "outcome_category": category, "n_trials": count, "total_trials_completed": total, "pct_trials_with_category": 100 * count / total if total else np.nan})
    annual = pd.DataFrame(annual_rows)
    annual["pct_trials_with_category_3y_centered_mean"] = annual.groupby("outcome_category", sort=False)["pct_trials_with_category"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    atomic_csv(annual, out / "annual_category_prevalence_2000_2025.csv", float_format="%.6f")

    period_rows = []
    for category in categories:
        row = {"outcome_category": category}
        for label, years in [("early", EARLY_YEARS), ("recent", RECENT_YEARS)]:
            mask_trials = cohort_trials["completion_year"].between(years[0], years[1], inclusive="both")
            ids = set(cohort_trials.loc[mask_trials, "nct_id"])
            n_total = len(ids)
            n_cat = trial_category.loc[(trial_category["nct_id"].isin(ids)) & (trial_category["outcome_category"] == category), "nct_id"].nunique()
            row[f"{label}_period"] = f"{years[0]}-{years[1]}"
            row[f"{label}_n_trials_with_category"] = int(n_cat)
            row[f"{label}_total_trials"] = int(n_total)
            row[f"{label}_pct_trials_with_category"] = 100 * n_cat / n_total if n_total else np.nan
        row["delta_percentage_points"] = row["recent_pct_trials_with_category"] - row["early_pct_trials_with_category"]
        row["relative_change_pct"] = 100 * row["delta_percentage_points"] / row["early_pct_trials_with_category"] if row["early_pct_trials_with_category"] else np.nan
        period_rows.append(row)
    period = pd.DataFrame(period_rows).sort_values(["delta_percentage_points", "outcome_category"], ascending=[False, True]).reset_index(drop=True)
    period.insert(0, "rank_by_delta", np.arange(1, len(period) + 1))
    atomic_csv(period, out / "period_comparison_2005_2010_vs_2020_2025.csv", float_format="%.6f")

    phase_matrix = phase.pivot(index="outcome_category", columns="phase", values="pct_trials_with_category").reindex(index=categories, columns=PHASE_ORDER)
    fig, ax = plt.subplots(figsize=(13, 14))
    sns.heatmap(phase_matrix, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.3, ax=ax, cbar_kws={"label": "Percent of trials in phase"})
    ax.set(title="Outcome-category prevalence by trial phase", xlabel="Trial phase", ylabel="Outcome category")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    savefig(fig, out / "phase_category_prevalence_heatmap.png")

    fig, axes = plt.subplots(7, 3, figsize=(17, 22), sharex=True)
    axes = axes.ravel()
    for ax, category in zip(axes, categories):
        data = annual[annual["outcome_category"] == category]
        ax.plot(data["year"], data["pct_trials_with_category"], color="#9ECAE1", linewidth=1, marker="o", markersize=2)
        ax.plot(data["year"], data["pct_trials_with_category_3y_centered_mean"], color="#08519C", linewidth=2)
        ax.set_title(category, fontsize=10, fontweight="bold")
        ax.set_ylabel("% trials", fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[len(categories):]:
        ax.axis("off")
    fig.suptitle("Annual prevalence of all 21 outcome categories, 2000-2025\n(light: annual; dark: centered 3-year mean)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, out / "annual_trends_all_21_categories.png")

    ordered = period.sort_values("delta_percentage_points")
    colors = np.where(ordered["delta_percentage_points"] >= 0, "#238B45", "#CB181D")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(ordered["outcome_category"], ordered["delta_percentage_points"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Change in trial prevalence by outcome category", xlabel="Percentage-point change: 2020-2025 minus 2005-2010", ylabel="")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, out / "period_change_all_21_categories.png")
    return phase, annual, period


def difficulty_casebook(mapping: pd.DataFrame, endpoint: pd.DataFrame, title_features: pd.DataFrame, out: Path, per_stratum: int) -> pd.DataFrame:
    feature_cols = ["aact_outcome_title", "word_count", "character_count", "semantic_pattern", *ASPECT_DEFINITIONS.keys()]
    cases = endpoint.merge(title_features[feature_cols], on="aact_outcome_title", how="left", validate="many_to_one")
    first_context = mapping.sort_values(["nct_id", "aact_outcome_title", "aact_condition_name", "snomed_condition_category_description"]).drop_duplicates(["nct_id", "aact_outcome_title"])
    cases = cases.merge(
        first_context[["nct_id", "aact_outcome_title", "aact_condition_name", "snomed_condition_category_description"]],
        on=["nct_id", "aact_outcome_title"], how="left", validate="one_to_one",
    )
    collapse = endpoint.groupby("outcome_normalized")["aact_outcome_title"].nunique().rename("n_distinct_titles_per_normalized_concept")
    cases = cases.merge(collapse, on="outcome_normalized", how="left")
    q99 = int(cases["word_count"].quantile(0.99))
    acronym = cases["aact_outcome_title"].str.fullmatch(r"\s*[A-Z][A-Z0-9-]{1,10}\s*", na=False)
    vague = cases["outcome_normalized"].str.lower().str.fullmatch(r"(?:change|score|response|safety|efficacy|outcome|assessment|evaluation|other|unknown)", na=False)

    strata = OrderedDict(
        [
            ("very_long_title", (cases["word_count"] >= q99, "At or above the 99th percentile of title word count; long titles can combine construct, timing, instrument, and constraints.")),
            ("very_short_or_acronym", ((cases["word_count"] <= 2) | acronym, "One/two-token or acronym-like title with limited lexical context.")),
            ("composite_or_multiple_objects", (cases["composite_or_multiple_objects"], "Lexical rules indicate a composite or multiple outcome objects.")),
            ("other_clinical_endpoint", (cases["outcome_category"].eq("Other Clinical Endpoints"), "Mapped to the residual category, a useful coverage-review stratum.")),
            ("many_to_one_lexical_collapse", (cases["n_distinct_titles_per_normalized_concept"] >= 20, "Normalized concept absorbs at least 20 distinct raw title strings.")),
            ("context_sensitive_mapping", ((cases["n_normalized_values"] > 1) | (cases["n_category_values"] > 1), "The same trial-title record has more than one normalized value or category across source rows.")),
            ("multi_domain_endpoint", (cases["n_disease_domains"] >= 4, "The endpoint record is linked to at least four disease domains through trial condition mappings.")),
            ("vague_normalized_label", (vague, "The normalized label is lexically generic and may need contextual review.")),
        ]
    )
    selected = []
    used: set[tuple[str, str]] = set()
    for reason, (mask, explanation) in strata.items():
        candidates = cases.loc[mask].copy()
        candidates = candidates.sort_values(
            ["n_distinct_titles_per_normalized_concept", "word_count", "aact_outcome_title", "nct_id"],
            ascending=[False, False, True, True],
        )
        kept = []
        for row in candidates.itertuples(index=False):
            key = (str(row.nct_id), str(row.aact_outcome_title))
            if key in used:
                continue
            used.add(key)
            kept.append(row)
            if len(kept) >= per_stratum:
                break
        for rank, row in enumerate(kept, 1):
            selected.append(
                {
                    "selection_reason": reason,
                    "rank_within_reason": rank,
                    "nct_id": row.nct_id,
                    "aact_condition_name": row.aact_condition_name,
                    "disease_domain": row.snomed_condition_category_description,
                    "aact_outcome_title": row.aact_outcome_title,
                    "outcome_normalized": row.outcome_normalized,
                    "outcome_category": row.outcome_category,
                    "word_count": row.word_count,
                    "semantic_pattern": row.semantic_pattern,
                    "n_normalized_values_for_trial_title": row.n_normalized_values,
                    "n_category_values_for_trial_title": row.n_category_values,
                    "n_disease_domains_for_trial_title": row.n_disease_domains,
                    "n_distinct_titles_per_normalized_concept": row.n_distinct_titles_per_normalized_concept,
                    "why_potentially_difficult": explanation,
                    "formal_validation_status": "DESCRIPTIVE EXAMPLE - not an error-rate estimate",
                }
            )
    casebook = pd.DataFrame(selected)
    atomic_csv(casebook, out / "descriptive_mapping_difficulty_casebook.csv")
    disclaimer = """# Descriptive mapping-difficulty casebook

This sample illustrates difficult mapping cases; it is not formal validation or
an error-rate estimate. Cases are selected deterministically from transparent
lexical and multiplicity strata to support manual review and manuscript
examples. Do not use this casebook to estimate model performance.
"""
    atomic_text(disclaimer, out / "descriptive_mapping_difficulty_casebook_README.md")
    return casebook


def write_readme(out: Path, audit: pd.DataFrame, ranking: pd.DataFrame, association_long: pd.DataFrame, period: pd.DataFrame, patterns: pd.DataFrame) -> None:
    metrics = dict(zip(audit["metric"], audit["value"]))
    top_categories = ranking.head(5)[["outcome_category", "n_endpoint_records", "pct_endpoint_records"]]
    top_residuals = association_long.reindex(association_long["standardized_residual"].abs().sort_values(ascending=False).index).head(8)
    biggest_increases = period.nlargest(5, "delta_percentage_points")
    biggest_decreases = period.nsmallest(5, "delta_percentage_points")
    top_patterns = patterns.head(8)

    def markdown_table(frame: pd.DataFrame) -> str:
        return frame.to_markdown(index=False, floatfmt=".2f")

    text = f"""# Advisor revision publication analysis

## Locked cohort

- Mapping rows: {int(metrics['mapping_rows']):,}
- Trials: {int(metrics['unique_trials']):,}
- Unique trial-title endpoint records: {int(metrics['unique_trial_raw_title_records']):,}
- Distinct title strings: {int(metrics['unique_raw_outcome_title_strings']):,}
- Normalized outcome strings: {int(metrics['unique_normalized_outcome_strings']):,}
- Outcome categories: {int(metrics['outcome_categories'])}
- Disease domains: {int(metrics['disease_domain_descriptions'])}

The mapping CSV is the cohort authority. Trial metadata is joined only to add
phase and year. Older 75,680-trial research artifacts are not blended into the
73,427-trial mapping cohort.

## Analysis units

1. Overall category frequency: unique NCT ID + raw outcome title.
2. Category-domain association: unique NCT ID + raw outcome title + disease domain.
3. Phase/time prevalence: unique NCT ID + outcome category presence.
4. Lexical complexity: unique raw outcome-title string.

The chi-square association is descriptive because trials can contribute
multiple endpoints and domains. Cramer's V measures association strength, and
Pearson standardized residuals identify cells driving the association; the
p-value is not cluster-robust.

## Leading categories

{markdown_table(top_categories)}

## Most common semantic signatures

{markdown_table(top_patterns[['rank', 'semantic_pattern', 'n_endpoint_records', 'pct_endpoint_records']])}

## Largest absolute category-domain residuals

{markdown_table(top_residuals[['disease_domain', 'outcome_category', 'within_domain_pct', 'standardized_residual']])}

## Largest period increases

{markdown_table(biggest_increases[['outcome_category', 'early_pct_trials_with_category', 'recent_pct_trials_with_category', 'delta_percentage_points']])}

## Largest period decreases

{markdown_table(biggest_decreases[['outcome_category', 'early_pct_trials_with_category', 'recent_pct_trials_with_category', 'delta_percentage_points']])}

## Reproducibility notes

- Source hashes and modification times are in `source_manifest.csv`.
- Semantic rules and regexes are in `semantic_pattern_definitions.csv`.
- All ordering, tie-breaking, sampling, and year/phase windows are deterministic.
- The casebook is descriptive and explicitly separated from formal validation.
"""
    atomic_text(text, out / "README.md")


def main() -> None:
    args = parse_args()
    if not args.execute:
        print("DRY RUN: no inputs read and no outputs written")
        print(f"mapping: {args.mapping.resolve()}")
        print(f"trials: {args.trials.resolve()}")
        print(f"planned output directory: {args.output_dir.resolve()}")
        print("To compute these descriptive analyses, use --execute.")
        return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping, trials = load_inputs(args.mapping.resolve(), args.trials.resolve())
    build_manifest(args.mapping.resolve(), args.trials.resolve(), out)
    endpoint = build_endpoint_records(mapping)
    audit = cohort_audit(mapping, trials, endpoint, out)
    title_features, patterns = complexity_and_patterns(endpoint, out)
    ranking = category_ranking(endpoint, int(mapping["nct_id"].nunique()), out)
    association_long = association_analysis(mapping, ranking, out)
    _, _, period = phase_time_trends(endpoint, trials, ranking, out)
    difficulty_casebook(mapping, endpoint, title_features, out, args.casebook_per_stratum)
    write_readme(out, audit, ranking, association_long, period, patterns)

    expected = {
        "cohort_count_audit.csv",
        "outcome_complexity_summary.csv",
        "semantic_pattern_frequencies.csv",
        "all_21_outcome_categories_ranked.csv",
        "category_by_disease_domain_association_summary.csv",
        "category_by_disease_domain_standardized_residuals.csv",
        "phase_category_prevalence.csv",
        "annual_category_prevalence_2000_2025.csv",
        "period_comparison_2005_2010_vs_2020_2025.csv",
        "descriptive_mapping_difficulty_casebook.csv",
        "README.md",
    }
    missing = sorted(name for name in expected if not (out / name).exists())
    if missing:
        raise RuntimeError(f"Missing expected outputs: {missing}")
    print(f"Wrote advisor revision analysis to {out}")


if __name__ == "__main__":
    main()
