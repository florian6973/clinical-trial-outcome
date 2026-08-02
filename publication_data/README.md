# Publication data release

This directory contains the outcome extract, disease aggregates, annotations,
and descriptive analyses for the JAMIA manuscript.

## Release files

The generated files in [`release/`](release/) are:

| File | Rows | Purpose |
|---|---:|---|
| `disease_crosswalk.csv.gz` | 274 | Analytic disease area, mapped disease label, and unique cohort-trial count |
| `disease_year_counts.csv.gz` | 6,102 | Unique trial counts by mapped disease and trial start year |
| `disease_year_outcome_counts.csv.gz` | 54,895 | Outcome-record and trial counts by disease, start year, and 21-category outcome class |
| `linked_outcome_records.csv.gz` | 467,903 | Reproducible outcome-only extract keyed by NCT ID and raw outcome title |
| `release_manifest.json` | - | Label-release schemas, counts, source checksums, and release checksums |
| `outcome_release_manifest.json` | - | Outcome-release schema and transformation record |

The 24 disease areas and 274 disease names are project analysis labels. See the
[data dictionary](DATA_DICTIONARY.md) for definitions and counts, and
[license and provenance](LICENSE_AND_PROVENANCE.md) for source and licensing
details.

## Other reproducibility materials

- [`recovered_study_artifacts/`](recovered_study_artifacts/): annotations and
  supplement figures.
- [`advisor_requested_descriptive_analyses/`](advisor_requested_descriptive_analyses/):
  outcome examples, category rankings, and disease-area analyses.
- [`dashboard/`](dashboard/): aggregate tables for the interactive companion.

## Build and validate

Run from the repository root in an environment with pandas and Parquet support:

```bash
python publication_data/build_outcome_release.py \
  --source /path/to/outcome_and_condition_aact_mapping.csv
python publication_data/build_publication_data.py \
  --condition-crosswalk /path/to/condition_disease_area_labels.csv \
  --trials /path/to/research_trials.parquet
python publication_data/validate_outcome_release.py --deep
python publication_data/validate_publication_data.py --deep
python -m unittest discover -s publication_data/tests -v
```

The builders use stable ordering. Validation checks schemas, counts, and file
hashes.
