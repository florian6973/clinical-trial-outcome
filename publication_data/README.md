# Publication data release

This directory contains the public data package for the JAMIA manuscript,
including the outcome extract, disease aggregates, study annotations, and
descriptive analyses.

## Public release contents

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
[data dictionary](DATA_DICTIONARY.md) for table definitions and counts.

Licensing and source details are documented in
[LICENSE_AND_PROVENANCE.md](LICENSE_AND_PROVENANCE.md).

## Other reproducibility materials

- [`recovered_study_artifacts/`](recovered_study_artifacts/) contains the study
  annotations and supplement figures.
- [`advisor_requested_descriptive_analyses/`](advisor_requested_descriptive_analyses/)
  contains the outcome characterization, examples, category ranking, and
  disease-area analysis.
- [`dashboard/`](dashboard/) contains the aggregate tables used by the
  interactive companion.

## Rebuild and validation

From the repository root, using an environment with pandas and parquet support:

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

The builders use stable ordering and deterministic gzip metadata. Validation
checks the release schemas, counts, and file hashes.
