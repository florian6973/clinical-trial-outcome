# Publication data release

This directory contains the public, identifier-free data package for the JAMIA
manuscript. The release includes the reproducible outcome-only extract,
label-only disease aggregates used by the dashboard, study annotations, and the
advisor-requested descriptive characterization.

## Public release contents

The generated files in [`release/`](release/) are:

| File | Rows | Purpose |
|---|---:|---|
| `disease_crosswalk.csv.gz` | 274 | Analytic disease area, mapped disease label, and unique cohort-trial count |
| `disease_year_counts.csv.gz` | 6,102 | Unique trial counts by mapped disease and trial start year |
| `disease_year_outcome_counts.csv.gz` | 54,895 | Outcome-record and trial counts by disease, start year, and 21-category outcome class |
| `linked_outcome_records.csv.gz` | 467,903 | Reproducible outcome-only extract keyed by NCT ID and raw outcome title |
| `release_manifest.json` | - | Label-release schemas, counts, source checksums, and release checksums |
| `outcome_release_manifest.json` | - | Outcome-release transformation and denominator contract |

No public release table contains SNOMED CT identifiers, terminology
descriptions, hierarchy relationships, reference-set content, or raw condition
strings. The 24 disease areas and 274 disease names are project analytic labels.

The disease table counts unique trials in the locked 73,427-trial cohort. Exactly
213 diseases have more than 100 unique trials. Annual files contain 73,410 dated
trials; 17 cohort trials without a usable start year remain in the all-cohort
disease table but are excluded from annual views. The annual outcome file covers
467,865 linked records; 38 records belong to the 17 undated trials.

## Licensing policy

SNOMED International states that organizations distributing products or
services that include or provide access to SNOMED CT must use the applicable
affiliate, sublicense, and national-license framework. The Global Patient Set
guidance permits specified identifier-level uses under CC BY 4.0 while reserving
semantic and ontology-based use for licensed access. To make the public boundary
unambiguous, this project publishes no SNOMED CT identifiers.

- [SNOMED CT licensing guidance](https://docs.snomed.org/snomed-ct-practical-guides/vendor-introduction-to-snomed-ct/7-licensing)
- [SNOMED CT distribution guidance](https://docs.snomed.org/snomed-ct-practical-guides/snomed-nrc-guide/distribution-of-snomed-ct)
- [Global Patient Set technical guidance](https://docs.snomed.org/implementation-guides/gps-implementation-guide/technical-application)

See [LICENSE_AND_PROVENANCE.md](LICENSE_AND_PROVENANCE.md) and
[DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Other reproducibility materials

[`recovered_study_artifacts/`](recovered_study_artifacts/) contains
outcome-structuring and term-grouping annotations and the figures embedded in
the original supplement.

[`advisor_requested_descriptive_analyses/`](advisor_requested_descriptive_analyses/)
contains the checksum-covered complexity characterization, difficulty casebook,
complete 21-category ranking, and 24-by-21 disease-area crosswalk used to answer
the advisor's requests.

[`dashboard/`](dashboard/) contains the checksum-covered condition and outcome
summary tables used by the interactive companion. Keeping these aggregates in
the paper repository lets the public dashboard refresh without access to
licensed terminology.

## Outcome-denominator reconciliation

The locked condition-expanded source contains 467,903 reproducible unique
`(nct_id, aact_outcome_title)` records. The manuscript reports 480,273 row-level
AACT outcome records. The public extract preserves the reproducible key and does
not coerce either denominator.

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

The builders use stable ordering and deterministic gzip metadata. The validation
code checks schemas, locked counts, file hashes, and the absence of terminology
identifier columns or code-like numeric values in public text fields.

The 404 MB internal analytic source is not committed because it exceeds GitHub's
ordinary file-size limit and contains more information than the public release
requires.
