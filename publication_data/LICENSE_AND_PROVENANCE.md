# Licensing and provenance notice

This notice documents the public-release boundary. It is not legal advice.

## SNOMED CT boundary

The mapping workflow used licensed terminology internally. The public release
does not distribute SNOMED CT identifiers, terminology descriptions, hierarchy
relationships, reference-set content, or a terminology-enabled service.
Instead, it contains project disease labels, analytic disease-area labels,
years, outcome-category labels, and aggregate counts.

SNOMED International's licensing guide states that organizations developing or
distributing products or services that include or provide access to SNOMED CT
must use the applicable Affiliate, national, and sublicense framework. Its
distribution guide describes maps and subsets derived from terminology
components as SNOMED CT derivatives. Its Global Patient Set guidance separately
permits specified flat identifier-level uses under CC BY 4.0 but requires
licensed access for semantic or ontology-based use. The project adopts the
stricter and simpler public rule: no SNOMED CT identifiers are released.

Authoritative guidance:

- https://docs.snomed.org/snomed-ct-practical-guides/vendor-introduction-to-snomed-ct/7-licensing
- https://docs.snomed.org/snomed-ct-practical-guides/snomed-nrc-guide/distribution-of-snomed-ct
- https://docs.snomed.org/implementation-guides/gps-implementation-guide/technical-application
- https://www.snomed.org/get-snomed

## Source lineage

The label-only disease aggregates are derived from three locked local inputs:

1. the internal condition-mapping crosswalk, used only to associate project
   disease labels with 24 analytic disease-area labels;
2. `data/output/research/research_trials.parquet`, used for one mapped disease
   and start year per cohort trial; and
3. `release/linked_outcome_records.csv.gz`, used to lock the 73,427-trial cohort
   and attach the 21-category outcome labels.

Only the aggregate label-only outputs are public. Internal terminology-bearing
inputs are not copied into the release.

## Deterministic transformations

`build_publication_data.py`:

- restricts the trial table to the 73,427 NCT IDs in the outcome release;
- validates one trial-level mapped disease per NCT ID;
- joins project disease and disease-area labels;
- counts unique trials by disease and start year;
- counts linked outcome records and contributing trials by disease, start year,
  and outcome category;
- rejects prohibited terminology columns and code-like numeric values in public
  text fields; and
- writes deterministic gzip files, schemas, counts, and checksums.

The builder does not infer terminology, alter disease assignments, train a
model, or change manuscript results.

## Registry-data attribution

Trial and outcome text trace to AACT/ClinicalTrials.gov-derived project data.
Users should review the applicable terms and attribution requirements:

- https://aact.ctti-clinicaltrials.org/
- https://clinicaltrials.gov/data-api/about-api/terms

The release carries no endorsement by ClinicalTrials.gov, the U.S. National
Library of Medicine, AACT, CTTI, or SNOMED International.
