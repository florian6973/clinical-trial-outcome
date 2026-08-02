# Public release data dictionary

All release tables are UTF-8 CSV files compressed with deterministic gzip. They
contain no SNOMED CT identifiers or terminology descriptions.

## `disease_crosswalk.csv.gz` - 274 rows

| Column | Type | Definition |
|---|---|---|
| `disease_area` | string | One of 24 project analytic disease-area labels |
| `disease` | string | Project-normalized mapped disease label |
| `trial_count` | integer | Unique trials assigned to the disease in the 73,427-trial cohort |

Exactly 213 diseases have `trial_count > 100`. Each trial has one mapped disease
in this table, so the 274 counts sum to 73,427.

## `disease_year_counts.csv.gz` - 6,102 rows

| Column | Type | Definition |
|---|---|---|
| `disease_area` | string | Project analytic disease-area label |
| `disease` | string | Project-normalized mapped disease label |
| `start_year` | integer | Trial start year |
| `trial_count` | integer | Unique trials in the disease-year cell |

The table covers 73,410 dated trials from 1966 through 2025. Seventeen cohort
trials without a usable start year are excluded from annual views.

## `disease_year_outcome_counts.csv.gz` - 54,895 rows

| Column | Type | Definition |
|---|---|---|
| `disease_area` | string | Project analytic disease-area label |
| `disease` | string | Project-normalized mapped disease label |
| `start_year` | integer | Start year of the contributing trial |
| `outcome_category` | string | One of the 21 manuscript outcome categories |
| `outcome_record_count` | integer | Linked outcome records in the cell |
| `trial_count` | integer | Unique trials contributing at least one outcome record to the cell |

The table contains 467,865 dated outcome records. A trial can contribute to
multiple outcome categories, so `trial_count` must not be summed across
categories.

## Notes

- The strict `>100` rule is a descriptive display threshold, not an inferential
  cutoff.
- Recent start years are affected by the lag between trial initiation and
  posted results in the analytic cohort.
- Disease areas are analytic groupings, not a published terminology hierarchy.
