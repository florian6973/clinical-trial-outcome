# Advisor-requested descriptive analyses

This folder contains the outcome characterization prepared during manuscript
review.

## Source

The source is `data/output/research/outcome_and_condition_aact_mapping.csv`:

- SHA-256: `1065422cb2479a667fa4017787f9090d7293e63c6e7c1dfe933ab95ebc6799b4`
- 792,091 condition-expanded rows from 73,427 trials
- 21 outcome categories and 24 disease domains

The relative source path, checksum, byte size, and timestamp at generation are
also recorded in `source_manifest.csv`.

## Analysis units

Each analysis uses the unit appropriate to its question: mapped outcome
records, distinct raw outcome titles, or trial-outcome-disease links. The unit
is recorded in each output table.

## Contents

- `outcome_complexity_*` and `semantic_*`: title length, detected aspects, and
  common signatures.
- `descriptive_mapping_difficulty_casebook*`: real mapped examples selected for
  review strata such as very long titles, ambiguous short titles, composites,
  and residual-category cases.
- `all_21_outcome_categories_ranked.csv`: complete category ranking from the
  linked cohort.
- `category_by_disease_domain_*`: the complete 24-by-21 crosswalk, within-domain
  percentages, standardized residuals, and descriptive association summary.
- `figures/`: publication-ready complexity and disease-domain figures.

The crosswalk reports descriptive percentages, Cramer's V, and standardized
residuals.

## Reproduction boundary

The source code is
[`../scripts/descriptive_analyses.py`](../scripts/descriptive_analyses.py).
Use `--execute` to regenerate the outputs.
