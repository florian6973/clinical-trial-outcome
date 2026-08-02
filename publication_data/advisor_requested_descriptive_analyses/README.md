# Descriptive analyses

These files describe the outcomes used in the manuscript.

## Source

The source is `data/output/research/outcome_and_condition_aact_mapping.csv`:

- SHA-256: `1065422cb2479a667fa4017787f9090d7293e63c6e7c1dfe933ab95ebc6799b4`
- 792,091 condition-expanded rows from 73,427 trials
- 21 outcome categories and 24 disease domains

`source_manifest.csv` records the source metadata.

## Analysis units

Each output records its analysis unit: mapped outcome records, distinct raw
titles, or trial-outcome-disease links.

## Contents

- `outcome_complexity_*` and `semantic_*`: title length, detected features, and
  common patterns.
- `descriptive_mapping_difficulty_casebook*`: mapped examples across long,
  short, composite, and residual-category cases.
- `all_21_outcome_categories_ranked.csv`: category ranking for the linked cohort.
- `category_by_disease_domain_*`: the 24-by-21 crosswalk, percentages,
  standardized residuals, and association summary.
- `figures/`: publication-ready complexity and disease-domain figures.

## Reproduce

Run from the repository root:

```bash
python publication_data/scripts/descriptive_analyses.py --execute
```
