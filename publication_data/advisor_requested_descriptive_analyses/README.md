# Advisor-requested descriptive analyses

This folder preserves the already-generated characterization requested during
Chunhua Weng's review. The files were copied from the locked mapping-file output;
they were **not recomputed during manuscript revision**.

## Cohort authority

The source is `data/output/research/outcome_and_condition_aact_mapping.csv`:

- 423,325,898 bytes (approximately 404 MiB)
- SHA-256: `1065422cb2479a667fa4017787f9090d7293e63c6e7c1dfe933ab95ebc6799b4`
- 792,091 condition-expanded rows from 73,427 trials
- 21 outcome categories and 24 disease domains

The relative source path, checksum, byte size, and timestamp at generation are
also recorded in `source_manifest.csv`.

## Analysis units

The files use four deliberately different units:

1. The manuscript's 480,273 mapped outcome records are row-level AACT outcome
   entries carried through normalization and remain the paper's headline count.
2. Overall mapping-file summaries use 467,903 unique `(NCT ID, raw outcome
   title)` records because the recovered locked file lacks an AACT outcome-row ID.
3. Lexical complexity uses 390,228 distinct raw outcome-title strings.
4. The category-by-disease crosswalk uses 561,609 unique `(NCT ID, raw outcome
   title, disease domain)` links.

The latter three counts characterize the recovered mapping file and must not be
substituted for the manuscript's 480,273-record denominator.

## Contents

- `outcome_complexity_*` and `semantic_*`: transparent lexical-rule
  characterization of title length, detected aspects, and common signatures.
- `descriptive_mapping_difficulty_casebook*`: real mapped examples selected for
  review strata such as very long titles, ambiguous short titles, composites,
  and residual-category cases. This is descriptive triage, not a formal model
  error-rate estimate.
- `all_21_outcome_categories_ranked.csv`: complete category ranking from the
  recovered linked cohort.
- `category_by_disease_domain_*`: the complete 24-by-21 crosswalk, within-domain
  percentages, standardized residuals, and descriptive association summary.
- `figures/`: publication-ready complexity and disease-domain figures.

The crosswalk's chi-square p-value is not treated as cluster-robust inference,
because one trial can contribute multiple outcomes and disease domains. Cramer's
V and standardized residuals are reported descriptively.

## Reproduction boundary

The deterministic source code is
[`../scripts/descriptive_analyses.py`](../scripts/descriptive_analyses.py). Its
default invocation is a dry run; `--execute` is required to regenerate these
outputs. The present revision reuses the checksum-covered files in this folder
and does not execute that analysis.
