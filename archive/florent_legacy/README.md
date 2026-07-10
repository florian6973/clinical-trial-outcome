# Florent legacy implementation

This directory preserves the repository's pre-refactor research scripts and
their Git history. They are retained for provenance and are not the supported
paper-reproduction interface.

| Previous directory | Archived location | Supported replacement |
|---|---|---|
| `Outcome Structuring/` | `archive/florent_legacy/outcome_structuring/` | `src/clinical_trial_outcome/outcome_structuring/` |
| `Term to Concept Normalization/` | `archive/florent_legacy/term_to_concept_normalization/` | `src/clinical_trial_outcome/outcome_normalization/` |
| `Condition Mapping/` | `archive/florent_legacy/condition_mapping/` | `src/clinical_trial_outcome/condition_normalization/` |
| `SNOMED Outcome Normalization/` | `archive/florent_legacy/snomed_outcome_normalization/` | Qwen retrieval and normalization modules |
| `Outcome x Condition x Trial Mapping/` | `archive/florent_legacy/outcome_condition_trial_mapping/` | configured derived-data outputs |
| `Final Analysis/` | `archive/florent_legacy/final_analysis/` | `src/clinical_trial_outcome/analysis/` |

Hard-coded Hugging Face credentials in the historical scripts were replaced
with optional `HF_TOKEN` environment-variable reads before archival. The
archived scripts may depend on paths, compute, and data not distributed here.

