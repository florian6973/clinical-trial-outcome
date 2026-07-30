# Legacy implementation

`original_2024/` contains the files that were at the repository root before the
paper-aligned reproducibility update. The move preserves Florent Pollet's
original code, data file, commit history, and Weng Lab attribution.

The archived implementation includes:

- outcome structuring experiments;
- term-to-concept Qwen scripts;
- condition and SNOMED CT mapping scripts;
- final trend-analysis scripts; and
- the original outcome-condition-trial mapping.

These files document the study's development history. They are not the current
entry point because they mix several experimental generations, assume
environment-specific paths, and do not provide the consolidated configuration,
schemas, validation, and audit records used by the current package.

Use [`../qwen_pipeline/`](../qwen_pipeline/) for the paper-aligned method and
[`../publication_data/`](../publication_data/) for the public study artifacts.
