# Clinical outcome measure characterization and trends

This repository supports the manuscript **A Large-Scale Characterization and
Trend Analysis of Clinical Outcome Measures in ClinicalTrials.gov
(2000-2025)**.

The pipeline uses task-specific Qwen2.5 LoRA models to normalize AACT outcome
titles and condition names. It maps conditions to SNOMED CT concepts and study
disease areas, and outcomes to approved groups and analysis categories.

## Start here

- [`qwen_pipeline/`](qwen_pipeline/) contains the paper-aligned normalization
  pipeline and evaluation code.
- [`publication_data/`](publication_data/) contains the public study data,
  annotations, and descriptive analyses.
- [`docs/REPRODUCIBILITY_STATUS.md`](docs/REPRODUCIBILITY_STATUS.md) summarizes
  the available study materials.
- The interactive companion is maintained in the
  [Clinical Trial Outcome Atlas](https://github.com/jamesbbaker/ClinicalTrialOutcomeTrends)
  repository.

## Workflow

1. Structure each free-text outcome title into its main object and modifiers.
2. Retrieve approved outcome groups or SNOMED CT concepts with NV-Embed-v2 and
   FAISS.
3. Process each category in turn. Qwen selects a match or proposes a new outcome
   group when no approved group fits.
4. Review proposed groups before adding them to later passes. Then map outcomes
   to 21 categories, group conditions into disease areas, and join the labels to
   trial dates and phases.

## Validate the checked-in artifacts

Run these checks from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=qwen_pipeline/src \
python3 -m qwen_pipeline.cli validate-config

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=qwen_pipeline/src \
python3 -m qwen_pipeline.cli experiments

PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest discover -s qwen_pipeline/tests -v

PYTHONDONTWRITEBYTECODE=1 \
python3 publication_data/validate_publication_data.py --deep

PYTHONDONTWRITEBYTECODE=1 \
python3 publication_data/validate_outcome_release.py --deep

PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest discover -s publication_data/tests -v
```

Qwen commands that use a GPU or write files require `--execute`. See
[`qwen_pipeline/README.md`](qwen_pipeline/README.md) for source preparation and
licensing requirements.

## Data and license

The publication release contains study identifiers, raw outcome titles,
project disease labels, analysis categories, and aggregate counts. See
[`publication_data/LICENSE_AND_PROVENANCE.md`](publication_data/LICENSE_AND_PROVENANCE.md).
Code is available under the MIT License. Data and third-party
terminology remain subject to their source licenses.
