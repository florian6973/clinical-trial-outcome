# Clinical outcome measure characterization and trends

This repository supports the manuscript **A Large-Scale Characterization and
Trend Analysis of Clinical Outcome Measures in ClinicalTrials.gov
(2000-2025)**.

The workflow combines ontology retrieval with task-specific Qwen2.5 LoRA
models to normalize free-text outcome titles and condition names. Conditions
are mapped to SNOMED CT concepts and study disease areas. Outcomes are mapped
to a curated set of normalized groups and analysis categories.

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

## Current workflow

1. Structure each free-text outcome title into its main object and modifiers.
2. Retrieve the closest existing labels with NV-Embed-v2 and FAISS.
3. Work category by category through the unlabeled values. The condition model
   selects a SNOMED CT concept. The outcome model selects a previously approved
   outcome group or proposes a new group when no existing label fits.
4. Review new groups and uncertain matches, then add approved groups to the
   reference vocabulary for subsequent passes.
5. Assign normalized outcomes to one of the 21 selected outcome categories,
   group conditions into study disease areas, and join the labels to trial
   dates and phases.

This iterative process keeps recurring concepts consistent while allowing the
outcome vocabulary to expand when a genuinely new measure is encountered.

## Validate the checked-in artifacts

Use these commands to validate the checked-in artifacts:

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

GPU- or write-intensive Qwen commands require an explicit `--execute` flag.
See [`qwen_pipeline/README.md`](qwen_pipeline/README.md) for reviewed-input and
licensed-terminology requirements.

## Public data

The publication release contains study identifiers, raw outcome titles,
project disease labels, analysis categories, and aggregate counts. Licensing
details are documented in
[`publication_data/LICENSE_AND_PROVENANCE.md`](publication_data/LICENSE_AND_PROVENANCE.md).

## License

The repository is released under the existing Weng Lab MIT License. Data and
third-party terminology remain subject to their source licenses.
