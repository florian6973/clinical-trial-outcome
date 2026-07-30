# Clinical outcome measure characterization and trends

This repository supports the manuscript **A Large-Scale Characterization and
Trend Analysis of Clinical Outcome Measures in ClinicalTrials.gov
(2000-2025)**.

The current workflow uses NV-Embed-v2 and FAISS to retrieve candidate concepts,
then uses task-specific Qwen2.5 LoRA selectors to normalize free-text outcome
titles and condition names. The resulting concepts support analyses by outcome
category, disease area, trial phase, and calendar period.

## Start here

- [`qwen_pipeline/`](qwen_pipeline/) contains the paper-aligned Qwen training,
  retrieval, inference, evaluation, experiment, prompt, schema, and audit code.
- [`publication_data/`](publication_data/) contains recovered study
  annotations, descriptive analyses, public aggregate data, and the
  reproducible outcome-only extract.
- [`docs/REPRODUCIBILITY_STATUS.md`](docs/REPRODUCIBILITY_STATUS.md) states
  which claims are verified by current artifacts, which values come from the
  manuscript, and which analyses were not rerun.
- [`legacy/`](legacy/) preserves the original implementation and repository
  history for provenance. It is not the current reproduction path.
- The interactive companion is maintained in the
  [Clinical Trial Outcome Atlas](https://github.com/jamesbbaker/ClinicalTrialOutcomeTrends)
  repository.

## Current workflow

1. Structure each free-text outcome title into its main object and optional
   modifiers.
2. Retrieve candidate terms and normalized groups with NV-Embed-v2 and FAISS.
3. Use task-specific Qwen2.5 LoRA selectors for outcome and condition mapping.
4. Validate model output against constrained schemas and retain the source
   text, candidates, scores, raw response, parsed label, configuration digest,
   and artifact hashes.
5. Join normalized outcomes and conditions to trial dates and phases for
   descriptive analyses.

The outcome selector can place a proposed new group in a review queue, but that
proposal cannot enter the approved vocabulary or downstream analysis without
human adjudication. The condition selector can choose a retrieved candidate or
abstain; it cannot create terminology identifiers.

## Evidence boundary

The paper reports 480,273 outcome rows. The available reproducible extract
contains 467,903 distinct combinations of ClinicalTrials.gov identifier and raw
outcome title. Both counts are reported as they arise from their respective
sources; no records were changed to force agreement.

## Validate the checked-in artifacts

These commands do not download a model, train an adapter, or rerun the study:

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

## Public data and terminology boundary

The publication release contains ClinicalTrials.gov study identifiers, raw
outcome titles, project disease labels, analytic disease-area labels, years,
outcome categories, and aggregate counts. It contains no SNOMED CT
identifiers, terminology descriptions, hierarchy relationships, or reference
set content. See
[`publication_data/LICENSE_AND_PROVENANCE.md`](publication_data/LICENSE_AND_PROVENANCE.md).

## License

The repository is released under the existing Weng Lab MIT License. Data and
third-party terminology remain subject to their source licenses.
