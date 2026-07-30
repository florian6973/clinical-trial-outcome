# Reproducibility status

This document is the integration boundary between the JAMIA manuscript, the
condition-data release, and the paper-aligned Qwen code. It records what the
current repository verifies and what comes from the paper or executable
configuration. It does not replace the manuscript Methods or Results.

## Claim-to-artifact status

| Manuscript or repository claim | Current status | Supporting artifact(s) | What may be concluded |
|---|---|---|---|
| The cohort contains 73,427 trials with posted results. | **Verified in the locked repository source** | [`../publication_data/release/release_manifest.json`](../publication_data/release/release_manifest.json); [`../publication_data/validate_publication_data.py`](../publication_data/validate_publication_data.py) | A deterministic scan of the locked outcome-by-condition source finds 73,427 distinct `nct_id` values. This verifies the cohort denominator, not the manuscript's outcome-record denominator. |
| The condition input contains 26,682 distinct raw strings mapped internally to 238 identifier values and 24 disease areas. | **Count-verified internally; public release is label-only** | [`../publication_data/release/disease_crosswalk.csv.gz`](../publication_data/release/disease_crosswalk.csv.gz); [`../publication_data/release/release_manifest.json`](../publication_data/release/release_manifest.json); [`../publication_data/LICENSE_AND_PROVENANCE.md`](../publication_data/LICENSE_AND_PROVENANCE.md) | The public artifact verifies 274 mapped disease labels, 24 disease-area labels, and 73,427 unique cohort trials without exposing terminology identifiers or descriptions. It does not establish that every internal terminology assignment is correct. |
| The 238 internal identifiers form a clean condition-to-area crosswalk. | **Not established and not distributed publicly** | [`../publication_data/LICENSE_AND_PROVENANCE.md`](../publication_data/LICENSE_AND_PROVENANCE.md); [`../publication_data/README.md`](../publication_data/README.md) | Internal QA found reused or cross-area assignments. Public files avoid this unresolved terminology layer and contain only project labels and aggregate counts. |
| The analysis contains 480,273 outcome records and 237,820 normalized outcome concepts. | **Canonical manuscript values; not reproduced by the available locked dataset** | [`../publication_data/README.md`](../publication_data/README.md); [`../publication_data/release/outcome_release_manifest.json`](../publication_data/release/outcome_release_manifest.json) | The available source instead yields 467,903 unique `(nct_id, aact_outcome_title)` records and 243,916 unique normalized strings. The outcome-only release preserves the reproducible 467,903-row key, and no records or keys were fabricated to force agreement. |
| Outcome and condition normalization use NV-Embed-v2 retrieval and Qwen2.5-32B-Instruct LoRA selectors with the documented settings. | **Implementation and configuration present; runs not executed for this revision** | [`../qwen_pipeline/config/pipeline.json`](../qwen_pipeline/config/pipeline.json); [`../qwen_pipeline/src/qwen_pipeline/`](../qwen_pipeline/src/qwen_pipeline/); [`../qwen_pipeline/README.md`](../qwen_pipeline/README.md) | The repository validates the written model, retrieval, LoRA, decoding, and audit configuration and provides dry-run-first entry points. It does not provide evidence that model training, inference, or corpus-scale normalization was rerun. |
| Qwen2.5-32B was selected from six checkpoints, and the learning curve used 0, 50, 100, 150, and 200 training examples. | **Paper-reported result; experiment design encoded** | [`../qwen_pipeline/config/experiments.json`](../qwen_pipeline/config/experiments.json); [`../qwen_pipeline/src/qwen_pipeline/experiments.py`](../qwen_pipeline/src/qwen_pipeline/experiments.py) | The design matrix records the paper's model labels and training sizes. The command plans those cells and does not claim a rerun. |
| Outcome accuracy was 88% (44/50), and the remaining reported evaluation values are those shown in the manuscript. | **Paper-reported result; not recomputed here** | [`../qwen_pipeline/fixtures/canonical_paper_metrics.example.json`](../qwen_pipeline/fixtures/canonical_paper_metrics.example.json); [`../qwen_pipeline/schemas/canonical_paper_metrics.schema.json`](../qwen_pipeline/schemas/canonical_paper_metrics.schema.json) | The schema validates supplied values and labels their status. It does not recompute accuracy or F1. |
| Files under `qwen_pipeline/fixtures/` reproduce the study. | **False; fixtures are illustrative only** | [`../qwen_pipeline/fixtures/`](../qwen_pipeline/fixtures/); [`../qwen_pipeline/README.md`](../qwen_pipeline/README.md) | Fixtures test schemas and prompts using synthetic or placeholder identifiers. They must never be substituted for study annotations, predictions, terminology, or results. |

## Safe verification commands

Run these commands from the repository root. They validate existing artifacts
or enumerate configuration; they do not download models, train adapters, run
inference, or change manuscript results.

```bash
# Reconcile the identifier-free disease release with the locked local inputs.
PYTHONDONTWRITEBYTECODE=1 python3 \
  publication_data/validate_publication_data.py --deep

# Run the condition-release regression test.
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s publication_data/tests -v

# Validate the documented Qwen/NV-Embed/LoRA configuration only.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=qwen_pipeline/src \
python3 -m qwen_pipeline.cli validate-config

# Enumerate the planned model-comparison and learning-curve cells only.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=qwen_pipeline/src \
python3 -m qwen_pipeline.cli experiments
```

Do not add `--execute` to Qwen commands until reviewed task data, vocabularies,
model identifiers, licensed terminology, and a run-provenance destination have
been supplied. A future executed run
must be reported separately from the manuscript's canonical supplied values
unless its input hashes and outputs are shown to reproduce the original analysis.

## Next reproducibility steps

1. Locate and hash the exact manuscript outcome-level dataset and reconcile its
   480,273/237,820 counts with the currently available 467,903/243,916 artifact.
2. Train the task-specific adapters under a versioned environment
   and preserve adapter, input, configuration, and output hashes.
3. Complete internal terminology QA and document the licensed SNOMED CT edition;
   keep terminology identifiers outside the public repository.
4. Only after those gates are met, execute training/inference/evaluation and
   compare the new run with the manuscript values without overwriting either.
