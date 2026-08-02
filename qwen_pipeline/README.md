# Qwen normalization pipeline

This directory contains the JAMIA manuscript's normalization pipeline:

1. map outcome objects to normalized outcome groups; and
2. map condition names to SNOMED CT disorder concepts and study disease areas.

The pipeline uses NV-Embed-v2, FAISS, and task-specific Qwen2.5 LoRA models. It
processes each category in turn. The condition model selects a retrieved SNOMED
CT concept. The outcome model selects an approved group or proposes a new one
for review. Approved groups enter later passes. The pipeline then assigns
normalized outcomes to one of 21 categories.

See [`../legacy/README.md`](../legacy/README.md) for the earlier implementation.

## Study configuration

`config/experiments.json` records the manuscript's 200-example outcome adapter,
50-example held-out set, and model-size and learning-curve comparisons. Small
fixtures test schemas, prompts, and commands.

## Directory map

- `config/`: retrieval, training, inference, and experiment settings.
- `schemas/`: input and output schemas.
- `src/qwen_pipeline/`: preparation, retrieval, training, inference, and checks.
- `fixtures/`: small synthetic test inputs; no licensed terminology.
- `tests/`: tests that need no GPU or model download.
- `outputs/`: local run files, ignored by Git.

## Install

Python 3.10+ is required. The basic tests use only the standard library:

```bash
cd qwen_pipeline
python -m unittest discover -s tests -v
```

For retrieval and Qwen LoRA runs, create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Setup does not download large models or licensed terminology.

## Prepare source data

To run the full pipeline, provide:

- an AACT flat-file export with trial outcomes and conditions; and
- a licensed SNOMED CT Snapshot release for condition normalization.

```bash
python -m qwen_pipeline.cli prepare-sources \
  --aact-dir /path/to/aact-flat-export \
  --snomed-rf2-dir /path/to/snomed/Snapshot \
  --output-dir outputs/source_inputs \
  --execute
```

Without `--execute`, the command validates and prepares the sources in memory
but writes nothing. With `--execute`, it writes `outcomes.raw.jsonl`,
`conditions.raw.jsonl`, `condition_vocabulary.jsonl`, and
`source_manifest.json`. Use `--aact-snapshot-id` and `--snomed-edition` to add
release identifiers when the directory names do not supply them.

This command prepares source files only. Outcome structuring, model inference,
review, and assignment to the 21 outcome categories use the commands below.
Those steps also require the reviewed outcome vocabulary and trained adapters.
SNOMED CT is not included in this repository and remains subject to its license.

## Check the setup

Commands that use a GPU or write files require `--execute`.

```bash
export PYTHONPATH="$PWD/src"

# Check configuration and examples.
python -m qwen_pipeline.cli validate-config

# Preview training records.
python -m qwen_pipeline.cli prepare \
  --task outcome \
  --annotations fixtures/outcome_annotations.jsonl \
  --output outputs/outcome_chat.jsonl

# List the planned experiments.
python -m qwen_pipeline.cli experiments

# Check manuscript metrics without recomputing them.
python -m qwen_pipeline.cli ingest-paper \
  --metrics fixtures/canonical_paper_metrics.example.json
```

Add `--execute` to `prepare` to write the chat JSONL.

## Run the models

```bash
# Build a FAISS vocabulary index.
python -m qwen_pipeline.cli build-index \
  --task outcome --vocabulary /path/to/outcome_vocabulary.jsonl \
  --output-dir outputs/outcome_index --execute

# Train an adapter with the fixed split.
python -m qwen_pipeline.cli train \
  --task outcome --train-jsonl /path/to/outcome_train.chat.jsonl \
  --validation-jsonl /path/to/outcome_validation.chat.jsonl \
  --output-dir outputs/outcome_adapter --execute

# Retrieve candidates and select with Qwen.
python -m qwen_pipeline.cli infer \
  --task outcome --input /path/to/outcomes.jsonl \
  --index-dir outputs/outcome_index --adapter outputs/outcome_adapter \
  --output outputs/outcome_predictions.jsonl --execute

# Evaluate against held-out labels.
python -m qwen_pipeline.cli evaluate \
  --task outcome --predictions outputs/outcome_predictions.jsonl \
  --gold /path/to/outcome_heldout.jsonl \
  --output outputs/outcome_evaluation.json --execute
```

## Reproducibility rules

- Keep record identifiers and explicit `train`, `validation`, and `test` labels.
  Preparation does not choose a random split.
- Train with the fixed splits and settings in `config/`. Use `--model-id` and
  `--max-train-records` to change an experiment through the CLI.
- Fit each retrieval index on its named reference vocabulary.
- The condition model may select a retrieved concept or abstain. It cannot
  create a terminology identifier.
- The outcome model returns one approved group or `PROPOSED_NEW_GROUP`. A
  reviewer must approve a new group before it enters the vocabulary or analysis.
- Save candidates, scores, model output, parsed labels, model identifiers,
  configuration, software versions, input hashes, and run metrics.
- Do not distribute SNOMED CT content unless its license permits it.
