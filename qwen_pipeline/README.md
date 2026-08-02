# Paper-aligned Qwen normalization pipeline

This directory implements the normalization workflow described in the JAMIA
manuscript:

1. outcome object to normalized outcome group; and
2. free-text condition to a candidate SNOMED CT disorder concept, followed by a
   separately supplied disease-area crosswalk.

The workflow uses NV-Embed-v2 embeddings, FAISS retrieval, and task-specific
Qwen2.5 LoRA selectors. It proceeds category by category through the unlabeled
values. Condition names are matched to retrieved SNOMED CT concepts. Outcome
objects are matched to previously approved outcome groups; when no group fits,
a new group is proposed for review and, once approved, becomes available in
later passes. The normalized groups are then assigned to one of the 21 selected
outcome categories.

See [`../legacy/README.md`](../legacy/README.md) for the earlier implementation.

## Study configuration

The manuscript reports a 200-example outcome adapter, a separate 50-example
held-out set, and a model-size/learning-curve comparison. The experiment matrix
in `config/experiments.json` records those cells. Small fixtures support schema,
prompt, and command-line tests.

## Directory map

- `config/`: paper-aligned retrieval, LoRA, inference, and experiment settings.
- `schemas/`: JSON Schemas for vocabularies, annotations, predictions, canonical
  metrics, and disease-area crosswalks.
- `src/qwen_pipeline/`: preparation, retrieval, training, inference, validation,
  experiment planning, and canonical-output ingestion.
- `fixtures/`: small illustrative inputs and a supplied-metric example.
- `tests/`: standard-library unit tests; no GPU or model download required.
- `outputs/`: ignored runtime destination except for its explanatory README.

## Install

Python 3.10+ is required. The lightweight contract tests use only the standard
library:

```bash
cd qwen_pipeline
python -m unittest discover -s tests -v
```

For full retrieval and Qwen LoRA execution, create an isolated environment and
install the pinned-minimum dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Large models and licensed terminology are not downloaded by setup.

## Dry-run walkthrough

All commands default to validation/planning. `--execute` is required for
GPU-intensive or output-writing operations.

```bash
export PYTHONPATH="$PWD/src"

# Validate package configuration and example records.
python -m qwen_pipeline.cli validate-config

# Preview the exact chat records that preparation would produce.
python -m qwen_pipeline.cli prepare \
  --task outcome \
  --annotations fixtures/outcome_annotations.jsonl \
  --output outputs/outcome_chat.jsonl

# Inspect, but do not run, the model-size and learning-curve matrix.
python -m qwen_pipeline.cli experiments

# Validate canonical manuscript metrics without recomputing them.
python -m qwen_pipeline.cli ingest-paper \
  --metrics fixtures/canonical_paper_metrics.example.json
```

Add `--execute` to `prepare` to write the chat JSONL. Full examples:

```bash
# Build a FAISS vocabulary index.
python -m qwen_pipeline.cli build-index \
  --task outcome --vocabulary /path/to/outcome_vocabulary.jsonl \
  --output-dir outputs/outcome_index --execute

# Train a task-specific adapter. Use the exact author-verified split artifact.
python -m qwen_pipeline.cli train \
  --task outcome --train-jsonl /path/to/outcome_train.chat.jsonl \
  --validation-jsonl /path/to/outcome_validation.chat.jsonl \
  --output-dir outputs/outcome_adapter --execute

# Retrieve candidates, select with Qwen, and save item-level audit records.
python -m qwen_pipeline.cli infer \
  --task outcome --input /path/to/outcomes.jsonl \
  --index-dir outputs/outcome_index --adapter outputs/outcome_adapter \
  --output outputs/outcome_predictions.jsonl --execute

# Evaluate against author-verified held-out labels.
python -m qwen_pipeline.cli evaluate \
  --task outcome --predictions outputs/outcome_predictions.jsonl \
  --gold /path/to/outcome_heldout.jsonl \
  --output outputs/outcome_evaluation.json --execute
```

## Required real-data artifacts

Full execution requires the following reviewed inputs:

- the exact 200/50 outcome annotation split and record identifiers;
- the original held-out predictions for every compared checkpoint;
- the condition benchmark rows used for the reported evaluation;
- trained adapter weights and their hashes;
- a licensed SNOMED CT release and its edition/version identifier;
- an independently validated terminology crosswalk. The publication-data
  package contains the observed 274-row/238-identifier/24-domain mapping and
  collision diagnostics, but that observed artifact is not yet an independently
  validated concept-to-area crosswalk; and
- the normalized outcome release supporting the manuscript's corpus counts.

Study annotations are available under
[`../publication_data/recovered_study_artifacts/`](../publication_data/recovered_study_artifacts/).
They include 250 outcome-title structuring annotations and two independently
completed 100-row term-grouping files.

Pass reviewed artifacts to the CLI using the paths shown above.

## Reproducibility and safety rules

- Preserve record identifiers and explicit `train`, `validation`, and `test`
  labels. Preparation never chooses a random split.
- The GPU trainer follows the historical Accelerate/PEFT loop: it loads the
  selected checkpoint in 8-bit mode, tokenizes prompt and response separately,
  masks prompt labels with `-100`, truncates or pads to 512 tokens, uses batch
  size 1 and AdamW at `1e-5`, evaluates loss after each of three epochs, and
  saves the LoRA adapter. Use `--model-id` and `--max-train-records` to run a
  reviewed model-size or learning-curve cell without changing the code.
- Fit retrieval indexes on the specified reference vocabulary only.
- The condition selector may choose only a retrieved candidate or abstain. It
  cannot create a terminology identifier.
- The outcome parser accepts exactly one normalized group. It can retain a
  `PROPOSED_NEW_GROUP` label in a non-aggregating review queue, but that proposal
  cannot enter the canonical vocabulary or category aggregation until human
  adjudication. Malformed or otherwise out-of-vocabulary labels are rejected.
- Record retrieval candidates, scores, raw model text, parsed output, adapter
  path, configuration digest, software versions, and input hashes.
- Store experiment outputs with their run-level configuration and metrics.
- Do not distribute licensed terminology unless the applicable license permits
  the proposed release.
