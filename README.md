# Clinical trial outcome harmonization

This repository is the paper-aligned implementation of the pipeline used to
structure, normalize, classify, and analyze ClinicalTrials.gov outcome
measures and study conditions. The supported workflow uses Qwen for the two
candidate-selection tasks described in the manuscript. Earlier exploratory
scripts are preserved under [`archive/florent_legacy/`](archive/florent_legacy/README.md).

## What this branch represents

The `paper-aligned-qwen-pipeline` branch reorganizes the research code around
the manuscript's actual methods. It does **not** claim that the study was
rerun during the refactor. Published counts and accuracy values in
`paper/expected_results.json` are validation targets copied from the paper;
they are never manufactured by a dry run.

The workflow has seven explicit stages:

1. select the paper cohort from read-only AACT exports;
2. structure free-text outcome titles into object, measure, and time frame;
3. embed outcome objects with NV-Embed-v2 and retrieve five candidate terms
   and five candidate groups;
4. fine-tune Qwen2.5-32B-Instruct with LoRA and select one normalized outcome
   group using the required `GROUP:` response;
5. retrieve SNOMED CT condition candidates with FAISS, then use the separately
   trained Qwen selector to choose the condition concept;
6. roll outputs into 21 outcome categories, four FDA COA types, and 24 disease
   areas; and
7. evaluate held-out annotations and generate the manuscript analysis tables.

See [`paper/METHOD_CONTRACT.md`](paper/METHOD_CONTRACT.md) for the exact model,
prompt, split, and hyperparameter contract and
[`paper/OUTPUT_CONTRACT.md`](paper/OUTPUT_CONTRACT.md) for every expected table.

## Safety and reproducibility contract

All commands are dry runs unless `--execute` is supplied. Model or full-pipeline
execution additionally requires an explicit confirmation flag. A dry run may
inspect configuration and input paths, but it does not train a model, submit a
remote job, write derived data, or recompute the paper's findings.

```bash
python -m clinical_trial_outcome --config configs/paper_pipeline.yaml
```

The command prints the planned stages, model, and expected outputs. To request
execution after installing the ML dependencies and supplying the licensed
inputs, use both gates:

```bash
python -m clinical_trial_outcome \
  --config configs/paper_pipeline.yaml \
  --execute --confirm-full-run
```

The top-level gate intentionally stops before expensive work and directs users
to stage-specific commands. This prevents an accidental end-to-end rerun while
keeping each method executable and testable.

### Stage-specific training entry points

Both Qwen selectors have runnable training implementations, but neither runs
on import or during tests. Outcome training is exposed as the documented
`train_qwen_lora(...)` Python API in
`clinical_trial_outcome.outcome_normalization.training`; it requires both
`execute=True` and `confirm_execute=True`. Condition-selector training also has
a command-line entry point:

```bash
# Dry-run plan: reads the 250-row annotation JSONL but does not train or write.
python -m clinical_trial_outcome.condition_normalization.training_cli \
  --annotations /path/to/manual_condition_annotations.jsonl \
  --output-dir /path/to/condition_adapter

# Authorized execution requires both gates.
python -m clinical_trial_outcome.condition_normalization.training_cli \
  --annotations /path/to/manual_condition_annotations.jsonl \
  --output-dir /path/to/condition_adapter \
  --execute --confirm-execute
```

The expected annotation and output fields are defined under `paper/schemas/`
and in `paper/OUTPUT_CONTRACT.md`. Supplying the flags authorizes local model
work; it does not turn manuscript expectations into reproduced results.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Install the optional local-ML stack only on a compatible compute environment:

```bash
python -m pip install -e '.[ml,dev]'
```

If gated Hugging Face files require authentication, set `HF_TOKEN` in the
environment. Never put credentials in code or configuration.

## Inputs

Copy `configs/paths.example.yaml` to a local, untracked paths file. The pipeline
expects read-only AACT `studies`, `outcomes`, and `conditions` exports; a SNOMED
CT terminology release available under its applicable license; and the two
manually annotated task datasets described in the paper. Raw AACT and SNOMED CT
files are not redistributed by this repository.

The paper uses 200 training examples and 50 held-out validation examples for
each Qwen candidate-selection task. The repository validates that split and
will not silently replace it with a random 90/10 partition.

## Models

- Selector: `Qwen/Qwen2.5-32B-Instruct`
- Embeddings: `nvidia/NV-Embed-v2`, FP16, L2-normalized
- Qwen loading: 8-bit quantization
- LoRA: rank 8, alpha 32, dropout 0.1
- Training: AdamW, learning rate `1e-5`, batch size 1, three epochs, maximum
  sequence length 512
- Retrieval: five nearest outcome terms and five nearest outcome groups;
  FAISS candidate retrieval for conditions

The checkpoint-comparison configuration also records Qwen2.5 7B/14B/32B and
Llama 3 8B/15B/34B at annotation-set sizes 0, 50, 100, 150, and 200. These are
evaluation specifications, not results generated by this refactor.

## Outputs and validation

Derived artifacts are written only under configured output directories and are
ignored by Git. Schemas live in `paper/schemas/`. The final validator compares
a completed run with the manuscript's expected values, including 563,077
registered trials, 73,427 trials with results, 480,273 mapped outcome records,
237,820 normalized outcome concepts, 21 categories, 26,682 condition strings,
238 SNOMED CT condition concepts, and 24 disease areas.

The phrase **mapped outcome records** is deliberate: 480,273 is the number of
category-assigned outcome records in the paper pipeline, not a claim that every
raw title string is distinct.

## Tests

```bash
pytest
```

Tests cover the annotation split, prompts and parsers, model configuration,
retrieval cardinality, taxonomy sizes, output schemas, expected-count
validation, dry-run behavior, and the absence of embedded credentials. They do
not download models or rerun the research analysis.

## Provenance

The archived implementation corresponds to repository commit
`a81eb347e3e446dc58b6f36ca81dc2cd7479757c`. The refactor preserves those files
for auditability while making the supported code paths match the methodology
described in the revised manuscript.
