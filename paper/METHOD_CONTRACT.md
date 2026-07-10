# Paper-aligned method contract

## Status and execution boundary

This repository expresses the methods required by the paper as inspectable contracts. All entry points are **dry-run by default**. Model loading, FAISS construction, inference, or any other computational execution requires an explicit `--execute` request and configured local inputs. Importing a module, running unit tests, or reading a configuration file must never train a model or start full-data inference.

No training or full-data execution is part of this contract implementation. Reported result constants are validation expectations stored in `paper/expected_results.json`; they are not recomputed or treated as reproduced evidence.

## Locked annotation contract

- Training examples: **200**.
- Validation examples: **50**.
- The 200/50 contract applies independently to each Qwen candidate-selection task: outcome-group selection and condition-to-SNOMED selection.
- The split must be frozen before model fitting and preserved in a manifest.
- Validation examples must not be used for prompt development, retrieval-index tuning, model selection, or learning-curve fitting.
- Configuration uses seed 42 for a reproducible split when an archived split manifest is not supplied.

## Condition normalization

Condition normalization has a mandatory two-stage decision order:

1. **FAISS/SNOMED candidate retrieval.** Embed the raw AACT condition and active SNOMED CT fully specified names with `nvidia/NV-Embed-v2`. L2-normalize all vectors. Retrieve candidates with a FAISS IVFFlat inner-product index; with normalized vectors, inner product is cosine similarity. The paper configuration uses `nlist=min(4096, floor(n_candidates/30))`, `nprobe=64`, and five candidates.
2. **Qwen candidate selection.** Present the raw condition and retrieved candidates to `Qwen/Qwen2.5-32B-Instruct`. Qwen must select exactly one supplied `concept_id`; it may not invent or rewrite a concept. A selected concept inherits the candidate's SNOMED term, similarity, rank, and one of the locked 24 disease-area labels.

The condition selector has its own frozen 200/50 annotation split and LoRA training entry point. Its training module is dry-run by default, imports Transformers/PEFT/TRL only after both execution gates are supplied, and writes its adapter only to the configured output directory.

The output must preserve all retrieved candidates so the Qwen decision can be audited. A Qwen answer that is not valid JSON, adds fields, selects a missing concept, or implies a disease area not carried by the selected candidate is invalid.

The locked 24 disease areas are defined in `src/clinical_trial_outcome/condition_normalization/disease_areas.py`. This downstream paper taxonomy is distinct from the 30-area legacy dictionary in the archived prototype.

## Outcome normalization boundary

Outcome normalization is handled by the separate `outcome_normalization` package. The shared configuration fixes the paper split at 200 training and 50 validation examples and uses five similar terms and five similar groups. The condition and classification packages must not train or modify that model.

The source-aligned outcome prompt displays the main term, the five retrieved terms with scores, and the five retrieved groups with scores. Supervised examples request an `EXPLANATION:` line followed by exactly one `GROUP:` line; downstream parsing treats the `GROUP:` value as the prediction. The paper prompt is preserved at `paper/prompts/outcome_qwen_selection.md`.

## Outcome-category and COA classification

After outcome concept/group normalization, Qwen performs constrained selection into two locked analytic taxonomies:

- **21 outcome categories**, defined in `classification/taxonomy.py`.
- **Four FDA clinical outcome assessment types**: Patient-Reported Outcome (PRO), Clinician-Reported Outcome (ClinRO), Observer-Reported Outcome (ObsRO), and Performance Outcome (PerfO).

The normalized outcome is presented with all allowed labels. The response must be JSON with exactly `outcome_category` and `coa_type`, and both values must match a supplied label exactly. Classification does not create new labels and does not modify the normalized concept.

The 21-category taxonomy is a downstream analysis layer; it must not be conflated with the Qwen normalization vocabulary or the archived exploratory 13-category Llama classifier.

## Model-comparison contract

`configs/model_comparison.yaml` defines exactly six paper comparison checkpoints:

1. Qwen2.5 7B
2. Qwen2.5 14B
3. Qwen2.5 32B
4. Llama3 8B
5. Llama3 15B
6. Llama3 34B

Learning-curve sizes are **0, 50, 100, 150, and 200** training examples. Every checkpoint must use the identical frozen 200/50 split, prompts, retrieved candidates, parsing, and exact-match metric. The 15B and 34B Llama identifiers are paper comparison aliases until their archived checkpoint locations are supplied. No comparison result may be reported merely because it appears in `expected_results.json`; archived predictions and metrics are required.

## Manuscript validation targets

The exact count targets are 563,077 registered trials, 73,427 result trials, 480,273 mapped outcome records, 237,820 normalized outcome concepts, 21 outcome categories, 26,682 raw condition strings, 238 SNOMED CT condition concepts, and 24 disease areas. These values are validation-only manuscript expectations. A dry run never emits them as observed results, and a completed run must be compared against them rather than silently replacing them.

## Secrets and paths

- Secrets are forbidden in source, YAML, prompts, schemas, test fixtures, and paper artifacts.
- Authentication is referenced only by environment-variable name, such as `HF_TOKEN`.
- `configs/paths.example.yaml` contains placeholders only and must be copied to a local ignored configuration before execution.
- Runtime manifests must record input hashes, model identifiers, adapter identifiers, configuration, and output hashes without recording token values.

## Dry-run behavior

Dry-run output is a JSON-serializable plan. It may validate strings, taxonomy sizes, config values, and stage order, but it must not import optional GPU/model dependencies, open large data files, build an index, call a model, or write result artifacts. `--execute` changes authorization, not evidence: execution still fails closed when required inputs, retriever, or model generator are absent.
