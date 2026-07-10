# Paper pipeline output contract

## General rules

- Outputs are UTF-8 and use snake_case field names.
- A run manifest records `dry_run`, the explicit execution request, input hashes, model/adapter identifiers, configuration hash, timestamp, code commit, and output hashes.
- Dry-run plans and executed results are different schemas and must not be mixed in one table.
- Reported manuscript constants are never populated into row-level outputs as if they were computed.

## Condition-normalization result

Schema: `paper/schemas/condition_normalization_output.schema.json`.

| Field | Type | Contract |
|---|---|---|
| `condition_text` | string | Raw AACT condition text. |
| `snomed_concept_id` | string | Exactly one concept ID supplied by FAISS retrieval and selected by Qwen. |
| `snomed_term` | string | Term belonging to the selected candidate. |
| `disease_area` | string | Exactly one of the locked 24 labels. |
| `similarity` | number | Selected candidate's FAISS inner-product/cosine score, between -1 and 1. |
| `candidate_rank` | integer | One-based rank among retrieved candidates. |
| `qwen_model` | string | `Qwen/Qwen2.5-32B-Instruct`. |
| `selection_method` | string | `faiss_candidates_then_qwen_selection`. |
| `candidates` | array | Complete ordered candidate list with concept ID, term, disease area, score, and rank. |

The selected concept must appear exactly once in `candidates`; its term, disease area, score, and rank must agree with the top-level selected fields.

## Outcome-classification result

Schema: `paper/schemas/outcome_classification_output.schema.json`.

| Field | Type | Contract |
|---|---|---|
| `normalized_outcome` | string | Input normalized concept/group; unchanged by classification. |
| `outcome_category` | string | Exactly one of 21 locked categories. |
| `coa_type` | string | Exactly one of four locked COA labels. |
| `qwen_model` | string | `Qwen/Qwen2.5-32B-Instruct`. |
| `classification_method` | string | `qwen_constrained_taxonomy_selection`. |

## Dry-run plans

Schema: `paper/schemas/dry_run_plan.schema.json`. A dry run returns `mode: dry_run`, ordered stages, model name, taxonomy counts, and `execution_required: true`. It does not include predictions, performance metrics, or manuscript-result constants.

## Validation and comparison outputs

An executed validation artifact must contain at least:

- immutable example ID;
- split (`train` or `validation`);
- input text;
- retrieved candidates where applicable;
- exact prompt or prompt hash;
- raw model response;
- parsed prediction;
- frozen reference label;
- exact-match result;
- model/checkpoint identifier;
- learning-curve size;
- random seed and configuration hash.

Aggregate metrics must be derived solely from the 50 frozen validation records. Model comparison contains six checkpoint rows for each learning size 0/50/100/150/200. Absent archived predictions, no aggregate value is considered reproduced.

## Expected-results boundary

`paper/expected_results.json` contains constants reported or targeted by the paper. Its top-level `validation_expectations_only` flag is `true`, and `recompute` is `false`. Tests may validate schema, values, and tolerances, but must not manufacture prediction files or declare the paper results reproduced from these constants.

The exact expected counts are:

| Field | Validation target |
|---|---:|
| `registered_trials` | 563,077 |
| `result_trials` | 73,427 |
| `mapped_outcome_records` | 480,273 |
| `normalized_outcome_concepts` | 237,820 |
| `outcome_categories` | 21 |
| `raw_condition_strings` | 26,682 |
| `snomed_condition_concepts` | 238 |
| `disease_areas` | 24 |

These counts and the reported 0.88 outcome accuracy, 0.92 condition accuracy, and 0.95 condition macro F1 are paper targets only. They are not recomputed by the contract layer.
