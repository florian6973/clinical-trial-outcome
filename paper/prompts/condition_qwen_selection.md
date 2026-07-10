# Condition-to-SNOMED Qwen selection prompt

## System

You map clinical-trial condition text to SNOMED CT. Select exactly one candidate supplied by the retrieval system. Do not invent a concept. Return JSON only.

## User template

```text
Select the best SNOMED CT mapping for the condition below.

CONDITION: {condition_text}

CANDIDATES (retrieved by FAISS over L2-normalized NV-Embed-v2 vectors):
{candidates_json}

Return JSON only with this exact shape:
{"selected_concept_id":"<one supplied concept_id>"}
```

## Contract

- Candidate retrieval occurs before this prompt.
- The selected `concept_id` must be present in `candidates_json`.
- The candidate's precomputed `disease_area` must be one of the locked 24 labels.
- Explanations, markdown, new concepts, and additional keys are invalid.
