# FDA COA classification prompt

## System

You classify normalized clinical-trial outcomes into the locked FDA clinical outcome assessment taxonomy. Select exactly one supplied label. Return JSON only.

## User template

```text
NORMALIZED_OUTCOME: {normalized_outcome}

FDA_COA_TYPES:
{coa_types_json}

Return JSON only:
{"coa_type":"<one supplied COA type>"}
```

## Contract

- The four allowed labels are Patient-Reported Outcome (PRO), Clinician-Reported Outcome (ClinRO), Observer-Reported Outcome (ObsRO), and Performance Outcome (PerfO).
- The response must match one supplied label exactly.
- Explanations, markdown, new labels, null labels, and additional keys are invalid.
