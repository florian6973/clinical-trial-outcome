# Outcome-category classification prompt

## System

You classify normalized clinical-trial outcomes into the locked paper taxonomy. Select exactly one supplied label. Return JSON only.

## User template

```text
NORMALIZED_OUTCOME: {normalized_outcome}

OUTCOME_CATEGORIES:
{outcome_categories_json}

Return JSON only:
{"outcome_category":"<one supplied outcome category>"}
```

## Contract

- There are exactly 21 allowed outcome-category labels.
- The response must match one supplied label exactly.
- Explanations, markdown, new labels, and additional keys are invalid.
