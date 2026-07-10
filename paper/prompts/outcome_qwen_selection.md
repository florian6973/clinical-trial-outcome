# Retrieval-augmented Qwen outcome-selection prompt

## System

You are a helpful and harmless assistant. You should output the correct answer.

## User template

```text
Task: Categorize the following Clinical Trial Outcome Object into one of the standardized groups listed below or create a new one if none are suitable.

Main term: {main_term}

Five similar terms for context:
1. {term_1} ({term_score_1:.2f})
2. {term_2} ({term_score_2:.2f})
3. {term_3} ({term_score_3:.2f})
4. {term_4} ({term_score_4:.2f})
5. {term_5} ({term_score_5:.2f})

Five candidate groups:
1. {group_1} ({group_score_1:.2f})
2. {group_2} ({group_score_2:.2f})
3. {group_3} ({group_score_3:.2f})
4. {group_4} ({group_score_4:.2f})
5. {group_5} ({group_score_5:.2f})

First explain briefly why the main term matches an existing group or requires a new concise group. Then provide exactly one group.

EXPLANATION: <brief explanation>
GROUP: <group name>
```

## Contract

- The main term is followed by exactly five retrieved terms and five retrieved groups, each with a similarity score.
- The supervised response contains one `EXPLANATION:` line followed by one `GROUP:` line.
- The parsed prediction is the single value following `GROUP:`.
- When no candidate group is suitable, the model may create a concise normalization group; this open-set behavior does not apply to the downstream locked 21-category classifier.
- Generation and parsing settings remain those in the Qwen configuration and output contract.
