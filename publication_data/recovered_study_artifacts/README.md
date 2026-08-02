# Study annotations and supporting figures

These files come from
[`florian6973/clinical-trial-outcome`](https://github.com/florian6973/clinical-trial-outcome)
at commit `a81eb347e3e446dc58b6f36ca81dc2cd7479757c` and the supplementary Word
document in this repository.

## Annotation artifacts

| File | Records | Role |
|---|---:|---|
| `manual-ann-ner-fp.yaml` | 250 | Primary outcome-title structuring annotations (object, measure, specifier, time, unit, and range) |
| `manual-ann-ner-jb.yaml` | 108 | Second-annotator structuring records available at the pinned public commit |
| `group-ann-fp.csv` | 100 | First annotator's term-to-concept grouping labels, including the complete retrieval prompt and explanation |
| `group-ann-jb.csv` | 100 | Second annotator's labels for the same term-to-concept grouping examples |

The 250-title structuring task and 100-row Qwen grouping task are separate. Do
not combine their train/test counts. The public code splits the structuring set
80/20 with seed 0 and the grouping set 90/10 with seed 42. The manuscript
reports a separate 200-development/50-held-out design.

## Figure artifacts

The nine images under `figures/` show outcome complexity, model comparisons,
outcome use, condition frequency, and time trends. The supplementary appendix
provides their captions and interpretation.

## Integrity

`SHA256SUMS` records each file checksum. The annotations retain their original
CSV or YAML format; no labels were changed.
