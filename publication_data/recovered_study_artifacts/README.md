# Recovered study annotations and development figures

These files were recovered from the public study repository
[`florian6973/clinical-trial-outcome`](https://github.com/florian6973/clinical-trial-outcome)
at commit `a81eb347e3e446dc58b6f36ca81dc2cd7479757c` and from the original
supplementary Word document in this repository. They are included so the
annotation examples and development analyses reviewed by the authors are not
lost from the paper-aligned release.

## Annotation artifacts

| File | Records | Role |
|---|---:|---|
| `manual-ann-ner-fp.yaml` | 250 | Primary outcome-title structuring annotations (object, measure, specifier, time, unit, and range) |
| `manual-ann-ner-jb.yaml` | 108 | Second-annotator structuring records available at the pinned public commit |
| `group-ann-fp.csv` | 100 | First annotator's term-to-concept grouping labels, including the complete retrieval prompt and explanation |
| `group-ann-jb.csv` | 100 | Second annotator's labels for the same term-to-concept grouping examples |

The 250-title structuring task and the 100-row Qwen grouping task are different
annotation tasks. They must not be combined into a single train/test count. The
public code uses an 80/20 split with seed 0 for the 250-title structuring set and
a 90/10 split with seed 42 for the grouping data supplied to the Qwen
fine-tuning preparation script. The exact row manifest behind the manuscript's
reported 200-development/50-held-out Qwen result was not recovered.

The repository includes code for Cohen's kappa, but the printed kappa output is
not committed and the historical script names annotation files differently
across commits. The manuscript's kappa value should therefore be identified as
reported in the original study record rather than presented as newly verified
from this release.

## Figure artifacts

The files under `figures/` are byte-identical copies of nine images embedded in
the original supplementary Word document. They retain development-stage
evidence that was removed from an earlier revised appendix:

- outcome complexity in the 250-title annotation set;
- legacy BERT/GPT/Llama/Mistral structuring comparisons;
- older outcome-use, condition-frequency, and temporal descriptive displays.

These figures are provenance artifacts. They use earlier models, tasks, cohorts,
taxonomies, or time windows and are not Qwen2.5 validation results. Revised
captions in the supplementary appendix state those boundaries explicitly.

## Integrity

`SHA256SUMS` records the checksum of every recovered file. The source annotations
remain in their original CSV/YAML serialization; no labels were edited.
