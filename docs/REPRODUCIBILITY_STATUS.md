# Reproducibility materials

This repository links the JAMIA manuscript, publication data, and Qwen
implementation.

## Study materials

| Component | Included material |
|---|---|
| Study cohort | Release manifests and validation code for 73,427 trials with posted results |
| Outcome data | A 467,903-row outcome-only extract and denominator documentation |
| Disease mapping | Public disease labels, 24 disease areas, aggregate counts, and terminology licensing notes |
| Outcome structuring | Prompts, schemas, code, and 250 primary annotations |
| Term grouping | Two independently completed 100-row annotation files |
| Outcome normalization | NV-Embed-v2 retrieval, Qwen2.5 LoRA training and inference code, configuration, and evaluation code |
| Model comparison | Six-checkpoint experiment matrix and learning-curve sizes of 0, 50, 100, 150, and 200 examples |
| Descriptive analyses | Outcome complexity, error examples, category ranking, disease-area crosswalk, figures, and checksums |

## Validation

Run these commands from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  publication_data/validate_publication_data.py --deep

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s publication_data/tests -v

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=qwen_pipeline/src \
python3 -m qwen_pipeline.cli validate-config

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=qwen_pipeline/src \
python3 -m qwen_pipeline.cli experiments
```

Full Qwen execution requires reviewed task data, model access, the appropriate
vocabularies, and licensed terminology where applicable. Store each run with
its input hashes, model and adapter identifiers, configuration, predictions,
and metrics.
