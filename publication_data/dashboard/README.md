# Dashboard data package

These checksum-covered CSV files are the exact deterministic inputs used by the
[Clinical Trial Outcome Atlas](https://github.com/jamesbbaker/ClinicalTrialOutcomeTrends).
They contain condition, disease-area, normalized-outcome, phase, year, and
completion-period summaries derived from the paper's locked study files.

The package does not contain SNOMED CT identifiers or terminology content. It
reads the checked publication files and generates detail-page summaries
deterministically.
`manifest.json` records every included data file, byte size, and SHA-256 hash.

To refresh a local dashboard checkout from this repository:

```bash
python scripts/sync_paper_data.py \
  --paper-repo-root /path/to/clinical-trial-outcome
```
