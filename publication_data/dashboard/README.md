# Dashboard data package

These checksum-covered CSV files power the
[Clinical Trial Outcome Atlas](https://github.com/jamesbbaker/ClinicalTrialOutcomeTrends).
They summarize conditions, disease areas, normalized outcomes, phases, years,
and completion periods from the study files.

The package contains no SNOMED CT identifiers or terminology. `manifest.json`
lists each file, size, and SHA-256 hash.

To refresh a local dashboard checkout from this repository:

```bash
python scripts/sync_paper_data.py \
  --paper-repo-root /path/to/clinical-trial-outcome
```
