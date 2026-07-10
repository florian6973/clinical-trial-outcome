"""Publication-analysis specifications for the paper-aligned Qwen pipeline.

Every analysis is dry-run by default.  Callers must pass ``execute=True`` (or
``--execute`` on the command line) before source data are read or outputs are
written.

EVIDENCE BOUNDARY: execution computes a new descriptive artifact from the
caller's input; it is not reproduced paper evidence.  These modules never
populate ``paper/expected_results.json`` and cannot establish the manuscript's
reported values without the archived, lineage-verified analysis inputs and
predictions required by the paper contracts.  Their ``AnalysisPlan`` payloads
are analysis-specific specifications, not model-pipeline prediction outputs.
"""

from ._common import AnalysisPlan, OutputSchema

__all__ = ["AnalysisPlan", "OutputSchema"]
