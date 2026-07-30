"""Ingest supplied canonical manuscript metrics without recomputing them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import read_json
from .io import atomic_write_json, sha256_file
from .validation import ContractError, validate_schema


def ingest_metrics(path: str | Path) -> dict[str, Any]:
    document = read_json(path)
    validate_schema(document, "canonical_paper_metrics.schema.json")
    if document["status"] != "supplied_canonical_not_recomputed":
        raise ContractError("canonical metric file must preserve the supplied-not-recomputed status")
    warnings = []
    for name, metric in document["metrics"].items():
        numerator = metric.get("numerator")
        denominator = metric.get("denominator")
        value = metric["value"]
        if (numerator is None) != (denominator is None):
            warnings.append(f"{name}: numerator and denominator must be supplied together")
        if numerator is not None and denominator:
            calculated = numerator / denominator
            if isinstance(value, (int, float)) and abs(float(value) - calculated) > 1e-9:
                warnings.append(f"{name}: value disagrees with numerator/denominator")
    return {
        "status": document["status"],
        "source_document": document["source_document"],
        "artifact_version": document.get("artifact_version", ""),
        "metrics": document["metrics"],
        "metric_count": len(document["metrics"]),
        "input_sha256": sha256_file(path),
        "warnings": warnings,
        "note": "Values were validated and ingested, not recomputed from study records.",
    }


def ingest_to_file(path: str | Path, output_path: str | Path) -> dict[str, Any]:
    result = ingest_metrics(path)
    atomic_write_json(output_path, result)
    return result
