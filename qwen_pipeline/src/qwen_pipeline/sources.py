"""Prepare deterministic pipeline inputs from AACT and licensed SNOMED CT data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .io import sha256_file, write_jsonl
from .validation import validate_schema


FSN_TYPE_ID = "900000000000003001"
SYNONYM_TYPE_ID = "900000000000013009"
PREFERRED_ACCEPTABILITY_ID = "900000000000548007"
IS_A_TYPE_ID = "116680003"
DISORDER_ROOT_ID = "64572001"

OUTCOMES_FILE = "outcomes.raw.jsonl"
CONDITIONS_FILE = "conditions.raw.jsonl"
CONDITION_VOCABULARY_FILE = "condition_vocabulary.jsonl"
MANIFEST_FILE = "source_manifest.json"


class SourceContractError(ValueError):
    """Raised when an AACT or RF2 source violates the supported contract."""


@dataclass(frozen=True)
class SourcePreparation:
    outcomes: list[dict[str, Any]]
    conditions: list[dict[str, Any]]
    condition_vocabulary: list[dict[str, Any]]
    manifest_base: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "aact_snapshot_id": self.manifest_base["aact"]["snapshot_id"],
            "snomed_edition": self.manifest_base["snomed_ct"]["edition_identifier"],
            "outcomes": len(self.outcomes),
            "conditions": len(self.conditions),
            "condition_vocabulary": len(self.condition_vocabulary),
        }


def _open_table(path: Path) -> tuple[csv.DictReader, Any]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    first = handle.readline()
    handle.seek(0)
    if "\t" in first:
        delimiter = "\t"
    elif "|" in first:
        delimiter = "|"
    else:
        delimiter = ","
    return csv.DictReader(handle, delimiter=delimiter), handle


def _read_rows(path: Path, required: set[str]) -> Iterator[dict[str, str]]:
    reader, handle = _open_table(path)
    try:
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise SourceContractError(f"{path.name} is missing required columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise SourceContractError(f"{path.name}:{line_number} has more values than columns")
            yield {key: (value or "").strip() for key, value in row.items()}
    finally:
        handle.close()


def _find_aact_table(root: Path, table: str) -> Path:
    matches = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.stem.casefold() == table.casefold()
    )
    if len(matches) != 1:
        raise SourceContractError(
            f"expected exactly one AACT {table} table under {root}; found {len(matches)}"
        )
    return matches[0]


def _find_rf2_files(root: Path, token: str) -> list[Path]:
    token = token.casefold()
    matches = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "snapshot" in path.name.casefold()
        and token in path.name.casefold()
    )
    if not matches:
        raise SourceContractError(f"no RF2 Snapshot file matching {token!r} under {root}")
    return matches


def _find_relationship_files(root: Path) -> list[Path]:
    inferred = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "snapshot" in path.name.casefold()
        and "sct2_relationship_" in path.name.casefold()
    )
    if inferred:
        return inferred
    stated = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "snapshot" in path.name.casefold()
        and "sct2_statedrelationship_" in path.name.casefold()
    )
    if not stated:
        raise SourceContractError(f"no RF2 Relationship Snapshot file under {root}")
    return stated


def _source_file(path: Path, root: Path) -> dict[str, Any]:
    return {
        "file": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _date_from_text(value: str) -> str | None:
    matches = re.findall(r"(?<!\d)(20\d{6})(?!\d)", value)
    if matches:
        raw = sorted(matches)[-1]
    else:
        separated = re.findall(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)", value)
        if not separated:
            return None
        raw = "".join(sorted(separated)[-1])
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _prepare_aact(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    measurement_path = _find_aact_table(root, "outcome_measurements")
    outcome_path = _find_aact_table(root, "outcomes")
    condition_path = _find_aact_table(root, "conditions")

    measurement_rows = 0
    measured_outcome_ids: set[tuple[str, str]] = set()
    for row in _read_rows(measurement_path, {"nct_id", "outcome_id"}):
        measurement_rows += 1
        key = (row["nct_id"], row["outcome_id"])
        if not all(key):
            raise SourceContractError(
                f"{measurement_path.name} contains a blank nct_id or outcome_id"
            )
        measured_outcome_ids.add(key)

    outcomes: list[dict[str, Any]] = []
    outcome_ids: set[tuple[str, str]] = set()
    for row in _read_rows(
        outcome_path,
        {"id", "nct_id", "outcome_type", "title", "description", "time_frame"},
    ):
        key = (row["nct_id"], row["id"])
        if not all(key):
            raise SourceContractError(f"{outcome_path.name} contains a blank nct_id or id")
        if key in outcome_ids:
            raise SourceContractError(
                f"{outcome_path.name} contains duplicate nct_id plus outcome id: {key}"
            )
        outcome_ids.add(key)
        nct_id, outcome_id = key
        record = {
            "record_id": f"outcome:{nct_id}:{outcome_id}",
            "nct_id": nct_id,
            "outcome_id": outcome_id,
            "title": row["title"],
            "description": row["description"],
            "outcome_type": row["outcome_type"],
            "time_frame": row["time_frame"],
        }
        validate_schema(record, "raw_outcome.schema.json")
        outcomes.append(record)
    outcomes.sort(key=lambda row: (row["nct_id"], row["outcome_id"]))

    condition_rows = 0
    blank_condition_rows = 0
    unique_conditions: set[tuple[str, str]] = set()
    for row in _read_rows(condition_path, {"nct_id", "name"}):
        condition_rows += 1
        if not row["nct_id"]:
            raise SourceContractError(f"{condition_path.name} contains a blank nct_id")
        if not row["name"]:
            blank_condition_rows += 1
            continue
        unique_conditions.add((row["nct_id"], row["name"]))

    conditions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for nct_id, name in sorted(unique_conditions):
        suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        record_id = f"condition:{nct_id}:{suffix}"
        if record_id in seen_ids:
            raise SourceContractError(f"condition record_id collision: {record_id}")
        seen_ids.add(record_id)
        record = {"record_id": record_id, "nct_id": nct_id, "raw_condition": name}
        validate_schema(record, "raw_condition.schema.json")
        conditions.append(record)

    files = [measurement_path, outcome_path, condition_path]
    metadata = {
        "files": [_source_file(path, root) for path in files],
        "counts": {
            "outcome_measurement_rows": measurement_rows,
            "outcome_rows": len(outcomes),
            "unique_outcomes": len(outcomes),
            "outcomes_with_measurements": len(outcome_ids & measured_outcome_ids),
            "outcomes_without_measurements": len(outcome_ids - measured_outcome_ids),
            "measurement_links_without_outcome": len(measured_outcome_ids - outcome_ids),
            "condition_rows": condition_rows,
            "blank_condition_rows": blank_condition_rows,
            "unique_conditions": len(conditions),
        },
    }
    return outcomes, conditions, metadata


def _prepare_snomed(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    concept_files = _find_rf2_files(root, "sct2_concept_")
    description_files = _find_rf2_files(root, "sct2_description_")
    language_files = _find_rf2_files(root, "der2_crefset_language")
    relationship_files = _find_relationship_files(root)
    all_files = concept_files + description_files + language_files + relationship_files

    active_concepts: set[str] = set()
    module_ids: set[str] = set()
    effective_times: set[str] = set()
    for path in concept_files:
        for row in _read_rows(path, {"id", "effectiveTime", "active", "moduleId"}):
            if row["effectiveTime"]:
                effective_times.add(row["effectiveTime"])
            if row["active"] == "1":
                active_concepts.add(row["id"])
                module_ids.add(row["moduleId"])
    if DISORDER_ROOT_ID not in active_concepts:
        raise SourceContractError(
            f"active SNOMED CT disorder root {DISORDER_ROOT_ID} is absent from Concept Snapshot"
        )

    descriptions: dict[str, list[dict[str, str]]] = defaultdict(list)
    active_description_ids: set[str] = set()
    for path in description_files:
        for row in _read_rows(
            path,
            {"id", "effectiveTime", "active", "moduleId", "conceptId", "languageCode", "typeId", "term"},
        ):
            if row["effectiveTime"]:
                effective_times.add(row["effectiveTime"])
            if (
                row["active"] == "1"
                and row["conceptId"] in active_concepts
                and row["languageCode"].casefold().startswith("en")
            ):
                descriptions[row["conceptId"]].append(row)
                active_description_ids.add(row["id"])
                module_ids.add(row["moduleId"])

    preferred_description_ids: set[str] = set()
    for path in language_files:
        for row in _read_rows(
            path,
            {"id", "effectiveTime", "active", "moduleId", "referencedComponentId", "acceptabilityId"},
        ):
            if row["effectiveTime"]:
                effective_times.add(row["effectiveTime"])
            if (
                row["active"] == "1"
                and row["acceptabilityId"] == PREFERRED_ACCEPTABILITY_ID
                and row["referencedComponentId"] in active_description_ids
            ):
                preferred_description_ids.add(row["referencedComponentId"])
                module_ids.add(row["moduleId"])

    children: dict[str, set[str]] = defaultdict(set)
    active_is_a_rows = 0
    for path in relationship_files:
        for row in _read_rows(
            path,
            {"id", "effectiveTime", "active", "moduleId", "sourceId", "destinationId", "typeId"},
        ):
            if row["effectiveTime"]:
                effective_times.add(row["effectiveTime"])
            if (
                row["active"] == "1"
                and row["typeId"] == IS_A_TYPE_ID
                and row["sourceId"] in active_concepts
                and row["destinationId"] in active_concepts
            ):
                children[row["destinationId"]].add(row["sourceId"])
                active_is_a_rows += 1
                module_ids.add(row["moduleId"])

    descendants: set[str] = set()
    queue = deque(sorted(children.get(DISORDER_ROOT_ID, set())))
    while queue:
        concept_id = queue.popleft()
        if concept_id in descendants:
            continue
        descendants.add(concept_id)
        queue.extend(sorted(children.get(concept_id, set())))

    vocabulary: list[dict[str, Any]] = []
    missing_english_term = 0
    preferred_fallbacks = 0
    for concept_id in sorted(descendants):
        concept_descriptions = descriptions.get(concept_id, [])
        preferred = [
            row for row in concept_descriptions if row["id"] in preferred_description_ids
        ]
        preferred.sort(
            key=lambda row: (
                0 if row["typeId"] == SYNONYM_TYPE_ID else 1,
                row["term"].casefold(),
                row["id"],
            )
        )
        selected = preferred[0] if preferred else None
        if selected is None:
            fsns = sorted(
                (row for row in concept_descriptions if row["typeId"] == FSN_TYPE_ID),
                key=lambda row: (row["term"].casefold(), row["id"]),
            )
            if not fsns:
                missing_english_term += 1
                continue
            selected = fsns[0]
            preferred_fallbacks += 1
        term = selected["term"].strip()
        if selected["typeId"] == FSN_TYPE_ID:
            term = re.sub(r"\s+\(disorder\)\s*$", "", term, flags=re.IGNORECASE).strip()
        record = {
            "vocabulary_id": f"snomed:{concept_id}",
            "concept_id": concept_id,
            "preferred_term": term,
        }
        validate_schema(record, "vocabulary.schema.json")
        vocabulary.append(record)

    metadata = {
        "files": [_source_file(path, root) for path in all_files],
        "module_ids": sorted(module_ids),
        "effective_times": sorted(effective_times),
        "counts": {
            "active_concepts": len(active_concepts),
            "active_is_a_relationships": active_is_a_rows,
            "disorder_descendants": len(descendants),
            "vocabulary_records": len(vocabulary),
            "preferred_term_fallbacks": preferred_fallbacks,
            "disorder_concepts_without_english_term": missing_english_term,
        },
    }
    return vocabulary, metadata


def prepare_sources(
    aact_dir: str | Path,
    snomed_rf2_dir: str | Path,
    *,
    aact_snapshot_id: str | None = None,
    snomed_edition: str | None = None,
) -> SourcePreparation:
    """Validate and prepare AACT records and a SNOMED CT disorder vocabulary."""

    aact_root = Path(aact_dir).resolve()
    snomed_root = Path(snomed_rf2_dir).resolve()
    if not aact_root.is_dir():
        raise SourceContractError(f"AACT directory does not exist: {aact_root}")
    if not snomed_root.is_dir():
        raise SourceContractError(f"SNOMED CT RF2 directory does not exist: {snomed_root}")

    outcomes, conditions, aact_metadata = _prepare_aact(aact_root)
    vocabulary, snomed_metadata = _prepare_snomed(snomed_root)
    inferred_snapshot = aact_snapshot_id or aact_root.name
    inferred_date = _date_from_text(inferred_snapshot) or _date_from_text(aact_root.name)
    edition = snomed_edition or snomed_root.name
    version_date = (
        _date_from_text(sorted(snomed_metadata["effective_times"])[-1])
        if snomed_metadata["effective_times"]
        else None
    )
    manifest_base = {
        "manifest_version": "1.0",
        "status": "prepared_source_inputs",
        "aact": {
            "snapshot_id": inferred_snapshot,
            "snapshot_date": inferred_date,
            **aact_metadata,
        },
        "snomed_ct": {
            "edition_identifier": edition,
            "version_date": version_date,
            **snomed_metadata,
        },
        "filters": {
            "aact_outcome_unit": "one nct_id plus outcome_id",
            "snomed_root_concept_id": DISORDER_ROOT_ID,
            "snomed_relationship_type_id": IS_A_TYPE_ID,
            "active_rows_only": True,
            "english_descriptions_only": True,
        },
    }
    return SourcePreparation(outcomes, conditions, vocabulary, manifest_base)


def write_prepared_sources(prepared: SourcePreparation, output_dir: str | Path) -> dict[str, Any]:
    """Write source inputs and a deterministic provenance manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    assets = [
        (OUTCOMES_FILE, prepared.outcomes),
        (CONDITIONS_FILE, prepared.conditions),
        (CONDITION_VOCABULARY_FILE, prepared.condition_vocabulary),
    ]
    output_records: list[dict[str, Any]] = []
    for filename, records in assets:
        path = destination / filename
        count = write_jsonl(path, records)
        output_records.append(
            {
                "file": filename,
                "records": count,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {**prepared.manifest_base, "outputs": output_records}
    validate_schema(manifest, "source_manifest.schema.json")
    manifest_path = destination / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
