"""Read-only helpers for AACT pipe-delimited exports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


REQUIRED_TABLES = ("studies", "outcomes", "conditions")


def resolve_aact_tables(aact_dir: str | Path, tables: Iterable[str] = REQUIRED_TABLES) -> dict[str, Path]:
    """Resolve table files without reading or changing them."""

    root = Path(aact_dir)
    resolved: dict[str, Path] = {}
    for table in tables:
        candidates = (root / f"{table}.txt", root / f"{table}.csv")
        match = next((path for path in candidates if path.exists()), candidates[0])
        resolved[table] = match
    return resolved


def missing_tables(paths: dict[str, Path]) -> list[str]:
    return [name for name, path in paths.items() if not path.exists()]


def read_aact_table(path: str | Path, *, columns: list[str] | None = None):
    """Read one AACT export; pandas is imported only when execution is requested."""

    import pandas as pd

    source = Path(path)
    separator = "|" if source.suffix == ".txt" else ","
    return pd.read_csv(source, sep=separator, usecols=columns, low_memory=False)

