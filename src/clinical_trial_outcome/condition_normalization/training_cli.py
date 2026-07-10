"""CLI for guarded condition-selector Qwen LoRA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .contract import SnomedCandidate
from .training import ConditionAnnotation, train_condition_qwen_lora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="250-row JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-execute", action="store_true")
    return parser


def _candidate(payload: dict[str, Any]) -> SnomedCandidate:
    return SnomedCandidate(
        concept_id=str(payload["concept_id"]),
        term=str(payload["term"]),
        disease_area=str(payload["disease_area"]),
        similarity=float(payload["similarity"]),
        rank=int(payload["rank"]),
    )


def load_annotations(path: Path) -> list[ConditionAnnotation]:
    annotations: list[ConditionAnnotation] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                annotations.append(
                    ConditionAnnotation(
                        condition_text=str(payload["condition_text"]),
                        candidates=tuple(
                            _candidate(candidate) for candidate in payload["candidates"]
                        ),
                        selected_concept_id=str(payload["selected_concept_id"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid annotation at line {line_number}") from exc
    return annotations


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_condition_qwen_lora(
        load_annotations(args.annotations),
        output_dir=args.output_dir,
        execute=args.execute,
        confirm_execute=args.confirm_execute,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
