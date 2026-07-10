"""Dry-run-first command-line contract for condition normalization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .pipeline import ConditionNormalizationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="raw AACT condition text")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enable execution; default is a dependency-free dry run",
    )
    parser.add_argument(
        "--confirm-execute",
        action="store_true",
        help="second gate required for model execution",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = ConditionNormalizationPipeline()
    if args.execute:
        if not args.confirm_execute:
            raise SystemExit("--execute also requires --confirm-execute")
        raise SystemExit(
            "--execute requires programmatic injection of the configured embedding, "
            "FAISS retriever, and Qwen generator; no full run is started by this CLI"
        )
    print(json.dumps(pipeline.normalize(args.condition), indent=2))
    return 0
