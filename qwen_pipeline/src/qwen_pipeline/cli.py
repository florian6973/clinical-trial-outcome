"""Dry-run-first command-line interface for the paper-aligned Qwen workflow."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from .config import config_digest, load_configs, validate_configs
from .evaluation import evaluate, evaluate_to_file
from .experiments import experiment_plan, require_executable_plan
from .inference import run_inference
from .io import read_jsonl, write_jsonl
from .paper_outputs import ingest_metrics, ingest_to_file
from .preparation import prepare_records, summarize_splits
from .retrieval import build_index, inspect_vocabulary
from .training import inspect_training_file, train_lora


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="qwen-normalize",
        description="Paper-aligned Qwen workflow; expensive commands are dry-run unless --execute is supplied.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config")

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--task", choices=["outcome", "condition"], required=True)
    prepare.add_argument("--annotations", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split", choices=["train", "validation", "test"])
    prepare.add_argument("--execute", action="store_true")

    index = commands.add_parser("build-index")
    index.add_argument("--task", choices=["outcome", "condition"], required=True)
    index.add_argument("--vocabulary", required=True)
    index.add_argument("--output-dir", required=True)
    index.add_argument("--execute", action="store_true")

    train = commands.add_parser("train")
    train.add_argument("--task", choices=["outcome", "condition"], required=True)
    train.add_argument("--train-jsonl", required=True)
    train.add_argument("--validation-jsonl")
    train.add_argument("--model-id")
    train.add_argument("--max-train-records", type=int)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--execute", action="store_true")

    infer = commands.add_parser("infer")
    infer.add_argument("--task", choices=["outcome", "condition"], required=True)
    infer.add_argument("--input", required=True)
    infer.add_argument("--index-dir", required=True)
    infer.add_argument("--adapter", required=True)
    infer.add_argument("--crosswalk")
    infer.add_argument("--output", required=True)
    infer.add_argument("--execute", action="store_true")

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--task", choices=["outcome", "condition"], required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--gold", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--execute", action="store_true")

    experiments = commands.add_parser("experiments")
    experiments.add_argument("--execute", action="store_true")

    ingest = commands.add_parser("ingest-paper")
    ingest.add_argument("--metrics", required=True)
    ingest.add_argument("--output")
    ingest.add_argument("--execute", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-config":
        configs = load_configs()
        errors = validate_configs(configs)
        _print({"status": "valid" if not errors else "invalid", "errors": errors, "configuration_sha256": config_digest(configs)})
        return 0 if not errors else 2

    if args.command == "prepare":
        source = list(read_jsonl(args.annotations))
        prepared = list(prepare_records(args.task, source))
        if args.split:
            prepared = [record for record in prepared if record["split"] == args.split]
        summary = {
            "status": "written" if args.execute else "dry_run_not_written",
            "task": args.task,
            "input_records": len(source),
            "selected_records": len(prepared),
            "split_counts": summarize_splits(prepared),
            "output": str(args.output),
            "sample": prepared[:1],
        }
        if args.execute:
            write_jsonl(args.output, prepared)
        _print(summary)
        return 0

    if args.command == "build-index":
        summary = inspect_vocabulary(args.task, args.vocabulary)
        summary.update({"status": "dry_run_not_built", "output_dir": args.output_dir})
        if args.execute:
            summary = build_index(args.task, args.vocabulary, args.output_dir)
        _print(summary)
        return 0

    if args.command == "train":
        summary = inspect_training_file(args.task, args.train_jsonl)
        summary.update(
            {
                "status": "dry_run_not_trained",
                "output_dir": args.output_dir,
                "validation_jsonl": args.validation_jsonl,
                "model_id": args.model_id,
                "max_train_records": args.max_train_records,
                "training_pattern": "historical response-masked Accelerate/PEFT loop",
            }
        )
        if args.execute:
            summary = train_lora(
                args.task,
                args.train_jsonl,
                args.output_dir,
                validation_jsonl=args.validation_jsonl,
                model_id=args.model_id,
                max_train_records=args.max_train_records,
            )
        _print(summary)
        return 0

    if args.command == "infer":
        if not args.execute:
            records = list(read_jsonl(args.input))
            _print({
                "status": "dry_run_not_inferred",
                "task": args.task,
                "input_records": len(records),
                "index_dir": args.index_dir,
                "adapter": args.adapter,
                "crosswalk": args.crosswalk,
                "output": args.output,
            })
        else:
            _print(run_inference(args.task, args.input, args.index_dir, args.adapter, args.output, args.crosswalk))
        return 0

    if args.command == "evaluate":
        result = evaluate_to_file(args.task, args.predictions, args.gold, args.output) if args.execute else evaluate(args.task, args.predictions, args.gold)
        result["write_status"] = "written" if args.execute else "dry_run_not_written"
        _print(result)
        return 0

    if args.command == "experiments":
        plan = experiment_plan()
        if args.execute:
            require_executable_plan(plan)
        _print(plan)
        return 0

    if args.command == "ingest-paper":
        if args.execute and not args.output:
            raise SystemExit("--execute requires --output")
        result = ingest_to_file(args.metrics, args.output) if args.execute else ingest_metrics(args.metrics)
        result["write_status"] = "written" if args.execute else "dry_run_not_written"
        _print(result)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
