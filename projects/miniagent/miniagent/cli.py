"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import BenchmarkConfig
from .runner import run_benchmark


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _csv_engines(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or any(item not in {"vllm", "sglang"} for item in values):
        raise argparse.ArgumentTypeError("engines must be vllm and/or sglang")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniagent",
        description="Evidence-first LLM serving benchmark agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark", help="run the Qwen vLLM/SGLang serving comparison"
    )
    benchmark.add_argument(
        "suite", nargs="?", default="qwen-compare", choices=("qwen-compare",)
    )
    benchmark.add_argument("--model", type=Path)
    benchmark.add_argument("--engines", type=_csv_engines, default=("vllm", "sglang"))
    benchmark.add_argument("--concurrency", type=_csv_ints, default=(1, 8, 32))
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--num-prompts", type=int, default=256)
    benchmark.add_argument("--input-tokens", type=int, default=1024)
    benchmark.add_argument("--output-tokens", type=int, default=128)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--run-id")
    benchmark.add_argument("--output-root", type=Path)
    benchmark.add_argument(
        "--smoke",
        action="store_true",
        help="use a fast c1/c8 integration matrix unless explicitly overridden",
    )
    benchmark.add_argument(
        "--dry-run", action="store_true", help="print commands without launching servers"
    )

    report = subparsers.add_parser("report", help="regenerate a report from raw evidence")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--output", type=Path)
    return parser


def _benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    concurrency = args.concurrency
    repeats = args.repeats
    prompts = args.num_prompts
    input_tokens = args.input_tokens
    output_tokens = args.output_tokens
    if args.smoke:
        if concurrency == (1, 8, 32):
            concurrency = (1, 8)
        if repeats == 3:
            repeats = 1
        if prompts == 256:
            prompts = 8
        if input_tokens == 1024:
            input_tokens = 128
        if output_tokens == 128:
            output_tokens = 16
    defaults = BenchmarkConfig()
    return defaults.with_overrides(
        model_path=args.model or defaults.model_path,
        engines=args.engines,
        concurrencies=concurrency,
        repeats=repeats,
        num_prompts=prompts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        seed=args.seed,
        output_root=args.output_root or defaults.output_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        config = _benchmark_config(args)
        result = run_benchmark(config, run_id=args.run_id, dry_run=args.dry_run)
        if isinstance(result, dict):
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result)
        return 0
    if args.command == "report":
        from .analysis import analyze_run, write_summary
        from .report import generate_report

        summary = analyze_run(args.run_dir)
        write_summary(summary, args.run_dir / "summary.json")
        output = args.output or args.run_dir / "report.md"
        generate_report(summary, output)
        print(output)
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2

