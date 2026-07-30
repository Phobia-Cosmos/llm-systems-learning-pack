#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed 32K/48K proxy LM runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--minimum-relative-nats-improvement",
        type=float,
        default=0.03,
        help="Require this relative nats/byte gain before paying the 48K model cost.",
    )
    return parser.parse_args()


def read_json_lines(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def summarize(run_root: Path, candidate: str) -> dict:
    records = read_json_lines(run_root / f"{candidate}.log")
    parameter_records = [record for record in records if "parameters" in record]
    validation_records = [record for record in records if "validation_loss" in record]
    throughput = [
        float(record["tokens_per_second"])
        for record in records
        if "tokens_per_second" in record and int(record.get("step", 0)) >= 100
    ]
    if not parameter_records or not validation_records or not throughput:
        raise RuntimeError(f"incomplete proxy run: {candidate}")
    final_validation = max(validation_records, key=lambda record: int(record["step"]))
    return {
        "candidate": candidate,
        "parameters": int(parameter_records[-1]["parameters"]),
        "final_step": int(final_validation["step"]),
        "validation_loss": float(final_validation["validation_loss"]),
        "validation_nats_per_byte": float(final_validation["validation_nats_per_byte"]),
        "median_tokens_per_second_after_step_100": statistics.median(throughput),
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    summaries = [summarize(run_root, candidate) for candidate in ("proxy32", "proxy48")]
    by_candidate = {item["candidate"]: item for item in summaries}
    proxy32 = by_candidate["proxy32"]
    proxy48 = by_candidate["proxy48"]
    relative_nats_improvement = (
        proxy32["validation_nats_per_byte"] - proxy48["validation_nats_per_byte"]
    ) / proxy32["validation_nats_per_byte"]
    selected = (
        "proxy48"
        if relative_nats_improvement >= args.minimum_relative_nats_improvement
        else "proxy32"
    )
    payload = {
        "schema_version": 1,
        "raw_nats_per_byte_winner": min(
            summaries, key=lambda item: item["validation_nats_per_byte"]
        )["candidate"],
        "selection_policy": {
            "metric": "validation_nats_per_byte",
            "minimum_relative_improvement_for_48k": args.minimum_relative_nats_improvement,
        },
        "relative_nats_per_byte_improvement_48k_vs_32k": relative_nats_improvement,
        "relative_parameter_increase_48k_vs_32k": (
            proxy48["parameters"] - proxy32["parameters"]
        )
        / proxy32["parameters"],
        "relative_throughput_change_48k_vs_32k": (
            proxy48["median_tokens_per_second_after_step_100"]
            - proxy32["median_tokens_per_second_after_step_100"]
        )
        / proxy32["median_tokens_per_second_after_step_100"],
        "selected": selected,
        "runs": summaries,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
