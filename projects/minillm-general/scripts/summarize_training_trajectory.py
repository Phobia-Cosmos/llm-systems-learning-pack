#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("ceval", "cmmlu", "arc_easy", "arc_challenge", "hellaswag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize milestone validation and capability metrics.")
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def json_lines(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def normalized_score(task: str, payload: dict) -> float:
    metrics = payload["metrics"]
    metric = next(iter(metrics.values())) if task == "hellaswag" else metrics.get("content", metrics["label"])
    return float(metric["accuracy_normalized"])


def checkpoint_tokens(benchmark: dict) -> int | None:
    checkpoint = benchmark.get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("tokens_processed") is not None:
        return int(checkpoint["tokens_processed"])
    model = benchmark.get("model")
    path = model.get("path") if isinstance(model, dict) else None
    if not isinstance(path, str) or not Path(path).is_file():
        return None
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    value = payload.get("tokens_processed") if isinstance(payload, dict) else None
    return int(value) if value is not None else None


def main() -> None:
    args = parse_args()
    training_log = Path(args.training_log).expanduser().resolve()
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    records = json_lines(training_log)
    validation_by_step = {
        int(record["step"]): record
        for record in records
        if "validation_loss" in record
    }
    tokens_by_step = {
        int(record["step"]): int(record["tokens_processed"])
        for record in records
        if "tokens_processed" in record
    }
    milestones = []
    for benchmark_path in sorted(benchmark_dir.glob("step-*.json")):
        step = int(benchmark_path.stem.removeprefix("step-"))
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        validation = validation_by_step.get(step, {})
        milestones.append(
            {
                "step": step,
                "tokens_processed": tokens_by_step.get(step, checkpoint_tokens(benchmark)),
                "validation_loss": validation.get("validation_loss"),
                "validation_nats_per_byte": validation.get("validation_nats_per_byte"),
                "capability_normalized_accuracy": {
                    task: normalized_score(task, benchmark["tasks"][task])
                    for task in TASKS
                },
            }
        )

    payload = {"schema_version": 1, "milestones": milestones}
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| tokens | step | val loss | nats/byte | C-Eval | CMMLU | ARC-E | ARC-C | HellaSwag |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in milestones:
        scores = row["capability_normalized_accuracy"]
        tokens = row["tokens_processed"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{tokens:,}" if tokens is not None else "n/a",
                    f"{row['step']:,}",
                    f"{row['validation_loss']:.4f}" if row["validation_loss"] is not None else "n/a",
                    f"{row['validation_nats_per_byte']:.4f}"
                    if row["validation_nats_per_byte"] is not None
                    else "n/a",
                    *(f"{100 * scores[task]:.1f}%" for task in TASKS),
                ]
            )
            + " |"
        )
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
