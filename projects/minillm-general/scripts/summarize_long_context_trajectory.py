#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


TASKS = ("ceval", "cmmlu", "arc_easy", "arc_challenge", "hellaswag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the 8K continuation trajectory.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--long-eval-dir", required=True)
    parser.add_argument("--capability-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def normalized_score(task: str, payload: dict[str, Any]) -> float:
    metrics = payload["metrics"]
    metric = next(iter(metrics.values())) if task == "hellaswag" else metrics.get("content", metrics["label"])
    return float(metric["accuracy_normalized"])


def milestone_row(
    milestone: dict[str, Any],
    long_eval_dir: Path,
    capability_dir: Path,
) -> dict[str, Any]:
    step = int(milestone["target_step"])
    stem = f"step-{step:08d}"
    long_result = read_object(long_eval_dir / f"{stem}.json")
    capability = read_object(capability_dir / f"{stem}.json")
    row: dict[str, Any] = {
        **milestone,
        "long_evaluation": None,
        "capability_normalized_accuracy": None,
    }
    if long_result is not None:
        fixed = long_result.get("fixed_record_test") or {}
        regression = long_result.get("regression_loss") or {}
        passkey = (long_result.get("passkey_retrieval") or {}).get("aggregate", {})
        parity = long_result.get("cache_parity") or {}
        segments = fixed.get("position_loss_segments") or []
        row["long_evaluation"] = {
            "long_loss": fixed.get("overall_loss"),
            "tail_position_loss": segments[-1].get("loss") if segments else None,
            "regression_4k_loss": regression.get("overall_loss"),
            "passkey_teacher_forced_token_accuracy": passkey.get(
                "mean_teacher_forced_token_accuracy"
            ),
            "passkey_greedy_exact_fraction": passkey.get("greedy_exact_fraction"),
            "cache_parity_passed": parity.get("all_passed"),
            "checkpoint_tokens_processed": (long_result.get("checkpoint") or {}).get(
                "tokens_processed"
            ),
        }
    if capability is not None:
        tasks = capability.get("tasks")
        if isinstance(tasks, dict) and all(task in tasks for task in TASKS):
            row["capability_normalized_accuracy"] = {
                task: normalized_score(task, tasks[task]) for task in TASKS
            }
    return row


def format_float(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    long_eval_dir = Path(args.long_eval_dir).expanduser().resolve()
    capability_dir = Path(args.capability_dir).expanduser().resolve()
    rows = [
        milestone_row(milestone, long_eval_dir, capability_dir)
        for milestone in plan["milestones"]
    ]
    payload = {
        "schema_version": 1,
        "plan": str(plan_path),
        "base_checkpoint": plan["base_checkpoint"],
        "dataset": plan["dataset"],
        "milestones": rows,
    }
    atomic_write(
        Path(args.output_json).expanduser().resolve(),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )

    lines = [
        "| added tokens | step | 8K loss | tail loss | 4K regression | passkey TF | passkey exact | cache parity | C-Eval | CMMLU | ARC-E | ARC-C | HellaSwag |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        long_result = row["long_evaluation"] or {}
        scores = row["capability_normalized_accuracy"] or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{int(row['actual_additional_tokens']):,}",
                    f"{int(row['target_step']):,}",
                    format_float(long_result.get("long_loss")),
                    format_float(long_result.get("tail_position_loss")),
                    format_float(long_result.get("regression_4k_loss")),
                    format_float(long_result.get("passkey_teacher_forced_token_accuracy"), 3),
                    format_float(long_result.get("passkey_greedy_exact_fraction"), 3),
                    str(long_result.get("cache_parity_passed", "n/a")),
                    *(format_float(100 * scores[task], 1) if task in scores else "n/a" for task in TASKS),
                ]
            )
            + " |"
        )
    atomic_write(Path(args.output_markdown).expanduser().resolve(), "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
