#!/usr/bin/env python3
"""Build a compact machine-readable summary for the 13-hour experiment suite."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_status(path: Path) -> dict[str, str]:
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def summarize_matrix(path: Path) -> dict:
    rows = load_json(path).get("results", [])
    valid = [row for row in rows if not row.get("errors")]
    throughputs = [float(row["output_throughput_tps"]) for row in valid]
    return {
        "points": len(rows),
        "errors": sum(int(row.get("errors", 0)) for row in rows),
        "median_output_throughput_tps": statistics.median(throughputs) if throughputs else None,
        "peak_output_throughput_tps": max(throughputs) if throughputs else None,
    }


def last_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "requests" in row:
            return row
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result_root
    output = args.output or root / "summary.json"

    payload: dict = {
        "schema_version": 1,
        "result_root": str(root),
        "stages": {
            path.stem: read_status(path)
            for path in sorted((root / "stages").glob("*.status"))
        },
    }

    ddp_path = root / "ddp-scaling" / "summary.json"
    if ddp_path.exists():
        payload["ddp_scaling"] = load_json(ddp_path)

    serving_root = root / "serving-ab"
    payload["serving_ab"] = {
        path.parent.name: summarize_matrix(path)
        for path in sorted(serving_root.glob("*/results.json"))
    }

    tp_dp_root = root / "vllm-tp-dp"
    payload["vllm_tp_dp"] = {
        path.parent.name: summarize_matrix(path)
        for path in sorted(tp_dp_root.glob("*/results.json"))
    }

    long_progress = root / "vllm-dp4-soak" / "progress.log"
    payload["long_soak_last_progress"] = last_progress(long_progress)
    recovery_path = root / "soak-recovery" / "vllm-dp4-soak" / "results.json"
    if recovery_path.exists():
        recovery = load_json(recovery_path)
        payload["recovery_soak"] = {
            key: recovery.get(key)
            for key in (
                "duration_requested_s",
                "duration_actual_s",
                "concurrency",
                "requests",
                "errors",
                "output_throughput_tps",
            )
        }
        payload["recovery_soak"]["run_status"] = read_status(
            root / "soak-recovery" / "status"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
