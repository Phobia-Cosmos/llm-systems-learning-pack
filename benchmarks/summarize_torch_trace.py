#!/usr/bin/env python3
"""Reduce a PyTorch Chrome trace to stable, reviewable timing summaries."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


def open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def top_rows(values: dict[str, list[float]], limit: int) -> list[dict]:
    rows = [
        {"name": name, "calls": len(durations), "total_us": sum(durations)}
        for name, durations in values.items()
    ]
    rows.sort(key=lambda row: row["total_us"], reverse=True)
    return rows[:limit]


def kernel_group(name: str) -> str:
    lowered = name.lower()
    if any(value in lowered for value in ("flash", "fmha", "attention", "paged_attention")):
        return "attention"
    if any(value in lowered for value in ("gemm", "cublas", "matmul")):
        return "gemm"
    if "memcpy" in lowered or "memset" in lowered:
        return "memory_copy_or_set"
    if any(value in lowered for value in ("reduce", "softmax", "layer_norm", "rms")):
        return "reduction_or_normalization"
    return "other"


def merged_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    with open_trace(args.trace) as handle:
        payload = json.load(handle)

    by_category: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    kernel_intervals = []
    kernel_groups: dict[str, float] = collections.defaultdict(float)
    for event in payload.get("traceEvents", []):
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        category = str(event.get("cat", "uncategorized"))
        name = str(event.get("name", "unknown"))
        duration = float(event["dur"])
        by_category[category][name].append(duration)
        if category == "kernel":
            start = float(event.get("ts", 0.0))
            kernel_intervals.append((start, start + duration))
            kernel_groups[kernel_group(name)] += duration

    category_totals = {
        category: sum(sum(durations) for durations in names.values())
        for category, names in by_category.items()
    }
    kernel_sum = category_totals.get("kernel", 0.0)
    result = {
        "schema_version": 1,
        "trace": str(args.trace),
        "device_properties": payload.get("deviceProperties", []),
        "cuda_runtime_version": payload.get("cuda_runtime_version"),
        "cuda_driver_version": payload.get("cuda_driver_version"),
        "kernel_sum_us": kernel_sum,
        "gpu_active_union_us": merged_duration(kernel_intervals),
        "kernel_groups": [
            {
                "group": group,
                "total_us": duration,
                "share_of_kernel_sum": duration / kernel_sum if kernel_sum else None,
            }
            for group, duration in sorted(
                kernel_groups.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "category_totals_us": category_totals,
        "top_kernel": top_rows(by_category.get("kernel", {}), args.top),
        "top_cpu_op": top_rows(by_category.get("cpu_op", {}), args.top),
        "top_cuda_runtime": top_rows(by_category.get("cuda_runtime", {}), args.top),
        "top_user_annotation": top_rows(
            by_category.get("user_annotation", {}), args.top
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
