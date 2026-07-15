#!/usr/bin/env python3
"""Plot median and p95-latency-derived Vector Add bandwidth curves."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DTYPES = ("float32", "float16", "int32")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    failed = [row for row in rows if row["correctness"] != "PASS"]
    if failed:
        raise ValueError(f"{path} contains failed correctness rows")
    return rows


def group_rows(rows: list[dict[str, str]]):
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dtype"], row["requested_variant"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["block_size"]))
    return grouped


def plot_bandwidth(rows: list[dict[str, str]], prefix: Path) -> None:
    grouped = group_rows(rows)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for axis, dtype in zip(axes, DTYPES):
        variants = sorted(
            variant for current_dtype, variant in grouped if current_dtype == dtype
        )
        for variant in variants:
            values = grouped[(dtype, variant)]
            blocks = [int(row["block_size"]) for row in values]
            median = [float(row["median_effective_gbps"]) for row in values]
            p95_lower = [
                float(row["p95_latency_effective_gbps"]) for row in values
            ]
            (line,) = axis.plot(blocks, median, marker="o", label=variant)
            axis.fill_between(
                blocks,
                p95_lower,
                median,
                color=line.get_color(),
                alpha=0.14,
                label=f"{variant} median→p95 latency",
            )
        axis.set_title(dtype)
        axis.set_xlabel("threads per block")
        axis.grid(True, alpha=0.3)
        # The packed variant has a different name for every dtype (float4,
        # half2, int4), so each panel needs its own legend.
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("effective bandwidth (GB/s)")
    first = rows[0]
    figure.suptitle(
        f"Vector Add block-size sweep: N={int(first['N']):,}, "
        f"rounds={first['rounds']}, iterations/round={first['iterations']}"
    )
    figure.tight_layout()
    for suffix in ("png", "svg"):
        output = prefix.with_name(prefix.name + "_bandwidth").with_suffix(f".{suffix}")
        figure.savefig(output, dpi=180, bbox_inches="tight")
        print(output)
    plt.close(figure)


def plot_alignment(
    aligned_rows: list[dict[str, str]],
    unaligned_rows: list[dict[str, str]],
    prefix: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    datasets = (("aligned", aligned_rows), ("offset=1 fallback", unaligned_rows))
    for axis, dtype in zip(axes, DTYPES):
        for label, rows in datasets:
            values = [
                row
                for row in rows
                if row["dtype"] == dtype and row["requested_variant"] != "scalar"
            ]
            values.sort(key=lambda row: int(row["block_size"]))
            axis.plot(
                [int(row["block_size"]) for row in values],
                [float(row["median_effective_gbps"]) for row in values],
                marker="o",
                label=label,
            )
        axis.set_title(dtype)
        axis.set_xlabel("threads per block")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("median effective bandwidth (GB/s)")
    axes[-1].legend(fontsize=8, loc="best")
    figure.suptitle("Packed path: aligned access versus scalar alignment fallback")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        output = prefix.with_name(prefix.name + "_alignment").with_suffix(f".{suffix}")
        figure.savefig(output, dpi=180, bbox_inches="tight")
        print(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="aligned advanced CSV")
    parser.add_argument("--unaligned-csv", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    aligned = read_rows(args.csv)
    plot_bandwidth(aligned, args.output_prefix)
    if args.unaligned_csv:
        unaligned = read_rows(args.unaligned_csv)
        plot_alignment(aligned, unaligned, args.output_prefix)


if __name__ == "__main__":
    main()
