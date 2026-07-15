#!/usr/bin/env python3
"""Combine Nsight Compute CSV logs into one tidy metric table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_METRICS = (
        "smsp__sass_inst_executed_op_global_ld.sum",
        "smsp__sass_inst_executed_op_global_st.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "lts__t_sectors_srcunit_tex_op_write.sum",
        "lts__t_bytes.sum",
        "lts__t_bytes.sum.per_second",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__bytes_read.sum.per_second",
        "dram__bytes_write.sum.per_second",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__time_duration.sum",
)


def parse_log(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    long_header_index = next(
        (index for index, line in enumerate(lines) if '"Metric Name"' in line),
        None,
    )
    stem = path.stem.removeprefix("ncu_")
    dtype, variant = stem.split("_", 1)
    parsed: list[dict[str, str]] = []

    if long_header_index is not None:
        # Nsight Compute details-page CSV: one metric per row.
        reader = csv.DictReader(lines[long_header_index:])
        for row in reader:
            metric = (row.get("Metric Name") or "").strip()
            if not metric:
                continue
            parsed.append(
                {
                    "dtype": dtype,
                    "variant": variant,
                    "kernel": (row.get("Kernel Name") or "").strip(),
                    "metric": metric,
                    "unit": (row.get("Metric Unit") or "").strip(),
                    "value": (row.get("Metric Value") or "").strip(),
                }
            )
    else:
        # Nsight Compute 2025.3 raw-page CSV: metric identifiers are columns,
        # followed by one units row and one row per captured launch.
        wide_header_index = next(
            (index for index, line in enumerate(lines) if '"Kernel Name"' in line),
            None,
        )
        if wide_header_index is None:
            raise ValueError(f"NCU CSV header not found in {path}")
        table = list(csv.reader(lines[wide_header_index:]))
        if len(table) < 3:
            raise ValueError(f"NCU raw CSV has no units/data rows: {path}")
        header, units = table[0], table[1]
        column = {name: index for index, name in enumerate(header)}
        missing_columns = set(EXPECTED_METRICS) - set(column)
        if missing_columns:
            raise ValueError(
                f"{path} is missing requested metric columns: "
                + ", ".join(sorted(missing_columns))
            )
        if "ID" not in column or "Kernel Name" not in column:
            raise ValueError(f"NCU raw CSV lacks ID/Kernel Name columns: {path}")
        launches = [
            row
            for row in table[2:]
            if len(row) == len(header) and row[column["ID"]].strip()
        ]
        if len(launches) != 1:
            raise ValueError(
                f"expected one captured launch in {path}, found {len(launches)}"
            )
        launch = launches[0]
        kernel = launch[column["Kernel Name"]].strip()
        for metric in EXPECTED_METRICS:
            index = column[metric]
            parsed.append(
                {
                    "dtype": dtype,
                    "variant": variant,
                    "kernel": kernel,
                    "metric": metric,
                    "unit": units[index].strip() if index < len(units) else "",
                    "value": launch[index].strip(),
                }
            )

    if not parsed:
        raise ValueError(f"no metric rows found in {path}")
    missing = set(EXPECTED_METRICS) - {row["metric"] for row in parsed}
    if missing:
        raise ValueError(
            f"{path} is missing requested metrics: {', '.join(sorted(missing))}"
        )
    incomplete = [
        row["metric"]
        for row in parsed
        if not row["kernel"] or not row["value"] or row["value"].lower() == "n/a"
    ]
    if incomplete:
        raise ValueError(
            f"{path} contains incomplete metric rows: "
            + ", ".join(sorted(set(incomplete)))
        )
    return parsed


def write_comparison(rows: list[dict[str, str]], path: Path) -> None:
    values = {
        (row["dtype"], row["variant"], row["metric"]): float(row["value"])
        for row in rows
    }
    kernels = {
        (row["dtype"], row["variant"]): row["kernel"] for row in rows
    }

    load_metric = "smsp__sass_inst_executed_op_global_ld.sum"
    store_metric = "smsp__sass_inst_executed_op_global_st.sum"
    l2_rate_metric = "lts__t_bytes.sum.per_second"
    l2_peak_metric = "lts__throughput.avg.pct_of_peak_sustained_elapsed"
    dram_read_metric = "dram__bytes_read.sum.per_second"
    dram_write_metric = "dram__bytes_write.sum.per_second"
    dram_peak_metric = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    duration_metric = "gpu__time_duration.sum"

    comparison = []
    for dtype in ("float", "half", "int"):
        def value(variant: str, metric: str) -> float:
            return values[(dtype, variant, metric)]

        scalar_load = value("scalar", load_metric)
        vector_load = value("vector", load_metric)
        scalar_store = value("scalar", store_metric)
        vector_store = value("vector", store_metric)
        scalar_dram_gbps = (
            value("scalar", dram_read_metric)
            + value("scalar", dram_write_metric)
        ) / 1.0e9
        vector_dram_gbps = (
            value("vector", dram_read_metric)
            + value("vector", dram_write_metric)
        ) / 1.0e9
        comparison.append(
            {
                "dtype": dtype,
                "scalar_kernel": kernels[(dtype, "scalar")],
                "vector_kernel": kernels[(dtype, "vector")],
                "global_load_inst_scalar": f"{scalar_load:.0f}",
                "global_load_inst_vector": f"{vector_load:.0f}",
                "global_load_inst_vector_over_scalar": f"{vector_load / scalar_load:.6f}",
                "global_store_inst_scalar": f"{scalar_store:.0f}",
                "global_store_inst_vector": f"{vector_store:.0f}",
                "global_store_inst_vector_over_scalar": f"{vector_store / scalar_store:.6f}",
                "l2_gbps_scalar": f"{value('scalar', l2_rate_metric) / 1.0e9:.3f}",
                "l2_gbps_vector": f"{value('vector', l2_rate_metric) / 1.0e9:.3f}",
                "l2_peak_pct_scalar": f"{value('scalar', l2_peak_metric):.2f}",
                "l2_peak_pct_vector": f"{value('vector', l2_peak_metric):.2f}",
                "dram_total_gbps_scalar": f"{scalar_dram_gbps:.3f}",
                "dram_total_gbps_vector": f"{vector_dram_gbps:.3f}",
                "dram_peak_pct_scalar": f"{value('scalar', dram_peak_metric):.2f}",
                "dram_peak_pct_vector": f"{value('vector', dram_peak_metric):.2f}",
                "gpu_time_us_scalar": f"{value('scalar', duration_metric) / 1.0e3:.3f}",
                "gpu_time_us_vector": f"{value('vector', duration_metric) / 1.0e3:.3f}",
                "gpu_time_vector_over_scalar": (
                    f"{value('vector', duration_metric) / value('scalar', duration_metric):.6f}"
                ),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [row for path in args.logs for row in parse_log(path)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dtype", "variant", "kernel", "metric", "unit", "value"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    comparison_path = args.output.with_name("comparison.csv")
    write_comparison(rows, comparison_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
