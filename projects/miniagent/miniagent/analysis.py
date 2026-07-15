"""Deterministic analysis for MiniAgent serving benchmarks.

The benchmark client stores one detailed ``vllm bench serve`` JSON document for
every engine/concurrency/repetition cell.  This module treats those documents
as the primary latency evidence and the concurrently sampled Prometheus summary
as causal context.  It deliberately uses only the Python standard library so
that analysis does not depend on either serving engine's environment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any


SCHEMA_VERSION = "miniagent.analysis.v1"
PERCENTILES = (50, 90, 95, 99)


def _finite_numbers(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def linear_percentile(values: Iterable[float], percentile: float) -> float | None:
    """Return the exact linear-interpolation percentile used by NumPy.

    For ``n`` sorted samples, the zero-based rank is
    ``(n - 1) * percentile / 100``.  Empty inputs return ``None`` so an
    incomplete run can still produce a diagnostic summary.
    """

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(_finite_numbers(values))
    if not ordered:
        return None
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize finite samples without rounding away source precision."""

    samples = _finite_numbers(values)
    result: dict[str, float | int | None] = {
        "count": len(samples),
        "mean": fmean(samples) if samples else None,
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
    }
    for percentile in PERCENTILES:
        result[f"p{percentile}"] = linear_percentile(samples, percentile)
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _nested_number(value: Any, statistic: str) -> float | None:
    direct = _number(value)
    if direct is not None:
        return direct
    if not isinstance(value, Mapping):
        return None
    candidates = {
        "max": ("max", "maximum", "value"),
        "mean": (
            "mean",
            "delta_mean",
            "delta_mean_seconds",
            "mean_seconds",
            "value",
        ),
        "delta": ("delta", "counter_delta", "value"),
        "count": ("delta_count", "count_delta", "count", "value"),
    }[statistic]
    for key in candidates:
        if key in value:
            found = _number(value[key])
            if found is not None:
                return found
    return None


def _walk_mappings(mapping: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, value in mapping.items():
        yield str(key).lower(), value
        if isinstance(value, Mapping):
            yield from _walk_mappings(value)


def _first_direct_number(
    summary: Mapping[str, Any], aliases: Sequence[str], statistic: str
) -> float | None:
    for alias in aliases:
        if alias in summary:
            found = _nested_number(summary[alias], statistic)
            if found is not None:
                return found
    return None


def _find_metric(
    summary: Mapping[str, Any],
    *,
    all_terms: Sequence[str],
    statistic: str,
    excluded_terms: Sequence[str] = (),
) -> float | None:
    for key, value in _walk_mappings(summary):
        normalized = key.replace(":", "_").replace(".", "_").replace("-", "_")
        if all(term in normalized for term in all_terms) and not any(
            term in normalized for term in excluded_terms
        ):
            found = _nested_number(value, statistic)
            if found is not None:
                return found
    return None


def _queue_histogram(summary: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Find a queue-time histogram and return (mean_seconds, delta_count)."""

    direct_mean = _first_direct_number(
        summary,
        (
            "queue_time_mean_seconds",
            "queue_time_delta_mean_seconds",
            "mean_queue_time_seconds",
        ),
        "mean",
    )
    direct_count = _first_direct_number(
        summary,
        ("queue_time_delta_count", "queue_delta_count"),
        "count",
    )
    if direct_mean is not None:
        return direct_mean, direct_count

    for key, value in _walk_mappings(summary):
        normalized = key.replace(":", "_").replace(".", "_").replace("-", "_")
        if "queue" not in normalized or "time" not in normalized:
            continue
        if not isinstance(value, Mapping):
            continue
        mean_value = _nested_number(value, "mean")
        delta_sum = _number(value.get("delta_sum"))
        if delta_sum is None:
            delta_sum = _number(value.get("sum_delta"))
        delta_count = _number(value.get("delta_count"))
        if delta_count is None:
            delta_count = _number(value.get("count_delta"))
        if mean_value is None and delta_sum is not None and delta_count:
            mean_value = delta_sum / delta_count
        if mean_value is not None:
            return mean_value, delta_count
    return None, direct_count


def normalize_metrics(payload: Mapping[str, Any] | None) -> dict[str, float | None]:
    """Normalize the sampler's direct summary keys and common nested layouts.

    The runner's stable interchange format is ``{"samples": ..., "summary":
    ...}``.  The fallbacks make historical or hand-produced summaries useful,
    but raw samples are intentionally not re-aggregated here.
    """

    if not isinstance(payload, Mapping):
        payload = {}
    nested = payload.get("summary")
    summary = nested if isinstance(nested, Mapping) else payload

    max_waiting = _first_direct_number(
        summary,
        (
            "max_waiting",
            "max_num_waiting",
            "max_queue_reqs",
            "max_num_requests_waiting",
        ),
        "max",
    )
    if max_waiting is None:
        max_waiting = _find_metric(
            summary,
            all_terms=("waiting",),
            statistic="max",
        )
    if max_waiting is None:
        max_waiting = _find_metric(
            summary,
            all_terms=("queue", "req"),
            statistic="max",
            excluded_terms=("time",),
        )

    max_running = _first_direct_number(
        summary,
        (
            "max_running",
            "max_num_running",
            "max_running_reqs",
            "max_num_requests_running",
        ),
        "max",
    )
    if max_running is None:
        max_running = _find_metric(
            summary,
            all_terms=("running",),
            statistic="max",
        )

    queue_mean, queue_count = _queue_histogram(summary)

    preemptions = _first_direct_number(
        summary,
        ("preemptions_delta", "num_preemptions_delta", "preemption_delta"),
        "delta",
    )
    if preemptions is None:
        preemptions = _find_metric(
            summary,
            all_terms=("preempt",),
            statistic="delta",
        )

    retractions = _first_direct_number(
        summary,
        (
            "retractions_delta",
            "retracted_requests_delta",
            "num_retracted_requests_delta",
        ),
        "delta",
    )
    if retractions is None:
        retractions = _find_metric(
            summary,
            all_terms=("retract",),
            statistic="delta",
        )

    max_retracted = _find_metric(
        summary,
        all_terms=("retract", "req"),
        statistic="max",
        excluded_terms=("total",),
    )
    prefill_mean = _find_metric(
        summary,
        all_terms=("prefill", "time"),
        statistic="mean",
    )
    decode_mean = _find_metric(
        summary,
        all_terms=("decode", "time"),
        statistic="mean",
    )
    server_ttft_mean = _find_metric(
        summary,
        all_terms=("time", "first", "token"),
        statistic="mean",
    )
    kv_usage = _find_metric(
        summary,
        all_terms=("kv", "cache", "usage"),
        statistic="max",
    )
    if kv_usage is None:
        kv_usage = _find_metric(
            summary,
            all_terms=("token", "usage"),
            statistic="max",
        )
    cuda_graph_passes = _find_metric(
        summary,
        all_terms=("cuda", "graph", "passes"),
        statistic="delta",
    )
    gpu_utilization_max = _first_direct_number(
        summary,
        ("gpu_utilization_percent_max",),
        "max",
    )
    gpu_utilization_mean = _first_direct_number(
        summary,
        ("gpu_utilization_percent_mean",),
        "mean",
    )

    return {
        "max_waiting": max_waiting,
        "max_running": max_running,
        "queue_time_mean_seconds": queue_mean,
        "queue_time_delta_count": queue_count,
        "preemptions_delta": preemptions,
        "retractions_delta": retractions,
        "max_retracted": max_retracted,
        "prefill_time_mean_seconds": prefill_mean,
        "decode_time_mean_seconds": decode_mean,
        "server_ttft_mean_seconds": server_ttft_mean,
        "kv_usage_max": kv_usage,
        "cuda_graph_passes_delta": cuda_graph_passes,
        "gpu_utilization_percent_max": gpu_utilization_max,
        "gpu_utilization_percent_mean": gpu_utilization_mean,
    }


def _sequence(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key, [])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _expected_value(
    entry: Mapping[str, Any], manifest: Mapping[str, Any], names: Sequence[str]
) -> int | None:
    containers: list[Mapping[str, Any]] = [entry]
    for key in ("expected", "workload", "benchmark", "config", "protocol"):
        value = entry.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    containers.append(manifest)
    for key in ("expected", "workload", "benchmark", "config", "protocol"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for name in names:
            value = _integer(container.get(name))
            if value is not None:
                return value
    return None


def summarize_raw_result(
    raw: Mapping[str, Any],
    *,
    expected_requests: int | None = None,
    expected_input_len: int | None = None,
    expected_output_len: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Recompute latency statistics and validate one detailed result document.

    Latencies in the detailed client output are seconds.  Returned latency
    samples and distributions are milliseconds.
    """

    errors: list[str] = []
    warnings: list[str] = []
    ttft_seconds = _finite_numbers(_sequence(raw, "ttfts"))
    input_lens_raw = _sequence(raw, "input_lens")
    output_lens_raw = _sequence(raw, "output_lens")
    input_lens = [_integer(value) for value in input_lens_raw]
    output_lens = [_integer(value) for value in output_lens_raw]
    itls_raw = _sequence(raw, "itls")

    completed = _integer(raw.get("completed"))
    failed = _integer(raw.get("failed"))
    if completed is None:
        errors.append("raw.completed 缺失或不是整数")
    elif completed != len(ttft_seconds):
        errors.append(
            f"completed={completed}，但详细 TTFT 样本数={len(ttft_seconds)}"
        )
    if failed is None:
        warnings.append("raw.failed 缺失，无法证明失败请求数为 0")
    elif failed != 0:
        errors.append(f"failed={failed}，存在失败请求")
    if expected_requests is not None and completed != expected_requests:
        errors.append(f"completed={completed}，预期请求数={expected_requests}")

    sample_count = len(ttft_seconds)
    for key, values in (
        ("input_lens", input_lens),
        ("output_lens", output_lens),
        ("itls", itls_raw),
    ):
        if len(values) != sample_count:
            errors.append(f"{key} 样本数={len(values)}，TTFT 样本数={sample_count}")
    if any(value is None for value in input_lens):
        errors.append("input_lens 包含非整数")
    if any(value is None for value in output_lens):
        errors.append("output_lens 包含非整数")
    if expected_input_len is not None and any(
        value != expected_input_len for value in input_lens
    ):
        errors.append(f"存在 input_len 不等于预期值 {expected_input_len} 的请求")
    if expected_output_len is not None and any(
        value != expected_output_len for value in output_lens
    ):
        errors.append(f"存在 output_len 不等于预期值 {expected_output_len} 的请求")

    e2e_ms: list[float] = []
    tpot_ms: list[float] = []
    flattened_itl_ms: list[float] = []
    usable = min(sample_count, len(itls_raw), len(output_lens))
    for index in range(usable):
        row = itls_raw[index]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            errors.append(f"itls[{index}] 不是序列")
            continue
        row_seconds = _finite_numbers(row)
        if len(row_seconds) != len(row):
            errors.append(f"itls[{index}] 包含非有限数值")
            continue
        flattened_itl_ms.extend(value * 1000.0 for value in row_seconds)
        e2e_seconds = ttft_seconds[index] + sum(row_seconds)
        e2e_ms.append(e2e_seconds * 1000.0)
        output_len = output_lens[index]
        if output_len is None or output_len <= 1:
            warnings.append(f"output_lens[{index}] <= 1，无法计算该请求 TPOT")
            continue
        # The benchmark's first output token is represented by TTFT; all
        # remaining output-token intervals form TPOT.
        tpot_ms.append((e2e_seconds - ttft_seconds[index]) * 1000.0 / (output_len - 1))

    ttft_ms = [value * 1000.0 for value in ttft_seconds]
    duration = _number(raw.get("duration"))
    input_tokens = _integer(raw.get("total_input_tokens"))
    output_tokens = _integer(raw.get("total_output_tokens"))
    if duration is None or duration <= 0:
        errors.append("duration 缺失、非有限或不大于 0")
    if input_tokens is None or input_tokens < 0:
        errors.append("total_input_tokens 缺失或无效")
    if output_tokens is None or output_tokens < 0:
        errors.append("total_output_tokens 缺失或无效")

    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    throughput = {
        "input_tokens_per_second": (
            input_tokens / duration
            if input_tokens is not None and duration is not None and duration > 0
            else None
        ),
        "output_tokens_per_second": (
            output_tokens / duration
            if output_tokens is not None and duration is not None and duration > 0
            else None
        ),
        "total_tokens_per_second": (
            total_tokens / duration
            if total_tokens is not None and duration is not None and duration > 0
            else None
        ),
    }
    summary = {
        "completed": completed,
        "failed": failed,
        "duration_seconds": duration,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "stats": {
            "ttft_ms": distribution(ttft_ms),
            "tpot_ms": distribution(tpot_ms),
            "itl_ms": distribution(flattened_itl_ms),
            "e2e_ms": distribution(e2e_ms),
        },
        "throughput": throughput,
        "validation": {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        },
    }
    samples = {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "itl_ms": flattened_itl_ms,
        "e2e_ms": e2e_ms,
    }
    return summary, samples


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: Any = manifest.get("runs")
    if candidates is None:
        candidates = manifest.get("run_entries")
    if candidates is None and isinstance(manifest.get("benchmark"), Mapping):
        candidates = manifest["benchmark"].get("runs")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("manifest must contain a runs list")
    result = [value for value in candidates if isinstance(value, Mapping)]
    if len(result) != len(candidates):
        raise ValueError("every manifest run entry must be an object")
    return result


def _engine_order(engine: str) -> tuple[int, str]:
    preferred = {"vllm": 0, "sglang": 1}
    return preferred.get(engine.lower(), 2), engine.lower()


def _evidence_id(prefix: str, engine: str, concurrency: int, repeat: int) -> str:
    safe_engine = "".join(character if character.isalnum() else "-" for character in engine)
    return f"{prefix}-{safe_engine.upper()}-C{concurrency}-R{repeat}"


def _aggregate_metrics(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [run.get("metrics", {}) for run in runs]

    def values(key: str) -> list[float]:
        return _finite_numbers(
            item.get(key) for item in metrics if isinstance(item, Mapping)
        )

    waiting = values("max_waiting")
    running = values("max_running")
    preemptions = values("preemptions_delta")
    retractions = values("retractions_delta")
    max_retracted = values("max_retracted")
    kv_usage = values("kv_usage_max")
    cuda_graph_passes = values("cuda_graph_passes_delta")
    gpu_utilization_max = values("gpu_utilization_percent_max")
    gpu_utilization_mean = values("gpu_utilization_percent_mean")
    prefill_means = values("prefill_time_mean_seconds")
    decode_means = values("decode_time_mean_seconds")
    server_ttft_means = values("server_ttft_mean_seconds")
    queue_means: list[float] = []
    queue_weights: list[float | None] = []
    for item in metrics:
        if not isinstance(item, Mapping):
            continue
        mean_value = _number(item.get("queue_time_mean_seconds"))
        if mean_value is not None:
            queue_means.append(mean_value)
            queue_weights.append(_number(item.get("queue_time_delta_count")))
    if queue_means and all(weight is not None and weight > 0 for weight in queue_weights):
        total_weight = sum(weight for weight in queue_weights if weight is not None)
        queue_mean = sum(
            value * weight
            for value, weight in zip(queue_means, queue_weights, strict=True)
            if weight is not None
        ) / total_weight
        queue_count = total_weight
    else:
        queue_mean = fmean(queue_means) if queue_means else None
        valid_weights = [weight for weight in queue_weights if weight is not None]
        queue_count = sum(valid_weights) if valid_weights else None
    return {
        "runs_with_metrics": sum(bool(run.get("metrics_path")) for run in runs),
        "max_waiting": max(waiting) if waiting else None,
        "max_running": max(running) if running else None,
        "queue_time_mean_seconds": queue_mean,
        "queue_time_delta_count": queue_count,
        "preemptions_delta": sum(preemptions) if preemptions else None,
        "retractions_delta": sum(retractions) if retractions else None,
        "max_retracted": max(max_retracted) if max_retracted else None,
        "kv_usage_max": max(kv_usage) if kv_usage else None,
        "cuda_graph_passes_delta": (
            sum(cuda_graph_passes) if cuda_graph_passes else None
        ),
        "gpu_utilization_percent_max": (
            max(gpu_utilization_max) if gpu_utilization_max else None
        ),
        "gpu_utilization_percent_mean": (
            fmean(gpu_utilization_mean) if gpu_utilization_mean else None
        ),
        "prefill_time_mean_seconds": (
            fmean(prefill_means) if prefill_means else None
        ),
        "decode_time_mean_seconds": fmean(decode_means) if decode_means else None,
        "server_ttft_mean_seconds": (
            fmean(server_ttft_means) if server_ttft_means else None
        ),
    }


def _aggregate_group(
    engine: str,
    concurrency: int,
    runs: Sequence[Mapping[str, Any]],
    sample_sets: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, Any]:
    pooled: dict[str, list[float]] = {
        "ttft_ms": [],
        "tpot_ms": [],
        "itl_ms": [],
        "e2e_ms": [],
    }
    for run in runs:
        run_samples = sample_sets[str(run["run_id"])]
        for name in pooled:
            pooled[name].extend(run_samples[name])

    durations = _finite_numbers(run.get("duration_seconds") for run in runs)
    total_input = sum(
        value
        for value in (_integer(run.get("total_input_tokens")) for run in runs)
        if value is not None
    )
    total_output = sum(
        value
        for value in (_integer(run.get("total_output_tokens")) for run in runs)
        if value is not None
    )
    total_duration = sum(durations)
    repeat_p95 = _finite_numbers(
        run.get("stats", {}).get("ttft_ms", {}).get("p95") for run in runs
    )
    repeat_summary = {
        "values": repeat_p95,
        "median": median(repeat_p95) if repeat_p95 else None,
        "min": min(repeat_p95) if repeat_p95 else None,
        "max": max(repeat_p95) if repeat_p95 else None,
    }
    errors = [
        f"{run['run_id']}: {message}"
        for run in runs
        for message in run.get("validation", {}).get("errors", [])
    ]
    warnings = [
        f"{run['run_id']}: {message}"
        for run in runs
        for message in run.get("validation", {}).get("warnings", [])
    ]
    total_tokens = total_input + total_output
    return {
        "group_id": f"{engine}-c{concurrency}",
        "engine": engine,
        "concurrency": concurrency,
        "repetitions": len(runs),
        "request_samples": len(pooled["ttft_ms"]),
        "stats": {name: distribution(samples) for name, samples in pooled.items()},
        "repeat_p95_ttft_ms": repeat_summary,
        "totals": {
            "duration_seconds": total_duration,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "tokens": total_tokens,
        },
        "throughput": {
            "input_tokens_per_second": total_input / total_duration
            if total_duration > 0
            else None,
            "output_tokens_per_second": total_output / total_duration
            if total_duration > 0
            else None,
            "total_tokens_per_second": total_tokens / total_duration
            if total_duration > 0
            else None,
        },
        "metrics": _aggregate_metrics(runs),
        "evidence_ids": [
            evidence_id
            for run in runs
            for evidence_id in run.get("evidence_ids", [])
        ],
        "validation": {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        },
    }


def analyze_run(
    run_dir: str | Path,
    manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Analyze all manifest entries below ``run_dir``.

    ``manifest`` may be an already loaded object or a path.  If omitted,
    ``run_dir/manifest.json`` is used.  No wall-clock timestamp is inserted,
    making identical inputs produce byte-for-byte identical summaries.
    """

    root = Path(run_dir).resolve()
    manifest_path: Path | None = None
    if manifest is None:
        manifest_path = root / "manifest.json"
        manifest_data = _load_json(manifest_path)
    elif isinstance(manifest, Mapping):
        manifest_data = manifest
        candidate = root / "manifest.json"
        if candidate.exists():
            manifest_path = candidate
        candidate_manifest = root / "manifest.json"
        if candidate_manifest.is_file():
            manifest_path = candidate_manifest
    else:
        manifest_path = _resolve_path(root, manifest)
        manifest_data = _load_json(manifest_path)

    run_summaries: list[dict[str, Any]] = []
    sample_sets: dict[str, dict[str, list[float]]] = {}
    evidence: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for entry in _entries(manifest_data):
        engine = str(entry.get("engine", "")).strip()
        concurrency = _integer(entry.get("concurrency"))
        repeat = _integer(entry.get("repeat"))
        raw_value = entry.get("raw_path", entry.get("result_path"))
        if not engine or concurrency is None or repeat is None or raw_value is None:
            raise ValueError(
                "each run entry requires engine, concurrency, repeat and raw_path"
            )
        run_id = str(entry.get("run_id") or f"{engine}-c{concurrency}-r{repeat}")
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate run_id: {run_id}")
        seen_run_ids.add(run_id)

        raw_path = _resolve_path(root, str(raw_value))
        raw = _load_json(raw_path)
        run_summary, samples = summarize_raw_result(
            raw,
            expected_requests=_expected_value(
                entry,
                manifest_data,
                ("num_prompts", "request_count", "expected_requests"),
            ),
            expected_input_len=_expected_value(
                entry,
                manifest_data,
                ("input_len", "input_tokens", "random_input_len", "prompt_tokens"),
            ),
            expected_output_len=_expected_value(
                entry,
                manifest_data,
                ("output_len", "output_tokens", "random_output_len", "max_tokens"),
            ),
        )
        raw_evidence_id = _evidence_id("RAW", engine, concurrency, repeat)
        raw_reference = str(raw_value)
        evidence_ids = [raw_evidence_id]
        evidence.append(
            {
                "id": raw_evidence_id,
                "kind": "client_raw",
                "path": raw_reference,
                "sha256": sha256_file(raw_path),
                "engine": engine,
                "concurrency": concurrency,
                "repeat": repeat,
            }
        )

        metrics: dict[str, float | None] = normalize_metrics(None)
        metrics_reference: str | None = None
        metrics_value = entry.get("metrics_path")
        if metrics_value:
            metrics_path = _resolve_path(root, str(metrics_value))
            metrics_payload = _load_json(metrics_path)
            metrics = normalize_metrics(metrics_payload)
            metrics_reference = str(metrics_value)
            metrics_evidence_id = _evidence_id("METRICS", engine, concurrency, repeat)
            evidence_ids.append(metrics_evidence_id)
            evidence.append(
                {
                    "id": metrics_evidence_id,
                    "kind": "server_metrics",
                    "path": metrics_reference,
                    "sha256": sha256_file(metrics_path),
                    "engine": engine,
                    "concurrency": concurrency,
                    "repeat": repeat,
                }
            )

        run_record = {
            "run_id": run_id,
            "engine": engine,
            "concurrency": concurrency,
            "repeat": repeat,
            "raw_path": raw_reference,
            "metrics_path": metrics_reference,
            "evidence_ids": evidence_ids,
            **run_summary,
            "metrics": metrics,
        }
        run_summaries.append(run_record)
        sample_sets[run_id] = samples

    run_summaries.sort(
        key=lambda item: (
            _engine_order(str(item["engine"])),
            int(item["concurrency"]),
            int(item["repeat"]),
        )
    )
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for run in run_summaries:
        grouped.setdefault((str(run["engine"]), int(run["concurrency"])), []).append(run)
    groups = [
        _aggregate_group(engine, concurrency, runs, sample_sets)
        for (engine, concurrency), runs in sorted(
            grouped.items(), key=lambda item: (_engine_order(item[0][0]), item[0][1])
        )
    ]

    all_errors = [
        f"{run['run_id']}: {message}"
        for run in run_summaries
        for message in run["validation"]["errors"]
    ]
    all_warnings = [
        f"{run['run_id']}: {message}"
        for run in run_summaries
        for message in run["validation"]["warnings"]
    ]
    manifest_evidence: dict[str, Any] | None = None
    if manifest_path is not None:
        try:
            manifest_reference = str(manifest_path.relative_to(root))
        except ValueError:
            manifest_reference = str(manifest_path)
        manifest_evidence = {
            "id": "MANIFEST",
            "kind": "manifest",
            "path": manifest_reference,
            "sha256": sha256_file(manifest_path),
        }
        evidence.insert(0, manifest_evidence)

    workload = manifest_data.get("workload")
    if isinstance(workload, Mapping) and workload.get("path"):
        workload_path = _resolve_path(root, str(workload["path"]))
        if workload_path.exists():
            try:
                workload_reference = str(workload_path.relative_to(root))
            except ValueError:
                workload_reference = str(workload_path)
            evidence.insert(
                1 if manifest_evidence is not None else 0,
                {
                    "id": "WORKLOAD",
                    "kind": "exact_token_workload",
                    "path": workload_reference,
                    "sha256": sha256_file(workload_path),
                },
            )

    servers = manifest_data.get("servers")
    if isinstance(servers, Mapping):
        for engine, server in sorted(servers.items(), key=lambda item: str(item[0])):
            if not isinstance(server, Mapping) or not server.get("log_path"):
                continue
            server_log_path = _resolve_path(root, str(server["log_path"]))
            if not server_log_path.exists():
                continue
            evidence.append(
                {
                    "id": f"SERVER-{str(engine).upper()}",
                    "kind": "server_log",
                    "path": str(server["log_path"]),
                    "sha256": sha256_file(server_log_path),
                    "engine": str(engine),
                }
            )

    context = {
        key: manifest_data[key]
        for key in (
            "run_id",
            "experiment_id",
            "created_at",
            "benchmark",
            "config",
            "protocol",
            "environment",
            "environment_path",
            "model",
            "hardware",
        )
        if key in manifest_data
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "context": context,
        "runs": run_summaries,
        "groups": groups,
        "evidence": evidence,
        "validation": {
            "valid": not all_errors,
            "errors": all_errors,
            "warnings": all_warnings,
        },
    }


def write_summary(summary: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write deterministic, human-diffable JSON and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path
