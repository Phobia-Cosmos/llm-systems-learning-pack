#!/home/undefined/Disk/python-envs/sglang/bin/python
"""Boundary benchmark for MiniLLM's native teaching KV-cache implementation.

The input prompts are deterministic random token IDs, so tokenizer and disk I/O
are outside the measured region.  The benchmark never stops on EOS: every case
executes the requested amount of work.  For D requested output tokens, prefill
selects output token 1 and the model performs max(D - 1, 0) incremental decode
forwards.  Both numbers are written to every JSONL record.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MINILLM_ROOT = SCRIPT_DIR.parent / "minillm"
DEFAULT_CHECKPOINT = MINILLM_ROOT / "artifacts/checkpoints/minillm-rope.pt"
DEFAULT_PROFILE_DIR = Path(
    os.environ.get(
        "LLM_BENCH_PROFILE_DIR",
        "/home/undefined/Disk/build-tmp/llm-boundary/native-minillm",
    )
)

if str(MINILLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MINILLM_ROOT))

from minillm import GPTConfig, MiniGPT  # noqa: E402


DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    num_requests: int
    prompt_len: int
    decode_len: int
    expected_error: bool = False

    @property
    def decode_model_passes(self) -> int:
        return max(self.decode_len - 1, 0)

    @property
    def required_cache_len(self) -> int:
        return self.prompt_len + self.decode_model_passes


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"median": None, "p95": None, "min": None, "max": None}
    return {
        "median": statistics.median(materialized),
        "p95": _percentile(materialized, 0.95),
        "min": min(materialized),
        "max": max(materialized),
    }


def _safe_rate(tokens: int, milliseconds: float | None) -> float | None:
    if tokens <= 0 or milliseconds is None or milliseconds <= 0:
        return None
    return tokens * 1000.0 / milliseconds


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("native_minillm_bench supports CPU and CUDA devices")
    return device


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return DTYPES[requested]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _unique_parameter_bytes(model: torch.nn.Module) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.numel() == 0:
            continue
        key = (tensor.device.type, tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        if key in seen:
            continue
        seen.add(key)
        total += tensor.untyped_storage().nbytes()
    return total


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MiniGPT, dict[str, Any], float]:
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("checkpoint must be a dict containing 'config' and 'model'")
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, checkpoint, time.perf_counter() - started


def _validate_case(case: BenchCase, block_size: int) -> None:
    if case.num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if case.prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    if case.decode_len < 0:
        raise ValueError("decode_len must be non-negative")
    if case.prompt_len > block_size:
        raise ValueError(
            f"prompt_len={case.prompt_len} exceeds model block_size={block_size}"
        )
    if case.required_cache_len > block_size:
        raise ValueError(
            "teaching KV-cache boundary exceeded: "
            f"prompt_len + max(decode_len - 1, 0) = {case.required_cache_len} "
            f"> block_size={block_size}"
        )


def _phase_context(enabled: bool, name: str):
    if not enabled:
        return nullcontext()
    return torch.profiler.record_function(name)


@torch.inference_mode()
def _run_iteration(
    model: MiniGPT,
    input_ids: torch.Tensor,
    decode_len: int,
    device: torch.device,
    *,
    cache_mode: str,
    annotate: bool = False,
) -> dict[str, Any]:
    """Run one exact-length generation and return synchronized phase timings."""

    static_cache = None
    if cache_mode == "static":
        static_cache = model.allocate_static_kv_cache(
            batch_size=input_ids.size(0),
            max_len=input_ids.size(1) + max(decode_len - 1, 0),
            device=device,
        )
    elif cache_mode != "legacy":
        raise ValueError(f"unknown cache mode: {cache_mode}")

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(device)
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        prefill_start.record()
    prefill_wall_started = time.perf_counter()
    with _phase_context(annotate, "minillm_prefill"):
        if static_cache is None:
            logits, cache_state = model.forward_with_cache(input_ids)
        else:
            logits, cache_state = model.forward_with_static_cache(input_ids, static_cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True) if decode_len > 0 else None
    if use_cuda:
        prefill_end.record()
        torch.cuda.synchronize(device)
    prefill_wall_ms = (time.perf_counter() - prefill_wall_started) * 1000.0
    prefill_gpu_ms = prefill_start.elapsed_time(prefill_end) if use_cuda else None

    decode_passes = max(decode_len - 1, 0)
    decode_step_gpu_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    decode_step_wall_ms: list[float] = []
    decode_gpu_ms: float | None = 0.0 if use_cuda else None
    decode_wall_ms = 0.0

    if decode_passes:
        if next_token is None:
            raise AssertionError("decode requires a token selected during prefill")
        if use_cuda:
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            decode_start.record()
        decode_wall_started = time.perf_counter()
        for _ in range(decode_passes):
            if use_cuda:
                step_start = torch.cuda.Event(enable_timing=True)
                step_end = torch.cuda.Event(enable_timing=True)
                step_start.record()
                step_wall_started = 0.0
            else:
                step_wall_started = time.perf_counter()
            with _phase_context(annotate, "minillm_decode_step"):
                if static_cache is None:
                    logits, cache_state = model.forward_with_cache(next_token, cache_state)
                else:
                    logits, cache_state = model.forward_with_static_cache(next_token, cache_state)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if use_cuda:
                step_end.record()
                decode_step_gpu_events.append((step_start, step_end))
            else:
                decode_step_wall_ms.append((time.perf_counter() - step_wall_started) * 1000.0)
        if use_cuda:
            decode_end.record()
            torch.cuda.synchronize(device)
            decode_gpu_ms = decode_start.elapsed_time(decode_end)
            decode_step_gpu_ms = [start.elapsed_time(end) for start, end in decode_step_gpu_events]
        else:
            decode_step_gpu_ms = []
        decode_wall_ms = (time.perf_counter() - decode_wall_started) * 1000.0
    else:
        decode_step_gpu_ms = []

    final_token_checksum = int(next_token.sum().item()) if next_token is not None else None
    generation_wall_ms = prefill_wall_ms + decode_wall_ms
    generation_gpu_ms = (
        float(prefill_gpu_ms) + float(decode_gpu_ms)
        if prefill_gpu_ms is not None and decode_gpu_ms is not None
        else None
    )
    return {
        "prefill_wall_ms": prefill_wall_ms,
        "prefill_gpu_ms": prefill_gpu_ms,
        "decode_wall_ms": decode_wall_ms,
        "decode_gpu_ms": decode_gpu_ms,
        "generation_wall_ms": generation_wall_ms,
        "generation_gpu_ms": generation_gpu_ms,
        "decode_step_gpu_ms": decode_step_gpu_ms,
        "decode_step_wall_ms": decode_step_wall_ms,
        "final_token_checksum": final_token_checksum,
        "cache_mode": cache_mode,
    }


def _aggregate_repeats(
    repeats: list[dict[str, Any]],
    case: BenchCase,
) -> dict[str, Any]:
    scalar_names = (
        "prefill_wall_ms",
        "prefill_gpu_ms",
        "decode_wall_ms",
        "decode_gpu_ms",
        "generation_wall_ms",
        "generation_gpu_ms",
    )
    timing: dict[str, Any] = {}
    for name in scalar_names:
        timing[name] = _summary(
            repeat[name] for repeat in repeats if repeat.get(name) is not None
        )

    all_step_gpu_ms = [
        value for repeat in repeats for value in repeat.get("decode_step_gpu_ms", [])
    ]
    all_step_wall_ms = [
        value for repeat in repeats for value in repeat.get("decode_step_wall_ms", [])
    ]
    timing["decode_step_gpu_ms"] = _summary(all_step_gpu_ms)
    timing["decode_step_wall_ms"] = _summary(all_step_wall_ms)

    input_tokens = case.num_requests * case.prompt_len
    decode_forward_tokens = case.num_requests * case.decode_model_passes
    output_tokens = case.num_requests * case.decode_len
    rates = {
        "prefill_input_tokens_per_second_wall": _summary(
            value
            for repeat in repeats
            if (value := _safe_rate(input_tokens, repeat["prefill_wall_ms"])) is not None
        ),
        "decode_tokens_per_second_wall": _summary(
            value
            for repeat in repeats
            if (value := _safe_rate(decode_forward_tokens, repeat["decode_wall_ms"])) is not None
        ),
        "output_tokens_per_second_wall": _summary(
            value
            for repeat in repeats
            if (value := _safe_rate(output_tokens, repeat["generation_wall_ms"])) is not None
        ),
        "prefill_input_tokens_per_second_gpu": _summary(
            value
            for repeat in repeats
            if (value := _safe_rate(input_tokens, repeat["prefill_gpu_ms"])) is not None
        ),
        "decode_tokens_per_second_gpu": _summary(
            value
            for repeat in repeats
            if (value := _safe_rate(decode_forward_tokens, repeat["decode_gpu_ms"])) is not None
        ),
    }

    memory_names = (
        "gpu_allocated_before_bytes",
        "gpu_allocated_after_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
        "gpu_incremental_peak_allocated_bytes",
    )
    memory = {
        name: _summary(
            repeat[name] for repeat in repeats if repeat.get(name) is not None
        )
        for name in memory_names
    }
    return {"timing": timing, "throughput": rates, "memory": memory}


def _benchmark_case(
    model: MiniGPT,
    case: BenchCase,
    device: torch.device,
    *,
    warmup: int,
    repeat: int,
    seed: int,
    cache_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_case(case, model.config.block_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(case.num_requests, case.prompt_len),
        dtype=torch.long,
        generator=generator,
    ).to(device)

    for _ in range(warmup):
        _run_iteration(
            model,
            input_ids,
            case.decode_len,
            device,
            cache_mode=cache_mode,
        )

    measurements: list[dict[str, Any]] = []
    for repeat_index in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            allocated_before = None

        measurement = _run_iteration(
            model,
            input_ids,
            case.decode_len,
            device,
            cache_mode=cache_mode,
        )
        measurement["repeat_index"] = repeat_index
        if device.type == "cuda":
            allocated_after = torch.cuda.memory_allocated(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            measurement.update(
                {
                    "gpu_allocated_before_bytes": allocated_before,
                    "gpu_allocated_after_bytes": allocated_after,
                    "gpu_peak_allocated_bytes": peak_allocated,
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "gpu_incremental_peak_allocated_bytes": max(
                        peak_allocated - int(allocated_before), 0
                    ),
                }
            )
        else:
            measurement.update(
                {
                    "gpu_allocated_before_bytes": None,
                    "gpu_allocated_after_bytes": None,
                    "gpu_peak_allocated_bytes": None,
                    "gpu_peak_reserved_bytes": None,
                    "gpu_incremental_peak_allocated_bytes": None,
                }
            )
        measurements.append(measurement)
    return measurements, _aggregate_repeats(measurements, case)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "case"


def _profile_case(
    model: MiniGPT,
    case: BenchCase,
    device: torch.device,
    *,
    seed: int,
    output_dir: Path,
    row_limit: int,
    with_stack: bool,
    cache_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (case.num_requests, case.prompt_len),
        dtype=torch.long,
        generator=generator,
    ).to(device)

    # Profile a dedicated post-warmup pass so profiler overhead never pollutes
    # the normal benchmark repeats stored in the same record.
    _run_iteration(
        model,
        input_ids,
        case.decode_len,
        device,
        cache_mode=cache_mode,
    )
    # The Kineto profiler in the registered Torch 2.9.1+cu130 environment
    # advertises CUDA activity but currently emits CPU-only events. The
    # autograd profiler still records CUDA event timings correctly here.
    with torch.autograd.profiler.profile(
        use_device="cuda" if device.type == "cuda" else None,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    ) as profiler:
        _run_iteration(
            model,
            input_ids,
            case.decode_len,
            device,
            cache_mode=cache_mode,
            annotate=True,
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{_safe_filename(case.case_id)}"
    trace_path = (output_dir / f"{stem}.trace.json").resolve()
    table_path = (output_dir / f"{stem}.top.txt").resolve()
    profiler.export_chrome_trace(str(trace_path))
    sort_by = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = profiler.key_averages(group_by_input_shape=True).table(
        sort_by=sort_by,
        row_limit=row_limit,
    )
    table_path.write_text(table + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "trace_path": str(trace_path),
        "top_table_path": str(table_path),
        "sort_by": sort_by,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": with_stack,
        "note": "Dedicated profiled pass; profiler overhead is excluded from benchmark timings.",
    }


def _request_cases() -> list[BenchCase]:
    return [
        BenchCase(f"request_n{n}_p32_d32", n, 32, 32)
        for n in (1, 8, 32, 128, 256)
    ]


def _prefill_cases() -> list[BenchCase]:
    return [
        BenchCase(f"prefill_n1_p{prompt_len}_d8", 1, prompt_len, 8)
        for prompt_len in (8, 32, 64, 96, 120)
    ]


def _decode_cases() -> list[BenchCase]:
    return [
        BenchCase(f"decode_n1_p16_d{decode_len}", 1, 16, decode_len)
        for decode_len in (1, 8, 32, 64, 96)
    ]


def _boundary_cases(block_size: int) -> list[BenchCase]:
    prompt_len = min(32, block_size)
    cases: list[BenchCase] = []
    for required_cache_len in (block_size - 1, block_size, block_size + 1):
        decode_len = required_cache_len - prompt_len + 1
        cases.append(
            BenchCase(
                f"boundary_cache{required_cache_len}_p{prompt_len}_d{decode_len}",
                1,
                prompt_len,
                decode_len,
                expected_error=required_cache_len > block_size,
            )
        )
    return cases


def _build_cases(args: argparse.Namespace, block_size: int) -> list[BenchCase]:
    if args.matrix is None:
        return [
            BenchCase(
                f"single_n{args.num_requests}_p{args.prompt_len}_d{args.decode_len}",
                args.num_requests,
                args.prompt_len,
                args.decode_len,
            )
        ]
    if args.matrix == "smoke":
        return [BenchCase("smoke_n1_p8_d4", 1, 8, 4)]
    if args.matrix == "request":
        return _request_cases()
    if args.matrix == "prefill":
        return _prefill_cases()
    if args.matrix == "decode":
        return _decode_cases()
    if args.matrix == "boundary":
        return _boundary_cases(block_size)
    if args.matrix == "all":
        return (
            _request_cases()
            + _prefill_cases()
            + _decode_cases()
            + _boundary_cases(block_size)
        )
    raise AssertionError(f"unhandled matrix {args.matrix!r}")


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_free_memory_before_benchmark_bytes": free_bytes,
                "gpu_total_memory_visible_bytes": total_bytes,
            }
        )
    return result


def _base_record(
    case: BenchCase,
    args: argparse.Namespace,
    checkpoint_path: Path,
    model: MiniGPT,
    device: torch.device,
    dtype: torch.dtype,
    environment: dict[str, Any],
    model_load_wall_s: float,
    model_resident_bytes: int | None,
    case_seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "native_minillm_kv_cache",
        "timestamp_utc": _utc_now(),
        "case_id": case.case_id,
        "status": "pending",
        "expected_error": case.expected_error,
        "expectation_met": None,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "dtype": _dtype_name(dtype),
        "seed": case_seed,
        "ignore_eos": args.ignore_eos,
        "prompt_source": "deterministic_uniform_random_token_ids",
        "next_token_policy": "greedy_argmax",
        "cache_mode": args.cache_mode,
        "num_requests": case.num_requests,
        "prompt_len": case.prompt_len,
        "decode_len": case.decode_len,
        "decode_model_passes": case.decode_model_passes,
        "required_cache_len": case.required_cache_len,
        "input_tokens_per_repeat": case.num_requests * case.prompt_len,
        "output_tokens_per_repeat": case.num_requests * case.decode_len,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "semantics": (
            "Prefill consumes all P prompt tokens and selects output token 1. "
            "Generating D tokens therefore performs max(D-1,0) one-token "
            f"{args.cache_mode} KV-cache decode calls. EOS is never inspected."
        ),
        "model": {
            "config": asdict(model.config),
            "parameter_count": model.parameter_count(),
            "unique_parameter_and_buffer_bytes_at_runtime_dtype": _unique_parameter_bytes(model),
            "cuda_resident_bytes_after_load": model_resident_bytes,
            "load_wall_seconds": model_load_wall_s,
        },
        "environment": environment,
    }


def _emit(record: dict[str, Any], output_handle) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    if output_handle is not None:
        output_handle.write(line + "\n")
        output_handle.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark native MiniLLM prefill and incremental KV-cache decode. "
            "No matrix is run unless --matrix is explicitly supplied."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("-N", "--num-requests", type=int, default=1)
    parser.add_argument("-P", "--prompt-len", type=int, default=32)
    parser.add_argument("-D", "--decode-len", type=int, default=32)
    parser.add_argument(
        "--matrix",
        choices=("smoke", "request", "prefill", "decode", "boundary", "all"),
        help="Run an opt-in built-in matrix instead of the single N/P/D case.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--cache-mode",
        choices=("static", "legacy"),
        default="static",
        help="Use the new fixed-address cache or the legacy per-step torch.cat path.",
    )
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep exact D-token workloads (default and required for comparable boundary tests).",
    )
    parser.add_argument("--output", type=Path, help="Optional JSONL file; stdout always receives JSONL too.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --output instead of appending.")
    parser.add_argument("--profile", action="store_true", help="Profile one selected case after normal timing.")
    parser.add_argument("--profile-case-index", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--profile-row-limit", type=int, default=40)
    parser.add_argument("--profile-with-stack", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if not args.ignore_eos:
        parser.error("--no-ignore-eos is unsupported: boundary cases require exact D-token work")
    if args.profile_case_index < 0:
        parser.error("--profile-case-index must be non-negative")
    if args.profile_row_limit <= 0:
        parser.error("--profile-row-limit must be positive")
    return args


def main() -> int:
    args = _parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, _checkpoint, model_load_wall_s = _load_model(checkpoint_path, device, dtype)
    model_resident_bytes = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    )
    environment = _environment(device)
    cases = _build_cases(args, model.config.block_size)
    if args.profile and args.profile_case_index >= len(cases):
        raise ValueError(
            f"--profile-case-index={args.profile_case_index} is outside {len(cases)} cases"
        )

    output_handle = None
    unexpected_failures = 0
    unmet_expectations = 0
    try:
        if args.output is not None:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open(
                "w" if args.overwrite else "a",
                encoding="utf-8",
            )

        for case_index, case in enumerate(cases):
            case_seed = args.seed + case_index
            record = _base_record(
                case,
                args,
                checkpoint_path,
                model,
                device,
                dtype,
                environment,
                model_load_wall_s,
                model_resident_bytes,
                case_seed,
            )
            try:
                measurements, summary = _benchmark_case(
                    model,
                    case,
                    device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    seed=case_seed,
                    cache_mode=args.cache_mode,
                )
                record.update(
                    {
                        "status": "ok",
                        "expectation_met": not case.expected_error,
                        "repeats": measurements,
                        "summary": summary,
                    }
                )
                if args.profile and case_index == args.profile_case_index:
                    try:
                        record["profiler"] = _profile_case(
                            model,
                            case,
                            device,
                            seed=case_seed,
                            output_dir=args.profile_dir.expanduser().resolve(),
                            row_limit=args.profile_row_limit,
                            with_stack=args.profile_with_stack,
                            cache_mode=args.cache_mode,
                        )
                    except Exception as error:  # profiling must not erase valid timing data
                        record["profiler"] = {
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(limit=8),
                        }
                        unexpected_failures += 1
                if case.expected_error:
                    unmet_expectations += 1
            except Exception as error:
                record.update(
                    {
                        "status": "error",
                        "expectation_met": case.expected_error,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
                if not case.expected_error:
                    unexpected_failures += 1
            _emit(record, output_handle)
    finally:
        if output_handle is not None:
            output_handle.close()

    return 1 if unexpected_failures or unmet_expectations else 0


if __name__ == "__main__":
    raise SystemExit(main())
