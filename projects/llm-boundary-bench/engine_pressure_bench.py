#!/home/undefined/Disk/python-envs/sglang/bin/python
"""Pressure benchmark for the local nano-vLLM and official Mini-SGLang engines.

The benchmark feeds token IDs directly, so ``P`` is the exact prompt length and
``D`` is the exact number of generated tokens per request.  A single engine is
kept alive while all cases run, which avoids counting model loading and CUDA
graph capture in every case.

Examples:

    ./engine_pressure_bench.py --engine nano --model-kind minillm --preset smoke
    ./engine_pressure_bench.py --engine nano --model-kind qwen --preset request --mode eager
    ./engine_pressure_bench.py --engine minisgl --model-kind qwen --cases '1:128:8;8:128:8'

Nsight Systems capture (capture starts after engine initialization and warmup):

    nsys profile --trace=cuda,nvtx,osrt,cublas \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      -o /home/undefined/Disk/build-tmp/qwen-nano \
      ./engine_pressure_bench.py --engine nano --model-kind qwen \
      --cases '1:4096:1' --cuda-profiler-case 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
import traceback
from array import array
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch


ROOT = Path("/home/undefined/Desktop/ai")
DEFAULT_MODELS = {
    "minillm": ROOT / "projects/minillm/artifacts/hf_exports/minillm-rope",
    "qwen": Path("/home/undefined/Disk/cache/models/huggingface/Qwen3-0.6B"),
}
DEFAULT_OUTPUT_ROOT = Path("/home/undefined/Disk/build-tmp")


@dataclass(frozen=True)
class BenchCase:
    name: str
    requests: int
    prefill_tokens: int
    decode_tokens: int

    @property
    def input_tokens(self) -> int:
        return self.requests * self.prefill_tokens

    @property
    def output_tokens(self) -> int:
        return self.requests * self.decode_tokens

    @property
    def max_sequence_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens


@dataclass
class StepSample:
    phase: str
    logical_tokens: int
    batch_size: int
    padded_batch_size: int
    gpu_ms: float
    submit_or_step_wall_ms: float
    used_cuda_graph: bool | None = None


def _percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    data = list(values)
    return {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "p50": _percentile(data, 0.50),
        "p95": _percentile(data, 0.95),
        "max": max(data) if data else None,
    }


def _rate(count: int, milliseconds: float) -> float | None:
    return count * 1000.0 / milliseconds if count > 0 and milliseconds > 0 else None


def _read_model_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"model config does not exist: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"model config is not a JSON object: {config_path}")
    return payload


def _model_limits(model_path: Path) -> tuple[int, int]:
    config = _read_model_config(model_path)
    vocab_size = int(config["vocab_size"])
    max_position = int(
        config.get("max_position_embeddings", config.get("block_size", 0))
    )
    if vocab_size <= 1 or max_position <= 0:
        raise ValueError(f"invalid vocab/context fields in {model_path / 'config.json'}")
    return vocab_size, max_position


def _make_prompts(
    case: BenchCase,
    *,
    vocab_size: int,
    seed: int,
    salt: int,
) -> list[list[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed((seed + 1_000_003 * salt) % (2**63 - 1))
    prompt_tensor = torch.randint(
        1,
        vocab_size,
        (case.requests, case.prefill_tokens),
        generator=generator,
        dtype=torch.int64,
    )
    # Make the beginning request-specific as well as random.  This prevents an
    # accidental shared prefix from changing a cold-prefill benchmark.
    if case.prefill_tokens > 0:
        prompt_tensor[:, 0] = (
            torch.arange(case.requests, dtype=torch.int64)
            + seed
            + salt * 131
        ) % (vocab_size - 1) + 1
    prompts = prompt_tensor.tolist()
    assert len(prompts) == case.requests
    assert all(len(prompt) == case.prefill_tokens for prompt in prompts)
    return prompts


def _prompt_digest(prompts: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(len(prompt).to_bytes(8, "little"))
        ids = array("I", prompt)
        if sys.byteorder != "little":
            ids.byteswap()
        digest.update(ids.tobytes())
    return digest.hexdigest()


@contextmanager
def _nvtx_range(message: str):
    pushed = False
    try:
        torch.cuda.nvtx.range_push(message)
        pushed = True
    except Exception:
        pass
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


def _cuda_profiler_start() -> None:
    result = torch.cuda.cudart().cudaProfilerStart()
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"cudaProfilerStart failed with CUDA error {result}")


def _cuda_profiler_stop() -> None:
    result = torch.cuda.cudart().cudaProfilerStop()
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"cudaProfilerStop failed with CUDA error {result}")


class NanoAdapter:
    engine_name = "nano"

    def __init__(self, model_path: Path, args: argparse.Namespace):
        from nanovllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            str(model_path),
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_batched_tokens,
            gpu_memory_utilization=args.memory_ratio,
            tensor_parallel_size=1,
            enforce_eager=args.mode == "eager",
        )
        self._last_schedule: dict[str, Any] = {}
        self._original_schedule = self.llm.scheduler.schedule

        def schedule_with_metadata():
            seqs, is_prefill = self._original_schedule()
            self._last_schedule = {
                "phase": "prefill" if is_prefill else "decode",
                "batch_size": len(seqs),
                "tokens": (
                    sum(seq.num_scheduled_tokens for seq in seqs)
                    if is_prefill
                    else len(seqs)
                ),
            }
            return seqs, is_prefill

        self.llm.scheduler.schedule = schedule_with_metadata

    def capacity(self) -> dict[str, Any]:
        config = self.llm.model_runner.config
        return {
            "max_model_len": self.llm.max_model_len,
            "max_num_seqs": config.max_num_seqs,
            "max_batched_tokens": config.max_num_batched_tokens,
            "kv_block_size": config.kvcache_block_size,
            "kv_blocks": config.num_kvcache_blocks,
            "kv_token_slots": config.num_kvcache_blocks * config.kvcache_block_size,
            "cuda_graph_enabled": not config.enforce_eager,
        }

    def run_case(
        self,
        case: BenchCase,
        prompts: list[list[int]],
        *,
        record_steps: bool,
    ) -> dict[str, Any]:
        sampling = self.SamplingParams(
            temperature=0.1,
            max_tokens=case.decode_tokens,
            ignore_eos=True,
        )
        seqs = []
        for prompt in prompts:
            self.llm.add_request(prompt, sampling)
            seqs.append(self.llm.scheduler.waiting[-1])

        step_samples: list[StepSample] = []
        arrival_ms: dict[int, list[float]] = {seq.seq_id: [] for seq in seqs}
        previous_output_len = {seq.seq_id: 0 for seq in seqs}

        torch.cuda.synchronize()
        started = time.perf_counter()
        while not self.llm.is_finished():
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            step_started = time.perf_counter()
            cuda_start.record()
            with _nvtx_range("nano.step"):
                _, signed_num_tokens = self.llm.step()
            cuda_end.record()
            cuda_end.synchronize()
            now = time.perf_counter()

            phase = "prefill" if signed_num_tokens > 0 else "decode"
            logical_tokens = abs(signed_num_tokens)
            metadata = self._last_schedule
            if metadata:
                assert metadata["phase"] == phase
                assert metadata["tokens"] == logical_tokens
            step_samples.append(
                StepSample(
                    phase=phase,
                    logical_tokens=logical_tokens,
                    batch_size=int(metadata.get("batch_size", logical_tokens)),
                    padded_batch_size=int(metadata.get("batch_size", logical_tokens)),
                    gpu_ms=float(cuda_start.elapsed_time(cuda_end)),
                    submit_or_step_wall_ms=(now - step_started) * 1000.0,
                    used_cuda_graph=(phase == "decode" and self.llm.model_runner.enforce_eager is False),
                )
            )
            elapsed_ms = (now - started) * 1000.0
            for seq in seqs:
                current = seq.num_completion_tokens
                previous = previous_output_len[seq.seq_id]
                if current > previous:
                    arrival_ms[seq.seq_id].extend([elapsed_ms] * (current - previous))
                    previous_output_len[seq.seq_id] = current

        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000.0
        outputs = [seq.completion_token_ids for seq in seqs]
        return _summarize_case(
            case,
            wall_ms=wall_ms,
            outputs=outputs,
            steps=step_samples,
            arrival_ms=arrival_ms,
            record_steps=record_steps,
        )

    def close(self) -> None:
        self.llm.scheduler.schedule = self._original_schedule
        self.llm.exit()


class MiniSGLangAdapter:
    engine_name = "minisgl"

    def __init__(self, model_path: Path, args: argparse.Namespace):
        from minisgl.core import SamplingParams
        from minisgl.llm import LLM

        self.SamplingParams = SamplingParams
        graph_max_bs = 0 if args.mode == "eager" else args.cuda_graph_max_bs
        self.llm = LLM(
            str(model_path),
            dtype=torch.bfloat16,
            max_seq_len_override=args.max_model_len,
            max_running_req=args.max_num_seqs,
            max_extend_tokens=args.max_batched_tokens,
            cuda_graph_max_bs=graph_max_bs,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            cache_type=args.cache_type,
            attention_backend=args.attention_backend,
        )
        self._step_samples: list[StepSample] = []
        self._arrival_ms: dict[int, list[float]] = {}
        self._case_started = 0.0

        self._original_forward_batch = self.llm.engine.forward_batch

        def timed_forward_batch(batch, sample_args):
            phase = "prefill" if batch.is_prefill else "decode"
            logical_tokens = (
                sum(req.extend_len for req in batch.reqs)
                if batch.is_prefill
                else batch.size
            )
            graph_runner = self.llm.engine.graph_runner
            used_graph = bool(graph_runner.can_use_cuda_graph(batch))
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            submit_started = time.perf_counter()
            cuda_start.record(self.llm.engine.stream)
            with _nvtx_range(f"minisgl.{phase}"):
                output = self._original_forward_batch(batch, sample_args)
            cuda_end.record(self.llm.engine.stream)
            self._step_samples.append(
                StepSample(
                    phase=phase,
                    logical_tokens=logical_tokens,
                    batch_size=batch.size,
                    padded_batch_size=batch.padded_size,
                    gpu_ms=0.0,
                    submit_or_step_wall_ms=(time.perf_counter() - submit_started) * 1000.0,
                    used_cuda_graph=used_graph,
                )
            )
            # Keep the event pair beside the sample.  elapsed_time is only read
            # after generate() and a device synchronization.
            self._step_samples[-1]._events = (cuda_start, cuda_end)  # type: ignore[attr-defined]
            return output

        self.llm.engine.forward_batch = timed_forward_batch

        self._original_send_result = self.llm.send_result

        def timed_send_result(reply):
            lengths_before = {
                msg.uid: len(self.llm.status_map[msg.uid].output_ids)
                for msg in reply
                if msg.uid in self.llm.status_map
            }
            self._original_send_result(reply)
            # LLM.offline_send_result drops a final token when its ID happens
            # to equal EOS, even when ignore_eos=True.  The scheduler did run
            # that token, so retain it for an exact-D benchmark.
            for msg in reply:
                status = self.llm.status_map.get(msg.uid)
                if (
                    status is not None
                    and msg.finished
                    and msg.next_token == self.llm.eos_token_id
                    and len(status.output_ids) == lengths_before.get(msg.uid, -1)
                ):
                    status.output_ids.append(msg.next_token)
            if self._case_started:
                elapsed_ms = (time.perf_counter() - self._case_started) * 1000.0
                for msg in reply:
                    self._arrival_ms.setdefault(msg.uid, []).append(elapsed_ms)

        self.llm.send_result = timed_send_result

    def capacity(self) -> dict[str, Any]:
        engine = self.llm.engine
        graph_runner = engine.graph_runner
        return {
            "max_model_len": engine.max_seq_len,
            "max_num_seqs": int(engine.page_table.shape[0] - 1),
            "max_batched_tokens": self.llm.prefill_budget,
            "kv_page_size": engine.ctx.page_size,
            "kv_pages": engine.num_pages,
            "kv_token_slots": engine.num_pages * engine.ctx.page_size,
            "cuda_graph_enabled": graph_runner.max_graph_bs > 0,
            "cuda_graph_max_bs": graph_runner.max_graph_bs,
            "cuda_graph_batch_sizes": graph_runner.graph_bs_list,
        }

    def run_case(
        self,
        case: BenchCase,
        prompts: list[list[int]],
        *,
        record_steps: bool,
    ) -> dict[str, Any]:
        sampling = self.SamplingParams(
            temperature=0.0,
            max_tokens=case.decode_tokens,
            ignore_eos=True,
        )
        self._step_samples = []
        self._arrival_ms = {uid: [] for uid in range(case.requests)}
        torch.cuda.synchronize()
        self._case_started = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - self._case_started) * 1000.0
        self._case_started = 0.0

        for sample in self._step_samples:
            start, end = sample._events  # type: ignore[attr-defined]
            sample.gpu_ms = float(start.elapsed_time(end))
            del sample._events  # type: ignore[attr-defined]
        token_outputs = [list(result["token_ids"]) for result in outputs]
        return _summarize_case(
            case,
            wall_ms=wall_ms,
            outputs=token_outputs,
            steps=self._step_samples,
            arrival_ms=self._arrival_ms,
            record_steps=record_steps,
        )

    def close(self) -> None:
        self.llm.engine.forward_batch = self._original_forward_batch
        self.llm.send_result = self._original_send_result
        self.llm.shutdown()


def _summarize_case(
    case: BenchCase,
    *,
    wall_ms: float,
    outputs: list[list[int]],
    steps: list[StepSample],
    arrival_ms: dict[int, list[float]],
    record_steps: bool,
) -> dict[str, Any]:
    output_lengths = [len(output) for output in outputs]
    if len(outputs) != case.requests:
        raise RuntimeError(f"expected {case.requests} outputs, received {len(outputs)}")
    if any(length != case.decode_tokens for length in output_lengths):
        raise RuntimeError(
            f"expected exactly D={case.decode_tokens} tokens per request, got {output_lengths}"
        )

    prefill_steps = [step for step in steps if step.phase == "prefill"]
    decode_steps = [step for step in steps if step.phase == "decode"]
    prefill_gpu_ms = sum(step.gpu_ms for step in prefill_steps)
    decode_gpu_ms = sum(step.gpu_ms for step in decode_steps)
    prefill_scheduled = sum(step.logical_tokens for step in prefill_steps)
    decode_scheduled = sum(step.logical_tokens for step in decode_steps)

    ttft = [times[0] for times in arrival_ms.values() if times]
    itl = [
        later - earlier
        for times in arrival_ms.values()
        for earlier, later in zip(times, times[1:])
    ]
    summary: dict[str, Any] = {
        "wall_ms": wall_ms,
        "generated_tokens": sum(output_lengths),
        "output_lengths": output_lengths,
        "scheduled_prefill_tokens": prefill_scheduled,
        "scheduled_decode_tokens": decode_scheduled,
        "prefill_gpu_ms": prefill_gpu_ms,
        "decode_gpu_ms": decode_gpu_ms,
        "gpu_ms_sum": prefill_gpu_ms + decode_gpu_ms,
        "steps": {
            "total": len(steps),
            "prefill": len(prefill_steps),
            "decode": len(decode_steps),
        },
        "throughput": {
            "requests_per_s": _rate(case.requests, wall_ms),
            "input_tokens_per_s_wall": _rate(case.input_tokens, wall_ms),
            "output_tokens_per_s_wall": _rate(case.output_tokens, wall_ms),
            "prefill_scheduled_tokens_per_s_gpu": _rate(prefill_scheduled, prefill_gpu_ms),
            "decode_scheduled_tokens_per_s_gpu": _rate(decode_scheduled, decode_gpu_ms),
        },
        "latency_ms": {
            "ttft": _distribution(ttft),
            "inter_token": _distribution(itl),
            "step_gpu": _distribution(step.gpu_ms for step in steps),
            "prefill_step_gpu": _distribution(step.gpu_ms for step in prefill_steps),
            "decode_step_gpu": _distribution(step.gpu_ms for step in decode_steps),
            "step_submit_or_wall": _distribution(
                step.submit_or_step_wall_ms for step in steps
            ),
        },
        "batch": {
            "logical_size": _distribution(float(step.batch_size) for step in steps),
            "padded_size": _distribution(float(step.padded_batch_size) for step in steps),
            "cuda_graph_steps": sum(step.used_cuda_graph is True for step in steps),
            "eager_steps": sum(step.used_cuda_graph is False for step in steps),
        },
    }
    if record_steps:
        summary["step_samples"] = [asdict(step) for step in steps]
    return summary


def _case(name: str, n: int, p: int, d: int) -> BenchCase:
    return BenchCase(name=name, requests=n, prefill_tokens=p, decode_tokens=d)


def _preset_cases(preset: str, model_kind: str, engine: str) -> list[BenchCase]:
    if preset == "smoke":
        limit = 64 if model_kind == "minillm" else 128
        return [_case("smoke", 1, min(16, limit), 2)]

    if model_kind == "minillm":
        presets = {
            "request": [_case(f"request-n{n}", n, 32, 32) for n in (1, 8, 32, 128, 256)],
            "prefill": [_case(f"prefill-p{p}", 8, p, 1) for p in (8, 32, 64, 96, 120)],
            "decode": [_case(f"decode-d{d}", 8, 16, d) for d in (1, 8, 32, 64, 96)],
            "mixed": [
                _case("mixed-latency", 1, 64, 32),
                _case("mixed-throughput", 64, 32, 64),
                _case("mixed-many-short", 256, 16, 16),
            ],
            "boundary": [
                _case("context-total-127", 1, 96, 31),
                _case("context-total-128", 1, 96, 32),
            ],
        }
    else:
        presets = {
            "request": [_case(f"request-n{n}", n, 128, 32) for n in (1, 4, 16, 32, 64)],
            "prefill": [_case(f"prefill-p{p}", 1, p, 1) for p in (16, 128, 512, 2048, 4096)],
            "decode": [_case(f"decode-d{d}", 1, 128, d) for d in (1, 8, 32, 64, 128, 256)],
            "mixed": [
                _case("mixed-prefill", 16, 2048, 64),
                _case("mixed-throughput", 64, 512, 64),
                _case("mixed-many-short", 128, 128, 64),
            ],
            "boundary": (
                [
                    # Prefill predicts output token 1, so only D-1 output
                    # tokens are fed back into the model and stored in KV.
                    _case("nano-block-cache-255", 1, 224, 32),
                    _case("nano-block-cache-256", 1, 225, 32),
                    _case("nano-block-cache-257", 1, 226, 32),
                    _case("context-8k", 1, 8192, 1),
                ]
                if engine == "nano"
                else [
                    _case("context-8k", 1, 8192, 1),
                    _case("concurrency-64", 64, 128, 64),
                ]
            ),
        }
    return presets[preset]


def _parse_cases(raw: str) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for index, item in enumerate(part.strip() for part in raw.split(";") if part.strip()):
        fields = item.split(":")
        if len(fields) == 3:
            name = f"custom-{index}"
            n, p, d = fields
        elif len(fields) == 4:
            name, n, p, d = fields
        else:
            raise ValueError(
                f"invalid case {item!r}; use N:P:D or NAME:N:P:D, separated by ';'"
            )
        cases.append(_case(name, int(n), int(p), int(d)))
    if not cases:
        raise ValueError("--cases did not contain any cases")
    return cases


def _validate_cases(cases: list[BenchCase], model_context: int) -> None:
    for case in cases:
        if case.requests < 1 or case.prefill_tokens < 1 or case.decode_tokens < 1:
            raise ValueError(f"N, P and D must all be >= 1: {case}")
        if case.max_sequence_tokens > model_context:
            raise ValueError(
                f"{case.name}: P+D={case.max_sequence_tokens} exceeds model context {model_context}"
            )


def _resolve_runtime_args(
    args: argparse.Namespace,
    cases: list[BenchCase],
    model_context: int,
) -> None:
    needed_len = max(case.max_sequence_tokens for case in cases)
    needed_seqs = max(case.requests for case in cases)
    args.max_model_len = args.max_model_len or needed_len
    args.max_num_seqs = args.max_num_seqs or needed_seqs
    default_token_budget = 2048 if args.model_kind == "qwen" else 2048
    args.max_batched_tokens = args.max_batched_tokens or min(
        max(max(case.prefill_tokens for case in cases), 128), default_token_budget
    )
    args.cuda_graph_max_bs = args.cuda_graph_max_bs or min(args.max_num_seqs, 64)
    if args.memory_ratio is None:
        # MiniLLM's model and useful KV capacity are tiny; reserving 75% of a
        # 12 GiB GPU would only make this educational engine coexist poorly.
        args.memory_ratio = 0.20 if args.model_kind == "minillm" else 0.75

    if args.max_model_len < needed_len or args.max_model_len > model_context:
        raise ValueError(
            f"--max-model-len must be in [{needed_len}, {model_context}], got {args.max_model_len}"
        )
    if args.max_num_seqs < needed_seqs:
        raise ValueError(
            f"--max-num-seqs={args.max_num_seqs} cannot run N={needed_seqs}"
        )
    if args.max_batched_tokens < 1:
        raise ValueError("--max-batched-tokens must be positive")
    if not 0.1 <= args.memory_ratio <= 0.95:
        raise ValueError("--memory-ratio must be between 0.1 and 0.95")
    if not args.allow_large_config:
        if args.max_num_seqs > 512:
            raise ValueError("refusing max_num_seqs > 512 without --allow-large-config")
        if args.max_batched_tokens > 16384:
            raise ValueError("refusing token budget > 16384 without --allow-large-config")


def _gpu_memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "driver_free_bytes": free_bytes,
        "driver_total_bytes": total_bytes,
    }


def _default_output(args: argparse.Namespace) -> Path:
    now = datetime.now().astimezone()
    run_dir = DEFAULT_OUTPUT_ROOT / f"llm-boundary-{now:%Y%m%d}"
    return run_dir / f"{args.engine}-{args.model_kind}-{args.mode}-{now:%H%M%S}.jsonl"


def _write_jsonl(handle, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    handle.write(line + "\n")
    handle.flush()
    print(line, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run exact-token pressure cases on one persistent local inference engine.",
    )
    parser.add_argument("--engine", choices=("nano", "minisgl"), required=True)
    parser.add_argument("--model-kind", choices=("minillm", "qwen"), required=True)
    parser.add_argument("--model", type=Path, help="local Hugging Face model directory")
    parser.add_argument(
        "--preset",
        choices=("smoke", "request", "prefill", "decode", "mixed", "boundary"),
        default="smoke",
    )
    parser.add_argument(
        "--cases",
        help="override preset: semicolon-separated N:P:D or NAME:N:P:D entries",
    )
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1, help="number of tiny engine warmups")
    parser.add_argument("--warmup-each-case", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-batched-tokens", type=int)
    parser.add_argument("--memory-ratio", type=float)
    parser.add_argument("--cuda-graph-max-bs", type=int)
    parser.add_argument("--page-size", type=int, default=1, help="Mini-SGLang KV page size")
    parser.add_argument("--cache-type", choices=("naive", "radix"), default="naive")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--disable-overlap", action="store_true")
    parser.add_argument("--allow-large-config", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--record-steps", action="store_true")
    parser.add_argument("--record-prompt-ids", action="store_true")
    parser.add_argument(
        "--cuda-profiler-case",
        type=int,
        help="zero-based case index captured via cudaProfilerStart/Stop",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.engine == "minisgl" and args.model_kind != "qwen":
        raise ValueError("official Mini-SGLang in this workspace supports the Qwen test, not MiniLLM")
    if args.repeats < 1 or args.warmup < 0 or args.warmup_each_case < 0:
        raise ValueError("repeat and warmup counts are invalid")
    if args.disable_overlap:
        os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"

    model_path = (args.model or DEFAULT_MODELS[args.model_kind]).resolve()
    if not model_path.is_dir():
        raise ValueError(f"model directory does not exist: {model_path}")
    vocab_size, model_context = _model_limits(model_path)
    cases = _parse_cases(args.cases) if args.cases else _preset_cases(
        args.preset, args.model_kind, args.engine
    )
    _validate_cases(cases, model_context)
    _resolve_runtime_args(args, cases, model_context)
    if args.cuda_profiler_case is not None and not 0 <= args.cuda_profiler_case < len(cases):
        raise ValueError("--cuda-profiler-case is outside the case matrix")

    output_path = (args.output or _default_output(args)).resolve()
    config_summary = {
        "engine": args.engine,
        "model_kind": args.model_kind,
        "model_path": str(model_path),
        "mode": args.mode,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_batched_tokens": args.max_batched_tokens,
        "memory_ratio": args.memory_ratio,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "page_size": args.page_size if args.engine == "minisgl" else None,
        "cache_type": args.cache_type if args.engine == "minisgl" else None,
        "cases": [asdict(case) for case in cases],
        "output": str(output_path),
    }
    if args.dry_run:
        print(json.dumps(config_summary, ensure_ascii=False, indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Mini-SGLang Engine intentionally asserts that it owns CUDA lazy
    # initialization.  Calling set_device before constructing it violates that
    # contract, whereas nano-vLLM expects rank 0 to be the active device.
    if args.engine == "nano":
        torch.cuda.set_device(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now().astimezone().isoformat()}-{os.getpid()}"

    adapter = None
    exit_code = 0
    with output_path.open("a", encoding="utf-8") as output:
        initialization_started = time.perf_counter()
        try:
            adapter = (
                NanoAdapter(model_path, args)
                if args.engine == "nano"
                else MiniSGLangAdapter(model_path, args)
            )
            torch.cuda.synchronize()
            initialization_ms = (time.perf_counter() - initialization_started) * 1000.0
            _write_jsonl(
                output,
                {
                    "record_type": "run",
                    "schema_version": 1,
                    "run_id": run_id,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "host": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "initialization_ms": initialization_ms,
                    "config": config_summary,
                    "capacity": adapter.capacity(),
                    "memory_after_init": _gpu_memory_snapshot(),
                },
            )

            warmup_decode = min(2, args.max_model_len - 1)
            warmup_case = _case(
                "warmup",
                min(2, args.max_num_seqs),
                min(16, args.max_model_len - warmup_decode),
                warmup_decode,
            )
            for warmup_index in range(args.warmup):
                prompts = _make_prompts(
                    warmup_case,
                    vocab_size=vocab_size,
                    seed=args.seed,
                    salt=-10_000 - warmup_index,
                )
                adapter.run_case(warmup_case, prompts, record_steps=False)

            for case_index, case in enumerate(cases):
                for warmup_index in range(args.warmup_each_case):
                    prompts = _make_prompts(
                        case,
                        vocab_size=vocab_size,
                        seed=args.seed,
                        salt=-(case_index + 1) * 1_000 - warmup_index,
                    )
                    adapter.run_case(case, prompts, record_steps=False)

                for repeat_index in range(args.repeats):
                    salt = 1 + case_index * args.repeats + repeat_index
                    prompts = _make_prompts(
                        case,
                        vocab_size=vocab_size,
                        seed=args.seed,
                        salt=salt,
                    )
                    digest = _prompt_digest(prompts)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    memory_before = _gpu_memory_snapshot()
                    capture = args.cuda_profiler_case == case_index and repeat_index == 0
                    if capture:
                        _cuda_profiler_start()
                    try:
                        with _nvtx_range(
                            f"case.{case_index}.{case.name}.N{case.requests}.P{case.prefill_tokens}.D{case.decode_tokens}"
                        ):
                            metrics = adapter.run_case(
                                case,
                                prompts,
                                record_steps=args.record_steps,
                            )
                    finally:
                        if capture:
                            torch.cuda.synchronize()
                            _cuda_profiler_stop()
                    memory_after = _gpu_memory_snapshot()
                    payload = {
                        "record_type": "case",
                        "schema_version": 1,
                        "run_id": run_id,
                        "case_index": case_index,
                        "repeat_index": repeat_index,
                        "case": asdict(case),
                        "prompt_source": "direct_token_ids",
                        "prompt_seed": args.seed,
                        "prompt_salt": salt,
                        "prompt_sha256": digest,
                        "prompt_ids_preview": prompts[0][: min(16, case.prefill_tokens)],
                        "prompt_ids": prompts if args.record_prompt_ids else None,
                        "memory_before": memory_before,
                        "memory_after": memory_after,
                        "peak_allocated_delta_bytes": max(
                            0,
                            memory_after["max_allocated_bytes"]
                            - memory_before["allocated_bytes"],
                        ),
                        "metrics": metrics,
                    }
                    _write_jsonl(output, payload)
        except Exception as exc:
            exit_code = 1
            _write_jsonl(
                output,
                {
                    "record_type": "error",
                    "schema_version": 1,
                    "run_id": run_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            if not args.continue_on_error:
                print(f"benchmark failed: {exc}", file=sys.stderr)
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as close_error:
                    print(f"engine shutdown failed: {close_error}", file=sys.stderr)

    print(f"JSONL: {output_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
