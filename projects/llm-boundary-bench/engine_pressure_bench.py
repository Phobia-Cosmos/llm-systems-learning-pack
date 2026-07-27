#!/home/undefined/Disk/python-envs/sglang/bin/python
"""Pressure benchmark for the local nano-vLLM and official Mini-SGLang engines.

Random mode feeds token IDs directly, so ``P`` is the exact prompt length.
Text mode runs the model's local tokenizer without padding, so ``P`` is a
per-request truncation cap unless repeat-truncate is selected.  ``D`` is always
the exact number of generated tokens per request.  A single engine is kept
alive while all cases run, which avoids counting model loading and CUDA graph
capture in every case.

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
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch


ROOT = Path("/home/undefined/Desktop/ai")
DEFAULT_MODELS = {
    "minillm": ROOT / "projects/minillm/artifacts/hf_exports/minillm-rope",
    "qwen": Path("/home/undefined/Disk/cache/models/huggingface/Qwen3-0.6B"),
}
DEFAULT_OUTPUT_ROOT = Path("/home/undefined/Disk/build-tmp")


BUILTIN_TEXT_PROMPTS: dict[str, tuple[str, ...]] = {
    "chinese": (
        "请用三句话解释为什么推理服务需要区分首 token 延迟和逐 token 延迟，并给出一个生活中的类比。",
        "小明有十个苹果，送给同学三个，又买了两袋、每袋四个。请先列式，再说明最后有多少个苹果。",
    ),
    "english": (
        "Explain the difference between latency and throughput in an inference server, then give one concrete example.",
        "A train leaves at 09:15 and travels for 2 hours and 47 minutes. Show the calculation and state the arrival time.",
    ),
    "code": (
        "Write a Python function that returns the first non-repeating character in a string. Explain its time and space complexity.",
        "Review this Python expression for edge cases and propose a safer version:\n\n```python\naverage = sum(values) / len(values)\n```",
    ),
    "long": (
        "You are designing a small language-model inference service for a team chat application. The service receives short questions, long pasted documents, and code-review requests. During the morning peak, many users arrive at nearly the same time, while at night only one or two requests are active. The GPU has limited memory, so model weights, temporary activations, and the key-value cache must share the same capacity. Describe a design that controls admission, batches work continuously, reuses safe shared prefixes, reports time to first token and inter-token latency, and degrades predictably when the queue is full. Include the assumptions you would validate before deployment, the metrics you would put on a dashboard, and the failure tests you would run. Do not assume that an offline throughput number is sufficient evidence for production readiness.",
    ),
}


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


@dataclass
class PreparedPrompts:
    token_ids: list[list[int]]
    metadata: dict[str, Any]


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


def _parse_positive_int_csv(raw: str, option: str) -> list[int]:
    values: list[int] = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            continue
        try:
            value = int(field)
        except ValueError as exc:
            raise ValueError(f"{option} must be a comma-separated list of integers") from exc
        if value < 1:
            raise ValueError(f"{option} values must all be >= 1")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{option} did not contain any values")
    return sorted(values)


def _parse_temperature(raw: str) -> float | None:
    if raw.strip().lower() == "auto":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("temperature must be auto or a number >= 0") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("temperature must be finite and >= 0")
    return value


def _grid_cases(
    batches: Sequence[int],
    contexts: Sequence[int],
    decode_tokens: int,
) -> list[BenchCase]:
    if decode_tokens < 1:
        raise ValueError("grid decode tokens must be >= 1")
    return [
        _case(f"grid-n{batch}-p{context}-d{decode_tokens}", batch, context, decode_tokens)
        for context in sorted(set(contexts))
        for batch in sorted(set(batches))
    ]


def _token_batch_digest(token_batches: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for token_ids in token_batches:
        digest.update(len(token_ids).to_bytes(8, "little"))
        ids = array("I", token_ids)
        if sys.byteorder != "little":
            ids.byteswap()
        digest.update(ids.tobytes())
    return digest.hexdigest()


def _text_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in texts:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_prompt_file(path: Path) -> list[str]:
    """Read local prompts without guessing a remote dataset format.

    Plain-text files use one non-empty line per prompt.  JSONL accepts a JSON
    string or an object containing ``prompt`` or ``text`` on each non-empty
    line.  This deliberately keeps loading deterministic and local-only.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"prompt file does not exist: {path}") from exc
    prompts: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if path.suffix.lower() != ".jsonl":
            prompts.append(line)
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on {path}:{line_number}: {exc.msg}") from exc
        if isinstance(payload, str):
            value = payload
        elif isinstance(payload, dict):
            value = payload.get("prompt", payload.get("text"))
        else:
            value = None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{path}:{line_number} must be a JSON string or contain a non-empty prompt/text field"
            )
        prompts.append(value)
    if not prompts:
        raise ValueError(f"prompt file contains no non-empty prompts: {path}")
    return prompts


def _builtin_prompts(suite: str) -> list[str]:
    if suite == "all":
        # Interleave categories so even N=4 covers Chinese, English, code and
        # long-form text instead of exhausting one category first.
        return [
            values[index]
            for index in range(max(len(values) for values in BUILTIN_TEXT_PROMPTS.values()))
            for values in BUILTIN_TEXT_PROMPTS.values()
            if index < len(values)
        ]
    return list(BUILTIN_TEXT_PROMPTS[suite])


def _tokenization_metadata(
    *,
    source_texts: Sequence[str],
    decoded_texts: Sequence[str],
    token_ids: Sequence[Sequence[int]],
    full_token_lengths: Sequence[int],
    tokenization_ms: float,
    source: str,
    length_policy: str,
    token_cap: int,
) -> dict[str, Any]:
    token_lengths = [len(ids) for ids in token_ids]
    decoded_chars = [len(text) for text in decoded_texts]
    decoded_bytes = [len(text.encode("utf-8")) for text in decoded_texts]
    chars_per_token = [chars / tokens for chars, tokens in zip(decoded_chars, token_lengths)]
    bytes_per_token = [byte_count / tokens for byte_count, tokens in zip(decoded_bytes, token_lengths)]
    return {
        "source": source,
        "length_policy": length_policy,
        "token_cap_per_request": token_cap,
        "padding": "none",
        "tokenization_ms": tokenization_ms,
        "request_count": len(token_ids),
        "truncated_requests": sum(
            full_length > token_length
            for full_length, token_length in zip(full_token_lengths, token_lengths)
        ),
        "source_text_sha256": _text_digest(source_texts),
        "prompt_tokens": _distribution(float(value) for value in token_lengths),
        "decoded_chars": _distribution(float(value) for value in decoded_chars),
        "decoded_bytes": _distribution(float(value) for value in decoded_bytes),
        "chars_per_token": _distribution(chars_per_token),
        "bytes_per_token": _distribution(bytes_per_token),
    }


def _prepare_text_prompts(
    case: BenchCase,
    *,
    tokenizer: Any,
    source_texts: Sequence[str],
    source: str,
    length_policy: str,
) -> PreparedPrompts:
    if not source_texts:
        raise ValueError("text prompt source is empty")
    selected = [source_texts[index % len(source_texts)] for index in range(case.requests)]
    token_ids: list[list[int]] = []
    decoded_texts: list[str] = []
    full_token_lengths: list[int] = []
    started = time.perf_counter()
    for source_text in selected:
        full_ids = list(tokenizer.encode(source_text, add_special_tokens=False))
        if not full_ids:
            raise ValueError("tokenizer produced an empty prompt")
        if length_policy == "repeat-truncate" and len(full_ids) < case.prefill_tokens:
            repeats = math.ceil(case.prefill_tokens / len(full_ids)) + 1
            expanded = "\n\n".join(source_text for _ in range(repeats))
            full_ids = list(tokenizer.encode(expanded, add_special_tokens=False))
        full_token_lengths.append(len(full_ids))
        ids = full_ids[: case.prefill_tokens]
        if not ids:
            raise ValueError("text prompt became empty after applying the token cap")
        token_ids.append(ids)
        decoded_texts.append(
            tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    tokenization_ms = (time.perf_counter() - started) * 1000.0
    metadata = _tokenization_metadata(
        source_texts=selected,
        decoded_texts=decoded_texts,
        token_ids=token_ids,
        full_token_lengths=full_token_lengths,
        tokenization_ms=tokenization_ms,
        source=source,
        length_policy=length_policy,
        token_cap=case.prefill_tokens,
    )
    return PreparedPrompts(token_ids=token_ids, metadata=metadata)


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
    return _token_batch_digest(prompts)


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
        self.temperature = args.temperature_resolved
        self.record_output_ids = args.record_output_ids
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
        decode_output: Callable[[list[int]], str] | None = None,
    ) -> dict[str, Any]:
        sampling = self.SamplingParams(
            temperature=self.temperature,
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
            prompt_lengths=[len(prompt) for prompt in prompts],
            steps=step_samples,
            arrival_ms=arrival_ms,
            record_steps=record_steps,
            decode_output=decode_output,
            record_output_ids=self.record_output_ids,
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
        self.temperature = args.temperature_resolved
        self.record_output_ids = args.record_output_ids
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
        decode_output: Callable[[list[int]], str] | None = None,
    ) -> dict[str, Any]:
        sampling = self.SamplingParams(
            temperature=self.temperature,
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
            prompt_lengths=[len(prompt) for prompt in prompts],
            steps=self._step_samples,
            arrival_ms=self._arrival_ms,
            record_steps=record_steps,
            decode_output=decode_output,
            record_output_ids=self.record_output_ids,
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
    prompt_lengths: list[int],
    steps: list[StepSample],
    arrival_ms: dict[int, list[float]],
    record_steps: bool,
    decode_output: Callable[[list[int]], str] | None = None,
    record_output_ids: bool = False,
) -> dict[str, Any]:
    output_lengths = [len(output) for output in outputs]
    if len(outputs) != case.requests:
        raise RuntimeError(f"expected {case.requests} outputs, received {len(outputs)}")
    if any(length != case.decode_tokens for length in output_lengths):
        raise RuntimeError(
            f"expected exactly D={case.decode_tokens} tokens per request, got {output_lengths}"
        )
    if len(prompt_lengths) != case.requests or any(length < 1 for length in prompt_lengths):
        raise RuntimeError("prompt lengths do not match the request batch")

    prefill_steps = [step for step in steps if step.phase == "prefill"]
    decode_steps = [step for step in steps if step.phase == "decode"]
    prefill_gpu_ms = sum(step.gpu_ms for step in prefill_steps)
    decode_gpu_ms = sum(step.gpu_ms for step in decode_steps)
    prefill_scheduled = sum(step.logical_tokens for step in prefill_steps)
    decode_scheduled = sum(step.logical_tokens for step in decode_steps)
    input_tokens = sum(prompt_lengths)
    # Prefill computes output token 1.  Only the following D-1 generated
    # tokens are fed back into the model, so the last generated token is not
    # resident in KV when the request completes.
    cache_lengths = [
        prompt_length + output_length - 1
        for prompt_length, output_length in zip(prompt_lengths, output_lengths)
    ]

    ttft = [times[0] for times in arrival_ms.values() if times]
    itl = [
        later - earlier
        for times in arrival_ms.values()
        for earlier, later in zip(times, times[1:])
    ]
    summary: dict[str, Any] = {
        "wall_ms": wall_ms,
        "input_tokens": input_tokens,
        "nominal_input_tokens": case.input_tokens,
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
            "input_tokens_per_s_wall": _rate(input_tokens, wall_ms),
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
            "effective_prefill": _distribution(
                float(step.batch_size) for step in prefill_steps
            ),
            "effective_decode": _distribution(
                float(step.batch_size) for step in decode_steps
            ),
            "cuda_graph_steps": sum(step.used_cuda_graph is True for step in steps),
            "eager_steps": sum(step.used_cuda_graph is False for step in steps),
        },
        "prompt": {
            "token_lengths": _distribution(float(length) for length in prompt_lengths),
            "total_prefill_tokens": input_tokens,
            "nominal_total_prefill_tokens": case.input_tokens,
        },
        "cache": {
            "final_logical_length_per_request": _distribution(
                float(length) for length in cache_lengths
            ),
            "final_logical_tokens_total": sum(cache_lengths),
            "formula": "prompt_tokens + generated_tokens - 1",
        },
        "semantic_regression": {
            "output_token_sha256": _token_batch_digest(outputs),
            "per_request_output_token_sha256": [
                _token_batch_digest([output]) for output in outputs
            ],
            "output_ids_preview": [output[:16] for output in outputs[:4]],
            "output_ids": outputs if record_output_ids else None,
        },
    }
    if decode_output is not None:
        generated_text = [decode_output(output) for output in outputs]
        summary["semantic_regression"].update(
            {
                "generated_text": generated_text,
                "generated_text_sha256": _text_digest(generated_text),
            }
        )
    if record_steps:
        summary["step_samples"] = [asdict(step) for step in steps]
    return summary


SATURATION_METRICS = (
    "requests_per_s",
    "input_tokens_per_s_wall",
    "output_tokens_per_s_wall",
    "prefill_scheduled_tokens_per_s_gpu",
    "decode_scheduled_tokens_per_s_gpu",
)


def _analyze_saturation(
    case_records: Sequence[dict[str, Any]],
    *,
    metric: str,
    threshold: float,
) -> dict[str, Any]:
    """Aggregate repeats and locate the first low-gain batch transition.

    Cases are grouped by nominal context cap and decode length.  The cap is
    intentionally kept separate from actual prompt lengths because text mode
    truncates but never pads a natural prompt.
    """

    if metric not in SATURATION_METRICS:
        raise ValueError(f"unsupported saturation metric: {metric}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("saturation threshold must be in [0, 1]")

    grouped: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = {}
    for record in case_records:
        case = record["case"]
        grouped.setdefault(
            (int(case["prefill_tokens"]), int(case["decode_tokens"])), {}
        ).setdefault(int(case["requests"]), []).append(record["metrics"])

    groups: list[dict[str, Any]] = []
    for (context_cap, decode_tokens), by_batch in sorted(grouped.items()):
        points: list[dict[str, Any]] = []
        for batch_size, metrics_list in sorted(by_batch.items()):
            metric_values = [
                metrics["throughput"][metric]
                for metrics in metrics_list
                if metrics["throughput"].get(metric) is not None
            ]
            actual_prefill = [metrics["input_tokens"] for metrics in metrics_list]
            cache_tokens = [
                metrics["cache"]["final_logical_tokens_total"]
                for metrics in metrics_list
            ]
            effective_decode = [
                metrics["batch"]["effective_decode"]["mean"]
                for metrics in metrics_list
                if metrics["batch"]["effective_decode"]["mean"] is not None
            ]
            effective_prefill = [
                metrics["batch"]["effective_prefill"]["mean"]
                for metrics in metrics_list
                if metrics["batch"]["effective_prefill"]["mean"] is not None
            ]
            points.append(
                {
                    "requested_batch": batch_size,
                    "repeat_count": len(metrics_list),
                    "metric_median": statistics.median(metric_values)
                    if metric_values
                    else None,
                    "total_prefill_tokens_median": statistics.median(actual_prefill),
                    "cache_tokens_median": statistics.median(cache_tokens),
                    "effective_prefill_batch_median": statistics.median(effective_prefill)
                    if effective_prefill
                    else None,
                    "effective_decode_batch_median": statistics.median(effective_decode)
                    if effective_decode
                    else None,
                }
            )

        first_low_gain_batch: int | None = None
        transitions: list[dict[str, Any]] = []
        for previous, current in zip(points, points[1:]):
            previous_value = previous["metric_median"]
            current_value = current["metric_median"]
            relative_gain = (
                current_value / previous_value - 1.0
                if previous_value is not None
                and current_value is not None
                and previous_value > 0
                else None
            )
            relative_batch_increase = (
                current["requested_batch"] / previous["requested_batch"] - 1.0
            )
            scaling_efficiency = (
                relative_gain / relative_batch_increase
                if relative_gain is not None and relative_batch_increase > 0
                else None
            )
            below_threshold = relative_gain is not None and relative_gain < threshold
            if below_threshold and first_low_gain_batch is None:
                first_low_gain_batch = current["requested_batch"]
            transitions.append(
                {
                    "from_batch": previous["requested_batch"],
                    "to_batch": current["requested_batch"],
                    "relative_throughput_gain": relative_gain,
                    "relative_batch_increase": relative_batch_increase,
                    "scaling_efficiency": scaling_efficiency,
                    "below_threshold": below_threshold,
                }
            )
        groups.append(
            {
                "context_token_cap": context_cap,
                "decode_tokens": decode_tokens,
                "first_low_gain_batch": first_low_gain_batch,
                "points": points,
                "transitions": transitions,
            }
        )
    return {
        "metric": metric,
        "threshold_fraction": threshold,
        "threshold_interpretation": (
            "first transition where current_throughput / previous_throughput - 1 "
            "is below threshold"
        ),
        "groups": groups,
    }


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
        description="Run token-ID or local-text pressure cases on one persistent inference engine.",
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
    parser.add_argument(
        "--grid-batches",
        help="batch sizes for a Cartesian saturation grid, for example 1,2,4,8,16",
    )
    parser.add_argument(
        "--grid-contexts",
        help="per-request prompt token caps for a Cartesian grid, for example 128,512,2048",
    )
    parser.add_argument(
        "--grid-decode-tokens",
        type=int,
        default=16,
        help="generated tokens per request in a Cartesian grid",
    )
    parser.add_argument(
        "--saturation-metric",
        choices=SATURATION_METRICS,
        default="output_tokens_per_s_wall",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.05,
        help="relative throughput gain below which a grid transition is marked saturated",
    )
    parser.add_argument(
        "--prompt-source",
        choices=("random", "builtin", "file"),
        default="random",
        help="random gives exact P token IDs; text modes tokenize locally and use P only as a cap",
    )
    parser.add_argument(
        "--prompt-suite",
        choices=("all", *BUILTIN_TEXT_PROMPTS.keys()),
        default="all",
        help="small built-in semantic prompt suite",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 prompts: one non-empty line per prompt, or prompt/text records in JSONL",
    )
    parser.add_argument(
        "--text-length-policy",
        choices=("truncate", "repeat-truncate"),
        default="truncate",
        help="never pads; truncate caps natural text, repeat-truncate repeats then caps for controlled pressure",
    )
    parser.add_argument(
        "--record-generated-text",
        action="store_true",
        help="decode and store full generated text (token checksums are always stored)",
    )
    parser.add_argument(
        "--record-output-ids",
        action="store_true",
        help="store every generated token ID; disabled by default to keep large sweeps compact",
    )
    parser.add_argument(
        "--temperature",
        type=_parse_temperature,
        default="auto",
        metavar="FLOAT|auto",
        help="shared sampling temperature; auto preserves nano=0.1 and minisgl=0.0",
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
    if not 0.0 <= args.saturation_threshold <= 1.0:
        raise ValueError("--saturation-threshold must be in [0, 1]")
    grid_requested = args.grid_batches is not None or args.grid_contexts is not None
    if grid_requested and (args.grid_batches is None or args.grid_contexts is None):
        raise ValueError("--grid-batches and --grid-contexts must be provided together")
    if grid_requested and args.cases:
        raise ValueError("--cases cannot be combined with a Cartesian grid")
    if args.prompt_source == "file" and args.prompt_file is None:
        raise ValueError("--prompt-source file requires --prompt-file")
    if args.prompt_source != "file" and args.prompt_file is not None:
        raise ValueError("--prompt-file requires --prompt-source file")
    args.temperature_resolved = (
        args.temperature
        if args.temperature is not None
        else (0.1 if args.engine == "nano" else 0.0)
    )
    if args.disable_overlap:
        os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"

    model_path = (args.model or DEFAULT_MODELS[args.model_kind]).resolve()
    if not model_path.is_dir():
        raise ValueError(f"model directory does not exist: {model_path}")
    vocab_size, model_context = _model_limits(model_path)
    if grid_requested:
        cases = _grid_cases(
            _parse_positive_int_csv(args.grid_batches, "--grid-batches"),
            _parse_positive_int_csv(args.grid_contexts, "--grid-contexts"),
            args.grid_decode_tokens,
        )
    else:
        cases = _parse_cases(args.cases) if args.cases else _preset_cases(
            args.preset, args.model_kind, args.engine
        )
    _validate_cases(cases, model_context)
    _resolve_runtime_args(args, cases, model_context)
    if args.cuda_profiler_case is not None and not 0 <= args.cuda_profiler_case < len(cases):
        raise ValueError("--cuda-profiler-case is outside the case matrix")

    output_path = (args.output or _default_output(args)).resolve()
    if args.prompt_source == "builtin":
        source_texts = _builtin_prompts(args.prompt_suite)
    elif args.prompt_source == "file":
        source_texts = _load_prompt_file(args.prompt_file.resolve())
    else:
        source_texts = []
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
        "prompt": {
            "source": args.prompt_source,
            "suite": args.prompt_suite if args.prompt_source == "builtin" else None,
            "file": str(args.prompt_file.resolve()) if args.prompt_file else None,
            "source_prompt_count": len(source_texts) if source_texts else None,
            "text_length_policy": (
                args.text_length_policy if args.prompt_source != "random" else None
            ),
            "padding": "none" if args.prompt_source != "random" else None,
            "record_generated_text": args.record_generated_text,
            "record_output_ids": args.record_output_ids,
        },
        "grid": {
            "enabled": grid_requested,
            "saturation_metric": args.saturation_metric if grid_requested else None,
            "saturation_threshold": args.saturation_threshold if grid_requested else None,
        },
        "sampling": {
            "temperature_requested": (
                args.temperature if args.temperature is not None else "auto"
            ),
            "temperature_resolved": args.temperature_resolved,
            "strategy": "greedy" if args.temperature_resolved == 0.0 else "stochastic",
            "ignore_eos": True,
            "max_tokens": "case.decode_tokens",
        },
        "cases": [asdict(case) for case in cases],
        "output": str(output_path),
    }
    if args.dry_run:
        print(json.dumps(config_summary, ensure_ascii=False, indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = None
    if args.prompt_source != "random" or args.record_generated_text:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=True,
        )

    def prepare_prompts(case: BenchCase, salt: int) -> PreparedPrompts:
        if args.prompt_source == "random":
            prompt_ids = _make_prompts(
                case,
                vocab_size=vocab_size,
                seed=args.seed,
                salt=salt,
            )
            return PreparedPrompts(
                token_ids=prompt_ids,
                metadata={
                    "source": "direct_token_ids",
                    "length_policy": "exact_random_ids",
                    "token_cap_per_request": case.prefill_tokens,
                    "padding": "none",
                    "tokenization_ms": None,
                    "request_count": case.requests,
                    "truncated_requests": 0,
                    "prompt_tokens": _distribution(
                        float(len(prompt)) for prompt in prompt_ids
                    ),
                    "decoded_chars": None,
                    "decoded_bytes": None,
                    "chars_per_token": None,
                    "bytes_per_token": None,
                },
            )
        assert tokenizer is not None
        return _prepare_text_prompts(
            case,
            tokenizer=tokenizer,
            source_texts=source_texts,
            source=(
                f"builtin:{args.prompt_suite}"
                if args.prompt_source == "builtin"
                else f"file:{args.prompt_file.resolve()}"
            ),
            length_policy=args.text_length_policy,
        )

    decode_output: Callable[[list[int]], str] | None = None
    if args.record_generated_text:
        assert tokenizer is not None

        def decode_output(token_ids: list[int]) -> str:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

    # Mini-SGLang Engine intentionally asserts that it owns CUDA lazy
    # initialization.  Calling set_device before constructing it violates that
    # contract, whereas nano-vLLM expects rank 0 to be the active device.
    if args.engine == "nano":
        torch.cuda.set_device(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now().astimezone().isoformat()}-{os.getpid()}"

    adapter = None
    exit_code = 0
    saturation_records: list[dict[str, Any]] = []
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
                    prepared = prepare_prompts(
                        case,
                        salt=-(case_index + 1) * 1_000 - warmup_index,
                    )
                    adapter.run_case(case, prepared.token_ids, record_steps=False)

                for repeat_index in range(args.repeats):
                    salt = 1 + case_index * args.repeats + repeat_index
                    prepared = prepare_prompts(case, salt=salt)
                    prompts = prepared.token_ids
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
                                decode_output=decode_output,
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
                        "prompt_source": prepared.metadata["source"],
                        "prompt_preparation": prepared.metadata,
                        "prompt_seed": args.seed if args.prompt_source == "random" else None,
                        "prompt_salt": salt if args.prompt_source == "random" else None,
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
                    saturation_records.append(payload)
            if grid_requested:
                _write_jsonl(
                    output,
                    {
                        "record_type": "saturation_analysis",
                        "schema_version": 1,
                        "run_id": run_id,
                        "analysis": _analyze_saturation(
                            saturation_records,
                            metric=args.saturation_metric,
                            threshold=args.saturation_threshold,
                        ),
                    },
                )
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
