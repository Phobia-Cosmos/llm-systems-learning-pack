"""Benchmark one school-gpu MiniLLM export through a local backend.

The output is intentionally backend-neutral so a matrix runner can compare
native MiniLLM, nano-vLLM, vLLM and the mini-sglang HTTP wrapper without
discarding prompt/completion token ids.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_MODEL_ROOT = Path("/home/undefined/Disk/models/school-gpu/minillm-general")


def cuda_snapshot() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"free_mb": None, "total_mb": None, "allocated_mb": None, "reserved_mb": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "free_mb": free / 2**20,
        "total_mb": total / 2**20,
        "allocated_mb": torch.cuda.memory_allocated() / 2**20,
        "reserved_mb": torch.cuda.memory_reserved() / 2**20,
    }


def wait_health(url: str, process: subprocess.Popen[str], timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    health = url.rsplit("/", 1)[0] + "/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"mini-sglang exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(health, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    raise TimeoutError(f"mini-sglang did not become healthy: {health}")


def request_http(url: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "greedy": True,
            "kv_cache": True,
        }
    ).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    choice = payload["choices"][0]
    return {
        "text": str(choice["text"]),
        "token_ids": [int(value) for value in choice.get("token_ids", payload.get("completion_token_ids", []))],
        "prompt_token_ids": [int(value) for value in payload.get("prompt_token_ids", [])],
    }


def run_native(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sys.path.insert(0, str(ROOT))
    from minillm.adapters import NativeMiniLLMBackend  # noqa: WPS433
    from minillm.contracts import GenerationRequest  # noqa: WPS433

    before = cuda_snapshot()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    backend = NativeMiniLLMBackend(args.checkpoint, args.device)
    load_ms = (time.perf_counter() - started) * 1000.0
    request = GenerationRequest(args.prompt, max_new_tokens=args.max_tokens, temperature=0.0, greedy=True)
    try:
        for _ in range(args.warmup):
            backend.generate(request)
        samples = [backend.generate(request).to_dict() for _ in range(args.samples)]
    finally:
        del backend
    after = cuda_snapshot()
    metrics = {
        "load_ms": load_ms,
        "timings_ms": [item["elapsed_ms"] for item in samples],
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "gpu_before": before,
        "gpu_after": after,
    }
    return metrics, samples


def run_nano(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nano_root = REPO_ROOT / "projects" / "nano-vllm"
    sys.path.insert(0, str(nano_root))
    from nanovllm import LLM, SamplingParams  # noqa: WPS433

    before = cuda_snapshot()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=max(128, args.max_tokens + 64),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    load_ms = (time.perf_counter() - started) * 1000.0
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    prompt_ids = [int(value) for value in engine.tokenizer.encode(args.prompt)]
    try:
        for _ in range(args.warmup):
            engine.generate([args.prompt], sampling, use_tqdm=False)
        samples = []
        for _ in range(args.samples):
            started = time.perf_counter()
            output = engine.generate([args.prompt], sampling, use_tqdm=False)[0]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            samples.append(
                {
                    "backend": "nano-vllm",
                    "prompt": args.prompt,
                    "text": str(output["text"]),
                    "token_ids": [int(value) for value in output["token_ids"]],
                    "prompt_token_ids": prompt_ids,
                    "elapsed_ms": elapsed_ms,
                    "metadata": {"engine_output": "completion-only"},
                }
            )
    finally:
        engine.exit()
    after = cuda_snapshot()
    return {
        "load_ms": load_ms,
        "timings_ms": [item["elapsed_ms"] for item in samples],
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "gpu_before": before,
        "gpu_after": after,
    }, samples


def run_vllm(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from vllm import LLM, SamplingParams  # noqa: WPS433

    before = cuda_snapshot()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    load_ms = (time.perf_counter() - started) * 1000.0
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    try:
        for _ in range(args.warmup):
            engine.generate([args.prompt], sampling, use_tqdm=False)
        samples = []
        for _ in range(args.samples):
            started = time.perf_counter()
            output = engine.generate([args.prompt], sampling, use_tqdm=False)[0]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            completion = output.outputs[0]
            prompt_ids = getattr(output, "prompt_token_ids", None) or []
            samples.append(
                {
                    "backend": "vllm",
                    "prompt": args.prompt,
                    "text": str(completion.text),
                    "token_ids": [int(value) for value in completion.token_ids],
                    "prompt_token_ids": [int(value) for value in prompt_ids],
                    "elapsed_ms": elapsed_ms,
                    "metadata": {"dtype": args.dtype},
                }
            )
    finally:
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    after = cuda_snapshot()
    return {
        "load_ms": load_ms,
        "timings_ms": [item["elapsed_ms"] for item in samples],
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "gpu_before": before,
        "gpu_after": after,
    }, samples


def run_mini_sglang(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    url = args.url or f"http://127.0.0.1:{args.port}/v1/completions"
    server_python = args.server_python or "/home/undefined/Disk/python-envs/sglang/bin/python"
    command = [
        server_python,
        str(REPO_ROOT / "projects/mini-sglang/mini_sglang_server.py"),
        "--checkpoint",
        args.checkpoint,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--device",
        args.device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        wait_health(url, process)
        load_ms = (time.perf_counter() - started) * 1000.0
        for _ in range(args.warmup):
            request_http(url, args.prompt, args.max_tokens)
        samples = []
        for _ in range(args.samples):
            started = time.perf_counter()
            output = request_http(url, args.prompt, args.max_tokens)
            samples.append(
                {
                    "backend": "mini-sglang-http",
                    "prompt": args.prompt,
                    "text": output["text"],
                    "token_ids": output["token_ids"],
                    "prompt_token_ids": output["prompt_token_ids"],
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "metadata": {"url": url},
                }
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return {"load_ms": load_ms, "timings_ms": [item["elapsed_ms"] for item in samples]}, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "nano", "vllm", "mini-sglang"), required=True)
    parser.add_argument("--model", type=Path, required=True, help="HF-like export for nano/vLLM")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Native .pt checkpoint")
    parser.add_argument("--prompt", default="embedding 是把离散 token id 映射成")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--port", type=int, default=18190)
    parser.add_argument("--url")
    parser.add_argument("--server-python")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.checkpoint = args.checkpoint.resolve()
    started = time.perf_counter()
    if args.backend == "native":
        metrics, samples = run_native(args)
    elif args.backend == "nano":
        metrics, samples = run_nano(args)
    elif args.backend == "vllm":
        metrics, samples = run_vllm(args)
    else:
        metrics, samples = run_mini_sglang(args)
    timings = [float(value) for value in metrics["timings_ms"]]
    reference = None
    parity = None
    if args.reference_json:
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))["samples"][0]["token_ids"]
        parity = all(item["token_ids"] == reference for item in samples)
    result = {
        "schema_version": 2,
        "backend": args.backend,
        "model": str(args.model),
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "sample_count": args.samples,
        "metrics": {
            **metrics,
            "median_ms": statistics.median(timings),
            "mean_ms": statistics.fmean(timings),
            "tokens_per_second": args.max_tokens / (statistics.median(timings) / 1000.0),
            "total_elapsed_ms": (time.perf_counter() - started) * 1000.0,
        },
        "samples": samples,
        "parity_reference_token_ids": reference,
        "parity_passed": parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
