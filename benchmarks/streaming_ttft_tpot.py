#!/usr/bin/env python3
"""Measure TTFT, TPOT, and E2E latency from OpenAI completion SSE streams."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import statistics
import time
import urllib.request
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def stream_request(
    base_url: str, prompt: str, output_tokens: int, model_name: str = "Qwen3-8B"
) -> dict:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "top_p": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_content = None
    last_content = None
    completion_count = None
    prompt_count = None
    text_chunks = 0
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                prompt_count = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                completion_count = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            if text:
                now = time.perf_counter()
                first_content = first_content or now
                last_content = now
                text_chunks += 1
    end = time.perf_counter()
    if first_content is None or last_content is None:
        raise RuntimeError("stream returned no non-empty completion chunk")
    completion_count = completion_count or text_chunks
    tpot = (last_content - first_content) / max(completion_count - 1, 1)
    return {
        "ttft_s": first_content - start,
        "tpot_s": tpot,
        "e2e_s": end - start,
        "completion_tokens": completion_count,
        "prompt_tokens": prompt_count,
        "text_chunks": text_chunks,
    }


PROMPT_PIECES = (
    " benchmark", " model", " data", " system", " request", " token", " cache",
    " compute", " memory", " network", " training", " inference", " attention",
    " kernel", " batch", " prompt", " output", " latency", " throughput", " test",
)


def prompt_pieces(tokenizer) -> list[str]:
    pieces = [
        piece
        for piece in PROMPT_PIECES
        if len(tokenizer.encode(piece, add_special_tokens=False)) == 1
    ]
    if len(pieces) < 8:
        raise RuntimeError("tokenizer did not retain enough one-token prompt pieces")
    return pieces


def build_prompts(tokenizer, input_tokens: int, count: int, mode: str, seed: int) -> list[str]:
    pieces = prompt_pieces(tokenizer)
    if mode == "identical":
        prompts = [pieces[0] * input_tokens] * count
    elif mode == "shared":
        suffix_tokens = min(16, max(1, input_tokens // 8))
        prefix = pieces[0] * (input_tokens - suffix_tokens)
        prompts = []
        for index in range(count):
            rng = random.Random(seed + index)
            prompts.append(prefix + "".join(rng.choice(pieces[1:]) for _ in range(suffix_tokens)))
    else:
        prompts = []
        for index in range(count):
            rng = random.Random(seed + index)
            prompts.append("".join(rng.choice(pieces) for _ in range(input_tokens)))
    lengths = {len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts}
    if lengths != {input_tokens}:
        raise RuntimeError(f"generated prompt lengths do not match {input_tokens}: {sorted(lengths)}")
    return prompts


def run_case(
    base_urls: list[str],
    tokenizer,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    prompt_mode: str,
    request_multiplier: int,
    min_requests: int,
    seed: int,
    model_name: str,
) -> dict:
    request_count = max(min_requests, concurrency * request_multiplier)
    prompts = build_prompts(tokenizer, input_tokens, request_count, prompt_mode, seed)
    if prompt_mode == "shared":
        stream_request(base_urls[0], prompts[0], 1, model_name)
    batch_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_request,
                base_urls[index % len(base_urls)],
                prompts[index],
                output_tokens,
                model_name,
            )
            for index in range(request_count)
        ]
        rows = [future.result() for future in futures]
    batch_wall = time.perf_counter() - batch_start
    total_completion_tokens = sum(row["completion_tokens"] for row in rows)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "prompt_mode": prompt_mode,
        "endpoints": len(base_urls),
        "requests": request_count,
        "batch_wall_s": batch_wall,
        "request_throughput_rps": request_count / batch_wall,
        "output_throughput_tps": total_completion_tokens / batch_wall,
        "reported_prompt_tokens": sorted(
            {row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None}
        ),
        "ttft_p50_s": statistics.median(row["ttft_s"] for row in rows),
        "ttft_p95_s": percentile([row["ttft_s"] for row in rows], 0.95),
        "tpot_p50_s": statistics.median(row["tpot_s"] for row in rows),
        "tpot_p95_s": percentile([row["tpot_s"] for row in rows], 0.95),
        "e2e_p50_s": statistics.median(row["e2e_s"] for row in rows),
        "e2e_p95_s": percentile([row["e2e_s"] for row in rows], 0.95),
        "raw": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=("vllm", "sglang"),
        default=("vllm", "sglang"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("512:32", "512:128", "4096:32", "4096:128", "7680:32"),
        metavar="INPUT:OUTPUT",
    )
    parser.add_argument("--concurrencies", nargs="+", type=int, default=(1, 8, 16))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        choices=("identical", "shared", "unique"),
        default=("identical",),
    )
    parser.add_argument("--request-multiplier", type=int, default=2)
    parser.add_argument("--min-requests", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--model-path", default="/public/home/u43077/lzh/models/Qwen3-8B")
    parser.add_argument("--served-model-name", default="Qwen3-8B")
    parser.add_argument("--vllm-urls", nargs="+", default=("http://127.0.0.1:18000",))
    parser.add_argument("--sglang-urls", nargs="+", default=("http://127.0.0.1:18001",))
    args = parser.parse_args()
    if (
        args.repeats <= 0
        or args.request_multiplier <= 0
        or args.min_requests <= 0
        or any(value <= 0 for value in args.concurrencies)
    ):
        parser.error("repeats, request counts, and concurrencies must be positive")
    engine_urls = {
        "vllm": args.vllm_urls,
        "sglang": args.sglang_urls,
    }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    try:
        cases = [tuple(map(int, value.split(":"))) for value in args.cases]
    except ValueError:
        parser.error("each case must use INPUT:OUTPUT integers")
    if any(len(case) != 2 or min(case) <= 0 or sum(case) > 8192 for case in cases):
        parser.error("cases must be positive INPUT:OUTPUT pairs with a total no greater than 8192")
    results = []
    for engine in args.engines:
        base_urls = engine_urls[engine]
        for input_tokens, output_tokens in cases:
            for concurrency in args.concurrencies:
                for prompt_mode in args.prompt_modes:
                    for repeat in range(args.repeats):
                        case_seed = args.seed + input_tokens * 1009 + concurrency * 17 + repeat
                        try:
                            result = run_case(
                                base_urls,
                                tokenizer,
                                input_tokens,
                                output_tokens,
                                concurrency,
                                prompt_mode,
                                args.request_multiplier,
                                args.min_requests,
                                case_seed,
                                args.served_model_name,
                            )
                            result.update({"engine": engine, "repeat": repeat, "errors": 0})
                        except Exception as error:
                            result = {
                                "engine": engine,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "concurrency": concurrency,
                                "prompt_mode": prompt_mode,
                                "repeat": repeat,
                                "errors": 1,
                                "error": repr(error),
                            }
                        print(json.dumps(result, ensure_ascii=False), flush=True)
                        results.append(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
