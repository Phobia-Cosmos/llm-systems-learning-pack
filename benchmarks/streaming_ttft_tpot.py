#!/usr/bin/env python3
"""Measure TTFT, TPOT, and E2E latency from OpenAI completion SSE streams."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
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


def stream_request(base_url: str, prompt: str, output_tokens: int) -> dict:
    payload = {
        "model": "Qwen3-8B",
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


def run_case(base_url: str, input_tokens: int, output_tokens: int, concurrency: int) -> dict:
    prompt = " benchmark" * input_tokens
    request_count = max(8, concurrency * 2)
    batch_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        rows = list(
            executor.map(
                lambda _: stream_request(base_url, prompt, output_tokens),
                range(request_count),
            )
        )
    batch_wall = time.perf_counter() - batch_start
    total_completion_tokens = sum(row["completion_tokens"] for row in rows)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
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
    args = parser.parse_args()
    if args.repeats <= 0 or any(value <= 0 for value in args.concurrencies):
        parser.error("repeats and concurrencies must be positive")
    engine_urls = {
        "vllm": "http://127.0.0.1:18000",
        "sglang": "http://127.0.0.1:18001",
    }
    try:
        cases = [tuple(map(int, value.split(":"))) for value in args.cases]
    except ValueError:
        parser.error("each case must use INPUT:OUTPUT integers")
    if any(len(case) != 2 or min(case) <= 0 or sum(case) > 8192 for case in cases):
        parser.error("cases must be positive INPUT:OUTPUT pairs with a total no greater than 8192")
    results = []
    for engine in args.engines:
        base_url = engine_urls[engine]
        for input_tokens, output_tokens in cases:
            for concurrency in args.concurrencies:
                for repeat in range(args.repeats):
                    try:
                        result = run_case(base_url, input_tokens, output_tokens, concurrency)
                        result.update({"engine": engine, "repeat": repeat, "errors": 0})
                    except Exception as error:
                        result = {
                            "engine": engine,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "concurrency": concurrency,
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
