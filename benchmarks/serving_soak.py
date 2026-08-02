#!/usr/bin/env python3
"""Run a bounded mixed-load streaming soak test and emit minute aggregates."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from streaming_ttft_tpot import build_prompts, percentile, stream_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default="/public/home/u43077/lzh/models/Qwen3-8B")
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.duration <= 0 or args.concurrency <= 0:
        parser.error("duration and concurrency must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    pool_size = max(256, args.concurrency * 2)
    prompt_pools = {
        length: build_prompts(tokenizer, length, pool_size, "unique", args.seed + length)
        for length in (512, 4096, 7680)
    }

    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    deadline = started + args.duration
    lock = threading.Lock()
    windows: dict[int, dict[str, list | int]] = {}

    def record(window: int, row: dict | None) -> None:
        with lock:
            bucket = windows.setdefault(
                window,
                {"requests": 0, "errors": 0, "completion_tokens": 0, "ttft": [], "tpot": [], "e2e": []},
            )
            if row is None:
                bucket["errors"] += 1
                return
            bucket["requests"] += 1
            bucket["completion_tokens"] += int(row["completion_tokens"])
            bucket["ttft"].append(float(row["ttft_s"]))
            bucket["tpot"].append(float(row["tpot_s"]))
            bucket["e2e"].append(float(row["e2e_s"]))

    def worker(worker_id: int) -> None:
        rng = random.Random(args.seed + worker_id * 100003)
        request_index = worker_id
        while time.monotonic() < deadline:
            choice = rng.random()
            input_tokens = 512 if choice < 0.5 else 4096 if choice < 0.85 else 7680
            output_tokens = 32 if input_tokens == 7680 or rng.random() < 0.45 else 128
            prompt = prompt_pools[input_tokens][request_index % pool_size]
            url = args.urls[request_index % len(args.urls)]
            try:
                row = stream_request(url, prompt, output_tokens, "Qwen3-8B")
            except Exception:
                row = None
            window = int((time.monotonic() - started) // 60)
            record(window, row)
            request_index += args.concurrency

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, index) for index in range(args.concurrency)]
        next_report = started + 60
        while any(not future.done() for future in futures):
            now = time.monotonic()
            if now >= next_report:
                minute = int((now - started) // 60)
                with lock:
                    completed = sum(int(bucket["requests"]) for bucket in windows.values())
                    errors = sum(int(bucket["errors"]) for bucket in windows.values())
                print(json.dumps({"elapsed_minutes": minute, "requests": completed, "errors": errors}), flush=True)
                next_report += 60
            time.sleep(min(5, max(0.1, deadline - now)))
        for future in futures:
            future.result()

    elapsed = time.monotonic() - started
    summaries = []
    for index, bucket in sorted(windows.items()):
        ttft = bucket.pop("ttft")
        tpot = bucket.pop("tpot")
        e2e = bucket.pop("e2e")
        summaries.append(
            {
                "minute": index,
                **bucket,
                "ttft_p50_s": statistics.median(ttft) if ttft else None,
                "ttft_p95_s": percentile(ttft, 0.95),
                "tpot_p50_s": statistics.median(tpot) if tpot else None,
                "tpot_p95_s": percentile(tpot, 0.95),
                "e2e_p50_s": statistics.median(e2e) if e2e else None,
                "e2e_p95_s": percentile(e2e, 0.95),
            }
        )
    total_requests = sum(row["requests"] for row in summaries)
    total_errors = sum(row["errors"] for row in summaries)
    total_tokens = sum(row["completion_tokens"] for row in summaries)
    payload = {
        "schema_version": 1,
        "started_at_utc": started_wall,
        "duration_requested_s": args.duration,
        "duration_actual_s": elapsed,
        "concurrency": args.concurrency,
        "endpoints": args.urls,
        "requests": total_requests,
        "errors": total_errors,
        "output_throughput_tps": total_tokens / elapsed,
        "windows": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
