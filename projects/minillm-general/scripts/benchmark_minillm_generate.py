"""Benchmark native MiniLLM generation on CPU or CUDA using the same prompt."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minillm import GPTConfig, MiniGPT
from minillm.tokenizer_registry import tokenizer_from_checkpoint
from minillm.tokenizer_variants import HFByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/minillm-rope.pt")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", default="embedding 是把离散 token id 映射成")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if "tokenizer" in checkpoint:
        tokenizer = tokenizer_from_checkpoint(checkpoint)
    else:
        tokenizer_path = Path(args.tokenizer) if getattr(args, "tokenizer", None) else checkpoint_path.parent / checkpoint.get("tokenizer_file", "tokenizer.json")
        tokenizer = HFByteBPETokenizer.from_file(tokenizer_path)
    prompt = args.prompt.replace("\\n", "\n")
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    load_ms = (time.perf_counter() - started) * 1000.0
    generate_fn = model.generate_with_kv_cache if args.kv_cache else model.generate

    def run():
        if device.type == "cuda":
            torch.cuda.synchronize()
        started_inner = time.perf_counter()
        result = generate_fn(input_ids, max_new_tokens=args.max_new_tokens, temperature=0.0, top_k=None, greedy=True)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - started_inner) * 1000.0, result.tolist()

    for _ in range(args.warmup):
        run()
    timings, token_ids = [], []
    with torch.inference_mode():
        for _ in range(args.samples):
            elapsed, token_ids = run()
            timings.append(elapsed)
    payload = {
        "schema_version": 1,
        "engine": "native-minillm",
        "device": str(device),
        "checkpoint": str(checkpoint_path.resolve()),
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "kv_cache": args.kv_cache,
        "warmup": args.warmup,
        "samples": args.samples,
        "load_ms": load_ms,
        "timings_ms": timings,
        "summary_ms": {"mean": statistics.fmean(timings), "median": statistics.median(timings), "min": min(timings), "max": max(timings)},
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids),
        "prompt_token_ids": input_ids[0].tolist(),
        "completion_token_ids": token_ids[len(input_ids[0]) :],
        "completion_text": tokenizer.decode(token_ids[len(input_ids[0]) :]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
