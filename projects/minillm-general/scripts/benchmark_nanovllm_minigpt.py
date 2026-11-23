"""Benchmark the real nano-vLLM MiniGPT engine on the local exported checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANO_ROOT = PROJECT_ROOT.parents[1] / "nano-vllm"
sys.path.insert(0, str(NANO_ROOT))

from nanovllm import LLM, SamplingParams  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(PROJECT_ROOT / "artifacts/hf_exports/minillm-rope"))
    parser.add_argument("--prompt", default="embedding 是把离散 token id 映射成")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    init_ms = (time.perf_counter() - started) * 1000.0
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    prompt = args.prompt.replace("\\n", "\n")
    try:
        for _ in range(args.warmup):
            llm.generate([prompt], sampling, use_tqdm=False)
        timings = []
        outputs = []
        for _ in range(args.samples):
            started = time.perf_counter()
            result = llm.generate([prompt], sampling, use_tqdm=False)[0]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            timings.append(elapsed_ms)
            outputs.append({"text": result["text"], "token_count": len(result["token_ids"]), "token_ids": result["token_ids"]})
    finally:
        llm.exit()
    payload = {
        "schema_version": 1,
        "engine": "nano-vllm",
        "model": str(Path(args.model).resolve()),
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "samples": args.samples,
        "init_ms": init_ms,
        "timings_ms": timings,
        "summary_ms": {
            "mean": statistics.fmean(timings),
            "median": statistics.median(timings),
            "min": min(timings),
            "max": max(timings),
        },
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
