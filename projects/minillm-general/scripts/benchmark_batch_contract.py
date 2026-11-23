"""Compare a small batch through native MiniLLM and nano-vLLM."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm import GPTConfig, MiniGPT  # noqa: E402
from minillm.tokenizer_registry import tokenizer_from_checkpoint  # noqa: E402


# MiniGPT's educational batch forward has no attention mask. Keep prompts at
# equal length so native batching does not silently treat padding as context.
PROMPTS = ["embedding 是把离散 token id 映射成"] * 4


def native(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    device = torch.device(args.device)
    model = MiniGPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prompt_ids = [tokenizer.encode(prompt) for prompt in PROMPTS]
    width = max(map(len, prompt_ids))
    batch = torch.tensor([ids + [0] * (width - len(ids)) for ids in prompt_ids], dtype=torch.long, device=device)
    fn = model.generate_with_kv_cache

    def run() -> tuple[float, list[list[int]]]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = fn(batch, max_new_tokens=args.max_new_tokens, temperature=0.0, top_k=None, greedy=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000.0, [row[width:].tolist() for row in output]

    for _ in range(args.warmup):
        run()
    timings, outputs = zip(*(run() for _ in range(args.samples)))
    return {"backend": "native-minillm", "batch_size": len(PROMPTS), "timings_ms": list(timings), "median_ms": statistics.median(timings), "completion_tokens": args.max_new_tokens * len(PROMPTS), "outputs": outputs[-1]}


def nano(args: argparse.Namespace) -> dict:
    nano_root = ROOT.parents[1] / "nano-vllm"
    sys.path.insert(0, str(nano_root))
    from nanovllm import LLM, SamplingParams  # noqa: WPS433

    engine = LLM(args.model, enforce_eager=True, max_num_seqs=4, max_num_batched_tokens=256, max_model_len=128, gpu_memory_utilization=0.7)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, ignore_eos=True)
    try:
        for _ in range(args.warmup):
            engine.generate(PROMPTS, sampling, use_tqdm=False)
        timings, outputs = [], []
        for _ in range(args.samples):
            started = time.perf_counter()
            result = engine.generate(PROMPTS, sampling, use_tqdm=False)
            timings.append((time.perf_counter() - started) * 1000.0)
            outputs = [item["token_ids"] for item in result]
    finally:
        engine.exit()
    return {"backend": "nano-vllm", "batch_size": len(PROMPTS), "timings_ms": timings, "median_ms": statistics.median(timings), "completion_tokens": args.max_new_tokens * len(PROMPTS), "outputs": outputs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "nano"), required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts/checkpoints/minillm-rope.pt"))
    parser.add_argument("--model", default=str(ROOT / "artifacts/hf_exports/minillm-rope"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"schema_version": 1, "prompts": PROMPTS, "max_new_tokens": args.max_new_tokens, "warmup": args.warmup, "samples": args.samples, "result": native(args) if args.backend == "native" else nano(args)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
