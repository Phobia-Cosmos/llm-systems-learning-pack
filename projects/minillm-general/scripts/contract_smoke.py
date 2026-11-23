"""Run one request through a contract adapter and write a parity-friendly record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.adapters import NanoVLLMBackend, NativeMiniLLMBackend  # noqa: E402
from minillm.contracts import GenerationRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "nano"), required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts/checkpoints/minillm-rope.pt"))
    parser.add_argument("--model", default=str(ROOT / "artifacts/hf_exports/minillm-rope"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt", default="embedding 是把离散 token id 映射成")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = GenerationRequest(args.prompt, max_new_tokens=args.max_new_tokens)
    if args.backend == "native":
        backend = NativeMiniLLMBackend(args.checkpoint, args.device)
    else:
        backend = NanoVLLMBackend(args.model, enforce_eager=True, max_num_seqs=2, max_num_batched_tokens=128, max_model_len=128, gpu_memory_utilization=0.7)
    try:
        result = backend.generate(request)
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
    request_payload = {"prompt": request.prompt, "max_new_tokens": request.max_new_tokens, "temperature": request.temperature, "top_k": request.top_k, "greedy": request.greedy, "kv_cache": request.kv_cache, "request_id": request.request_id}
    payload = {"schema_version": 1, "request": request_payload, "result": result.to_dict()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

