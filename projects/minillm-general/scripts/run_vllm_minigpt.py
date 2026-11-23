from __future__ import annotations

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams


MINILLM_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MiniLLM HF-like export through vLLM.")
    parser.add_argument("--model", default=str(MINILLM_DIR / "artifacts" / "hf_exports" / "minillm-rope"))
    parser.add_argument("--prompt", default="embedding 是把离散 token id 映射成")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-model-len", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    output = llm.generate(
        [args.prompt.replace("\\n", "\n")],
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
    )[0]
    print(args.prompt.replace("\\n", "\n") + output.outputs[0].text)
    print(f"generated_token_ids={output.outputs[0].token_ids}")


if __name__ == "__main__":
    main()
