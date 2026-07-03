from __future__ import annotations

import argparse
import sys
from pathlib import Path


MINILLM_DIR = Path(__file__).resolve().parents[1]
PROJECTS_DIR = Path(__file__).resolve().parents[2]
NANO_VLLM_DIR = PROJECTS_DIR / "nano-vllm"
sys.path.insert(0, str(NANO_VLLM_DIR))

from nanovllm import LLM, SamplingParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniLLM HF-like export through the nano-vLLM teaching backend.")
    parser.add_argument("--model", default=str(MINILLM_DIR / "hf_exports" / "minillm"))
    parser.add_argument("--prompt", default="用户: 什么是 decoder-only Transformer？\\n助手:")
    parser.add_argument("--max-tokens", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        max_model_len=128,
        gpu_memory_utilization=0.2,
    )
    try:
        outputs = llm.generate(
            [args.prompt.replace("\\n", "\n")],
            SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True),
            use_tqdm=False,
        )
        print(outputs[0]["text"])
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
