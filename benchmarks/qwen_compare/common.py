from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/home/undefined/Desktop/ai")
MODEL_PATH = ROOT / ".model_cache" / "huggingface" / "Qwen3-0.6B"

PROMPTS = [
    "用三句话解释 KV cache 为什么能加速大模型推理。",
    "比较 vLLM 和 SGLang 在 LLM serving 中分别解决什么问题。",
    "写一个 Python 函数，输入整数 n，返回前 n 个斐波那契数。",
]

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.6
TOP_P = 0.95


@dataclass
class BenchResult:
    engine: str
    elapsed_s: float
    output_tokens: int
    texts: list[str]

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_s = time.perf_counter() - self.start


def print_result(result: BenchResult) -> None:
    print(f"engine: {result.engine}")
    print(f"elapsed_s: {result.elapsed_s:.3f}")
    print(f"output_tokens: {result.output_tokens}")
    print(f"tokens_per_s: {result.tokens_per_s:.2f}")
    for i, text in enumerate(result.texts, 1):
        print(f"\n--- output {i} ---")
        print(text)
