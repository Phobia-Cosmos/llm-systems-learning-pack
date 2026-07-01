from __future__ import annotations

import argparse

import sglang as sgl
from transformers import AutoTokenizer

from common import (
    BenchResult,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    PROMPTS,
    TEMPERATURE,
    TOP_P,
    Timer,
    print_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SGLang backend variant.")
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--sampling-backend", default="pytorch")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--mem-fraction-static", type=float, default=0.45)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--max-prefill-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in PROMPTS
    ]
    llm = sgl.Engine(
        model_path=str(MODEL_PATH),
        trust_remote_code=True,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        chunked_prefill_size=args.chunked_prefill_size,
        max_prefill_tokens=args.max_prefill_tokens,
        disable_cuda_graph=args.disable_cuda_graph,
        attention_backend=args.attention_backend,
        sampling_backend=args.sampling_backend,
        log_level="error",
    )

    try:
        with Timer() as timer:
            outputs = llm.generate(
                prompts,
                sampling_params={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            )

        texts = [out["text"] for out in outputs]
        output_tokens = sum(len(tokenizer.encode(text)) for text in texts)
        graph_label = "no-cudagraph" if args.disable_cuda_graph else "cudagraph"
        result = BenchResult(
            engine=(
                "sglang-"
                f"{args.attention_backend}-attn-"
                f"{args.sampling_backend}-sampling-"
                f"{graph_label}"
            ),
            elapsed_s=timer.elapsed_s,
            output_tokens=output_tokens,
            texts=texts,
        )
        print_result(result)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
