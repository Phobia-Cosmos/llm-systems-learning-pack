from __future__ import annotations

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


def main() -> None:
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
        mem_fraction_static=0.45,
        context_length=2048,
        chunked_prefill_size=1024,
        max_prefill_tokens=2048,
        disable_cuda_graph=False,
        attention_backend="flashinfer",
        sampling_backend="flashinfer",
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
        result = BenchResult(
            engine="sglang-full-flashinfer-attn-sampler-cudagraph",
            elapsed_s=timer.elapsed_s,
            output_tokens=output_tokens,
            texts=texts,
        )
        print_result(result)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
