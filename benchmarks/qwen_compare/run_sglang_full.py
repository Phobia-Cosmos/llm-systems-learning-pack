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
    # TODO:AutoTokenizer是什么？apply_chat_template除了这个还可以做哪些事情分词器？
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            # TODO:这个属性是什么意思？
            add_generation_prompt=True,
        )
        for prompt in PROMPTS
    ]
    # TODO:请你帮我解释以下各个参数的作用是什么以及为什么要使用这些参数？
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
        # TODO:为什么要使用tokenizer单独编码然后才能计算长度？
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
