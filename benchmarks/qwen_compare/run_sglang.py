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
        mem_fraction_static=0.25,
        context_length=512,
        chunked_prefill_size=512,
        max_prefill_tokens=1024,
        disable_cuda_graph=True,
        attention_backend="triton",
        sampling_backend="pytorch",
        log_level="error",
    )

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
        engine="sglang",
        elapsed_s=timer.elapsed_s,
        output_tokens=output_tokens,
        texts=texts,
    )
    print_result(result)
    llm.shutdown()


if __name__ == "__main__":
    main()
