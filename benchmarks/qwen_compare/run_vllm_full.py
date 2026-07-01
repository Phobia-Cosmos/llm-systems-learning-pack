from __future__ import annotations

import os

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
    llm = LLM(
        model=str(MODEL_PATH),
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        max_model_len=2048,
    )
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )

    with Timer() as timer:
        outputs = llm.generate(prompts, sampling, use_tqdm=False)

    texts = [out.outputs[0].text for out in outputs]
    output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    result = BenchResult(
        engine="vllm-full-flashinfer-sampler",
        elapsed_s=timer.elapsed_s,
        output_tokens=output_tokens,
        texts=texts,
    )
    print_result(result)


if __name__ == "__main__":
    main()
