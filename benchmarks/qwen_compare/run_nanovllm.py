from __future__ import annotations

from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

from common import (
    BenchResult,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    PROMPTS,
    TEMPERATURE,
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
        str(MODEL_PATH),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=8,
        gpu_memory_utilization=0.6,
    )
    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)

    with Timer() as timer:
        outputs = llm.generate(prompts, sampling, use_tqdm=False)

    texts = [out["text"] for out in outputs]
    output_tokens = sum(len(out["token_ids"]) for out in outputs)
    result = BenchResult(
        engine="nano-vllm",
        elapsed_s=timer.elapsed_s,
        output_tokens=output_tokens,
        texts=texts,
    )
    print_result(result)


if __name__ == "__main__":
    main()
