from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    messages = [[{"role": "user", "content": prompt}] for prompt in PROMPTS]
    rendered = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode(), Timer() as timer:
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[:, input_len:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    result = BenchResult(
        engine="transformers",
        elapsed_s=timer.elapsed_s,
        output_tokens=int(generated.numel()),
        texts=texts,
    )
    print_result(result)


if __name__ == "__main__":
    main()
