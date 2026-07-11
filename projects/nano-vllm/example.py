import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # TODO：我们该如何写一个nano vllm的入口函数 可以选择模型 现在只能选择一个Qwen吧？
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # TODO：最小的AutoTokenizer如何实现 这个库的作用是什么 开源的学习的仓库是否存在？
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
