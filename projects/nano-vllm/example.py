import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 问题（已回答）：入口如何选择模型，现在只能用 Qwen 吗？
    # 回答：实际入口可用 argparse 接收 path/prompt/采样参数。LLM 读取 config.json 后由 registry 选后端；
    # 当前已支持 Qwen3 和 MiniGPT，不只是一份固定 Qwen checkpoint。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 问题（已回答）：AutoTokenizer 做什么，最小实现是什么，有学习项目吗？
    # 回答：它从 tokenizer 配置自动选择实现，负责文本/id、special tokens 和 chat template。
    # 最小字符版就是 MiniLLM 的 stoi/itos；标准实现可学习 HF tokenizers、SentencePiece、tiktoken 源码。
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
