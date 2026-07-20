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
    # 问题（已回答）：这里是一位用户的多个问题吗，同一模型怎样处理多用户并发请求？
    # 回答：prompts 只是两条独立请求，不记录用户归属；离线示例会先全部入队，再由 scheduler 动态组成批次。
    # 在线多用户场景应在 LLMEngine 外提供服务层，把各用户请求汇入一个由单线程/协程独占的 engine step 循环，
    # 并维护 request_id、响应队列、取消与限流。不要让多个线程并发调用同一个同步 generate 实例。
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
