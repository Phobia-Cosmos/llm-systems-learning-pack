from __future__ import annotations

import argparse

import torch

from minillm import CharTokenizer, GPTConfig, MiniGPT
from train import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniGPT checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/minillm.pt")
    parser.add_argument("--prompt", default="LLM")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="Set <= 0 to disable top-k filtering.")
    parser.add_argument("--greedy", action="store_true", help="Use argmax decoding instead of random sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)
    # 问题（已回答）:checkpoint代表什么？weights_only是什么意思？checkpoint是什么类型？
    # 回答：checkpoint 是训练后保存的快照，用来恢复模型结构、权重和 tokenizer。这里 torch.load 读出来的是一个 dict。
    # weights_only 是 PyTorch 新版本的安全选项；True 时倾向只加载权重类对象，降低反序列化任意 Python 对象的风险。
    # 本项目 checkpoint 里还有 config/tokenizer/args 这些普通 Python 对象，所以这里用 weights_only=False。
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(device)
    # 问题（已回答）:为什么这几个函数都不需要我们自己写就可以运行？
    # 回答：to、load_state_dict、eval 都来自 nn.Module 基类。to(device) 迁移参数到设备；
    # load_state_dict 把 checkpoint 里的权重张量按名字装回模型；eval 关闭 Dropout 等训练行为。
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = args.prompt.replace("\\n", "\n")
    token_ids = tokenizer.encode(prompt)
    # 问题（已回答）:为什么要变成tensor以及转换后有何不同吗？
    # 回答：tokenizer.encode 得到的是 Python list[int]，只能表示数据，不能直接进入 nn.Embedding/GPU 计算。
    # torch.tensor([token_ids], dtype=torch.long, device=device) 把它变成 shape [1, seq_len] 的 LongTensor，
    # 多出来的外层 [] 是 batch 维，device 让它和模型在同一设备。
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    # 问题（已回答）:为什么只选择第零个？
    # 回答：generate 返回 shape [batch, total_seq_len]。这里一次只生成一个 prompt，所以 batch=1，
    # [0] 取出第一条也是唯一一条生成序列，再 tolist() 转成 tokenizer.decode 能处理的 id 列表。
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=None if args.top_k <= 0 else args.top_k,
        greedy=args.greedy,
    )[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
