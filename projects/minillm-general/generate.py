from __future__ import annotations

import argparse

import torch

from minillm import GPTConfig, MiniGPT
from minillm.tokenizer_registry import tokenizer_from_checkpoint
from train import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniGPT checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/minillm.pt")
    parser.add_argument("--prompt", default="LLM")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="Set <= 0 to disable top-k filtering.")
    parser.add_argument("--greedy", action="store_true", help="Use argmax decoding instead of random sampling.")
    parser.add_argument("--kv-cache", action="store_true", help="Use the teaching KV-cache decode path.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 问题（已回答）：为什么设置 seed，reproducible sampling 是什么？
    # 回答：采样会调用随机数；相同模型、prompt、参数和 seed 通常得到同一 token 序列，便于对比实验。
    # 某些 GPU 并行 kernel 仍可能非完全确定，因此 seed 是可复现的重要条件但不是跨设备绝对保证。
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)
    # 问题（已回答）:checkpoint代表什么？weights_only是什么意思？checkpoint是什么类型？
    # 回答：checkpoint 是训练后保存的快照，用来恢复模型结构、权重和 tokenizer。这里 torch.load 读出来的是一个 dict。
    # weights_only 是 PyTorch 新版本的安全选项；True 时倾向只加载权重类对象，降低反序列化任意 Python 对象的风险。
    # 本项目 checkpoint 里还有 config/tokenizer/args 这些普通 Python 对象，所以这里用 weights_only=False。
    # 问题（已回答）：任意 Python 反序列化有什么风险，checkpoint 为什么需要它？
    # 回答：pickle 可在加载恶意文件时执行构造代码，所以只能加载可信 checkpoint。这里 torch.save 明确保存了
    # model/config/tokenizer/args 的字典；weights_only=False 是为了恢复这些非纯权重元数据。公开权重更推荐 safetensors。
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    tokenizer = tokenizer_from_checkpoint(checkpoint)
    # 问题（已回答）：checkpoint、** 和 to(device) 分别是什么？
    # 回答：这里 checkpoint 是 dict；**checkpoint["config"] 把键值展开为 GPTConfig 的命名参数。
    # nn.Module.to(device) 将模型参数和 buffer 移到 CPU/CUDA/MPS，并返回模型自身。
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(device)
    # 问题（已回答）:为什么这几个函数都不需要我们自己写就可以运行？
    # 回答：to、load_state_dict、eval 都来自 nn.Module 基类。to(device) 迁移参数到设备；
    # load_state_dict 把 checkpoint 里的权重张量按名字装回模型；eval 关闭 Dropout 等训练行为。
    model.load_state_dict(checkpoint["model"])
    # 问题（已回答）：不调用 model.eval() 会怎样？
    # 回答：模型仍处于训练模式，Dropout 会随机丢激活，导致同样输入不稳定且与部署行为不一致；eval() 不会关闭梯度，
    # 这里只是切换模块模式，若还要省显存应配合 no_grad/inference_mode。
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
    # 问题（已回答）：一个/多个 prompt、batch=1 和 [0] 分别是什么？
    # 回答：这里 idx 外层只放了一条 token 序列，所以 shape 为 [1,T]，batch=1；可以把多条等长/padding 后的序列
    # 堆成 [B,T] 批量生成，但当前简单 generate 对不同长度/停止状态支持有限。返回 [B,total_T]，[0] 取第一个样本。
    generate_fn = model.generate_with_kv_cache if args.kv_cache else model.generate
    out = generate_fn(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=None if args.top_k <= 0 else args.top_k,
        greedy=args.greedy,
    )[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
