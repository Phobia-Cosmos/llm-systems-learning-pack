from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from minillm import CharTokenizer, GPTConfig, MiniGPT
from minillm.data import get_batch, read_text, split_train_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal decoder-only LLM.")
    parser.add_argument("--data", default="data/tiny_corpus.txt")
    parser.add_argument("--out-dir", default="checkpoints")
    # 问题（已回答）:作用？eval-interval、max-steps、block-size、n-embd、lr请你帮我解释每个参数的意义是什么？
    # 回答：这些是训练超参数。max-steps 是优化更新多少步；eval-interval 是每隔多少步评估/打印一次 loss；
    # eval-iters 是评估 loss 时抽多少个 batch 求平均；batch-size 是一次训练多少条序列；block-size 是上下文长度；
    # n-layer 是 TransformerBlock 层数；n-head 是 attention 头数；n-embd 是每个 token 向量维度；
    # dropout 是训练时随机丢激活的比例；lr 是 learning rate，控制每步参数更新幅度；seed 控制随机数可复现；device 选择 CPU/GPU。
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 问题（已回答）:这个是什么？
    # 回答：MPS 是 Apple Silicon/Mac GPU 的 PyTorch 后端。这里的顺序是：优先 CUDA，其次 Mac MPS，最后 CPU。
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    # 问题（已回答）:为什么MiniGPT中没有这个函数？
    # 回答：eval() 来自 nn.Module 基类，不需要 MiniGPT 自己实现。它会把模型切到评估模式，主要影响 Dropout/BatchNorm 等层。
    model.eval()
    result: dict[str, float] = {}
    # 问题（已回答）:这个循环实在做什么？为什么训练前要先计算loss？
    # 回答：这个循环分别在训练集和验证集上抽 eval_iters 个 batch，估计平均 loss。
    # step 0 先算一次 loss 是为了看到“未训练前”的基线；之后对比 loss 是否下降，判断训练是否真的有效以及是否过拟合。
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            # 问题（已回答）:为什么传入model xy就可以自动出来loss等值了？loss是个什么？为什么还要使用item？
            # 回答：model(x, y) 会触发 MiniGPT.forward(idx=x, targets=y)。forward 里如果 targets 不为 None 就计算 cross_entropy loss。
            # loss 是一个标量 Tensor，表示当前 batch 的预测错误程度；item() 把单元素 Tensor 转成 Python float，方便记录和打印。
            _, loss = model(x, y)
            losses[i] = loss.item()
        result[split] = losses.mean().item()
    # 问题（已回答）:为什么不用自己写这个函数就可以调用？
    # 回答：train() 也来自 nn.Module 基类。它把模型切回训练模式，Dropout 会重新启用。
    model.train()
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    text = read_text(args.data)
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    train_data, val_data = split_train_val(token_ids)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)
    # 问题（已回答）:为什么要使用一个优化器，这个优化器的作用是什么？难道MiniGPT不能完成任务吗？
    # 回答：MiniGPT 只定义“如何从输入算出 logits/loss”，不会自己改变参数。loss.backward() 只计算梯度；
    # optimizer 根据梯度真正更新参数。AdamW 是常用优化器，带自适应学习率和 decoupled weight decay。
    # 没有优化器，模型会一直停留在随机初始化状态，任务不会被学会。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={device}, vocab_size={tokenizer.vocab_size}, parameters={model.parameter_count():,}")
    for step in range(args.max_steps + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.block_size,
                args.batch_size,
                args.eval_iters,
                device,
            )
            print(f"step {step:04d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        # 问题（已回答）:这里是在做什么？
        # 回答：PyTorch 默认会累加梯度，所以每次更新前要清空上一轮梯度。set_to_none=True 会把 grad 设为 None，
        # 通常比填 0 更省一点内存/时间。
        optimizer.zero_grad(set_to_none=True)
        # 问题（已回答）:这个是如何实现的？为什么不需要我们手动实现？clip_grad_norm_又是在做什么？优化器step代表什么？
        # 回答：PyTorch autograd 在前向计算时记录计算图，loss.backward() 会按链式法则自动反向传播，给每个参数填充 .grad。
        # clip_grad_norm_ 把所有参数梯度的总范数限制到 1.0，防止梯度爆炸导致训练不稳定。optimizer.step() 根据当前梯度和 AdamW 规则更新权重。
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "minillm.pt"
    # 问题（已回答）:pt格式是什么？自动保存为pt格式吗？
    # 回答：.pt/.pth 是 PyTorch 社区常用 checkpoint 后缀，本质是 torch.save 序列化出来的文件，不是强制文件格式标准。
    # 这里因为路径写成 minillm.pt，所以保存出来就是 .pt。内容是一个 dict：模型权重、config、tokenizer、训练参数。
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "tokenizer": tokenizer.to_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
