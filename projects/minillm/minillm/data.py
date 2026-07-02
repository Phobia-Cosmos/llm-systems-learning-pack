from __future__ import annotations

from pathlib import Path

import torch


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_train_val(token_ids: list[int], val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    # 问题（已回答）:为什么要使用torch.long？这个long和普通的long有和区别？
    # 回答：token id 是离散类别编号，nn.Embedding 和 F.cross_entropy 的 target 都要求 LongTensor。
    # torch.long 是 PyTorch 的 int64 张量 dtype；Python 里的 int/long 是普通标量概念，
    # 不能直接表达“整批数据在 GPU/CPU 上以 int64 张量方式参与运算”。
    data = torch.tensor(token_ids, dtype=torch.long)
    split = max(1, int(len(data) * (1.0 - val_fraction)))
    return data[:split], data[split:]


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("dataset is too small for the configured block_size")
    # 问题（已回答）:为什么第三个参数里面有个,？为什么start是随机数?为什么要使用stack？x和y的区别是什么？为什么后需要使用to？
    # 回答：(batch_size,) 是一维 shape tuple，逗号表示“这是只有一个元素的元组”，不是括号表达式。
    # starts 用随机数是为了每次从语料中抽不同片段，形成随机 mini-batch，训练不会只记住固定顺序。
    # torch.stack 把 batch_size 个长度为 block_size 的一维片段堆成二维张量 [batch, block_size]。
    # x 是输入序列，y 是右移一位的目标序列；模型看到 x[t]，训练目标是预测 y[t] 也就是下一个 token。
    # .to(device) 把数据移动到 CPU/CUDA/MPS 中当前模型所在设备，否则模型和数据不在同一设备会报错。
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
