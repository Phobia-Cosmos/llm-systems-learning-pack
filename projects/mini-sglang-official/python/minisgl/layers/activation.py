from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul

    # TODO：为什么要使用flashinfer的silu_and_mul，我们不能使用原始的吗？
    # 解答：可以用 x.chunk(2, dim=-1) 后计算 silu(x1) * x2；FlashInfer 把切分、激活和乘法融合为一次专用 kernel，减少启动次数、中间张量和显存读写，语义不变。
    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
