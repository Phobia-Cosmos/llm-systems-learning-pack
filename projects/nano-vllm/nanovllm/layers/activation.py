import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    # 问题（已回答）：为什么加 torch.compile？
    # 回答：它可把 chunk、SiLU、逐元素乘融合成更少 kernel，减少 Python 和 GPU launch 开销；首次调用有编译成本。
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
