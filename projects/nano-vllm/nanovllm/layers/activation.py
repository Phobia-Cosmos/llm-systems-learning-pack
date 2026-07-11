import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    # TODO：为什么要compile？
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
