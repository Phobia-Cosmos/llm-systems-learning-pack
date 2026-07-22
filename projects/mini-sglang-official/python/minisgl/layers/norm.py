from typing import Tuple

import torch

from .base import BaseOP


# TODO：为什么这些Norm都是BaseOP的子类？
# 解答：RMSNorm 含可训练/待加载的 weight，继承 BaseOP 后能被模型的递归 state_dict/load_state_dict 发现，并与其他推理算子使用统一 forward 接口。
class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    # TODO：inplace是否一直都会比普通的forward性能更好？
    # 解答：不一定；原地写通常省一次分配和显存流量，但只有调用方不再需要原值、没有别名冲突且不依赖 autograd 时才安全，形状和 kernel 实现也会影响性能。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
