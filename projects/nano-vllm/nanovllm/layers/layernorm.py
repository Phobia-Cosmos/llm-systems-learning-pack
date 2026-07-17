import torch
from torch import nn
import math


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 问题（已回答）：为什么先加残差，又把相加结果保存成 residual？
        # 回答：传入的 x 是当前子层输出，residual 是此前累计的残差流；这里先得到 s=x+residual，
        # 把 s 的低精度副本作为下一子层继续使用的 residual，同时对同一个 s 做 RMSNorm。融合可少一次显存读写，
        # 并不是用 x “反算”旧 residual。
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        # 问题（已回答）：RMSNorm 后为什么还要乘 weight？
        # 回答：归一化把每个 token 的整体 RMS 固定到约 1；可训练的逐通道 weight 再恢复各特征不同的幅度，
        # 让模型不因归一化而失去通道缩放能力。推理时它是训练后固定下来的参数。
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class ScaleNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # 问题（已回答）：ScaleNorm 为什么初始化为 sqrt(hidden_size)？
        # 回答：若隐藏向量各维方差约为 1，其期望 L2 范数约为 sqrt(C)。把可学习尺度 g 初始化为 sqrt(C)，
        # 可让归一化后的初始幅度接近输入，而不是骤降到单位 L2 范数；之后 g 会由训练调整。
        self.scale = nn.Parameter(torch.tensor(math.sqrt(hidden_size), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 问题（已回答）：ScaleNorm 的原理和作用是什么？
        # 回答：它计算 y = g*x/max(||x||_2, eps)，用一个可训练标量 g 控制整个向量长度，而不做均值中心化。
        # 它限制激活尺度、改善深层网络的数值和梯度稳定性；相比 RMSNorm，参数更少但不能逐通道缩放。
        norm = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True).clamp_min(self.eps)
        return (x.float() * (self.scale.float() / norm)).to(x.dtype)


def build_norm(hidden_size: int, norm_type: str, eps: float, bias: bool) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps, bias=bias)
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_size, eps)
    if norm_type == "scalenorm":
        return ScaleNorm(hidden_size, eps)
    # 问题（已回答）：norm_type="none" 是什么 Norm？
    # 回答：它不是一种归一化，而是 nn.Identity 恒等映射，原样返回输入；保留该选项是为了消融实验和教学对比。
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"unsupported norm_type: {norm_type}")
