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
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
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
        self.scale = nn.Parameter(torch.tensor(math.sqrt(hidden_size), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True).clamp_min(self.eps)
        return (x.float() * (self.scale.float() / norm)).to(x.dtype)


def build_norm(hidden_size: int, norm_type: str, eps: float, bias: bool) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps, bias=bias)
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_size, eps)
    if norm_type == "scalenorm":
        return ScaleNorm(hidden_size, eps)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"unsupported norm_type: {norm_type}")
