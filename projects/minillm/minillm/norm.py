from __future__ import annotations

import math

import torch
from torch import nn


SUPPORTED_NORMS = ("layernorm", "rmsnorm", "scalenorm", "none")


class RMSNorm(nn.Module):
    """Root mean square normalization used by Llama/Qwen/Mistral."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized.to(x.dtype) * self.weight).to(x.dtype)


class ScaleNorm(nn.Module):
    """Scale vectors by their L2 norm and one learned scalar."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(math.sqrt(hidden_size), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True).clamp_min(self.eps)
        return (x.float() * (self.scale.float() / norm)).to(x.dtype)


def build_norm(
    hidden_size: int,
    norm_type: str,
    *,
    eps: float,
    bias: bool,
) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps, bias=bias)
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_size, eps)
    if norm_type == "scalenorm":
        return ScaleNorm(hidden_size, eps)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"unsupported norm_type: {norm_type}")
