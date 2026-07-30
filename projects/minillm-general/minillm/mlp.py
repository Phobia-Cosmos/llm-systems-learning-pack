from __future__ import annotations

import torch
from torch import nn

from .activations import build_activation


SUPPORTED_MLP_TYPES = ("dense", "swiglu", "geglu", "reglu")


def default_intermediate_size(hidden_size: int, mlp_type: str) -> int:
    if mlp_type == "dense":
        return 4 * hidden_size
    # A gated MLP has three matrices, so ~8/3*C keeps its parameter count
    # close to a dense 4*C MLP. Round to 8 for hardware-friendly shapes.
    return 8 * math_ceil_div(8 * hidden_size, 3 * 8)


def math_ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias),
            build_activation(config.activation),
            nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        gate_activation = {
            "swiglu": "silu",
            "geglu": "gelu",
            "reglu": "relu",
        }[config.mlp_type]
        self.gate_up_proj = nn.Linear(
            config.n_embd,
            2 * config.intermediate_size,
            bias=config.bias,
        )
        self.activation = build_activation(gate_activation)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.dropout(self.down_proj(self.activation(gate) * value))


def build_mlp(config) -> nn.Module:
    if config.mlp_type == "dense":
        return DenseMLP(config)
    if config.mlp_type in {"swiglu", "geglu", "reglu"}:
        return GatedMLP(config)
    raise ValueError(f"unsupported mlp_type: {config.mlp_type}")
