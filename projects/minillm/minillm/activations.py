from __future__ import annotations

import torch
from torch import nn
# TODO:functional是什么？作用是什么？
from torch.nn import functional as F


SUPPORTED_ACTIVATIONS = (
    "gelu",
    "gelu_tanh",
    "relu",
    "relu_squared",
    "leaky_relu",
    "elu",
    "silu",
    "mish",
    "tanh",
    "sigmoid",
    "identity",
)


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


def build_activation(name: str) -> nn.Module:
    factories = {
        "gelu": lambda: nn.GELU(approximate="none"),
        "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
        "relu": nn.ReLU,
        "relu_squared": SquaredReLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation: {name}") from exc
