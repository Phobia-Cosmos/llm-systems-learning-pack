from __future__ import annotations

import torch
from torch import nn
# 问题（已回答）：torch.nn.functional 是什么，为什么这里写成 F？
# 回答：torch.nn.functional 提供 ReLU、softmax、cross_entropy 等“直接接收 Tensor 并返回 Tensor”的函数式接口，
# 通常不保存可训练参数或 train/eval 状态。`as F` 只是社区惯用的短别名，所以 F.relu(x) 就是直接对 x 做 ReLU。
# nn.ReLU() 则先构造一个 nn.Module，适合放进 Sequential、跟随 model.train()/eval() 并显示在模型结构中。
# 这里 SquaredReLU 需要先 ReLU 再平方，直接调用无状态的 F.relu 最清楚；普通 ReLU 仍由下方工厂返回 nn.ReLU 模块。
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

# TODO：类似于大的GPT、llama也都是这种方式实现激活函数的吗？使用下面的factory？
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
