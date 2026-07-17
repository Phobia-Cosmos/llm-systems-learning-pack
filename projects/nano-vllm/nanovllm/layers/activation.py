import torch
from torch import nn
import torch.nn.functional as F


# 问题（已回答）：SiluAndMul 和 ActivationAndMul 有什么区别？
# 回答：两者都实现 GLU 形式 act(gate) * value。SiluAndMul 把激活固定为 SiLU 并用 torch.compile
# 优化，是 Qwen3/SwiGLU 的专用快速路径；ActivationAndMul 接受激活名称，支持 SwiGLU/GEGLU/ReGLU 等教学变体。
class SiluAndMul(nn.Module):

    # 问题（已回答）：为什么加 torch.compile？
    # 回答：它可把 chunk、SiLU、逐元素乘融合成更少 kernel，减少 Python 和 GPU launch 开销；首次调用有编译成本。
    # 问题（已回答）：这里是后端自动融合，而不是手写 Triton 吗？
    # 回答：是。源码只描述 chunk、SiLU 和乘法；torch.compile 通过 Dynamo/Inductor 捕获图并选择融合方案，
    # NVIDIA GPU 上可能生成 Triton 或其他 CUDA kernel。这里没有人工编写 Triton kernel，融合结果也依赖版本和 shape。
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()

# 问题（已回答）：激活和乘法是两个概念吗？
# 回答：是。build_activation 只构造逐元素非线性函数，如 GELU/SiLU；“乘”是门控 MLP 额外执行的
# 逐元素 gate/value 结合。普通 dense MLP 只有 Linear -> activation -> Linear，不一定存在门控乘法。
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


class ActivationAndMul(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        self.activation = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 问题（已回答）：为什么先 chunk，gate 和 value 分别是什么？
        # 回答：上游 fused Linear 沿最后一维输出 [gate | value]，宽度是 2*intermediate_size。
        # chunk(2) 返回两个等宽视图；gate 经非线性后控制 value 每个通道保留多少，结果仍为 intermediate_size 宽。
        gate, value = x.chunk(2, dim=-1)
        return self.activation(gate) * value
