from __future__ import annotations

import math

import torch
import torch.nn.functional as F

torch._dynamo.config.capture_dynamic_output_shape_ops = True


# TODO：这里的backend除了可以是inductor还可以是哪些？各个参数分别代表什么意思
# backend 也可接已注册后端名称或用户自定义 compiler callable；fullgraph=True 要求整函数进入一张图，dynamic=True 尝试让尺寸符号化。
def _compile(function):
    return torch.compile(function, backend="inductor", fullgraph=True, dynamic=True)


@_compile
def elementwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


@_compile
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


@_compile
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@_compile
def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return F.elu(x, alpha=alpha)


@_compile
def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


@_compile
def swish(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


@_compile
def hardswish(x: torch.Tensor) -> torch.Tensor:
    return F.hardswish(x)


@_compile
def hardshrink(x: torch.Tensor, lambd: float = 0.5) -> torch.Tensor:
    return F.hardshrink(x, lambd=lambd)


@_compile
def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() * b.float()).sum()


@_compile
def gemv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.matmul(matrix, vector)


@_compile
def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


@_compile
def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


@_compile
def layer_norm(
    x: torch.Tensor,
    scale: float = 1.0,
    bias: float = 0.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    normalized = F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps)
    return normalized * scale + bias


@_compile
def layer_norm_affine(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


@_compile
def layer_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    columns = x.shape[-1]
    # TODO：mean.reshape(*x.shape[:-1], 1)是什么意思？rstd是什么？
    # mean 每个归一化行一个值；reshape 成 [...,1] 可沿最后维广播。rstd 是 reciprocal standard deviation，即 1/sqrt(variance + eps)。
    normalized = (x.float() - mean.reshape(*x.shape[:-1], 1)) * rstd.reshape(*x.shape[:-1], 1)
    weighted_grad = grad_output.float() * weight.float()
    # TODO：sum中传入的keepdim是什么意思？
    # keepdim=True 保留被归约维度但把长度设为 1，例如 [B,K] 沿 K 求和仍得到 [B,1]，便于后续广播。
    normalized_projection = (normalized * weighted_grad).sum(dim=-1, keepdim=True) / columns
    mean_gradient = weighted_grad.sum(dim=-1, keepdim=True) / columns


    grad_input = ((weighted_grad - normalized * normalized_projection - mean_gradient) * rstd.reshape(*x.shape[:-1], 1)).to(x.dtype)
    flattened_grad = grad_output.float().reshape(-1, columns)
    flattened_normalized = normalized.reshape(-1, columns)
    grad_weight = (flattened_grad * flattened_normalized).sum(dim=0).to(weight.dtype)
    grad_bias = flattened_grad.sum(dim=0).to(weight.dtype)
    return grad_input, grad_weight, grad_bias


@_compile
def rms_norm(x: torch.Tensor, scale: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight=None, eps=eps) * scale


@_compile
def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int8:
        return torch.sum(x, dtype=torch.int32)
    return x.float().sum()


@_compile
def embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.embedding(indices, weight)


@_compile
def matrix_transpose(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 1).contiguous()


@_compile
def rope(x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    sequence_length, hidden_size = x.shape
    # TODO：为什么要变成这个行状？
    # RoPE 按相邻两个 hidden 元素组成二维旋转平面；[seq,hidden] 变为 [seq,hidden/2,2] 后，最后一维就是每个旋转对。
    pairs = x.float().reshape(sequence_length, hidden_size // 2, 2)

    pair_indices = torch.arange(hidden_size // 2, device=x.device, dtype=torch.float32)
    positions = torch.arange(sequence_length, device=x.device, dtype=torch.float32)
    frequencies = torch.exp((-math.log(theta) * 2.0 / hidden_size) * pair_indices)
    angles = positions[:, None] * frequencies[None, :]
    first = pairs[..., 0]
    second = pairs[..., 1]
    output = torch.stack(
        (
            first * torch.cos(angles) - second * torch.sin(angles),
            first * torch.sin(angles) + second * torch.cos(angles),
        ),
        dim=-1,
    )
    return output.reshape_as(x).to(x.dtype)


@_compile
def _histogram_impl(values: torch.Tensor, num_bins: int) -> torch.Tensor:
    output = torch.zeros((num_bins,), device=values.device, dtype=torch.int32)
    ones = torch.ones_like(values, dtype=torch.int32)
    return output.scatter_add(0, values.to(torch.int64), ones)


def histogram(values: torch.Tensor, num_bins: int | None = None) -> torch.Tensor:
    if num_bins is None:
        num_bins = int(values.max().item()) + 1
    return _histogram_impl(values, num_bins)


@_compile
def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=causal,
        scale=scale,
    )


@_compile
def _merge_attention_states_impl(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_infinity = torch.tensor(float("inf"), device=prefix_lse.device)
    negative_infinity = torch.tensor(float("-inf"), device=prefix_lse.device)
    prefix_lse = torch.where(prefix_lse == positive_infinity, negative_infinity, prefix_lse)
    suffix_lse = torch.where(suffix_lse == positive_infinity, negative_infinity, suffix_lse)
    
    output_lse = torch.logaddexp(prefix_lse, suffix_lse)
    prefix_scale = torch.exp(prefix_lse - output_lse).transpose(0, 1).unsqueeze(-1)
    suffix_scale = torch.exp(suffix_lse - output_lse).transpose(0, 1).unsqueeze(-1)
    output = (prefix_output * prefix_scale + suffix_output * suffix_scale).to(prefix_output.dtype)
    return output, output_lse


def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    *,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    output, output_lse = _merge_attention_states_impl(
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
    )
    return (output, output_lse) if return_lse else output


try:
    from torchvision.ops import nms as _torchvision_nms
except (ImportError, RuntimeError):
    _torchvision_nms = None


if _torchvision_nms is not None:
    nms = _compile(_torchvision_nms)
else:
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        raise RuntimeError("torch_compile_backend.nms requires torchvision")
