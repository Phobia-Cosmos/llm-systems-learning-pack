from __future__ import annotations

from typing import Final

import torch
import triton
import triton.language as tl

from ._common import check_cuda_contiguous, check_float_tensor, check_same_shape


_BLOCK_SIZE: Final = 256


@triton.jit
def _binary_add_kernel(a_ptr, b_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, a + b, mask=mask)


@triton.jit
def _activation_kernel(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    parameter: tl.constexpr,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    if OP == 0:
        output = tl.maximum(x, 0.0)
    elif OP == 1:
        output = 1.0 / (1.0 + tl.exp(-x))
    elif OP == 2:
        output = tl.where(x > 0.0, x, parameter * (tl.exp(x) - 1.0))
    elif OP == 3:
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        output = 0.5 * x * (1.0 + 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0)
    elif OP == 4:
        output = x / (1.0 + tl.exp(-x))
    elif OP == 5:
        output = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    else:
        output = tl.where((x >= -parameter) & (x <= parameter), 0.0, x)

    tl.store(output_ptr + offsets, output, mask=mask)


def elementwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    check_cuda_contiguous(a, b)
    check_same_shape(a, b)
    if a.dtype != b.dtype:
        raise TypeError("a and b must have the same dtype")
    output = torch.empty_like(a)
    grid = (triton.cdiv(a.numel(), _BLOCK_SIZE),)
    _binary_add_kernel[grid](a, b, output, a.numel(), BLOCK_SIZE=_BLOCK_SIZE)
    return output


def _activation(x: torch.Tensor, op: int, parameter: float = 0.0) -> torch.Tensor:
    check_cuda_contiguous(x)
    check_float_tensor(x)
    output = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), _BLOCK_SIZE),)
    _activation_kernel[grid](
        x,
        output,
        x.numel(),
        parameter=parameter,
        OP=op,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


def relu(x: torch.Tensor) -> torch.Tensor:
    return _activation(x, 0)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return _activation(x, 1)


def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _activation(x, 2, alpha)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return _activation(x, 3)


def swish(x: torch.Tensor) -> torch.Tensor:
    return _activation(x, 4)


def hardswish(x: torch.Tensor) -> torch.Tensor:
    return _activation(x, 5)


def hardshrink(x: torch.Tensor, lambd: float = 0.5) -> torch.Tensor:
    return _activation(x, 6, lambd)
