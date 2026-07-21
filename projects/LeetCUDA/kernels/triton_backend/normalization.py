from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import check_cuda_contiguous, check_float_tensor


@triton.jit
def _softmax_kernel(input_ptr, output_ptr, rows, columns, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < columns
    values = tl.load(input_ptr + row * columns + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(output_ptr + row * columns + offsets, numerator / denominator, mask=mask)


@triton.jit
def _layer_norm_kernel(
    input_ptr,
    output_ptr,
    columns,
    scale: tl.constexpr,
    bias: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < columns
    values = tl.load(input_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    output = centered * tl.rsqrt(variance + eps) * scale + bias
    tl.store(output_ptr + row * columns + offsets, output, mask=mask)


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    output_ptr,
    columns,
    scale: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < columns
    values = tl.load(input_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / columns
    output = values * tl.rsqrt(mean_square + eps) * scale
    tl.store(output_ptr + row * columns + offsets, output, mask=mask)


@triton.jit
def _layer_norm_affine_forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    columns,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < columns
    values = tl.load(input_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    rstd = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(output_ptr + row * columns + offsets, centered * rstd * weight + bias, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layer_norm_backward_dx_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    grad_input_ptr,
    columns,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < columns
    grad_output = tl.load(grad_output_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.load(input_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    normalized = tl.where(mask, (values - mean) * rstd, 0.0)
    weighted_grad = grad_output * weight
    normalized_projection = tl.sum(normalized * weighted_grad, axis=0) / columns
    mean_gradient = tl.sum(weighted_grad, axis=0) / columns
    grad_input = (weighted_grad - normalized * normalized_projection - mean_gradient) * rstd
    tl.store(grad_input_ptr + row * columns + offsets, grad_input, mask=mask)


@triton.jit
def _layer_norm_backward_weight_bias_kernel(
    grad_output_ptr,
    input_ptr,
    mean_ptr,
    rstd_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    rows,
    columns,
    BLOCK_ROWS: tl.constexpr,
):
    column = tl.program_id(0)
    row_offsets = tl.arange(0, BLOCK_ROWS)
    mask = row_offsets < rows
    offsets = row_offsets * columns + column
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row_offsets, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + row_offsets, mask=mask, other=0.0)
    normalized = (values - mean) * rstd
    tl.store(grad_weight_ptr + column, tl.sum(grad_output * normalized, axis=0))
    tl.store(grad_bias_ptr + column, tl.sum(grad_output, axis=0))


def _row_shape(x: torch.Tensor) -> tuple[int, int]:
    if x.ndim < 1:
        raise ValueError("expected a tensor with at least one dimension")
    columns = x.shape[-1]
    rows = x.numel() // columns
    if columns == 0:
        raise ValueError("the normalized dimension cannot be empty")
    return rows, columns


def _block_size(columns: int) -> int:
    block_size = triton.next_power_of_2(columns)
    if block_size > 65536:
        raise ValueError("last dimension must be <= 65536")
    return block_size


def softmax(x: torch.Tensor) -> torch.Tensor:
    check_cuda_contiguous(x)
    check_float_tensor(x)
    rows, columns = _row_shape(x)
    output = torch.empty_like(x)
    block_size = _block_size(columns)
    _softmax_kernel[(rows,)](
        x,
        output,
        rows,
        columns,
        BLOCK_SIZE=block_size,
        num_warps=min(8, max(1, block_size // 256)),
    )
    return output


def layer_norm(
    x: torch.Tensor,
    scale: float = 1.0,
    bias: float = 0.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    check_cuda_contiguous(x)
    check_float_tensor(x)
    rows, columns = _row_shape(x)
    output = torch.empty_like(x)
    block_size = _block_size(columns)
    _layer_norm_kernel[(rows,)](
        x,
        output,
        columns,
        scale=scale,
        bias=bias,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=min(8, max(1, block_size // 256)),
    )
    return output


def rms_norm(x: torch.Tensor, scale: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    check_cuda_contiguous(x)
    check_float_tensor(x)
    rows, columns = _row_shape(x)
    output = torch.empty_like(x)
    block_size = _block_size(columns)
    _rms_norm_kernel[(rows,)](
        x,
        output,
        columns,
        scale=scale,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=min(8, max(1, block_size // 256)),
    )
    return output


def _layer_norm_affine_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    check_cuda_contiguous(x, weight, bias)
    check_float_tensor(x)
    rows, columns = _row_shape(x)
    if weight.shape != (columns,) or bias.shape != (columns,):
        raise ValueError("weight and bias must match the last input dimension")
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise TypeError("x, weight, and bias must have the same dtype")
    output = torch.empty_like(x)
    mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    block_size = _block_size(columns)
    _layer_norm_affine_forward_kernel[(rows,)](
        x,
        weight,
        bias,
        output,
        mean,
        rstd,
        columns,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=min(8, max(1, block_size // 256)),
    )
    return output, mean, rstd


def layer_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    check_cuda_contiguous(grad_output, x, weight, mean, rstd)
    rows, columns = _row_shape(x)
    if grad_output.shape != x.shape or weight.shape != (columns,):
        raise ValueError("gradient/input shapes are incompatible")
    if mean.shape != (rows,) or rstd.shape != (rows,):
        raise ValueError("mean and rstd must contain one value per row")
    grad_input = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty_like(weight)
    block_size = _block_size(columns)
    block_rows = triton.next_power_of_2(rows)
    if block_rows > 65536:
        raise ValueError("layer_norm_backward supports at most 65536 rows")
    _layer_norm_backward_dx_kernel[(rows,)](
        grad_output,
        x,
        weight,
        mean,
        rstd,
        grad_input,
        columns,
        BLOCK_SIZE=block_size,
        num_warps=min(8, max(1, block_size // 256)),
    )
    _layer_norm_backward_weight_bias_kernel[(columns,)](
        grad_output,
        x,
        mean,
        rstd,
        grad_weight,
        grad_bias,
        rows,
        columns,
        BLOCK_ROWS=block_rows,
        num_warps=min(8, max(1, block_rows // 256)),
    )
    return grad_input, grad_weight, grad_bias


class _LayerNormAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        output, mean, rstd = _layer_norm_affine_forward(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = layer_norm_backward(grad_output, x, weight, mean, rstd)
        return grad_input, grad_weight, grad_bias, None


def layer_norm_affine(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return _LayerNormAffine.apply(x, weight, bias, eps)
