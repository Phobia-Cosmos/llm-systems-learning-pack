from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import check_cuda_contiguous, check_float_tensor


@triton.jit
def _embedding_kernel(indices_ptr, weight_ptr, output_ptr, num_indices, embedding_dim, BLOCK_SIZE: tl.constexpr):
    index_row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < embedding_dim
    index = tl.load(indices_ptr + index_row).to(tl.int64)
    values = tl.load(weight_ptr + index * embedding_dim + offsets, mask=mask)
    tl.store(output_ptr + index_row * embedding_dim + offsets, values, mask=mask)


@triton.jit
def _transpose_kernel(
    input_ptr,
    output_ptr,
    rows,
    columns,
    BLOCK: tl.constexpr,
):
    block_row = tl.program_id(0)
    block_column = tl.program_id(1)
    row_offsets = block_row * BLOCK + tl.arange(0, BLOCK)
    column_offsets = block_column * BLOCK + tl.arange(0, BLOCK)
    input_offsets = row_offsets[:, None] * columns + column_offsets[None, :]
    mask = (row_offsets[:, None] < rows) & (column_offsets[None, :] < columns)
    tile = tl.load(input_ptr + input_offsets, mask=mask)
    output_offsets = column_offsets[:, None] * rows + row_offsets[None, :]
    tl.store(output_ptr + output_offsets, tl.trans(tile), mask=tl.trans(mask))


@triton.jit
def _rope_kernel(input_ptr, output_ptr, sequence_length, hidden_size, theta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    token = tl.program_id(0)
    pair_offsets = tl.arange(0, BLOCK_SIZE)
    pair_count = hidden_size // 2
    mask = pair_offsets < pair_count
    first_offsets = token * hidden_size + pair_offsets * 2
    first = tl.load(input_ptr + first_offsets, mask=mask).to(tl.float32)
    second = tl.load(input_ptr + first_offsets + 1, mask=mask).to(tl.float32)
    frequency = tl.exp(-tl.log(theta) * (2.0 * pair_offsets / hidden_size))
    angle = token * frequency
    cosine = tl.cos(angle)
    sine = tl.sin(angle)
    tl.store(output_ptr + first_offsets, first * cosine - second * sine, mask=mask)
    tl.store(output_ptr + first_offsets + 1, first * sine + second * cosine, mask=mask)


@triton.jit
def _histogram_kernel(values_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(values_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(output_ptr + values, 1, mask=mask)


def embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    check_cuda_contiguous(indices, weight)
    check_float_tensor(weight)
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be int32 or int64")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [vocabulary, embedding_dim]")
    output = torch.empty((*indices.shape, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    block_size = triton.next_power_of_2(weight.shape[1])
    if block_size > 65536:
        raise ValueError("embedding dimension must be <= 65536")
    _embedding_kernel[(indices.numel(),)](
        indices,
        weight,
        output,
        indices.numel(),
        weight.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=min(8, max(1, block_size // 256)),
    )
    return output


def matrix_transpose(x: torch.Tensor) -> torch.Tensor:
    check_cuda_contiguous(x)
    if x.ndim != 2:
        raise ValueError("matrix_transpose expects a rank-2 tensor")
    rows, columns = x.shape
    output = torch.empty((columns, rows), device=x.device, dtype=x.dtype)
    block = 32
    _transpose_kernel[(triton.cdiv(rows, block), triton.cdiv(columns, block))](
        x,
        output,
        rows,
        columns,
        BLOCK=block,
        num_warps=8,
    )
    return output


def rope(x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    check_cuda_contiguous(x)
    check_float_tensor(x)
    if x.ndim != 2 or x.shape[1] % 2:
        raise ValueError("rope expects [sequence_length, even_hidden_size]")
    sequence_length, hidden_size = x.shape
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(hidden_size // 2)
    _rope_kernel[(sequence_length,)](
        x,
        output,
        sequence_length,
        hidden_size,
        theta=theta,
        BLOCK_SIZE=block_size,
        num_warps=min(8, max(1, block_size // 256)),
    )
    return output


def histogram(values: torch.Tensor, num_bins: int | None = None) -> torch.Tensor:
    check_cuda_contiguous(values)
    if values.dtype != torch.int32:
        raise TypeError("histogram expects int32 values")
    if num_bins is None:
        num_bins = int(values.max().item()) + 1
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    output = torch.zeros((num_bins,), device=values.device, dtype=torch.int32)
    block_size = 256
    _histogram_kernel[(triton.cdiv(values.numel(), block_size),)](
        values,
        output,
        values.numel(),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output
