from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import check_cuda_contiguous, check_float_tensor


@triton.jit
def _dot_partial_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS: tl.constexpr,
):
    program_offset = tl.program_id(0) * ELEMENTS
    lane_offsets = tl.arange(0, BLOCK_SIZE)
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for chunk_offset in range(0, ELEMENTS, BLOCK_SIZE):
        offsets = program_offset + chunk_offset + lane_offsets
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        accumulator += a * b
    tl.store(output_ptr + tl.program_id(0), tl.sum(accumulator, axis=0))


@triton.jit
def _gemv_kernel(
    matrix_ptr,
    vector_ptr,
    output_ptr,
    rows,
    columns,
    stride_matrix_row,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    column_offsets = tl.arange(0, BLOCK_K)
    mask = column_offsets < columns
    matrix = tl.load(
        matrix_ptr + row * stride_matrix_row + column_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    vector = tl.load(vector_ptr + column_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + row, tl.sum(matrix * vector, axis=0))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    m,
    n,
    k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program_id = tl.program_id(0)
    programs_n = tl.cdiv(n, BLOCK_N)
    program_m = program_id // programs_n
    program_n = program_id % programs_n

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] + k_offset * BLOCK_K < k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offsets_k[:, None] + k_offset * BLOCK_K < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    output_ptrs = output_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on
    tl.store(output_ptrs, accumulator, mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n))


def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    check_cuda_contiguous(a, b)
    check_float_tensor(a)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("a and b must have identical shapes and dtypes")
    block_size = 1024
    elements = 4096
    programs = triton.cdiv(a.numel(), elements)
    partial = torch.empty((programs,), device=a.device, dtype=torch.float32)
    _dot_partial_kernel[(programs,)](
        a,
        b,
        partial,
        a.numel(),
        BLOCK_SIZE=block_size,
        ELEMENTS=elements,
        num_warps=8,
    )
    from .reduction import reduce_sum

    return reduce_sum(partial)


def gemv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    check_cuda_contiguous(matrix, vector)
    check_float_tensor(matrix)
    if matrix.ndim != 2 or vector.numel() != matrix.shape[1]:
        raise ValueError("expected matrix [M, K] and vector [K] or [K, 1]")
    if matrix.dtype != vector.dtype:
        raise TypeError("matrix and vector must have the same dtype")
    rows, columns = matrix.shape
    output = torch.empty((rows,), device=matrix.device, dtype=matrix.dtype)
    block_k = triton.next_power_of_2(columns)
    if block_k > 65536:
        raise ValueError("gemv supports K <= 65536")
    _gemv_kernel[(rows,)](
        matrix,
        vector,
        output,
        rows,
        columns,
        matrix.stride(0),
        BLOCK_K=block_k,
        num_warps=min(8, max(1, block_k // 256)),
    )
    return output[:, None] if vector.ndim == 2 else output


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    check_cuda_contiguous(a, b)
    check_float_tensor(a)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible rank-2 matrices")
    if a.dtype != b.dtype:
        raise TypeError("a and b must have the same dtype")
    m, k = a.shape
    _, n = b.shape
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    _gemm_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
    )
    return output
