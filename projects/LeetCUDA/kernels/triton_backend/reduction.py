from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import check_cuda_contiguous


_BLOCK_SIZE = 1024
_ELEMENTS_PER_PROGRAM = 4096


@triton.jit
def _reduce_sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    IS_INTEGER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS: tl.constexpr,
):
    program_offset = tl.program_id(0) * ELEMENTS
    lane_offsets = tl.arange(0, BLOCK_SIZE)
    if IS_INTEGER:
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for chunk_offset in range(0, ELEMENTS, BLOCK_SIZE):
        offsets = program_offset + chunk_offset + lane_offsets
        mask = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=mask)
        if IS_INTEGER:
            values = values.to(tl.int32)
        else:
            values = values.to(tl.float32)
        values = tl.where(mask, values, 0)
        accumulator += values
    tl.store(output_ptr + tl.program_id(0), tl.sum(accumulator, axis=0))


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    check_cuda_contiguous(x)
    float8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dtype is not None
    )
    if x.dtype not in (torch.int8, torch.float16, torch.bfloat16, torch.float32, *float8_dtypes):
        raise TypeError(f"unsupported dtype {x.dtype}")
    is_integer = x.dtype == torch.int8
    output_dtype = torch.int32 if is_integer else torch.float32
    current = x
    while current.numel() > 1:
        programs = triton.cdiv(current.numel(), _ELEMENTS_PER_PROGRAM)
        output = torch.empty((programs,), device=x.device, dtype=output_dtype)
        _reduce_sum_kernel[(programs,)](
            current,
            output,
            current.numel(),
            IS_INTEGER=is_integer,
            BLOCK_SIZE=_BLOCK_SIZE,
            ELEMENTS=_ELEMENTS_PER_PROGRAM,
            num_warps=8,
        )
        current = output
    return current.reshape(())
