from __future__ import annotations

import torch


SUPPORTED_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def check_cuda_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError("Triton backend requires CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("Triton backend requires contiguous tensors")


def check_same_shape(*tensors: torch.Tensor) -> None:
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("all tensors must have the same shape")


def check_float_tensor(tensor: torch.Tensor) -> None:
    if tensor.dtype not in SUPPORTED_FLOAT_DTYPES:
        raise TypeError(f"expected a floating tensor, got {tensor.dtype}")
