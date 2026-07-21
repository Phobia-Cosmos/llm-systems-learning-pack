"""PyTorch expressions compiled with TorchInductor.

This module mirrors :mod:`kernels.triton_backend`.  Each public callable is a
``torch.compile`` wrapper around a pure PyTorch implementation, so the first
call compiles for the observed shapes and later calls reuse the generated code.
"""

from .ops import (
    dot_product,
    elementwise_add,
    elu,
    embedding,
    flash_attention,
    gelu,
    gemm,
    gemv,
    hardshrink,
    hardswish,
    histogram,
    layer_norm,
    layer_norm_affine,
    layer_norm_backward,
    matrix_transpose,
    merge_attention_states,
    nms,
    reduce_sum,
    relu,
    rms_norm,
    rope,
    sigmoid,
    softmax,
    swish,
)

__all__ = [
    "dot_product",
    "elementwise_add",
    "elu",
    "embedding",
    "flash_attention",
    "gelu",
    "gemm",
    "gemv",
    "hardshrink",
    "hardswish",
    "histogram",
    "layer_norm",
    "layer_norm_affine",
    "layer_norm_backward",
    "matrix_transpose",
    "merge_attention_states",
    "nms",
    "reduce_sum",
    "relu",
    "rms_norm",
    "rope",
    "sigmoid",
    "softmax",
    "swish",
]
