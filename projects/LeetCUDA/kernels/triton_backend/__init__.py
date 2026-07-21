"""Reusable Triton implementations for the CUDA examples in ``kernels``.

The package exposes one implementation per mathematical operation.  The CUDA
tree often contains several scalar, packed, tiled, or pipelined versions of the
same operation; those variants map to the same public function here because
Triton chooses vectorization and instruction selection during compilation.
"""

from .attention import flash_attention, merge_attention_states, nms
from .elementwise import (
    elementwise_add,
    elu,
    gelu,
    hardshrink,
    hardswish,
    relu,
    sigmoid,
    swish,
)
from .indexing import embedding, histogram, matrix_transpose, rope
from .linear import dot_product, gemm, gemv
from .normalization import layer_norm, layer_norm_affine, layer_norm_backward, rms_norm, softmax
from .reduction import reduce_sum

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
