from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)

# TODO：为什么index也需要使用kernel重写？为什么要单独写一个kernel？
# 解答：语义仍是按 token id 取 embedding 行；专用 CUDA kernel 可直接复制整行，并在词表并行时把非本 rank 的 id 写零，省去通用索引、掩码和临时张量的多次 kernel 启动。
@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)

    # TODO：这里的“index”（第一个参数）是什么意思？这些wrapper是什么？函数吗？
    # 解答："index" 是编译模块名/缓存键的一部分；wrapper 元组把 Python 名 launch 映射到 C++ 的 IndexKernel<...>::run，生成的是 FFI 导出函数入口。
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


# TODO：这里的weight和indices都是什么形状的以及output的形状要求是什么？
# 解答：weights 为 [本地词表行数, D]，indices 为 [L]，output 为同设备同 dtype 的 [L, D]；给出 vocab_range 时 indices 是全局 id，范围外的行由 kernel 填零。
def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1

    # TODO：这里返回的是什么？下面调用launch会返回什么结果？
    # 解答：module 是带 launch 方法的 tvm_ffi.Module；launch 异步启动 CUDA kernel、原地填充 output，返回值不用，indexing 最后返回该 output Tensor。
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
