from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    # TODO:为什么获取的是shape 0?kcache的不同维度分别代表什么意思?
    # 解答：传到这里的是单层缓存 [物理 token 槽数, 本地 KV 头数, head_dim]，所以 shape[0] 是可寻址槽数；完整池在外层还有 K/V、layer、page、page 内位置等维度。
    num_tokens = k_cache.shape[0]
    # TODO：为什么要传递这个形状的view？
    # 解答：kernel 把每个槽的一整行当作连续字节块复制；展平尾部维度为 [槽数, 每槽元素数] 不改数据，只统一了 FFI 的二维形状和步长检查。
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    # TODO：shape1表示的又是什么？k_cache的element size是什么？
    # 解答：shape[1] 是每个 token 的本地 KV 头数乘 head_dim；element_size 是这一整行占用的字节数，而不是单个标量的字节数。
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
