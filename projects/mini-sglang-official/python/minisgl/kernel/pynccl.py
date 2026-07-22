from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    # TODO：这个class中并没有property 为什么后续init时返回的cls要传入参数？
    # 解答：这里仅为静态类型检查描述运行时 FFI 对象的方法接口，并不定义其构造器；真正的 cls 代理 C++ NCCLWrapper，其已注册的 __init__ 接受 rank、world size、buffer 大小和 UID。
    class PyNCCLCommunicator:
        @abstractmethod
        # TODO：为什么还需要额外传入一个op参数？
        # 解答：all-reduce 还必须指定如何合并各 rank 的元素；当前调用方只传 "sum"，底层还映射了 prod/max/min/avg，故接口保留 op。
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        # TODO：这个是返回buffer的size是吗？
        # 解答：不是；C++ get_buffer() 返回内部对称通信缓冲区的指针，FFI 在 Python 侧以整数/句柄表示，大小由初始化时的 max_size_bytes 决定。
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


# TODO：这个又是什么缓存？
# 解答：functools.cache 缓存的是本进程已编译并加载的 tvm_ffi.Module，避免每次初始化都重复编译、链接和 dlopen；它不是 KV cache 或 NCCL 数据缓冲区。
@functools.cache
def _load_nccl_module() -> Module:
    # TODO：这个是在做什么？
    # 解答：它编译/加载 pynccl.cu，并链接系统 NCCL 库；返回模块暴露 create_nccl_uid 等 C++ 符号。
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    # TODO 这个是一个固定的写法是吗？为什么一定要实现一个PyNCCLImpl？
    # 解答：这是 TVM-FFI 对象绑定模式：注册键必须与 C++ 的类型键一致；PyNCCLImpl 是 Python 代理，负责调用 __ffi_init__ 构造 C++ 对象并暴露其方法，并非 NCCL 算法的另一份实现。
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    # TODO：到这里是初始化出来这个类 后续如何使用呢？
    # 解答：此处只取得代理类，真正实例在函数末尾创建；它随后被包装进 PyNCCLDistributedImpl，供线性层的 all_reduce/all_gather 调用。
    cls = _get_pynccl_wrapper_cls()

    # TODO：为什么tp rank不为0时也要做广播？
    # 解答：broadcast_object_list 是所有 rank 都必须参加的 collective；rank 0 提供 UID，其余 rank 进入同一次调用接收 UID，否则其他进程会一直等待。
    if tp_rank == 0:
        # TODO：这个会返回什么东西？
        # 解答：create_nccl_uid() 返回底层 ncclGetUniqueId 生成的唯一通信 ID 的 FFI 字节数组；列表只是 broadcast_object_list 可原地改写的容器。
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    # TODO：这里返回的是什么类型？为什么要返回这个 以及后续谁会使用？
    # 解答：运行时返回 PyNCCLImpl（静态视作 PyNCCLCommunicator），其中持有 C++ NCCL communicator 和缓冲区；distributed.impl 会保存它并执行张量并行 collective。
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
