from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        # TODO：这里的shape 0是什么？为什么要放大？
        # 解答：shape[0] 是张量的第 0 维；all_gather_into_tensor 会把每个 rank 的输入沿该维拼接，因此输出需预留 tp_size 倍的长度。
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    # TODO:这个PyNCCLCommunicator作用是什么？这个在参数传入时会自动赋值（@dataclass的功能）？
    # 解答：它封装本 rank 的 NCCL collective 通信；@dataclass 生成的 __init__(comm) 会把传入对象赋给该字段。
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size

        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    # TODO：TorchDistributedImpl为什么会有一个()？这个不是调用的意思吗？这里的plugin是用来做什么的？主要就是实现reduce和gather？
    # 解答：TorchDistributedImpl() 是实例化默认后端；plugins 是同一 all_reduce/all_gather 接口的后端栈，便于之后切换到 PyNCCL。
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    # TODO：为什么要使用最后一个plugin？其他的不可以使用吗？
    # 解答：新启用的后端会 append 到末尾，取 [-1] 就是让最新后端覆盖默认实现；旧对象仍在列表中，但当前不会被调用。
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


# TODO：为什么pynccl_distributed需要额外开启 那torch版本的不需要？
# 解答：Torch 实现已预先注册，只需 init_process_group；PyNCCL 还需在 rank/group/device 已知后创建 communicator 和通信缓冲区，所以显式启用。
def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
