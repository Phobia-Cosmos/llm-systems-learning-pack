from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    # TODO:为什么传入的是batch size？什么叫做last indices？为什么MetaData只有这一个函数，连属性都没有？
    # 解答：bs 是真实请求数（批次可能还含 CUDA Graph 的 dummy 请求）；返回每个真实请求最后一个 query token 在扁平输出中的下标，LM Head 只需这些位置。这里只定义跨后端接口，FA/FI 的具体 Metadata 自有字段。
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    # TODO：准备哪些metadata,给谁准备metadata？
    # 解答：它把序列长度、累积长度、页表等整理成当前注意力内核需要的格式并写入 batch.attn_metadata，随后由该后端的 forward 消费。
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        # TODO：难道使用这个backend时还要传入两个抽象函数？
        # 解答：传入的是两个已经实现 BaseAttnBackend 接口的具体后端对象，不是抽象函数；HybridBackend 据阶段把 prefill 和 decode 分派给不同实现。
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    # TODO：为什么下面三个都是decode才可以进行的？也就是说只要是decode阶段 这三个函数是一定会执行的？
    # 解答：本项目只为形状稳定的 decode 捕获 CUDA Graph，所以这些操作委托给 decode 后端；init/capture 在启动时按批大小执行，replay 也只在批次满足图条件时执行，并非每次 decode 都执行三者。
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
