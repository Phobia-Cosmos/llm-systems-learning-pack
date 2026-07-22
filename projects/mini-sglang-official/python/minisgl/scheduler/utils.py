from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
# todo：为什么有了Req还需要PendingReq？这两个类型有何区别？
# 解答：PendingReq 是尚在队列中的轻量请求，只保留原始输入和参数；Req 是获准运行后才创建的状态，额外持有 table slot、cache handle 及 cached/device 长度，避免排队请求提前占 GPU 资源。
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    # TODO:这个是什么意思？输出下标是什么？
    # 解答：从类型设计看它原计划携带每个已调度请求对应的输出槽位索引；但 ScheduleResult 当前没有任何调用方，实际实现已由 Batch 和 _make_write_tuple 负责映射，因而该字段目前无运行时作用。
    output_indices: List[torch.Tensor]
