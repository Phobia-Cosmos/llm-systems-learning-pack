from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
# TODO：一个engine的作用是什么？一个GPU内部一个engine是吗？
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # TODO：代表我们的sglang最多一次性处理256请求是吗？每一个请求的大小有限制吗？
    max_running_req: int = 256
    
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    # TODO：这个指的是kvcache的page大小是吗？
    page_size: int = 1
    # TODO：GPU内存使用率的最大值吗？
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    # TODO：这个dummy weight是什么以及何时会使用？
    use_dummy_weight: bool = False
    use_pynccl: bool = True

    # TODO：这两个属性分别代表什么意思？
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    # TODO：这个修饰符的作用是什么 会存储在哪里？CPU还是GPU？
    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    # TODO：这个返回的是要存储的有关位置编码相关的信息是吗？
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
