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
# 解答：Engine 拥有一个 TP rank 的模型分片、KV cache、attention/sampling 后端和 CUDA stream；当前启动方式通常是每个 GPU 一个 Scheduler 进程和一个 Engine。
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # TODO：代表我们的sglang最多一次性处理256请求是吗？每一个请求的大小有限制吗？
    # 解答：它限制同时占用 table slot/KV 资源的运行中请求数，不是每个 forward 必须有 256 条；单请求长度还受 max_seq_len 和可用 KV 容量限制。
    max_running_req: int = 256
    
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    # TODO：这个指的是kvcache的page大小是吗？
    # 解答：是，它表示 KV cache 一个物理页包含的 token 槽位数，也是分配、回收和 radix 匹配的对齐粒度。
    page_size: int = 1
    # TODO：GPU内存使用率的最大值吗？
    # 解答：它是按加载模型前可用显存估算“模型 + KV cache”目标预算的比例，用于推导页数，不是对进程的实时硬性显存上限。
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    # TODO：这个dummy weight是什么以及何时会使用？
    # 解答：开启后用同形状随机张量代替真实 checkpoint，用于不关心输出语义的启动、性能或内核测试。
    use_dummy_weight: bool = False
    use_pynccl: bool = True

    # TODO：这两个属性分别代表什么意思？
    # 解答：max_seq_len_override 覆盖模型声明的最大序列长度；num_page_override 覆盖按显存自动估算的 KV cache 页数。
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    # TODO：这个修饰符的作用是什么 会存储在哪里？CPU还是GPU？
    # 解答：@cached_property 在首次访问时计算并把结果缓存到该 Python 实例中；这里缓存的是 CPU 上的 Hugging Face 配置对象，不是 GPU 张量。
    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    # TODO：这个返回的是要存储的有关位置编码相关的信息是吗？
    # 解答：不是位置编码内容，而是允许的最大逻辑序列长度；未覆盖时借用 rotary_config.max_position 作为模型上限。
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
