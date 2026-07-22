from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    # TODO：为什么要这样判断？
    # 解答：temperature <= 0 或 top_k == 1 都会去掉随机选择，top_p == 1 则表示不启用 nucleus 截断，因此可走 argmax 快速路径。
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    # TODO：这个对应的是kvcache的映射下标吗？
    # 解答：它是 TableManager 分配的请求行号，用来索引 token_pool 和 page_table；page_table 的表项才继续映射到物理 KV cache 槽位。
    table_idx: int

    # TODO：这个值是如何获取的？是否要提前查询kvcache然后匹配？
    # 解答：是的，PrefillAdder 先通过 CacheManager.match_req 做前缀匹配，再从返回的 cache handle 取出已复用的 token 数。
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        # TODO：为什么要判断是否是cpu的输入？
        # 解答：input_ids 是请求的主机端状态，后续会在 CPU 上追加输出并从 pinned memory 异步拷贝；GPU 上另有 token_pool 副本。
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        # TODO：为什么cached len长度要比这两者小？
        # 解答：cached_len 只能是当前序列的已缓存前缀，且新批次必须至少有一个 token 待计算；device_len 又不能超过“输入+生成预算”。
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    # TODO：这个函数是在做什么？为什么令cache=device然后在自增就可以完成一个？这里完成的是什么？
    # 解答：一次 forward 后，原 device_len 范围的 KV 已写入缓存，故先令 cached_len 追上它；再为本次新采样的 token 将逻辑序列长度加一。
    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    # TODO：为什么cat不需要写出拼接的维度？decode过程中每完成一个都要append一次吧？
    # 解答：torch.cat 的 dim 默认为 0，这里两者都是一维 token 序列；每个非 chunked 请求在每次产生 token 后都会追加一次。
    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    # TODO：这个函数的作用是什么？为什么要写成__？
    # 解答：__repr__ 是 Python 的特殊方法，repr(req)、交互式显示和容器/日志调试时会用它生成明确的对象表示。
    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)

    # TODO：这个是什么？由谁来赋值的？这里指的是kvcache的地址是吗？
    # 解答：Scheduler._prepare_batch 通过 page_table 为当前 input token 计算并赋值物理 KV 槽位索引；它是整数位置，不是原始指针地址。
    out_loc: torch.Tensor = field(init=False)
    # TODO：什么是padded_reqs？为什么会出现这种req？
    # 解答：GraphRunner 为复用固定 batch shape 的 CUDA Graph，会在真实 reqs 后补 dummy_req；padded_reqs 是补齐后供图和 attention metadata 使用的列表。
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    # TODO：这里的page指的是kvcache吗还是其他存储区域的page？
    # 解答：这里专指 paged KV cache 的页大小，即一个物理 KV 页容纳的 token 数，不是操作系统内存页。
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)

    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        # TODO：为什么batch为None就代表是nested batch？
        # 解答：恰好相反，_batch is None 表示当前没有活动 batch；若它已非 None 还再进入此上下文，才是会覆盖全局状态的嵌套调用。
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
