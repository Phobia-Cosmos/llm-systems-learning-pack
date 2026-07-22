import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


# TODO：为什么不能直接实现NaiveCache而是需要一个Handle？
# 解答：调度器只依赖统一的 BaseCacheHandle；NaiveHandle 用“cached_len=0、索引为空”表示永不命中，使 Naive 与 Radix 能走同一套匹配、锁定和释放流程。
class NaiveCacheHandle(BaseCacheHandle):
    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        super().__init__(cached_len=0)

    # TODO：为什么这里返回的是empty_tensor而不是一个indice？
    # 解答：接口返回的是整个命中前缀对应的一维槽位索引张量；Naive 从不复用前缀，所以正确结果是长度为 0 的张量，而不是某个标量下标。
    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor

# TODO：为什么有些函数这个类都是pass实现？
# 解答：NaivePrefixCache 是“禁用前缀缓存”的空实现，没有树、引用计数或可驱逐条目，因此 lock/reset/check_integrity 无状态可改，但保留方法以满足统一接口。
class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # TODO：为什么要set一下？目的是什么？
        # 解答：把当前 device 上的空张量设为 Handle 的类属性，使无状态 Handle 无需逐个保存 device，也能返回设备正确的空索引张量。
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    # TODO：为什么这里不实现lock？
    # 解答：NaiveHandle 不引用任何缓存节点，没有内容会被驱逐，因而锁定和解锁都是无操作。
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    # TODO：这个函数的作用是什么？返回的是什么？为什么要把NaiveCacheHandle放在返回里面？
    # 解答：match_prefix 应返回可复用前缀；Naive 永远返回 cached_len=0 的 Handle，放进 MatchResult 是为了与 Radix 的返回协议一致。
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle())

    # TODO：为什么这里返回也是Handle？为什么不使用输入参数input ids和indice？
    # 解答：Naive 模式刻意不记录前缀和槽位，所以参数无需使用；返回 InsertResult(0, 空 Handle) 表示插入前没有任何已缓存部分，之后也没有可复用引用。
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle())

    # TODO：如果不支持驱逐 那么如果存满了怎么办？
    # 解答：物理页满时 CacheManager 会尝试调用 evict；Naive 没有已保存前缀可回收，非零驱逐会明确报错，因此请求不能继续分配，而不会覆盖仍在使用的 KV。
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    # TODO：为什么返回的大小永远都是0吗？
    # 解答：是；这里只统计 PrefixCache 持有的可复用/受保护前缀，Naive 不持有任何前缀。正在运行请求占用的页由 CacheManager.free_slots 的减少体现。
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
