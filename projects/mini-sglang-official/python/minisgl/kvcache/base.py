from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch

# TODO：BaseKVCachePool和BasePrefixCache是并列等级的两类Cache吧？
# 解答：二者是协作但职责正交的抽象：KVCachePool 保存各层真实 K/V 张量，PrefixCache 保存 token 前缀到 KV 槽位的映射并负责复用、锁定和驱逐。
class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    # TODO：out_loc这个是什么 指向的是kvcache？layer id表示存储的是哪一层的kvcache是吗？
    # 解答：out_loc 是每个新 token 在扁平 KV 池中的物理槽位下标；layer_id 确实选择模型的哪一层 K/V 缓冲区。
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

# TODO：为什么一定需要这样一个类 在成熟的sglang中 这个类的作用不会只有这么一点吧？还有哪些额外的功能？handle和直接的类有何区别？
# 解答：Handle 是一次前缀匹配得到的轻量引用/租约，携带 cached_len 并隐藏 Radix 节点等内部结构，调度器可用统一接口锁定和取索引；完整系统还可让它表示 GPU、主机或分层缓存中的命中。
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class SizeInfo(NamedTuple):
    # TODO:难道kvcache中存储的就只有这两类数据？
    # 解答：这两个字段不是 KV 数据类型，而是 PrefixCache 管理的槽位按“可驱逐”和“被活动请求保护”划分的容量统计；真实 K/V 仍在 KVCachePool 中。
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # TODO: support HiCache 这个Hicache是什么？还有为什么要有一个cuda handle？这个handle是什么？
    # 解答：HiCache 是把 GPU KV 与主机/更低层存储组合的分层缓存；当前结果只有 cuda_handle，表示本次 GPU 前缀命中的引用，未来可再携带 host handle。

# TODO：这里的prefix存储的数据类型是什么，前缀树吗？mha以及naive、radix都会基于这个BasePrefix吗？
# 解答：PrefixCache 存的是“token 序列片段 -> KV 物理槽位索引”的元数据；Radix 用压缩前缀树实现，Naive 不保存映射，而 MHAKVCache 是物理 K/V 池，不继承它。
class BasePrefixCache(ABC):
    @abstractmethod
    # TODO：处于lcoked状态的handle是否是可读的？
    # 解答：可以且正应在锁定后读取；锁不是互斥读锁，而是提高引用计数，保证 handle 对应节点在使用期间不会被驱逐。
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """

    @abstractmethod
    # TODO：为什么evict 0 is always safe and does nothing.？actual evict size may be larger than the requested size.？这里只是给了size evict策略是什么？
    # 解答：请求 0 无需释放任何节点；Radix 按最久未使用的可驱逐叶节点整段删除，节点长度不可拆时会超过目标 size，因此返回真实被释放的全部槽位索引。
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    # TODO：这个会被重置到什么状态，所有的cache全部清空？
    # 解答：接口语义是清空前缀索引及其容量/引用统计，回到刚初始化的空状态；本项目 Radix 实现目前仍明确抛出 NotImplementedError。
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""

    @abstractmethod
    # TODO：何时cache会被corrupted？
    # 解答：这些情况在语义上都属于损坏，例如节点父子关系、key/value 长度、ref_count 与容量统计不一致，或槽位被重复释放/同时占用；但当前 Radix/Naive 的 check_integrity 多为空实现，CacheManager 只做总页数和对齐检查，尚未全面验证内部结构。
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
