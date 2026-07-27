from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    # TODO：为什么是下划线命名的函数？作用是什么？
    # 解答：__call__ 是 Python 的特殊方法，表示对象可像函数一样调用；这里用 Protocol 描述“接收 device、返回 PrefixCache”的工厂签名，普通函数也满足该结构类型。
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


# TODO：为什么[CacheManagerCreator]("Cache Manager")这个是什么语法？这个是返回一个_type是Cache Manager的注册点吗？然后泛型是CacheManagerCreator？CACHE_MANAGER是用来管理kvcache的吗？
# 解答：Registry[CacheManagerCreator] 是给类型检查器标明注册项类型，("Cache Manager") 才是实例化并设置报错名称；该注册表保存 PrefixCache 工厂，而不是保存实际 KV 数据。
SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")

# TODO：为什么MHACache和NaivePrefixCache的创建方法不一样？MHA不是Prefix Cache吗？
# 解答：MHAKVCache 按模型形状创建真实 K/V 张量池，Naive/Radix PrefixCache 只管理前缀到槽位的映射；它们是上下两层而非同一种 Cache，因此构造参数和工厂不同。
def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)
    # 解答：当前工厂只实现普通 MHA/GQA 的逐层 K/V 张量布局；MLA 会缓存低秩 latent 与独立 RoPE 分量，shape、每 token 字节数和 attention backend 接口都不同，不能只把类名替换掉，需增加新的 pool、store kernel、容量公式和 backend metadata。

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


# TODO：这个是什么语法？
# 解答：这是装饰器注册语法；register("naive") 返回的装饰器把下方工厂函数存入注册表的 "naive" 键。该装饰器返回 None，因此全局同名会变成 None，预期入口是注册表按名称取出的函数。
@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)

# TODO：为什么这里不使用@SUPPORTED_CACHE_MANAGER.register？
# 解答：该函数是注册表的统一查询/创建入口而不是一种新实现；它用 type 选出先前已注册的工厂并传入 device，所以不应再次注册自己。
def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
