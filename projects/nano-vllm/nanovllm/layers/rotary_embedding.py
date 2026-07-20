from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 问题（已回答）：chunk(2) 是按奇偶维拆分吗？
    # 回答：不是。这里沿最后一维拆成连续的前半和后半，这是 NeoX/Llama 常用的 half-split RoPE 约定；
    # 若要奇偶配对需要 0::2/1::2 切片。最后一维必须为偶数，否则两半宽度不同，后续旋转运算无法对应。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        # 问题（已回答）：RoPE 的 base 有什么作用？
        # 回答：base（常称 theta）控制各维旋转频率和波长；base 越大，高维旋转越慢，通常可覆盖更长位置。
        # 它是模型训练配置的一部分，加载 checkpoint 时必须一致，不能在推理时随意修改。
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # 问题（已回答）：为什么 rotary_dim 必须等于 head_size？
        # 回答：这个精简实现会旋转整个 attention head，并且没有“只旋转前 rotary_dim、其余维原样拼回”的分支，
        # 所以要求相等。更通用的实现可以让 rotary_dim < head_size，但要显式保留未旋转的尾部。
        assert rotary_dim == head_size
        # 问题（已回答）：inv_freq、t、freqs 分别是什么，为什么除以 rotary_dim？
        # 回答：inv_freq[i]=base^(-2i/rotary_dim) 是每个二维旋转平面的角频率；除以 rotary_dim
        # 将指数均匀铺满整个旋转维度。t 是位置 0..max_position-1；einsum("i,j->ij") 在这里是外积，
        # 得到 freqs[position,dim_pair]=position*inv_freq，随后对它取 cos/sin 作为旋转系数。

        # 问题（已回答）：这里的 arange 会生成 rotary_dim/2 个数吗，步长 2 是什么，各元素相同吗？
        # 回答：rotary_dim 为偶数时，arange(0, rotary_dim, 2) 生成 [0,2,...,rotary_dim-2]，长度正好
        # 是 rotary_dim/2；第三个位置参数 2 是步长。各元素不同，所以除以 rotary_dim 并以 base 为底
        # 计算后，会得到从快到慢的不同旋转频率，而不是一组相同数值。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        cos = freqs.cos()
        sin = freqs.sin()
        # 问题（已回答）：为什么 unsqueeze，输入参数有什么要求？
        # 回答：cos/sin 拼接后是 [max_position, head_size]，unsqueeze(1) 变成 [S,1,D]，从而能沿 head 维
        # 广播到 query/key 的 [N,H,D]。head_size/rotary_dim 必须相等且为偶数，base 为正，position id 必须小于 S；
        # query 和 key 的 token 维要与 positions 对应，最后一维必须为 D。

        # 问题（已回答）：为什么要沿 head 维广播，[S,1,D] 与 [N,H,D] 中的 S、N、H、D 是什么？
        # 回答：S 是预计算支持的最大位置数，N 是本轮打包后的 token 数，H 是本 rank 的 attention head 数，
        # D 是每个 head 的维度。相同 token 位置应对所有 head 使用同一组旋转角，因此缓存保留大小为 1 的 head
        # 轴即可同时广播到 Q heads 和数量可能不同的 KV heads，也避免为每个 head 重复存表。
        # 问题（已回答）：为什么 position id 必须小于 S，query/key 的 token 维为什么要与 positions 对应？
        # 回答：forward 用 positions 直接索引缓存的 S 行，越界 id 没有可用的 cos/sin；每个 positions[n]
        # 都描述 query[n] 和 key[n] 的绝对位置，所以三者的 N 维必须一一对应，才能给每个 token 施加正确旋转。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 问题（已回答）：为什么注册 buffer，它的生命周期是什么？
        # 回答：cos/sin 表不是可训练参数，但必须随 Module 一起迁移 device/dtype，并能被 PyTorch 发现，因此注册为 buffer。
        # 它与 RotaryEmbedding 实例同生命周期；persistent=False 表示不写入 state_dict，因为可由构造参数重新计算。
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# 问题（已回答）：@lru_cache(1) 的作用是什么？
# 回答：它按 get_rope 的四个参数缓存最近一个 RotaryEmbedding 实例；相同配置再次调用可复用预计算 cos/sin 表，
# 避免每层重复构造。缓存的是 Module 实例本身，maxsize=1 表示换一组配置后旧缓存会被淘汰。
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
