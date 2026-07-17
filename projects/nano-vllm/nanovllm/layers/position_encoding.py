from __future__ import annotations

import math

import torch
from torch import nn

# 问题（已回答）：固定正弦位置编码的原理是什么？
# 回答：它为位置 pos 构造固定向量：PE(pos,2i)=sin(pos/base^(2i/C))，
# PE(pos,2i+1)=cos(pos/base^(2i/C))。不同维度使用不同波长，向量与 token embedding 相加，
# 使没有顺序概念的注意力获得绝对位置信息；这些值不参与训练。
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, hidden_size: int, base: float = 10000.0):
        super().__init__()
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        # 问题（已回答）：frequencies 表示什么，为什么需要它？
        # 回答：它等于 base^(-2i/C)，为每一对 sin/cos 通道指定角频率。低维旋转较快、高维较慢，
        # 让有限隐藏维同时表达短距离和长距离位置变化。
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(base) / hidden_size)
        )
        table = torch.zeros(max_seq_len, hidden_size, dtype=torch.float32)
        # 问题（已回答）：两次切片和 sin/cos 表是怎样计算的？
        # 回答：positions [S,1] 与 frequencies [ceil(C/2)] 广播成每个“位置-频率”的角度。
        # 0::2 表示第 0、2、4... 个偶数通道写 sin；1::2 表示第 1、3、5... 个奇数通道写 cos。
        # 奇数 hidden_size 时，cos 侧少一列，所以用实际切片宽度截断 frequencies。
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # 问题（已回答）：forward 中的 index_select 和 view 在做什么？
        # 回答：先把任意形状的 position id 拉平成一维，从预计算表按行查出对应向量，再恢复为
        # positions.shape + [hidden_size]；例如 [N] 变成 [N,C]，[B,T] 变成 [B,T,C]。
        return self.table.index_select(0, positions.reshape(-1)).view(*positions.shape, -1)

# 问题（已回答）：get_alibi_slopes 在做什么，ALiBi 原理是什么？
# 回答：ALiBi 不向隐藏状态加入位置向量，而是在每个注意力头的 score 上加 m_h*(k_pos-q_pos)。
# 因果注意力中历史 key 的差值为负，因此越远惩罚越大；不同 head 使用不同斜率 m_h，覆盖多种距离尺度。
# 该函数按论文参考实现生成这些固定斜率，并处理 head 数不是 2 的幂的情况。
def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    closest_power_of_two = 2 ** math.floor(math.log2(num_heads))
    # 问题（已回答）：base 公式在做什么，为什么出现 -3？
    # 回答：它生成按几何级数衰减的 head slope。常数 3 来自 ALiBi 参考实现的经验标定，
    # 使 2 的幂个 head 的斜率大致覆盖从较强局部偏置到约 1/256 的缓慢偏置；它不是训练推导出的必然常数。
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_two) - 3)))
    slopes = torch.pow(
        torch.tensor(base, dtype=torch.float32),
        torch.arange(1, closest_power_of_two + 1, dtype=torch.float32),
    )
    if closest_power_of_two == num_heads:
        return slopes
    extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_two) - 3)))
    remaining = num_heads - closest_power_of_two
    extra_powers = torch.arange(1, 2 * remaining + 1, 2, dtype=torch.float32)
    return torch.cat((slopes, torch.pow(torch.tensor(extra_base), extra_powers)))
