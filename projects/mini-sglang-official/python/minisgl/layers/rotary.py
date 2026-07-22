from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


# TODO：为什么我们的emdedding是StateLessOP？
# 解答：这里是 RotaryEmbedding。它没有 checkpoint 中学习得到的参数；inv_freq 和 cos/sin 都由配置确定，且缓存放在以下划线开头的运行时字段中，所以按项目约定属于 StateLessOP。
class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        # TODO：为什么还存在一个post_process？一般都是处理什么东西？
        # 解答：post_process 是可选的频率变换钩子，在生成 cos/sin cache 前修改 inv_freq；Llama 3、YaRN 等长上下文方案借此复用同一 RoPE 主流程。
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        if post_process is not None:
            inv_freq = post_process(inv_freq)

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)

        # TODO：为什么要加上这一行 为什么head size必须是这几个？
        # 解答：这是 FlashInfer 当前 apply_rope kernel 已实例化/支持的 head_size 集合，不是 RoPE 数学本身只能取这些值；提前断言可避免进入 kernel 后失败。
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            # TODO：这几个factor分别代表什么意思？
            # 解答：factor 是低频整体缩放倍率；low/high_freq_factor 用原训练窗口内的旋转次数划出低频、过渡、高频区域；original_max_position 是计算这些边界的原上下文长度。
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            # TODO：这里处理的原理是什么？公式以及为什么要这样做？
            # 解答：先以 wavelength=2*pi/inv_freq 衡量频率：高频保持原值、低频用 inv_freq/factor 拉长波长，中间区按 smooth 线性插值，从而扩展长程位置又尽量保留短程分辨率。
            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta

                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                # TODO：这是在做什么？原理公式是什么？为什么要这样做？
                # 解答：它反解 RoPE 频率公式，求在原窗口内恰好旋转 num_rotations 圈所对应的频率维下标；beta_fast/beta_slow 得到过渡区两端，再对该区做 ramp 混合。
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    # TODO：为什么是一个Tuple[Tuple[str, Any], ...]类型 这个如何解读？可以直接转换为dict类型？
    # 解答：它表示若干 (键, 值) 对组成的不可变元组；functools.cache 要求参数可哈希而 dict 不可哈希，所以调用处先转 tuple，这里可以且确实再用 dict(...) 还原。
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None

    # TODO：这里是定义一个什么维度的tensor？
    # 解答：torch.tensor([]) 是 shape=[0] 的一维空张量；这里只借它探测当前默认 device，并不承载 RoPE 数据。
    t = torch.tensor([])

    # TODO：这里在判断什么东西？device（“meta“）是什么？为什么要分开_get_rope？
    # 解答：Engine 会在 meta device 上无存储地构造模型以节省初始化显存，但 cos/sin 必须真实计算；检测到 meta 后临时切到预设 CUDA device。分出 _get_rope 可让这个设备切换包住真正的张量创建。
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
