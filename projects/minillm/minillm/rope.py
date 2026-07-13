from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Full-dimension RoPE using the NeoX/Llama half-split convention."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if base <= 0:
            raise ValueError("rope_theta must be positive")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_rotary(query, positions), self._apply_rotary(key, positions)

    def _apply_rotary(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x is [B,H,T,D]; positions may be [T] or [B,T].
        cos = self.cos.index_select(0, positions.reshape(-1)).view(*positions.shape, -1)
        sin = self.sin.index_select(0, positions.reshape(-1)).view(*positions.shape, -1)
        if positions.dim() == 1:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif positions.dim() == 2:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            raise ValueError("positions must have shape [T] or [B,T]")

        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        first, second = torch.chunk(x.float(), 2, dim=-1)
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        return rotated.to(x.dtype)
