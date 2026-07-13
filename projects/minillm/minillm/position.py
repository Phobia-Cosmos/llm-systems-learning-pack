from __future__ import annotations

import math

import torch
from torch import nn

from .rope import RotaryEmbedding


SUPPORTED_POSITION_ENCODINGS = ("learned", "sinusoidal", "rope", "alibi", "none")


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed absolute position encoding from Attention Is All You Need."""

    def __init__(self, max_seq_len: int, hidden_size: int, base: float = 10000.0):
        super().__init__()
        if base <= 0:
            raise ValueError("sinusoidal_theta must be positive")

        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(base) / hidden_size)
        )
        table = torch.zeros(max_seq_len, hidden_size, dtype=torch.float32)
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        flat = positions.reshape(-1)
        return self.table.index_select(0, flat).view(*positions.shape, -1)


def build_input_position_encoding(
    kind: str,
    max_seq_len: int,
    hidden_size: int,
    sinusoidal_theta: float,
) -> nn.Module | None:
    if kind == "learned":
        # Returning nn.Embedding preserves the historical position_embedding.weight key.
        return nn.Embedding(max_seq_len, hidden_size)
    if kind == "sinusoidal":
        return SinusoidalPositionEmbedding(max_seq_len, hidden_size, sinusoidal_theta)
    return None


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Return the head slopes from the ALiBi reference implementation."""

    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    closest_power_of_two = 2 ** math.floor(math.log2(num_heads))
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
    return torch.cat(
        (slopes, torch.pow(torch.tensor(extra_base), extra_powers)),
        dim=0,
    )


class AttentionPositionEncoding(nn.Module):
    """Position hooks that run inside attention."""

    def apply_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key

    def attention_bias(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        return None


class RotaryAttentionPositionEncoding(AttentionPositionEncoding):
    def __init__(self, head_dim: int, max_seq_len: int, base: float):
        super().__init__()
        self.rotary = RotaryEmbedding(head_dim, max_seq_len, base)

    def apply_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary(query, key, positions)


class ALiBiAttentionPositionEncoding(AttentionPositionEncoding):
    def __init__(self, num_heads: int):
        super().__init__()
        self.register_buffer("slopes", get_alibi_slopes(num_heads), persistent=False)

    def attention_bias(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = query_positions.reshape(-1).to(device=device, dtype=torch.float32)
        key_positions = key_positions.reshape(-1).to(device=device, dtype=torch.float32)
        relative_distance = key_positions.unsqueeze(0) - query_positions.unsqueeze(1)
        slopes = self.slopes.to(device=device).view(1, -1, 1, 1)
        return (slopes * relative_distance.view(1, 1, *relative_distance.shape)).to(dtype)


def build_attention_position_encoding(
    kind: str,
    head_dim: int,
    num_heads: int,
    max_seq_len: int,
    rope_theta: float,
) -> AttentionPositionEncoding:
    if kind == "rope":
        return RotaryAttentionPositionEncoding(head_dim, max_seq_len, rope_theta)
    if kind == "alibi":
        return ALiBiAttentionPositionEncoding(num_heads)
    return AttentionPositionEncoding()
