from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, hidden_size: int, base: float = 10000.0):
        super().__init__()
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
        return self.table.index_select(0, positions.reshape(-1)).view(*positions.shape, -1)


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
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
    return torch.cat((slopes, torch.pow(torch.tensor(extra_base), extra_powers)))
