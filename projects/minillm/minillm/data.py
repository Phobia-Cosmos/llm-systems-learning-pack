from __future__ import annotations

from pathlib import Path

import torch


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_train_val(token_ids: list[int], val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(token_ids, dtype=torch.long)
    split = max(1, int(len(data) * (1.0 - val_fraction)))
    return data[:split], data[split:]


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("dataset is too small for the configured block_size")

    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
