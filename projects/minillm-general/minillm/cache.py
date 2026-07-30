from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StaticKVCache:
    """Fixed-size, inference-only KV storage shared by all decode steps.

    Each item in ``key_caches`` and ``value_caches`` belongs to one
    Transformer layer and has shape ``[B, Hkv, max_len, head_dim]``. Only the
    prefix ``:length`` is valid; ``reset`` makes the storage reusable without
    reallocating or clearing it.
    """

    key_caches: list[torch.Tensor]
    value_caches: list[torch.Tensor]
    max_len: int
    length: int = 0

    def __post_init__(self) -> None:
        if self.max_len <= 0:
            raise ValueError("max_len must be positive")
        if len(self.key_caches) == 0:
            raise ValueError("a static KV cache requires at least one layer")
        if len(self.key_caches) != len(self.value_caches):
            raise ValueError("key_caches and value_caches must have the same number of layers")
        if not 0 <= self.length <= self.max_len:
            raise ValueError("cache length must be between zero and max_len")

        expected_shape = self.key_caches[0].shape
        if len(expected_shape) != 4 or expected_shape[2] != self.max_len:
            raise ValueError("static KV tensors must have shape [B, Hkv, max_len, head_dim]")
        for key, value in zip(self.key_caches, self.value_caches):
            if key.shape != expected_shape or value.shape != expected_shape:
                raise ValueError("all static KV tensors must have the same shape")
            if key.device != value.device or key.dtype != value.dtype:
                raise ValueError("key and value storage must use the same device and dtype")

    @property
    def batch_size(self) -> int:
        return self.key_caches[0].size(0)

    @property
    def num_layers(self) -> int:
        return len(self.key_caches)

    @property
    def device(self) -> torch.device:
        return self.key_caches[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.key_caches[0].dtype

    def reset(self) -> None:
        """Mark the cache empty while retaining all allocated storage."""

        self.length = 0
