from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
