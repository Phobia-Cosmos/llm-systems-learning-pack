from dataclasses import dataclass


@dataclass
# TODO:为什么这几个属性会被赋值成这些值？这里的layer指的是transformer还是transformer+mlp？还有block size指代的是什么？
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
