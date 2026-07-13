from dataclasses import dataclass


@dataclass
# 问题（已回答）：这些默认值为何这样设置，layer 和 block_size 指什么？
# 回答：它们是便于 CPU/单卡教学的小模型超参数，不是固定标准。n_layer 表示完整 TransformerBlock 数量，
# 每个 block 包含 attention、MLP、两次 norm 和残差；block_size 是单条序列最多处理的 token 数/上下文窗口。
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
    position_encoding: str = "learned"
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding must be 'learned' or 'rope'")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.position_encoding == "rope" and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
