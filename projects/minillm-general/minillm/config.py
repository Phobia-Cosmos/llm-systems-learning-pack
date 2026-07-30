from dataclasses import dataclass

from .activations import SUPPORTED_ACTIVATIONS
from .mlp import SUPPORTED_MLP_TYPES, default_intermediate_size
from .norm import SUPPORTED_NORMS
from .position import SUPPORTED_POSITION_ENCODINGS


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
    fused_qkv: bool = True
    dropout: float = 0.1
    bias: bool = True
    position_encoding: str = "learned"
    rope_theta: float = 10000.0
    sinusoidal_theta: float = 10000.0
    norm_type: str = "layernorm"
    norm_eps: float = 1e-5
    mlp_type: str = "dense"
    activation: str = "gelu"
    intermediate_size: int | None = None
    # Keep this field at the end so positional construction of older configs
    # keeps the same argument order. ``None`` is the legacy MHA default.
    num_key_value_heads: int | None = None
    # General-training extensions stay opt-in so copied teaching checkpoints
    # retain their original parameter names and numerical path.
    qk_norm: bool = False
    use_sdpa: bool = False
    scale_residual_init: bool = False

    def __post_init__(self) -> None:
        if self.position_encoding not in SUPPORTED_POSITION_ENCODINGS:
            raise ValueError(f"position_encoding must be one of {SUPPORTED_POSITION_ENCODINGS}")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.n_head
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_key_value_heads > self.n_head:
            raise ValueError("num_key_value_heads must not exceed n_head")
        if self.n_head % self.num_key_value_heads != 0:
            raise ValueError("n_head must be divisible by num_key_value_heads")
        if self.position_encoding == "rope" and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.sinusoidal_theta <= 0:
            raise ValueError("sinusoidal_theta must be positive")
        if self.norm_type not in SUPPORTED_NORMS:
            raise ValueError(f"norm_type must be one of {SUPPORTED_NORMS}")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.mlp_type not in SUPPORTED_MLP_TYPES:
            raise ValueError(f"mlp_type must be one of {SUPPORTED_MLP_TYPES}")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"activation must be one of {SUPPORTED_ACTIVATIONS}")
        if self.intermediate_size is None:
            self.intermediate_size = default_intermediate_size(self.n_embd, self.mlp_type)
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
