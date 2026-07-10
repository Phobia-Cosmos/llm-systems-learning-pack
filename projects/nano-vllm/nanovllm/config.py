import os
from dataclasses import dataclass
from transformers import AutoConfig

# TODO：slots是什么？
@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    # TODO：max_num_seqs、max_model_len的作用是什么？tensor_parallel_size可以随意更改吗？
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    # TODO：是什么由谁？
    enforce_eager: bool = False
    # TODO：AutoConfig是什么？为什么eos为-1,代表什么东西？kvcache_block_size大小限制的是什么？
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # TODO：为什么一定要是256的倍数？为什么并行只能在1-8？我们的model只能是AutoConfig中定义过的是吗？
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
