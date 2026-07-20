import os
from dataclasses import dataclass
from transformers import PretrainedConfig

from nanovllm.models.registry import load_model_config

# 问题（已回答）：dataclass(slots=True) 的 slots 是什么？
# 回答：它生成固定字段槽，减少实例内存并阻止因拼错字段名而动态添加属性。
@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    # 问题（已回答）：max_num_seqs、max_model_len 和 tensor_parallel_size 是什么？
    # 回答：前两者限制同时运行请求数和单序列最大 token 数；TP size 是切分同一模型的 GPU 数，
    # 必须不超过可见 GPU，并满足模型头/维度可整除和进程数约束，不能随意设置。
    # 问题（已回答）：单序列最大 token 数指的是一个请求的最大 token 数量吗？单 GPU 是否不需要 TP？
    # 回答：是，max_model_len 限制一条请求的总上下文长度，通常包括 prompt 和已经生成的 token；它不同于
    # max_num_batched_tokens，后者限制一次调度中所有请求合计处理的 token。只有一张 GPU 时 TP 必须为 1，
    # 不会带来切分收益。TP 这一技术本身可扩展到跨节点，但当前 ModelRunner 固定用 localhost rendezvous，并按 rank
    # 选择本机 GPU，所以本仓库实现的是单机多卡，不能直接当作已支持跨节点。
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    # 问题（已回答）：enforce_eager 是什么？
    # 回答：True 强制 eager execution 并跳过 CUDA graph，启动简单但 decode 较慢；False 会 warmup 和捕获图。
    # 问题（已回答）：enforce_eager=False 时会自动构建 CUDA Graph 吗？
    # 回答：会，但这是本仓库的 ModelRunner 显式执行 warmup_model() 和 capture_cudagraph()，不是 PyTorch
    # 对任意程序自动完成。这里只捕获预设 batch size 的 decode 图；prefill、超出图范围的 batch 等仍走 eager。
    enforce_eager: bool = False
    # 问题（已回答）：hf_config、eos 和 kvcache_block_size 是什么？
    # 回答：hf_config 是 config.json 解析出的 PretrainedConfig；eos 是结束 token id，-1 是加载前占位值。
    # block_size 是每个物理 KV block 的 token 槽位数；槽中存每层 K/V 向量，不是 token id。
    hf_config: PretrainedConfig | None = None
    eos: int | None = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # 问题（已回答）：为什么使用 __post_init__？
    # 回答：dataclass 自动 __init__ 先写入参数，随后 __post_init__ 做跨字段校验、加载模型 config 并收紧长度。
    def __post_init__(self):
        assert os.path.isdir(self.model)
        # 问题（已回答）：为什么要求 256 倍数、TP 为 1-8，模型支持范围是什么？
        # 回答：这是本教学实现/kernel 的限制，不是通用规律；固定块简化 page 管理，1-8 控制实现范围。
        # 模型也不是任意 AutoConfig 都能跑，必须在 nanovllm model registry 注册匹配的 config/model 后端。
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = load_model_config(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
