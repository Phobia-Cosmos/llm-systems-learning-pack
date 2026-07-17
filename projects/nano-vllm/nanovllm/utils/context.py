from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    # 问题（已回答）：cu_seqlens_q 和 cu_seqlens_k 是什么？
    # 回答：它们是 packed variable-length batch 的累计长度（prefix sum），形如 [0,L0,L0+L1,...]。
    # 相邻元素之差给出每条序列的 Q/K 长度；Q 是本轮调度 token，K 可再包含 prefix cache 中的历史 token。
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 问题（已回答）：slot_mapping 保存什么？
    # 回答：它为本轮每个扁平 token 指定物理 KV 槽位，槽位通常是 block_id*block_size+offset；
    # attention 投影出 K/V 后据此散写 paged cache，值 -1 可表示 CUDA Graph 的无效填充位置。
    slot_mapping: torch.Tensor | None = None
    # 问题（已回答）：context_lens 保存什么？
    # 回答：decode 时每条序列当前可见的 KV 总长度，包含刚写入的当前 token；paged attention 用它限制每条请求读取多少槽位。
    context_lens: torch.Tensor | None = None
    # 问题（已回答）：block_tables 保存什么？
    # 回答：shape 通常为 [num_sequences,max_blocks]，第 i 行把该序列的逻辑 block 序号映射到物理 KV block id；
    # 尾部用 -1 填充。decode 总是需要它，prefill 只有读取已缓存 prefix 时才需要。
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
