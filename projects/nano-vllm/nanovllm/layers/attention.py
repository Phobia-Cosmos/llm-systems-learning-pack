import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


flash_attn_varlen_func = None
flash_attn_with_kvcache = None
_flash_attn_backend = None

# 问题（已回答）：为什么需要 FlashAttention？
# 回答：它不是“算对结果”的必要条件，下面的 torch fallback 也能得到同一数学结果；它是长上下文高性能服务的关键。
# 普通实现会显式生成 [H,Q,K] score/probability，显存读写和临时空间为 O(QK)。FlashAttention 分块读取 Q/K/V、
# 在线计算 softmax 并融合 P@V，避免落盘完整矩阵，所以显存占用更低、速度更高；没有它时本项目仍可运行但并发性能很差。
def load_flash_attn_backend():
    # 问题（已回答）：三个全局变量有什么区别？
    # 回答：flash_attn_varlen_func 是 packed 变长序列的 prefill 接口；flash_attn_with_kvcache 是 upstream
    # flash-attn 专门读取 paged KV cache 的 decode 接口；_flash_attn_backend 是一次性探测状态，取
    # None/"flash_attn"/"vllm"/"torch"，用于避免每次 forward 重复 import 并决定 decode 分支。
    global flash_attn_varlen_func, flash_attn_with_kvcache, _flash_attn_backend
    if _flash_attn_backend is not None:
        return
    try:
        from flash_attn import flash_attn_varlen_func as varlen_func
        from flash_attn import flash_attn_with_kvcache as kvcache_func

        flash_attn_varlen_func = varlen_func
        flash_attn_with_kvcache = kvcache_func
        _flash_attn_backend = "flash_attn"
        return
    except ImportError:
        pass

    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func as varlen_func

        flash_attn_varlen_func = varlen_func
        _flash_attn_backend = "vllm"
        return
    except ImportError:
        _flash_attn_backend = "torch"


# 问题（已回答）：为什么 KV cache 写入使用 Triton？
# 回答：这里需要按任意 slot_mapping 把每个 token 的整行 K/V 散写到 paged cache。Triton 可用一个 kernel
# 同时写 K 和 V，支持 fused-QKV 产生的非连续 token stride，避免多个 PyTorch indexing kernel 和临时张量；
# 这种短小、规则的内存算子也比维护 C++/CUDA 扩展更适合教学实现。
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    # 问题（已回答）：tl.program_id(0) 是什么？
    # 回答：Triton 把 launch grid 中的每个实例称为 program；axis=0 取得一维 grid 上当前 program 的编号。
    # 本 kernel 以 grid=(N,) 启动，因此 idx 恰好对应第 idx 个本轮 token，一个 program 复制一整行 K/V。
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 问题（已回答）：cache_offsets 表示什么？
    # 回答：slot 是目标物理 token 槽位，D=num_kv_heads*head_dim 是每个槽位的元素数；
    # slot*D + [0,D) 就是把 paged cache 视为二维 [physical_slots,D] 后，该槽位整行的扁平地址。
    cache_offsets = slot * D + tl.arange(0, D)

    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # 问题（已回答）：N 表示什么？
    # 回答：key/value 的布局是 [N,num_kv_heads,head_dim]；N 是本轮实际处理的扁平 token 数。
    # prefill 时等于所有请求 scheduled token 数之和，decode 时通常等于当前 batch 的序列数。
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 问题（已回答）：这些 stride 约束分别表示什么？
    # 回答：stride(-1)=1 表示一个 head 内的 head_dim 连续；stride(1)=head_dim 表示同一 token 的各 KV head
    # 紧邻，因此可把后两维展平为 D。stride(0) 是相邻 token 起点的距离，fused QKV view 下可能大于 D，
    # 所以显式传入 kernel。cache.stride(1)=D 表示 cache 中相邻 token 槽位正好相隔一整行 D。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 问题（已回答）：[(N,)]、stride(0) 和 slot_mapping_ptr 分别是什么？
    # 回答：Triton 的 kernel[grid](args) 是 launch 语法，(N,) 是一维 grid，启动 N 个 program。
    # program 已把 head/head_dim 展平成一行，只需 stride(0) 从 token idx 跳到该行起点；slot_mapping_ptr
    # 是 slot_mapping Tensor 的设备指针，第 idx 项告诉该 token 应写入哪个物理 cache slot。
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        alibi_slopes=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        slopes = torch.empty(0, dtype=torch.float32) if alibi_slopes is None else torch.as_tensor(alibi_slopes)
        self.register_buffer("alibi_slopes", slopes.float(), persistent=False)
        self.k_cache = self.v_cache = torch.tensor([])

    def expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads == self.num_kv_heads:
            return x
        repeat = self.num_heads // self.num_kv_heads
        # 问题（已回答）：为什么要 repeat_interleave KV heads？
        # 回答：GQA/MQA 的 Q head 数多于 KV head 数，每组多个 Q head 共享一个 K/V head。这个纯 PyTorch
        # fallback 为了使用统一 einsum，把每个 KV head 重复 num_heads/num_kv_heads 次以对齐 Q；高性能 kernel
        # 通常直接理解这种分组关系，不会真的复制 K/V。
        return x.repeat_interleave(repeat, dim=1)

    # 问题（已回答）：torch_attention 最终计算什么？
    # 回答：它对一条序列计算 scaled causal attention：softmax(QK^T/sqrt(D)+position_bias+mask)V，
    # 输入为 [Q,H,D]、[K,Hkv,D]，返回每个 query/head 聚合后的上下文向量 [Q,H,D]。
    def torch_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: int) -> torch.Tensor:
        k = self.expand_kv_heads(k)
        v = self.expand_kv_heads(v)
        # 问题（已回答）：einsum 的字符串是什么意思？
        # 回答：第一个参数是 Einstein 求和下标表达式，不是普通数据字符串。"qhd,khd->hqk" 表示
        # Q 的 query/head/dim 与 K 的 key/head/dim 在 d 维点积，保留 h、q、k，得到 [H,Q,K] scores。
        scores = torch.einsum("qhd,khd->hqk", q, k).float() * self.scale
        # 问题（已回答）：q_pos/k_pos 是什么，后续有什么用？
        # 回答：torch.arange 会创建位置 id Tensor，而不是为 KV cache 分配空间。key 覆盖绝对位置 0..K-1；
        # query 可能只是历史末尾的一段，因此从 query_start 开始。这两者用于 causal mask 和可选 ALiBi 相对距离。
        q_pos = torch.arange(query_start, query_start + q.size(0), device=q.device)
        k_pos = torch.arange(k.size(0), device=q.device)
        if self.alibi_slopes.numel():
            # 问题（已回答）：为什么 relative_distance=k_pos-q_pos，unsqueeze 维度为何不同？
            # 回答：k_pos[None,:] 形状 [1,K]，q_pos[:,None] 形状 [Q,1]，广播相减得到每个 query-key 对的
            # [Q,K] 距离。因果范围内 k<=q，所以距离不大于 0；乘正 slope 后，越久远的 key 获得越大负偏置。
            relative_distance = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)
            # 问题（已回答）：两个 None 索引和这次乘法在做什么？
            # 回答：slopes[:,None,None] 将 [H] 变为 [H,1,1]，distance[None,:,:] 将 [Q,K] 变为 [1,Q,K]；
            # 广播后得到 [H,Q,K] ALiBi bias，每个 head 使用自己的距离惩罚并与 scores 同形相加。
            scores = scores + self.alibi_slopes[:, None, None] * relative_distance[None, :, :]
        causal_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        # 问题（已回答）：torch.finfo(scores.dtype).min 是什么？
        # 回答：它是该浮点 dtype 可表示的最小有限值；这里 scores 已提升为 float32。把未来位置写成极大负数后，
        # softmax 的指数近似 0，等价于屏蔽未来 token，同时避免某些 kernel 对直接 -inf 的特殊处理。
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        # 问题（已回答）：第二个 einsum 在做什么？
        # 回答："hqk,khd->qhd" 在 key 轴 k 上求和，即每个 query/head 用概率 probs 对所有 value 加权；
        # 返回 [Q,H,D] attention context，随后模型会把 H、D 展平并经过输出投影。
        return torch.einsum("hqk,khd->qhd", probs, v)

    def gather_paged_kv(self, cache: torch.Tensor, block_table: torch.Tensor, seq_len: int) -> torch.Tensor:
        block_size = cache.size(1)
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_table[:num_blocks].long()
        # 问题（已回答）：paged KV 的 reshape 得到什么？
        # 回答：index_select 先按 block_table 取出该序列的物理页，形状为 [num_blocks,block_size,Hkv,D]；
        # reshape 合并前两维，恢复逻辑 token 顺序 [num_blocks*block_size,Hkv,D]，最后 [:seq_len] 去掉末页空槽。
        return cache.index_select(0, block_ids).reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]

    # 问题（已回答）：为什么 attention 要区分 prefill？
    # 回答：prefill 一次处理每条请求的多个 prompt/scheduled token，Q 长度可大于 1 且序列长度不一；
    # decode 通常每条序列只处理一个新 token，并从 paged cache 读取全部历史。两者 shape、数据来源和最优 kernel 都不同。
    def torch_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        outputs = []
        for i in range(context.cu_seqlens_q.numel() - 1):
            q_start = int(context.cu_seqlens_q[i].item())
            # 问题（已回答）：cu_seqlens_q 是 Tensor，为什么可以索引？
            # 回答：Tensor 支持 [] 索引，得到 0 维标量 Tensor；.item() 再把它取成 Python int 用于切片。
            # 这会引入 GPU 到 CPU 同步，所以这里只是正确性 fallback；FlashAttention 直接在设备端读取累计长度。
            q_end = int(context.cu_seqlens_q[i + 1].item())
            k_end = int(context.cu_seqlens_k[i + 1].item())
            q_len = q_end - q_start
            k_len = k_end - int(context.cu_seqlens_k[i].item())
            if context.block_tables is None:
                # 问题（已回答）：为什么 K/V 都使用 k_len，并从 q_start 开始？
                # 回答：K 和 V 必须在同一 token 轴逐项对应，所以都取 k_len。block_tables=None 表示没有已缓存 prefix，
                # 此时该序列本轮 Q/K/V 在 packed 张量中从同一 q_start 开始，且通常 k_len=q_len；若 K 还含历史 prefix，
                # 历史并不在本轮 k/v 张量中，必须走下面的 paged-cache gather 分支。
                k_seq = k[q_start:q_start + k_len]
                v_seq = v[q_start:q_start + k_len]
            else:
                # 问题（已回答）：prefix-cache 分支为什么需要 gather，和上面的切片有什么不同？
                # 回答：命中 prefix cache 时，本轮 k/v 参数只含新 token，而完整 K/V 历史分散在物理 cache pages 中；
                # block_table 给出该请求的页面顺序，因此必须 gather 才能重建 [0..k_len) 的逻辑序列。无 prefix 时直接切当前 k/v 即可。
                k_seq = self.gather_paged_kv(self.k_cache, context.block_tables[i], k_len)
                v_seq = self.gather_paged_kv(self.v_cache, context.block_tables[i], k_len)
            # 问题（已回答）：为什么 query_start=k_len-q_len，prefill 返回什么？
            # 回答：当前 q 是完整 key 历史末尾的 q_len 个位置，所以其首个绝对位置是 k_len-q_len；这让 mask/ALiBi
            # 使用正确位置。torch_attention 返回该请求 [q_len,H,D] 的上下文，循环末尾 cat 后得到 packed [sum_q,H,D]。
            outputs.append(self.torch_attention(q[q_start:q_end], k_seq, v_seq, k_len - q_len))
        return torch.cat(outputs, dim=0)

    def torch_decode(self, q: torch.Tensor) -> torch.Tensor:
        context = get_context()
        outputs = []
        for i in range(q.size(0)):
            # 问题（已回答）：torch decode fallback 为什么逐序列遍历？
            # 回答：每条请求的 context_lens 和 block_table 都可能不同，简单 dense Tensor 无法直接堆叠而不 padding。
            # 这个 fallback 逐条恢复历史以保持逻辑清晰；生产 paged-attention/FlashAttention kernel 会在一个 kernel 中批处理。
            seq_len = int(context.context_lens[i].item())
            # 问题（已回答）：decode 为什么从 cache gather K/V？
            # 回答：decode 的函数参数 k/v 只包含本轮新 token；历史 token 已在先前步骤写入分页 cache，且物理页不连续，
            # 所以必须根据该请求的 block_table 按逻辑顺序取回长度 seq_len 的完整 K/V。
            k_seq = self.gather_paged_kv(self.k_cache, context.block_tables[i], seq_len)
            v_seq = self.gather_paged_kv(self.v_cache, context.block_tables[i], seq_len)
            # 问题（已回答）：query_start 为什么是 seq_len-1？
            # 回答：本轮 q 只有最新一个 token，而 cache 已包含位置 0..seq_len-1，因此它的绝对位置正是最后一位
            # seq_len-1；传入该值后 causal mask 会允许它查看全部历史及自身。
            outputs.append(self.torch_attention(q[i:i + 1], k_seq, v_seq, seq_len - 1))
        return torch.cat(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        load_flash_attn_backend()
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 问题（已回答）：forward 开头为什么先 store_kvcache？
        # 回答：如果 ModelRunner 已为各层分配 cache，就把本轮投影出的 K/V 按 slot_mapping 写入物理槽位。
        # decode 当前 token 随后即可从完整 cache 参与 attention，未来 token 也能复用；prefill 写入后则形成后续 decode 的历史。
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # 问题（已回答）：为什么 flash varlen 函数为空时调用 torch_prefill？
            # 回答：这里已经由 context.is_prefill 确定处于 prefill；若两个 FlashAttention 包都未提供 varlen 接口，
            # 只能退回数学等价的 PyTorch 实现。不是“为空就进入 prefill”，而是“prefill 中没有加速后端就 fallback”。
            if flash_attn_varlen_func is None:
                return self.torch_prefill(q, k, v)
            # 问题（已回答）：block_tables 非空时为什么把 k/v 换成 cache？
            # 回答：它表示本次 prefill 正在复用已缓存 prefix，完整 K/V 位于分页 cache 而非当前 compact k/v。
            # FlashAttention 会结合 block_table 从 k_cache/v_cache 找页面；未命中 prefix 时直接使用当前 packed k/v 更简单。
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # 问题（已回答）：flash_attn_varlen_func 计算什么？
            # 回答：它对 packed 的多条变长序列执行 causal scaled attention，cu_seqlens 划分每条 Q/K，max_seqlen
            # 用于 kernel 配置，block_table 可启用 paged prefix，ALiBi 可选；输出 o 与 q 同为 [sum_q,H,D]。
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables,
                                       alibi_slopes=self.alibi_slopes if self.alibi_slopes.numel() else None)
        # 问题（已回答）：decode 的三个后端分支有什么区别？
        # 回答：优先使用 upstream flash_attn_with_kvcache，它有专用 decode API，直接接收 [B,1,H,D] Q、页表和长度；
        # vLLM bundled 版本这里只暴露 varlen API，因此构造每条 q_len=1 的 cu_seqlens，并用 seqused_k/block_table
        # 描述 paged K/V；两者都在 GPU 中批量完成。都不可用时 torch_decode 逐序列 gather+einsum，仅保证正确性。
        elif flash_attn_with_kvcache is not None:    # decode with upstream flash-attn
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True,
                                        alibi_slopes=self.alibi_slopes if self.alibi_slopes.numel() else None)
        elif _flash_attn_backend == "vllm":    # decode with vLLM's bundled FlashAttention
            cu_seqlens_q = torch.arange(q.size(0) + 1, dtype=torch.int32, device=q.device)
            o = flash_attn_varlen_func(
                q,
                k_cache,
                v_cache,
                max_seqlen_q=1,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=context.block_tables.size(1) * k_cache.size(1),
                seqused_k=context.context_lens,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
                alibi_slopes=self.alibi_slopes if self.alibi_slopes.numel() else None,
            )
        else:
            return self.torch_decode(q)
        return o
