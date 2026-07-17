import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

# 问题（已回答）：为什么使用并行 Embedding，还有哪些实现？
# 回答：词表矩阵 [V,C] 在大模型中很大，Vocab Parallel 按词表行把它分到 TP GPU，每张卡只保存 V/TP 行。
# 其他选择包括每卡完整复制的 replicated embedding、按 hidden 维切分的 embedding，以及 CPU/offload 方案；
# 具体选择取决于显存、通信和后续 LM head 是否共享权重，本引擎固定采用词表切分。
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 问题（已回答）：为什么 vocab_size 必须整除 tp_size？
        # 回答：当前实现给每个 rank 分配完全相同的连续行数，并用 rank*partition_size 计算边界，所以必须整除。
        # 这不是理论限制；生产实现可把词表 padding 到 TP 的倍数或支持不等长 shard，但本类没有对应逻辑。
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 问题（已回答）：weight 和 Embedding 的作用、来源及模型位置是什么？
        # 回答：weight 是本 rank 的 [V_local,C] token embedding 矩阵，按 token id 查一行，把离散 id 变成隐藏向量。
        # 它在训练时是可学习参数，推理引擎这里只加载 checkpoint 后固定使用；模型中位于所有 Transformer 层之前，
        # Qwen3 叫 embed_tokens，MiniGPT 叫 token_embedding，tie_word_embeddings 时还与 lm_head 共享。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 问题（已回答）：为什么 Parameter 需要自定义 weight_loader？
        # 回答：checkpoint 通常保存完整 [V,C] 权重，而当前 Parameter 只有本 rank 的 V_local 行；通用 copy 不知道
        # TP rank 和切分维度。loader 属性让 utils/loader.py 在读 safetensors 时调用本类逻辑，只复制正确 shard。
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 问题（已回答）：这里的 shard 是什么？
        # 回答：shard 是完整词表矩阵按第 0 维划分后属于一个 TP rank 的连续行块；shard_size 就是本地 Parameter 行数，
        # start_idx 决定当前 rank 在全局词表中的起始 token id。
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 问题（已回答）：为什么先取 param.data 再 copy_？
        # 回答：param_data 只是已注册 Parameter 存储的引用，并没有创建或重新赋值参数；copy_ 把 shard 原地写入它，
        # 可保留 Parameter 对象、device/dtype、模块注册关系及附加的 weight_loader。直接 param=loaded_weight 会丢失这些信息。
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 问题（已回答）：为什么从全局 token id 减 vocab_start_idx？
            # 回答：每张卡的 weight 行号从 0 开始，而输入 x 是全局词表 id；减起点把本 rank 负责的 id 映射为本地行号。
            # 非本 rank token 先由 mask 标记，再乘成安全索引 0，查询结果稍后会被 mask 清零。
            x = mask * (x - self.vocab_start_idx)
        # 问题（已回答）：F.embedding 是把 x 和 weight 相乘吗？
        # 回答：不是普通矩阵乘法，它按整数 x 从 weight gather 对应行；输入 [N] 得到 [N,C]。
        # 数学上可类比 one-hot(x) @ weight，但实际不会构造巨大 one-hot 矩阵。
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 问题（已回答）：mask 和 all_reduce 两行分别做什么？
            # 回答：mask.unsqueeze(1)*y 将不属于本 rank 的伪查询结果置零；随后 all_reduce(sum) 汇总所有 rank。
            # 每个 token 只有负责其词表区间的 rank 提供非零向量，因此求和后每张卡都得到完整 embedding，供后续 TP 层使用。
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
