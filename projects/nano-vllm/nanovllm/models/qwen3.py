import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.registry import register_model


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 问题（已回答）：q_size/kv_size 为什么等于 head 数乘 head_dim？
        # 回答：一个 head 是 head_dim 维向量，把本 rank 的所有 head 沿最后一维拼接后，Q 宽度就是 Hq_local*D，
        # K/V 宽度是 Hkv_local*D。GQA 中 Hkv 可小于 Hq，所以两者需分开记录，用于切分 fused QKV 输出。
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 问题（已回答）：scaling=head_dim**-0.5 有什么作用？
        # 回答：QK 点积是 D 项之和，若各维方差相近，其尺度会随 D 增大；乘 1/sqrt(D) 可稳定 score 方差，
        # 防止 softmax 过早饱和，这是 scaled dot-product attention 的标准因子。
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # 问题（已回答）：q_norm/k_norm 在做什么，为什么层 Norm 之外还需要它？
        # 回答：这是 QK-Norm：在线性投影后按每个 head 的 D 维分别做 RMSNorm，再送入 RoPE/QK 点积，
        # 用于控制 attention logits 尺度并改善长上下文训练稳定性；它与 hidden_size 上的 layer RMSNorm 作用位置不同。
        # 此实现用 qkv_bias 作为支持模型变体的开关；“无 bias 必然需要 QK-Norm”不是普遍数学规律，而是 checkpoint 约定。
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 问题（已回答）：qkv_proj 得到什么，为什么一次能得到 Q/K/V？
        # 回答：它是一个输出通道按 [Q|K|V] 打包的 Linear，权重等价于把三个独立投影矩阵沿输出维拼接；
        # 一次 GEMM 读 hidden_states 就能生成三段结果，随后按 q_size/kv_size split，数学上与三次 Linear 完全等价。
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 问题（已回答）：为什么要把 Q/K/V view 成三维？
        # 回答：Linear 输出把所有 head 展平在最后一维 [tokens,H*D]；RoPE 和 Attention 需要显式知道 head 维，
        # 所以零拷贝重解释为 [tokens,H,D]。Q 使用 num_heads，K/V 使用较小 num_kv_heads 以表达 GQA。
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        # 问题（已回答）：Attention 得到什么，为什么还需要 o_proj？
        # 回答：self.attn 返回每个 token、每个 Q head 的上下文 [tokens,Hq,D]。flatten 只是拼接各 head；o_proj=W_o
        # 再学习跨 head 混合，并把 Hq*D 映射回 hidden_size，使结果能与 residual 相加。attention 加权和本身不承担该投影。
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            # 问题（已回答）：Qwen3 MLP 为什么 bias=False？
            # 回答：这是 Qwen3 训练架构的参数约定，官方 gate/up/down 投影不含 bias；checkpoint 也没有对应向量。
            # 添加 bias 会改变函数、参数量和权重名称，无法直接加载。无 bias 也可减少少量参数和 kernel 融合负担。
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            # 问题（已回答）：rope_scaling 和 qkv_bias 是什么，Q/K/V bias 相同吗？
            # 回答：rope_scaling 描述扩展 RoPE 上下文的方法及 factor/original length/theta 等；不同模型可能使用
            # linear、dynamic、YaRN 等规则。本精简实现只读取其中 rope_theta，其他策略并未完整实现。
            # qkv_bias 控制 fused QKV Linear 是否带加法向量；原始 Q/K/V 各有独立 bias 段，打包后位于同一向量，数值并不共享。
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 问题（已回答）：input_layernorm 和 post_attention_layernorm 有何区别？
        # 回答：两者都是 RMSNorm 且 shape 相同，但参数独立、位置不同：前者把进入 attention 的残差流归一化；
        # 后者先把 attention 输出加回 residual，再把结果归一化后送入 MLP。add_rms_forward 同时维护未归一化 residual，
        # 下一层再把 MLP 输出融合进去。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 问题（已回答）：为什么 Qwen 不选择 Embedding 类型？
        # 回答：Qwen3 架构已规定使用一个 token embedding，config 只决定 vocab_size/hidden_size，不把“Embedding 算法”
        # 作为可切换超参数。这里的 VocabParallelEmbedding 是引擎对同一数学 lookup 的 TP 实现；Qwen 使用 RoPE，
        # 因此也不需要像教学 MiniGPT 那样选择 learned/sinusoidal position embedding。
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_model(
    model_type="qwen3",
    architectures=("Qwen3ForCausalLM",),
    config_class=Qwen3Config,
)
class Qwen3ForCausalLM(nn.Module):
    # 问题（已回答）：packed_modules_mapping 是什么，为什么需要？
    # 回答：HF checkpoint 分开保存 q_proj/k_proj/v_proj 和 gate_proj/up_proj，而推理模型为减少 GEMM/launch，
    # 把它们分别打包成 qkv_proj 和 gate_up_proj。loader 用该映射改参数名，并把 "q"/"k"/"v" 或 0/1
    # 作为 shard_id 传给自定义 weight_loader，确保每块权重进入 packed Parameter 的正确区间。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
