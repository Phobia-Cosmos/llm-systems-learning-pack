from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig

from nanovllm.layers.attention import Attention
from nanovllm.layers.activation import ActivationAndMul, build_activation
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import build_norm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.position_encoding import SinusoidalPositionEmbedding, get_alibi_slopes
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.registry import register_model


def _resolve_alias(name: str, value, alias_name: str, alias_value, default):
    if value is not None and alias_value is not None and value != alias_value:
        raise ValueError(f"Conflicting {name}={value!r} and {alias_name}={alias_value!r}")
    if value is not None:
        return value
    if alias_value is not None:
        return alias_value
    return default

# 问题（已回答）：为什么 MiniGPT 需要自定义 Config，而 Qwen3 不需要？
# 回答：Hugging Face Transformers 已提供标准 Qwen3Config，可直接解析官方 config.json；MiniGPT 是本项目自定义架构，
# HF 没有认识其 block_size/n_embd 等字段的类。继承 PretrainedConfig 可让 registry/from_pretrained 正常工作，
# 同时在这里把 MiniGPT 命名与 HF 标准命名互相映射。这不是模型数学上的额外层。
class MiniGPTConfig(PretrainedConfig):
    model_type = "minigpt"

    def __init__(
        self,
        vocab_size: int = 0,
        block_size: int | None = None,
        n_layer: int | None = None,
        n_head: int | None = None,
        n_embd: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        # 问题（已回答）：n_head、num_attention_heads、num_key_value_heads 和 tie_word_embeddings 是什么？
        # 回答：n_head 是 MiniGPT 原始字段，num_attention_heads 是 HF 标准别名，二者都表示 Q head 数；
        # num_key_value_heads 是独立 K/V head 数：MHA 中等于 Q heads，GQA/MQA 中更少，且必须整除 Q heads。
        # tie_word_embeddings=True 表示输入 token embedding 与输出 lm_head 共用同一权重矩阵，节省参数并共享词义空间。
        max_position_embeddings: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        hidden_size: int | None = None,
        tie_word_embeddings: bool = True,
        position_encoding: str = "learned",
        # 问题（已回答）：RoPE/sinusoidal theta 能随便设置吗？两个 Norm epsilon 有何区别？
        # 回答：theta/base 决定位置频率和波长；rope_theta 用于旋转 Q/K，sinusoidal_theta 用于固定绝对位置表。
        # 它们可在训练前作为超参数选择，但训练后属于 checkpoint 配置，推理必须一致，不能随意改。
        # norm_eps 是本项目名，layer_norm_epsilon 是 HF 常见别名；_resolve_alias 最终合并成同一个数值，防止除零/数值不稳定。
        rope_theta: float = 10000.0,
        sinusoidal_theta: float = 10000.0,
        norm_type: str = "layernorm",
        norm_eps: float | None = None,
        layer_norm_epsilon: float | None = None,
        # 问题（已回答）：为什么 MLP 有 type，不同类型和 hidden_act 是什么？
        # 回答：dense 是 fc1->activation->fc2；SwiGLU/GEGLU/ReGLU 则生成 gate/value 两支并计算
        # act(gate)*value，再 down projection，表达力和参数布局不同。hidden_act 是 HF config 对激活函数的标准字段，
        # 这里与 MiniGPT 的 activation 互为别名；门控类型会进一步固定 SiLU/GELU/ReLU。
        mlp_type: str = "dense",
        activation: str | None = None,
        hidden_act: str | None = None,
        intermediate_size: int | None = None,
        **kwargs,
    ) -> None:
        block_size = _resolve_alias(
            "block_size", block_size, "max_position_embeddings", max_position_embeddings, 64
        )
        n_layer = _resolve_alias("n_layer", n_layer, "num_hidden_layers", num_hidden_layers, 2)
        n_head = _resolve_alias("n_head", n_head, "num_attention_heads", num_attention_heads, 4)
        n_embd = _resolve_alias("n_embd", n_embd, "hidden_size", hidden_size, 128)
        norm_eps = _resolve_alias("norm_eps", norm_eps, "layer_norm_epsilon", layer_norm_epsilon, 1e-5)
        activation = _resolve_alias("activation", activation, "hidden_act", hidden_act, "gelu")
        if num_key_value_heads is None:
            num_key_value_heads = n_head
        if num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if n_head % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if position_encoding not in {"learned", "sinusoidal", "rope", "alibi", "none"}:
            raise ValueError("unsupported position_encoding")
        if position_encoding == "rope" and (n_embd // n_head) % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if rope_theta <= 0 or sinusoidal_theta <= 0:
            raise ValueError("position theta values must be positive")
        if norm_type not in {"layernorm", "rmsnorm", "scalenorm", "none"}:
            raise ValueError("unsupported norm_type")
        if mlp_type not in {"dense", "swiglu", "geglu", "reglu"}:
            raise ValueError("unsupported mlp_type")
        if activation not in {
            "gelu", "gelu_tanh", "relu", "relu_squared", "leaky_relu",
            "elu", "silu", "mish", "tanh", "sigmoid", "identity",
        }:
            raise ValueError("unsupported activation")
        if norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if intermediate_size is None:
            intermediate_size = 4 * n_embd if mlp_type == "dense" else 8 * ((n_embd + 2) // 3)
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.max_position_embeddings = block_size
        self.n_layer = n_layer
        self.num_hidden_layers = n_layer
        self.n_head = n_head
        self.num_attention_heads = n_head
        self.num_key_value_heads = num_key_value_heads
        self.n_embd = n_embd
        self.hidden_size = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.bias = bias
        self.position_encoding = position_encoding
        self.rope_theta = rope_theta
        self.sinusoidal_theta = sinusoidal_theta
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.layer_norm_epsilon = norm_eps
        self.mlp_type = mlp_type
        self.activation = activation
        self.hidden_act = activation
        self.intermediate_size = intermediate_size


@dataclass(frozen=True)
class MiniGPTCharTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    unk_token: str = "<unk>"

    @classmethod
    def from_pretrained(cls, model_path: str) -> "MiniGPTCharTokenizer":
        tokenizer_path = Path(model_path) / "tokenizer.json"
        payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        tokenizer = cls(
            stoi={str(token): int(token_id) for token, token_id in payload["stoi"].items()},
            itos=[str(token) for token in payload["itos"]],
            unk_token=str(payload.get("unk_token", "<unk>")),
        )
        if tokenizer.unk_token not in tokenizer.stoi:
            raise ValueError(f"Unknown token {tokenizer.unk_token!r} is missing from {tokenizer_path}")
        if len(tokenizer.stoi) != len(tokenizer.itos):
            raise ValueError(f"Inconsistent character vocabulary in {tokenizer_path}")
        return tokenizer

    @property
    def eos_token_id(self) -> None:
        return None

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, **_) -> list[int]:
        unknown_id = self.stoi[self.unk_token]
        return [self.stoi.get(character, unknown_id) for character in text]

    def decode(self, token_ids: list[int], **_) -> str:
        pieces = []
        for token_id in token_ids:
            token = self.itos[int(token_id)]
            pieces.append("?" if token == self.unk_token else token)
        return "".join(pieces)


def load_minigpt_tokenizer(model_path: str):
    config_path = Path(model_path) / "tokenizer_config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("tokenizer_class") == "CharTokenizer":
            return MiniGPTCharTokenizer.from_pretrained(model_path)
    # 问题（已回答）：这里返回什么样的分词器？
    # 回答：若不是上面的自定义 CharTokenizer，就由 HF AutoTokenizer 根据目录中的 tokenizer_config.json/
    # tokenizer.json 等文件选择实现；use_fast=True 优先返回 Rust tokenizers 后端的 PreTrainedTokenizerFast，
    # 提供 encode/decode、special token、batch 和 chat template 等标准接口。
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


class MiniGPTAttention(nn.Module):

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if config.n_head % tp_size != 0:
            raise ValueError(f"n_head={config.n_head} must be divisible by tensor_parallel_size={tp_size}")
        if config.num_key_value_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads={config.num_key_value_heads} must be divisible by "
                f"tensor_parallel_size={tp_size}"
            )
        self.num_heads = config.n_head // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        # 问题（已回答）：q_size、kv_size 和 bias 是什么？
        # 回答：q_size 是本 rank 的 Q heads 总宽度，kv_size 是本 rank 的 K/V heads 总宽度；MHA 中两者相等，
        # GQA/MQA 中 kv_size 更小。bias 是 Linear 的可训练加法向量；config.bias 决定 QKV 和输出投影
        # 是否创建它，必须与训练 checkpoint 结构一致。KV cache 只保存较小的 K/V heads。
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rotary = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.block_size,
                base=config.rope_theta,
            )
            if config.position_encoding == "rope"
            else None
        )
        self.c_attn = QKVParallelLinear(
            config.n_embd,
            self.head_dim,
            config.n_head,
            config.num_key_value_heads,
            bias=config.bias,
        )
        self.c_proj = RowParallelLinear(config.n_embd, config.n_embd, bias=config.bias)

        alibi_slopes = None
        # 问题（已回答）：什么是 ALiBi 位置编码？
        # 回答：ALiBi 不修改 token hidden state，而是在 attention score 上加入每个 head 独立的线性相对位置偏置
        # slope_h*(key_position-query_position)。在因果注意力中越旧的 token 得分惩罚越大；无需位置 embedding 表，
        # 并常有较好的长度外推。TP 时每个 rank 只取自己负责 heads 的 slope。
        if config.position_encoding == "alibi":
            head_start = dist.get_rank() * self.num_heads
            alibi_slopes = get_alibi_slopes(config.n_head)[head_start : head_start + self.num_heads]

        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            alibi_slopes=alibi_slopes,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if self.rotary is not None:
            q, k = self.rotary(positions, q, k)
        output = self.attn(q, k, v)
        return self.c_proj(output.flatten(1, -1))

# 问题（已回答）：为什么 Qwen MLP 不选择 dense，MiniGPT dense 为什么用 Sequential？
# 回答：Qwen3 架构和 checkpoint 已固定为 SwiGLU，不是运行时可切换的超参数；MiniGPT 是教学模型，允许比较 dense
# 与多种 GLU。dense 的数据流是单路 Linear->activation->Linear，恰好可用 Sequential 顺序表达；门控 MLP
# 需要拆成 gate/value 再逐元素相乘，不能只靠简单 Sequential 表示。
class MiniGPTMLP(nn.Module):

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.mlp_type = config.mlp_type
        # 问题（已回答）：dense MLP 第一层为什么用 ColumnParallelLinear？
        # 回答：fc1 的 intermediate 输出通道彼此独立，可按列/输出行分给各 TP rank，不需立即通信；
        # 每张卡对自己的 intermediate shard 做激活，随后 RowParallelLinear 消费这些局部特征并 all-reduce 回 hidden_size。
        # 这是 Megatron 风格 MLP 的成对切分，可避免在两层之间聚合巨大中间激活。
        if self.mlp_type == "dense":
            self.net = nn.Sequential(
                ColumnParallelLinear(config.n_embd, config.intermediate_size, bias=config.bias),
                build_activation(config.activation),
                RowParallelLinear(config.intermediate_size, config.n_embd, bias=config.bias),
            )
        else:
            # 问题（已回答）：gate_activation 映射选出的是什么？
            # 回答：它把门控 MLP 名称转换成 gate 分支的具体逐元素激活：SwiGLU->SiLU、GEGLU->GELU、
            # ReGLU->ReLU；value 分支不激活，二者相乘后再做 down projection。
            gate_activation = {"swiglu": "silu", "geglu": "gelu", "reglu": "relu"}[self.mlp_type]
            self.gate_up_proj = MergedColumnParallelLinear(
                config.n_embd,
                [config.intermediate_size, config.intermediate_size],
                bias=config.bias,
            )
            # 问题（已回答）：为什么门控 MLP 要“激活并相乘”？
            # 回答：GLU 家族公式就是 down(act(gate_projection(x))*value_projection(x))。激活后的 gate 像逐通道开关，
            # value 携带内容；乘法让两个投影发生条件交互，这也是它区别于普通 dense MLP 的核心。
            self.activation_and_mul = ActivationAndMul(gate_activation)
            self.down_proj = RowParallelLinear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "dense":
            return self.net(hidden_states)
        return self.down_proj(self.activation_and_mul(self.gate_up_proj(hidden_states)))

# 问题（已回答）：Qwen 为什么没有同名 Block，它是 decode block 吗？
# 回答：Qwen 中对应类叫 Qwen3DecoderLayer，结构同样包含 pre-norm attention、残差、pre-norm MLP、残差，
# 只是用 fused residual+RMSNorm 保存 residual。MiniGPTBlock 不是“仅 decode 阶段”的 block；decoder-only 是模型架构名，
# 同一层同时用于 prompt prefill 和逐 token decode，阶段差异由内部 Attention/Context 处理。
class MiniGPTBlock(nn.Module):

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln_1 = build_norm(config.n_embd, config.norm_type, config.norm_eps, config.bias)
        self.attn = MiniGPTAttention(config)
        self.ln_2 = build_norm(config.n_embd, config.norm_type, config.norm_eps, config.bias)
        self.mlp = MiniGPTMLP(config)

    # 问题（已回答）：forward 为什么需要 positions，为什么不单独传 residual？
    # 回答：引擎把序列展平成 token 流，模型无法从 Tensor 轴推断每个 token 的绝对位置；RoPE/ALiBi/位置表都需要
    # positions。MiniGPT 直接在 block 内写 hidden + branch，自身持有残差流；Qwen 为融合 add+RMSNorm、减少显存读写，
    # 才把 residual 作为独立 Tensor 跨子层传递，两种写法数学上对应。
    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), positions)
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states

# 问题（已回答）：为什么 MiniGPT 注册时要提供专用 tokenizer_loader？
# 回答：MiniGPT 可能导出教学用的 stoi/itos 字符 tokenizer，它不是 HF AutoTokenizer 认识的标准 tokenizer.json。
# registry 先检查 tokenizer_class 并用 MiniGPTCharTokenizer 加载；若模型使用标准 BPE 等格式，则函数会回退 AutoTokenizer。
@register_model(
    model_type="minigpt",
    architectures=("MiniGPTForCausalLM",),
    config_class=MiniGPTConfig,
    tokenizer_loader=load_minigpt_tokenizer,
)
class MiniGPTForCausalLM(nn.Module):

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        # 问题（已回答）：为什么读取 TP size，词表为何必须整除，它从哪里来？
        # 回答：VocabParallelEmbedding/LMHead 要按词表行均分，所以这里提前验证 vocab_size%tp_size==0；
        # 当前实现不支持不等长 shard，生产系统也可选择 padding。tp_size 来自 ModelRunner 按 Config.tensor_parallel_size
        # 初始化的 torch.distributed process group，dist.get_world_size() 返回该组进程/GPU 数；单 GPU 时为 1。
        tp_size = dist.get_world_size()
        if config.vocab_size % tp_size != 0:
            raise ValueError(
                f"vocab_size={config.vocab_size} must be divisible by tensor_parallel_size={tp_size}"
            )

        self.token_embedding = VocabParallelEmbedding(config.vocab_size, config.n_embd)
        if config.position_encoding == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        elif config.position_encoding == "sinusoidal":
            self.position_embedding = SinusoidalPositionEmbedding(
                config.block_size, config.n_embd, config.sinusoidal_theta
            )
        else:
            self.position_embedding = None

        # 问题（已回答）：为什么 MiniGPT 不再单独封装 Base Model 类？
        # 回答：这是保持教学实现和导出权重命名简洁的设计选择，ForCausalLM 直接拥有 embedding、blocks、norm 和 head。
        # Qwen 拆成 Qwen3Model+Qwen3ForCausalLM 是为匹配 HF 结构、复用 base model 和 checkpoint 名称；MiniGPT 也可重构，
        # 但需同步修改 safetensors 参数名、loader 映射和兼容测试，并非功能上缺少一层。
        self.blocks = nn.ModuleList([MiniGPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = build_norm(config.n_embd, config.norm_type, config.norm_eps, config.bias)
        self.lm_head = ParallelLMHead(config.vocab_size, config.n_embd)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.token_embedding.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden_states = hidden_states + self.position_embedding(positions).to(hidden_states.dtype)
        for block in self.blocks:
            hidden_states = block(hidden_states, positions)
        return self.ln_f(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
