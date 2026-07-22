from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    # TODO：scaling的作用是什么 为什么需要加上scaling？
    # 解答：scaling 保存可选的 RoPE 长上下文策略及参数；超过原训练上下文时需按模型配置调整频率，普通模型则为 None 并使用原始 RoPE。
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    # TODO：默认使用多头注意力是吗？
    # 解答：这两个字段直接来自 HF 配置，并非这里强制默认 MHA；两者相等是 MHA，KV head 更少是 GQA，num_kv_heads=1 则是 MQA。
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int

    # TODO：这是词表的长度吧？intermediate_size是什么？
    # 解答：vocab_size 是可表示 token 的数量；intermediate_size 是普通 FFN/MLP 从 hidden_size 上投影后的中间宽度，通常再经激活和下投影回 hidden_size。
    vocab_size: int
    intermediate_size: int

    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str

    # TODO：tie_word_embeddings是什么？moe_intermediate_size？norm_topk_prob？
    # 解答：tie_word_embeddings 表示输入 embedding 与 LM head 共用权重；moe_intermediate_size 是每个专家 FFN 的中间宽度；norm_topk_prob 表示是否把入选专家的路由权重重新归一化到和为 1。
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool

    model_type: str
    architectures: list[str]

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        # TODO：text_config是什么？
        # 解答：某些复合/多模态 HF 配置把语言模型子配置放在 text_config，顶层还包含视觉等配置；Mini-SGLang这里只构建文本生成模型。
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            # TODO：为什么要先取出text_config？为什么一定要把top中的赋值到config中？
            # 解答：先切到文本子配置才能统一读取层数、hidden size 等；architectures/RoPE 信息有时只在顶层，因此仅在子配置缺失时继承，避免覆盖更具体的值。
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        # TODO：为什么默认注意力头数量可以是kvcache head数量？
        # 解答：旧式 MHA 配置常没有 num_key_value_heads，此时每个 query head 都有自己的 K/V head，所以用 num_attention_heads 回退；有 GQA/MQA 字段时则使用其更小值。
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
        )
