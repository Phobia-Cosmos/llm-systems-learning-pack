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
        max_position_embeddings: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        hidden_size: int | None = None,
        tie_word_embeddings: bool = True,
        position_encoding: str = "learned",
        rope_theta: float = 10000.0,
        sinusoidal_theta: float = 10000.0,
        norm_type: str = "layernorm",
        norm_eps: float | None = None,
        layer_norm_epsilon: float | None = None,
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
        if num_key_value_heads is not None and num_key_value_heads != n_head:
            raise ValueError("MiniGPT uses multi-head attention, so num_key_value_heads must equal n_head")
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
        self.num_key_value_heads = n_head
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
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


class MiniGPTAttention(nn.Module):

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if config.n_head % tp_size != 0:
            raise ValueError(f"n_head={config.n_head} must be divisible by tensor_parallel_size={tp_size}")
        self.num_heads = config.n_head // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
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
            config.n_head,
            bias=config.bias,
        )
        self.c_proj = RowParallelLinear(config.n_embd, config.n_embd, bias=config.bias)
        alibi_slopes = None
        if config.position_encoding == "alibi":
            head_start = dist.get_rank() * self.num_heads
            alibi_slopes = get_alibi_slopes(config.n_head)[head_start : head_start + self.num_heads]
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_heads,
            alibi_slopes=alibi_slopes,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split([self.q_size, self.q_size, self.q_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        if self.rotary is not None:
            q, k = self.rotary(positions, q, k)
        output = self.attn(q, k, v)
        return self.c_proj(output.flatten(1, -1))


class MiniGPTMLP(nn.Module):

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.mlp_type = config.mlp_type
        if self.mlp_type == "dense":
            self.net = nn.Sequential(
                ColumnParallelLinear(config.n_embd, config.intermediate_size, bias=config.bias),
                build_activation(config.activation),
                RowParallelLinear(config.intermediate_size, config.n_embd, bias=config.bias),
            )
        else:
            gate_activation = {"swiglu": "silu", "geglu": "gelu", "reglu": "relu"}[self.mlp_type]
            self.gate_up_proj = MergedColumnParallelLinear(
                config.n_embd,
                [config.intermediate_size, config.intermediate_size],
                bias=config.bias,
            )
            self.activation_and_mul = ActivationAndMul(gate_activation)
            self.down_proj = RowParallelLinear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "dense":
            return self.net(hidden_states)
        return self.down_proj(self.activation_and_mul(self.gate_up_proj(hidden_states)))


class MiniGPTBlock(nn.Module):

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln_1 = build_norm(config.n_embd, config.norm_type, config.norm_eps, config.bias)
        self.attn = MiniGPTAttention(config)
        self.ln_2 = build_norm(config.n_embd, config.norm_type, config.norm_eps, config.bias)
        self.mlp = MiniGPTMLP(config)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), positions)
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


@register_model(
    model_type="minigpt",
    architectures=("MiniGPTForCausalLM",),
    config_class=MiniGPTConfig,
    tokenizer_loader=load_minigpt_tokenizer,
)
class MiniGPTForCausalLM(nn.Module):

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
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
