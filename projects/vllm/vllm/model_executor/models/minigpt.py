# SPDX-License-Identifier: Apache-2.0
"""vLLM teaching backend for MiniLLM's MiniGPT architecture.

This is intentionally close to ``projects/minillm/minillm/model.py`` but uses
vLLM embedding, linear, logits, and attention layers so the model can enter the
normal vLLM KV-cache path. It is a learning implementation, not yet a tuned
production backend.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors


def _first_attr(config, *names: str):
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    raise AttributeError(f"none of {names} found in config")


def _hidden_size(config) -> int:
    return _first_attr(config, "hidden_size", "n_embd")


def _num_layers(config) -> int:
    return _first_attr(config, "num_hidden_layers", "n_layer")


def _num_heads(config) -> int:
    return _first_attr(config, "num_attention_heads", "n_head")


def _intermediate_size(config) -> int:
    return getattr(config, "intermediate_size", 4 * _hidden_size(config))


def _uses_rope(config) -> bool:
    return getattr(config, "position_encoding", "learned") == "rope"


def _map_checkpoint_name(name: str) -> str:
    """Map MiniLLM state-dict names to this backend's module names."""
    return name.replace(".mlp.net.0.", ".mlp.fc_in.").replace(
        ".mlp.net.2.", ".mlp.fc_out."
    )


class MiniGPTAttention(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = _hidden_size(config)
        total_num_heads = _num_heads(config)
        self.head_dim = hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5
        self.c_attn = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            total_num_heads,
            bias=getattr(config, "bias", True),
            quant_config=quant_config,
            prefix=f"{prefix}.c_attn",
        )
        self.c_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=getattr(config, "bias", True),
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=_first_attr(config, "max_position_embeddings", "block_size"),
                is_neox_style=True,
                rope_parameters={"rope_theta": getattr(config, "rope_theta", 10000.0)},
            )
            if _uses_rope(config)
            else None
        )
        self.attn = Attention(
            total_num_heads,
            self.head_dim,
            scale=self.scale,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        hidden_states = self.attn(q, k, v)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class MiniGPTMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = _hidden_size(config)
        self.fc_in = ColumnParallelLinear(
            hidden_size,
            _intermediate_size(config),
            bias=getattr(config, "bias", True),
            quant_config=quant_config,
            prefix=f"{prefix}.net.0",
        )
        self.fc_out = RowParallelLinear(
            _intermediate_size(config),
            hidden_size,
            bias=getattr(config, "bias", True),
            quant_config=quant_config,
            prefix=f"{prefix}.net.2",
        )
        self.act = get_act_fn(getattr(config, "hidden_act", "gelu"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class MiniGPTBlock(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = _hidden_size(config)
        self.ln_1 = nn.LayerNorm(hidden_size, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.attn = MiniGPTAttention(config, cache_config, quant_config, prefix=f"{prefix}.attn")
        self.ln_2 = nn.LayerNorm(hidden_size, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.mlp = MiniGPTMLP(config, quant_config, prefix=f"{prefix}.mlp")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(positions, self.ln_1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


@support_torch_compile
class MiniGPTModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        hidden_size = _hidden_size(config)

        self.token_embedding = VocabParallelEmbedding(
            config.vocab_size,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.token_embedding",
        )
        self.position_embedding = (
            None
            if _uses_rope(config)
            else nn.Embedding(
                _first_attr(config, "max_position_embeddings", "block_size"),
                hidden_size,
            )
        )
        self.blocks = nn.ModuleList(
            [
                MiniGPTBlock(
                    config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(_num_layers(config))
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=getattr(config, "layer_norm_epsilon", 1e-5))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        if self.position_embedding is not None:
            hidden_states = hidden_states + self.position_embedding(positions)
        for block in self.blocks:
            hidden_states = block(positions, hidden_states)
        return self.ln_f(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = _map_checkpoint_name(name)
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MiniGPTForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        hidden_size = _hidden_size(config)
        self.model = MiniGPTModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
        )
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head = self.lm_head.tie_weights(self.model.token_embedding)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, tensor in weights:
            name = _map_checkpoint_name(name)
            candidates = [name]
            if not name.startswith("model.") and not name.startswith("lm_head."):
                candidates.append("model." + name)
            for candidate in candidates:
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
                loaded_params.add(candidate)
                if candidate == "lm_head.weight":
                    loaded_params.add("model.token_embedding.weight")
                break
        return loaded_params
