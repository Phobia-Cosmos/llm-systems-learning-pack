import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.minigpt import (
    MiniGPTAttention,
    ScaleNorm,
    SinusoidalPositionEmbedding,
    _build_activation,
    _get_alibi_slopes,
    _map_checkpoint_name,
    _num_kv_heads,
)


class MiniGPTCheckpointNameMappingTests(unittest.TestCase):
    def test_minigpt_checkpoint_name_mapping(self) -> None:
        self.assertEqual(
            _map_checkpoint_name("blocks.0.mlp.net.0.weight"),
            "blocks.0.mlp.fc_in.weight",
        )
        self.assertEqual(
            _map_checkpoint_name("blocks.1.mlp.net.2.bias"),
            "blocks.1.mlp.fc_out.bias",
        )
        self.assertEqual(
            _map_checkpoint_name("blocks.0.attn.c_attn.weight"),
            "blocks.0.attn.c_attn.weight",
        )

    def test_new_component_helpers(self) -> None:
        sinusoidal = SinusoidalPositionEmbedding(8, 6)
        values = sinusoidal(torch.tensor([0]))
        torch.testing.assert_close(values[0, 0::2], torch.zeros(3))
        torch.testing.assert_close(values[0, 1::2], torch.ones(3))

        self.assertEqual(_get_alibi_slopes(3).numel(), 3)
        normalized = ScaleNorm(4)(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        torch.testing.assert_close(normalized.norm(dim=-1), torch.tensor([2.0]))
        torch.testing.assert_close(
            _build_activation("leaky_relu")(torch.tensor([-1.0, 2.0])),
            torch.tensor([-0.01, 2.0]),
        )

    def test_kv_heads_default_to_query_heads_and_accept_gqa(self) -> None:
        legacy = SimpleNamespace(num_attention_heads=8)
        grouped = SimpleNamespace(
            num_attention_heads=8,
            num_key_value_heads=2,
        )

        self.assertEqual(_num_kv_heads(legacy), 8)
        self.assertEqual(_num_kv_heads(grouped), 2)

    @patch("vllm.model_executor.models.minigpt.Attention")
    @patch("vllm.model_executor.models.minigpt.RowParallelLinear")
    @patch("vllm.model_executor.models.minigpt.QKVParallelLinear")
    @patch(
        "vllm.model_executor.models.minigpt.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    def test_attention_propagates_gqa_sizes_to_vllm_layers(
        self,
        _world_size,
        qkv_linear,
        _row_linear,
        attention_layer,
    ) -> None:
        qkv_module = MagicMock()
        qkv_module.num_kv_heads = 2
        qkv_linear.return_value = qkv_module
        config = SimpleNamespace(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            bias=True,
            position_encoding="none",
        )

        attention = MiniGPTAttention(config, prefix="blocks.0.attn")

        self.assertEqual(attention.q_size, 16)
        self.assertEqual(attention.kv_size, 8)
        qkv_linear.assert_called_once_with(
            16,
            4,
            4,
            2,
            bias=True,
            quant_config=None,
            prefix="blocks.0.attn.c_attn",
        )
        attention_layer.assert_called_once_with(
            4,
            4,
            scale=0.5,
            num_kv_heads=2,
            alibi_slopes=None,
            cache_config=None,
            quant_config=None,
            prefix="blocks.0.attn.attn",
        )

    def test_fused_gqa_checkpoint_weight_loads_with_compact_kv_rows(self) -> None:
        with (
            patch(
                "vllm.model_executor.layers.linear."
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.model_executor.parameter."
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
                return_value=0,
            ),
        ):
            layer = QKVParallelLinear(
                hidden_size=16,
                head_size=4,
                total_num_heads=4,
                total_num_kv_heads=2,
                bias=True,
            )

        checkpoint_weight = torch.arange(
            32 * 16, dtype=layer.weight.dtype
        ).reshape(32, 16)
        checkpoint_bias = torch.arange(32, dtype=layer.bias.dtype)
        layer.weight.weight_loader(layer.weight, checkpoint_weight)
        layer.bias.weight_loader(layer.bias, checkpoint_bias)

        self.assertEqual(tuple(layer.weight.shape), (32, 16))
        self.assertEqual(tuple(layer.bias.shape), (32,))
        torch.testing.assert_close(layer.weight, checkpoint_weight)
        torch.testing.assert_close(layer.bias, checkpoint_bias)


if __name__ == "__main__":
    unittest.main()
