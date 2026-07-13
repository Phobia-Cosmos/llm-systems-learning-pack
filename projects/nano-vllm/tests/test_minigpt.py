import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nanovllm.layers.linear import QKVParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.minigpt import MiniGPTCharTokenizer, MiniGPTConfig
from nanovllm.models.registry import create_model, load_model_config, load_tokenizer, supported_model_types


class MiniGPTConfigTests(unittest.TestCase):

    def test_canonical_and_minillm_names_are_normalized(self):
        config = MiniGPTConfig(
            vocab_size=32,
            block_size=64,
            n_layer=3,
            n_head=4,
            n_embd=128,
        )

        self.assertEqual(config.max_position_embeddings, 64)
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.head_dim, 32)
        self.assertEqual(config.position_encoding, "learned")
        self.assertEqual(config.rope_theta, 10000.0)

    def test_conflicting_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting n_layer"):
            MiniGPTConfig(n_layer=2, num_hidden_layers=3)


class MiniGPTRegistryTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name)
        (self.model_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "minigpt",
                    "architectures": ["MiniGPTForCausalLM"],
                    "vocab_size": 8,
                    "block_size": 16,
                    "max_position_embeddings": 16,
                    "n_layer": 2,
                    "num_hidden_layers": 2,
                    "n_head": 2,
                    "num_attention_heads": 2,
                    "n_embd": 8,
                    "hidden_size": 8,
                    "bias": True,
                    "tie_word_embeddings": True,
                    "torch_dtype": "float32",
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_registry_loads_config_and_model_with_exported_parameter_names(self):
        config = load_model_config(str(self.model_path))
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            model = create_model(config)

        expected_parameters = (
            "token_embedding.weight",
            "position_embedding.weight",
            "blocks.0.ln_1.weight",
            "blocks.0.attn.c_attn.weight",
            "blocks.0.attn.c_attn.bias",
            "blocks.0.attn.c_proj.weight",
            "blocks.0.mlp.net.0.weight",
            "blocks.0.mlp.net.2.weight",
            "ln_f.weight",
            "lm_head.weight",
        )
        for parameter_name in expected_parameters:
            self.assertIsNotNone(model.get_parameter(parameter_name))
        self.assertEqual(model.lm_head.weight.data_ptr(), model.token_embedding.weight.data_ptr())
        self.assertIn("minigpt", supported_model_types())
        self.assertIn("qwen3", supported_model_types())

    def test_registry_builds_rope_model_without_absolute_position_weight(self):
        payload = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        payload["position_encoding"] = "rope"
        payload["rope_theta"] = 10000.0
        (self.model_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

        config = load_model_config(str(self.model_path))
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            model = create_model(config)

        self.assertIsNone(model.position_embedding)
        self.assertNotIn("position_embedding.weight", model.state_dict())
        self.assertIsNotNone(model.blocks[0].attn.rotary)

    def test_registry_loads_minillm_character_tokenizer(self):
        (self.model_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "CharTokenizer"}), encoding="utf-8"
        )
        (self.model_path / "tokenizer.json").write_text(
            json.dumps(
                {
                    "stoi": {"<unk>": 0, "a": 1, "中": 2},
                    "itos": ["<unk>", "a", "中"],
                    "unk_token": "<unk>",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        config = load_model_config(str(self.model_path))
        tokenizer = load_tokenizer(str(self.model_path), config)

        self.assertIsInstance(tokenizer, MiniGPTCharTokenizer)
        self.assertEqual(tokenizer.encode("a中?"), [1, 2, 0])
        self.assertEqual(tokenizer.decode([1, 2, 0]), "a中?")
        self.assertIsNone(tokenizer.eos_token_id)


class FusedQKVLoaderTests(unittest.TestCase):

    def test_fused_qkv_weight_is_partitioned_per_component(self):
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
        ):
            layer = QKVParallelLinear(
                hidden_size=4,
                head_size=2,
                total_num_heads=2,
                total_num_kv_heads=2,
                bias=True,
            )

        loaded_weight = torch.arange(12 * 4, dtype=layer.weight.dtype).reshape(12, 4)
        loaded_bias = torch.arange(12, dtype=layer.bias.dtype)
        layer.weight.weight_loader(layer.weight, loaded_weight)
        layer.bias.weight_loader(layer.bias, loaded_bias)

        expected_rows = torch.tensor([2, 3, 6, 7, 10, 11])
        torch.testing.assert_close(layer.weight, loaded_weight[expected_rows])
        torch.testing.assert_close(layer.bias, loaded_bias[expected_rows])


class RotaryEmbeddingTests(unittest.TestCase):

    def test_flattened_rope_preserves_norm_and_position_zero(self):
        rope = get_rope(head_size=8, rotary_dim=8, max_position=16, base=10000.0)
        query = torch.randn(5, 2, 8)
        key = torch.randn(5, 2, 8)
        rotated_query, rotated_key = rope(torch.arange(5), query, key)

        torch.testing.assert_close(rotated_query[0], query[0])
        torch.testing.assert_close(rotated_key[0], key[0])
        torch.testing.assert_close(
            rotated_query.float().norm(dim=-1),
            query.float().norm(dim=-1),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
