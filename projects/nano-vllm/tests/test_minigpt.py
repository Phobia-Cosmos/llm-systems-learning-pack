import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nanovllm.layers.layernorm import RMSNorm, ScaleNorm
from nanovllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from nanovllm.layers.position_encoding import SinusoidalPositionEmbedding
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.minigpt import MiniGPTAttention, MiniGPTCharTokenizer, MiniGPTConfig
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

        # 问题（已回答）：为什么测试这些标准属性，它们代表什么？
        # 回答：MiniGPTConfig 同时接受 n_layer/n_head/n_embd 等教学命名和 HF 标准命名；这些断言验证二者被
        # 规范化为同一组层数、Q head 数、隐藏维度、每头维度及最大位置，供通用 engine 读取而不会出现歧义。
        # 问题（已回答）：num_key_value_heads、rope_theta、mlp_type 和 intermediate_size 分别是什么？
        # 回答：num_key_value_heads 是独立 K/V head 数，默认等于 Q heads 即 MHA；rope_theta 是训练配置确定的
        # RoPE 频率基数，加载 checkpoint 推理时不能随意改变。默认 dense MLP 是 Linear-激活-Linear，另支持
        # swiglu/geglu/reglu；intermediate_size 是 FFN 中间宽度，dense 默认是 hidden_size 的四倍。
        self.assertEqual(config.max_position_embeddings, 64)
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.head_dim, 32)
        self.assertEqual(config.position_encoding, "learned")
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertEqual(config.norm_type, "layernorm")
        self.assertEqual(config.mlp_type, "dense")
        self.assertEqual(config.activation, "gelu")
        self.assertEqual(config.intermediate_size, 512)

    def test_conflicting_aliases_are_rejected(self):
        # 问题（已回答）：这个 assertRaisesRegex 在测试什么？
        # 回答：n_layer 和 num_hidden_layers 是同一配置的两个别名；同时传入互相冲突的值必须抛出 ValueError，
        # 并让错误消息包含 "Conflicting n_layer"，防止实现静默选择其中一个而构造出意外层数的模型。
        with self.assertRaisesRegex(ValueError, "Conflicting n_layer"):
            MiniGPTConfig(n_layer=2, num_hidden_layers=3)

    def test_grouped_query_attention_heads_are_normalized(self):
        config = MiniGPTConfig(n_head=8, num_key_value_heads=2, n_embd=32)

        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.num_key_value_heads, 2)

    def test_query_heads_must_be_divisible_by_kv_heads(self):
        with self.assertRaisesRegex(
            ValueError, "num_attention_heads must be divisible by num_key_value_heads"
        ):
            MiniGPTConfig(n_head=6, num_key_value_heads=4, n_embd=24)


class MiniGPTAttentionTests(unittest.TestCase):

    def test_grouped_query_attention_uses_distinct_q_and_kv_sizes(self):
        config = MiniGPTConfig(n_head=4, num_key_value_heads=2, n_embd=16)
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            attention = MiniGPTAttention(config)

        self.assertEqual(attention.num_heads, 4)
        self.assertEqual(attention.num_kv_heads, 2)
        self.assertEqual(attention.q_size, 16)
        self.assertEqual(attention.kv_size, 8)
        self.assertEqual(attention.c_attn.total_num_kv_heads, 2)
        self.assertEqual(attention.attn.num_kv_heads, 2)

    def test_tensor_parallel_size_must_partition_kv_heads(self):
        config = MiniGPTConfig(n_head=4, num_key_value_heads=1, n_embd=16)
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=0),
            self.assertRaisesRegex(
                ValueError,
                "num_key_value_heads=1 must be divisible by tensor_parallel_size=2",
            ),
        ):
            MiniGPTAttention(config)


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

    def test_registry_builds_modular_position_norm_and_mlp_variants(self):
        base_payload = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        for position_encoding in ("sinusoidal", "alibi", "none"):
            with self.subTest(position_encoding=position_encoding):
                payload = {
                    **base_payload,
                    "position_encoding": position_encoding,
                    "norm_type": "rmsnorm",
                    "norm_eps": 1e-5,
                    "mlp_type": "swiglu",
                    "activation": "silu",
                    "intermediate_size": 24,
                }
                (self.model_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
                config = load_model_config(str(self.model_path))
                with (
                    patch("torch.distributed.get_world_size", return_value=1),
                    patch("torch.distributed.get_rank", return_value=0),
                ):
                    model = create_model(config)

                self.assertIsInstance(model.blocks[0].ln_1, RMSNorm)
                self.assertIsNotNone(model.get_parameter("blocks.0.mlp.gate_up_proj.weight"))
                self.assertIsNotNone(model.get_parameter("blocks.0.mlp.down_proj.weight"))
                if position_encoding == "sinusoidal":
                    self.assertIsInstance(model.position_embedding, SinusoidalPositionEmbedding)
                else:
                    self.assertIsNone(model.position_embedding)
                if position_encoding == "alibi":
                    self.assertEqual(model.blocks[0].attn.attn.alibi_slopes.numel(), 2)

    def test_scalenorm_model_uses_scale_parameter(self):
        payload = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        payload["norm_type"] = "scalenorm"
        (self.model_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        config = load_model_config(str(self.model_path))
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            model = create_model(config)
        self.assertIsInstance(model.blocks[0].ln_1, ScaleNorm)
        self.assertIsNotNone(model.get_parameter("blocks.0.ln_1.scale"))

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

    def test_fused_gqa_weight_is_partitioned_per_component(self):
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
        ):
            layer = QKVParallelLinear(
                hidden_size=16,
                head_size=4,
                total_num_heads=4,
                total_num_kv_heads=2,
                bias=True,
            )

        loaded_weight = torch.arange(32 * 16, dtype=layer.weight.dtype).reshape(32, 16)
        loaded_bias = torch.arange(32, dtype=layer.bias.dtype)
        layer.weight.weight_loader(layer.weight, loaded_weight)
        layer.bias.weight_loader(layer.bias, loaded_bias)

        expected_rows = torch.tensor(
            [*range(8, 16), *range(20, 24), *range(28, 32)]
        )
        torch.testing.assert_close(layer.weight, loaded_weight[expected_rows])
        torch.testing.assert_close(layer.bias, loaded_bias[expected_rows])

    def test_fused_merged_weight_loads_without_explicit_shard_id(self):
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
        ):
            layer = MergedColumnParallelLinear(4, [6, 6], bias=True)

        loaded_weight = torch.arange(12 * 4, dtype=layer.weight.dtype).reshape(12, 4)
        loaded_bias = torch.arange(12, dtype=layer.bias.dtype)
        layer.weight.weight_loader(layer.weight, loaded_weight)
        layer.bias.weight_loader(layer.bias, loaded_bias)
        expected_rows = torch.tensor([3, 4, 5, 9, 10, 11])
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
    # 问题（已回答）：这个主函数判断和 unittest.main() 做什么，如何运行上面的测试？
    # 回答：文件被直接执行时 __name__ 才等于 "__main__"；unittest.main() 会发现当前模块中继承
    # unittest.TestCase 且名称以 test 开头的方法并执行。可运行 python3 tests/test_minigpt.py，或在仓库根目录用
    # python3 -m unittest discover -s tests -p 'test_minigpt.py'；也可把具体测试类/方法路径交给 unittest runner。
    unittest.main()
