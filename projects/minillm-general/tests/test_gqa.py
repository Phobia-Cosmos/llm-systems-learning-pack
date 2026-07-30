from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from minillm import GPTConfig, MiniGPT
from minillm.debug import split_qkv_parameters, trace_forward
from minillm.position import SUPPORTED_POSITION_ENCODINGS


class GQAConfigAndProjectionTests(unittest.TestCase):
    def test_legacy_default_resolves_to_mha_and_invalid_groupings_are_rejected(self):
        config = GPTConfig(vocab_size=32, n_head=4, n_embd=16)
        self.assertEqual(config.num_key_value_heads, 4)

        for value in (0, 3, 5):
            with self.subTest(num_key_value_heads=value):
                with self.assertRaises(ValueError):
                    GPTConfig(
                        vocab_size=32,
                        n_head=4,
                        n_embd=16,
                        num_key_value_heads=value,
                    )

    def test_fused_and_separate_projections_use_compact_kv_width(self):
        hidden_states = torch.randn(2, 3, 16)
        for fused_qkv in (True, False):
            with self.subTest(fused_qkv=fused_qkv):
                model = MiniGPT(
                    GPTConfig(
                        vocab_size=32,
                        n_layer=1,
                        n_head=4,
                        n_embd=16,
                        num_key_value_heads=2,
                        fused_qkv=fused_qkv,
                        dropout=0.0,
                    )
                )
                attention = model.blocks[0].attn
                query, key, value = attention.project_qkv(hidden_states)

                self.assertEqual(tuple(query.shape), (2, 3, 16))
                self.assertEqual(tuple(key.shape), (2, 3, 8))
                self.assertEqual(tuple(value.shape), (2, 3, 8))
                self.assertEqual(attention.q_size, 16)
                self.assertEqual(attention.kv_size, 8)
                if fused_qkv:
                    self.assertEqual(attention.c_attn.out_features, 32)
                else:
                    self.assertEqual(attention.k_proj.out_features, 8)
                    self.assertEqual(attention.v_proj.out_features, 8)

                parameters = split_qkv_parameters(attention)
                self.assertEqual(parameters["q"][0].shape[0], 16)
                self.assertEqual(parameters["k"][0].shape[0], 8)
                self.assertEqual(parameters["v"][0].shape[0], 8)

    def test_expand_kv_heads_uses_contiguous_query_groups(self):
        attention = MiniGPT(
            GPTConfig(
                vocab_size=8,
                n_layer=1,
                n_head=4,
                n_embd=8,
                num_key_value_heads=2,
            )
        ).blocks[0].attn
        compact = torch.tensor([[[[10.0]], [[20.0]]]])
        expanded = attention.expand_kv_heads(compact)
        torch.testing.assert_close(
            expanded.flatten(),
            torch.tensor([10.0, 10.0, 20.0, 20.0]),
        )


class GQAForwardAndCacheTests(unittest.TestCase):
    def _assert_full_matches_cache(
        self,
        *,
        num_key_value_heads: int,
        position_encoding: str,
        fused_qkv: bool = True,
    ) -> None:
        torch.manual_seed(11)
        config = GPTConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=4,
            n_embd=16,
            num_key_value_heads=num_key_value_heads,
            fused_qkv=fused_qkv,
            dropout=0.0,
            position_encoding=position_encoding,
        )
        model = MiniGPT(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 6))

        with torch.no_grad():
            full_logits, _ = model(input_ids)
            cache = None
            cached_logits = []
            for index in range(input_ids.size(1)):
                logits, cache = model.forward_with_cache(
                    input_ids[:, index : index + 1],
                    cache,
                )
                cached_logits.append(logits)

        torch.testing.assert_close(
            torch.cat(cached_logits, dim=1),
            full_logits,
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertIsNotNone(cache)
        for key, value in cache:
            self.assertEqual(
                tuple(key.shape),
                (2, num_key_value_heads, input_ids.size(1), config.n_embd // config.n_head),
            )
            self.assertEqual(value.shape, key.shape)

    def test_mha_gqa_and_mqa_match_token_by_token_cache(self):
        for num_key_value_heads in (4, 2, 1):
            for fused_qkv in (True, False):
                with self.subTest(
                    num_key_value_heads=num_key_value_heads,
                    fused_qkv=fused_qkv,
                ):
                    self._assert_full_matches_cache(
                        num_key_value_heads=num_key_value_heads,
                        position_encoding="rope",
                        fused_qkv=fused_qkv,
                    )

    def test_gqa_cache_parity_for_every_position_encoding(self):
        for position_encoding in SUPPORTED_POSITION_ENCODINGS:
            with self.subTest(position_encoding=position_encoding):
                self._assert_full_matches_cache(
                    num_key_value_heads=2,
                    position_encoding=position_encoding,
                )

    def test_gqa_cache_has_half_the_mha_elements(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        cache_elements = {}
        for num_key_value_heads in (4, 2, 1):
            model = MiniGPT(
                GPTConfig(
                    vocab_size=8,
                    block_size=8,
                    n_layer=1,
                    n_head=4,
                    n_embd=16,
                    num_key_value_heads=num_key_value_heads,
                    dropout=0.0,
                    position_encoding="rope",
                )
            ).eval()
            with torch.no_grad():
                _, cache = model.forward_with_cache(input_ids)
            cache_elements[num_key_value_heads] = sum(
                key.numel() + value.numel() for key, value in cache
            )

        self.assertEqual(cache_elements[2] * 2, cache_elements[4])
        self.assertEqual(cache_elements[1] * 4, cache_elements[4])

    def test_debug_trace_exposes_compact_and_expanded_kv(self):
        model = MiniGPT(
            GPTConfig(
                vocab_size=16,
                block_size=8,
                n_layer=1,
                n_head=4,
                n_embd=16,
                num_key_value_heads=2,
                dropout=0.0,
                position_encoding="rope",
            )
        ).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        trace = trace_forward(model, input_ids, input_ids)

        self.assertTrue(all(trace["checks"].values()))
        block = trace["blocks"][0]
        self.assertEqual(tuple(block["k_heads_grouped"].shape), (1, 2, 4, 4))
        self.assertEqual(tuple(block["v_heads_grouped"].shape), (1, 2, 4, 4))
        self.assertEqual(tuple(block["k_heads"].shape), (1, 4, 4, 4))
        self.assertEqual(tuple(block["v_heads"].shape), (1, 4, 4, 4))

    def test_gqa_state_dict_roundtrip(self):
        config = GPTConfig(
            vocab_size=16,
            n_layer=1,
            n_head=4,
            n_embd=16,
            num_key_value_heads=2,
        )
        model = MiniGPT(config).eval()
        input_ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected, _ = model(input_ids)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gqa.pt"
            torch.save(model.state_dict(), path)
            restored = MiniGPT(config).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
        with torch.no_grad():
            actual, _ = restored(input_ids)
        torch.testing.assert_close(actual, expected)


class LegacyCompatibilityAndGenerationBoundaryTests(unittest.TestCase):
    def test_explicit_mha_is_bitwise_identical_to_legacy_default(self):
        common = dict(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=4,
            n_embd=16,
            dropout=0.0,
            position_encoding="rope",
        )
        torch.manual_seed(23)
        legacy = MiniGPT(GPTConfig(**common)).eval()
        torch.manual_seed(23)
        explicit = MiniGPT(GPTConfig(**common, num_key_value_heads=4)).eval()

        self.assertEqual(legacy.state_dict().keys(), explicit.state_dict().keys())
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            legacy_logits, _ = legacy(input_ids)
            explicit_logits, _ = explicit(input_ids)
        self.assertTrue(torch.equal(legacy_logits, explicit_logits))

    def test_existing_rope_checkpoint_loads_with_mha_default(self):
        checkpoint_path = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "checkpoints"
            / "minillm-rope.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.assertNotIn("num_key_value_heads", checkpoint["config"])
        config = GPTConfig(**checkpoint["config"])
        self.assertEqual(config.num_key_value_heads, config.n_head)
        MiniGPT(config).load_state_dict(checkpoint["model"])

    def test_kv_generation_accepts_the_true_block_size_boundary(self):
        torch.manual_seed(29)
        model = MiniGPT(
            GPTConfig(
                vocab_size=16,
                block_size=4,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
            )
        ).eval()

        for prompt, max_new_tokens in (
            (torch.tensor([[1, 2, 3, 4]]), 1),
            (torch.tensor([[1, 2, 3]]), 2),
        ):
            with self.subTest(prompt_len=prompt.size(1), max_new_tokens=max_new_tokens):
                ordinary = model.generate(
                    prompt.clone(), max_new_tokens=max_new_tokens, greedy=True
                )
                cached = model.generate_with_kv_cache(
                    prompt.clone(), max_new_tokens=max_new_tokens, greedy=True
                )
                torch.testing.assert_close(cached, ordinary)

        with self.assertRaisesRegex(ValueError, "max_new_tokens - 1"):
            model.generate_with_kv_cache(
                torch.tensor([[1, 2, 3]]), max_new_tokens=3, greedy=True
            )


if __name__ == "__main__":
    unittest.main()
