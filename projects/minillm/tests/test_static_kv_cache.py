from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from minillm import GPTConfig, MiniGPT, StaticKVCache


class StaticKVCacheParityTests(unittest.TestCase):
    def _make_model(self, num_key_value_heads: int, position_encoding: str) -> MiniGPT:
        torch.manual_seed(101)
        return MiniGPT(
            GPTConfig(
                vocab_size=32,
                block_size=12,
                n_layer=2,
                n_head=4,
                n_embd=16,
                num_key_value_heads=num_key_value_heads,
                dropout=0.0,
                position_encoding=position_encoding,
            )
        ).eval()

    def test_full_legacy_and_static_logits_match_for_mha_gqa_mqa_and_positions(self):
        prompt = torch.tensor([[1, 4, 2, 7], [3, 5, 6, 8]])
        for num_key_value_heads in (4, 2, 1):
            for position_encoding in ("learned", "rope"):
                with self.subTest(
                    num_key_value_heads=num_key_value_heads,
                    position_encoding=position_encoding,
                ):
                    model = self._make_model(num_key_value_heads, position_encoding)
                    full_prefill, _ = model(prompt)
                    legacy_prefill, legacy_cache = model.forward_with_cache(prompt)
                    static_cache = model.allocate_static_kv_cache(2, max_len=8)
                    static_prefill, returned_cache = model.forward_with_static_cache(
                        prompt, static_cache
                    )

                    self.assertIs(returned_cache, static_cache)
                    torch.testing.assert_close(legacy_prefill, full_prefill)
                    torch.testing.assert_close(static_prefill, full_prefill)

                    next_token = full_prefill[:, -1].argmax(dim=-1, keepdim=True)
                    full_decode, _ = model(torch.cat((prompt, next_token), dim=1))
                    legacy_decode, _ = model.forward_with_cache(next_token, legacy_cache)
                    static_decode, _ = model.forward_with_static_cache(next_token, static_cache)
                    torch.testing.assert_close(legacy_decode[:, -1], full_decode[:, -1])
                    torch.testing.assert_close(static_decode[:, -1], full_decode[:, -1])

                    ordinary = model.generate(prompt, max_new_tokens=3, greedy=True)
                    cached = model.generate_with_kv_cache(
                        prompt, max_new_tokens=3, greedy=True
                    )
                    torch.testing.assert_close(cached, ordinary)

    def test_storage_addresses_stay_fixed_across_decode_and_reset(self):
        model = self._make_model(num_key_value_heads=2, position_encoding="rope")
        cache = model.allocate_static_kv_cache(batch_size=1, max_len=8)
        pointers = [
            tensor.data_ptr()
            for pair in zip(cache.key_caches, cache.value_caches)
            for tensor in pair
        ]

        logits, _ = model.forward_with_static_cache(torch.tensor([[1, 2, 3]]), cache)
        for _ in range(3):
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            logits, _ = model.forward_with_static_cache(token, cache)

        self.assertEqual(cache.length, 6)
        self.assertEqual(
            pointers,
            [
                tensor.data_ptr()
                for pair in zip(cache.key_caches, cache.value_caches)
                for tensor in pair
            ],
        )
        cache.reset()
        self.assertEqual(cache.length, 0)
        model.forward_with_static_cache(torch.tensor([[4, 5]]), cache)
        self.assertEqual(cache.length, 2)
        self.assertEqual(pointers[0], cache.key_caches[0].data_ptr())

    def test_static_forward_does_not_call_torch_cat(self):
        model = self._make_model(num_key_value_heads=2, position_encoding="learned")
        cache = model.allocate_static_kv_cache(batch_size=1, max_len=6)
        with patch.object(torch, "cat", side_effect=AssertionError("unexpected torch.cat")):
            model.forward_with_static_cache(torch.tensor([[1, 2, 3]]), cache)
            model.forward_with_static_cache(torch.tensor([[4]]), cache)

    def test_cpu_autocast_allocates_and_uses_matching_cache_dtype(self):
        model = self._make_model(num_key_value_heads=2, position_encoding="rope")
        prompt = torch.tensor([[1, 2, 3]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            full, _ = model(prompt)
            cache = model.allocate_static_kv_cache(batch_size=1, max_len=5)
            static, _ = model.forward_with_static_cache(prompt, cache)

        self.assertEqual(cache.dtype, torch.bfloat16)
        torch.testing.assert_close(static, full)

    def test_generation_uses_legacy_prefill_once_then_only_static_decode(self):
        model = self._make_model(num_key_value_heads=2, position_encoding="rope")
        prompt = torch.tensor([[1, 2, 3]])
        with (
            patch.object(
                model,
                "forward_with_cache",
                wraps=model.forward_with_cache,
            ) as legacy_forward,
            patch.object(
                model,
                "forward_with_static_cache",
                wraps=model.forward_with_static_cache,
            ) as static_forward,
        ):
            model.generate_with_kv_cache(prompt, max_new_tokens=4, greedy=True)

        self.assertEqual(legacy_forward.call_count, 1)
        self.assertEqual(static_forward.call_count, 3)


class StaticKVCacheValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MiniGPT(
            GPTConfig(
                vocab_size=16,
                block_size=6,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
            )
        ).eval()

    def test_capacity_and_batch_boundaries(self):
        cache = self.model.allocate_static_kv_cache(batch_size=1, max_len=3)
        self.model.forward_with_static_cache(torch.tensor([[1, 2, 3]]), cache)
        with self.assertRaisesRegex(ValueError, "exceeds static KV capacity"):
            self.model.forward_with_static_cache(torch.tensor([[4]]), cache)

        cache.reset()
        with self.assertRaisesRegex(ValueError, "batch size"):
            self.model.forward_with_static_cache(torch.tensor([[1], [2]]), cache)

    def test_invalid_allocations_are_rejected(self):
        for batch_size, max_len in ((0, 2), (1, 0), (1, 7)):
            with self.subTest(batch_size=batch_size, max_len=max_len):
                with self.assertRaises(ValueError):
                    self.model.allocate_static_kv_cache(batch_size, max_len)

    def test_cache_dataclass_rejects_inconsistent_storage(self):
        key = torch.empty(1, 2, 4, 4)
        value = torch.empty(1, 2, 3, 4)
        with self.assertRaisesRegex(ValueError, "same shape"):
            StaticKVCache([key], [value], max_len=4)


if __name__ == "__main__":
    unittest.main()
