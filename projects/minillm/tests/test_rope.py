from __future__ import annotations

import unittest

import torch

from minillm import GPTConfig, MiniGPT
from minillm.rope import RotaryEmbedding


class RotaryEmbeddingTests(unittest.TestCase):
    def test_position_zero_is_identity_and_rotation_preserves_norm(self):
        rope = RotaryEmbedding(head_dim=8, max_seq_len=16, base=10000.0)
        query = torch.randn(2, 3, 4, 8)
        key = torch.randn(2, 3, 4, 8)
        positions = torch.arange(4)

        rotated_query, rotated_key = rope(query, key, positions)

        torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
        torch.testing.assert_close(rotated_key[:, :, 0], key[:, :, 0])
        torch.testing.assert_close(
            rotated_query.float().norm(dim=-1),
            query.float().norm(dim=-1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_invalid_rope_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "even head_dim"):
            GPTConfig(vocab_size=32, n_embd=12, n_head=4, position_encoding="rope")


class MiniGPTRoPETests(unittest.TestCase):
    def test_learned_default_is_checkpoint_compatible(self):
        config = GPTConfig(vocab_size=32)
        model = MiniGPT(config).to("cpu")

        self.assertEqual(config.position_encoding, "learned")
        self.assertIsNotNone(model.position_embedding)
        self.assertIn("position_embedding.weight", model.state_dict())

    def test_rope_replaces_absolute_position_parameters(self):
        config = GPTConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=4,
            n_embd=32,
            dropout=0.0,
            position_encoding="rope",
        )
        model = MiniGPT(config).to("cpu")

        self.assertIsNone(model.position_embedding)
        self.assertNotIn("position_embedding.weight", model.state_dict())
        self.assertIsNotNone(model.blocks[0].attn.rotary)

    def test_rope_full_forward_matches_token_by_token_kv_cache(self):
        torch.manual_seed(7)
        config = GPTConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=4,
            n_embd=32,
            dropout=0.0,
            position_encoding="rope",
        )
        model = MiniGPT(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 7))

        with torch.no_grad():
            full_logits, _ = model(input_ids)
            cached_logits = []
            past_key_values = None
            for position in range(input_ids.size(1)):
                step_logits, past_key_values = model.forward_with_cache(
                    input_ids[:, position : position + 1],
                    past_key_values,
                )
                cached_logits.append(step_logits)

        torch.testing.assert_close(
            torch.cat(cached_logits, dim=1),
            full_logits,
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
