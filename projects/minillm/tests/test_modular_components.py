from __future__ import annotations

import unittest

import torch

from minillm import GPTConfig, MiniGPT
from minillm.activations import SUPPORTED_ACTIVATIONS, build_activation
from minillm.mlp import SUPPORTED_MLP_TYPES, build_mlp
from minillm.norm import RMSNorm, ScaleNorm, build_norm
from minillm.position import (
    SUPPORTED_POSITION_ENCODINGS,
    ALiBiAttentionPositionEncoding,
    SinusoidalPositionEmbedding,
    get_alibi_slopes,
)


class PositionEncodingTests(unittest.TestCase):
    def test_fixed_sinusoidal_values_and_no_parameters(self):
        encoding = SinusoidalPositionEmbedding(max_seq_len=8, hidden_size=6)
        values = encoding(torch.tensor([0, 1]))

        torch.testing.assert_close(values[0, 0::2], torch.zeros(3))
        torch.testing.assert_close(values[0, 1::2], torch.ones(3))
        self.assertEqual(sum(parameter.numel() for parameter in encoding.parameters()), 0)
        self.assertEqual(encoding.state_dict(), {})

    def test_alibi_bias_penalizes_distant_history(self):
        encoding = ALiBiAttentionPositionEncoding(num_heads=2)
        bias = encoding.attention_bias(
            torch.tensor([2]),
            torch.tensor([0, 1, 2]),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(bias.shape), (1, 2, 1, 3))
        self.assertTrue(torch.all(bias[:, :, :, 0] < bias[:, :, :, 1]))
        self.assertTrue(torch.all(bias[:, :, :, 1] < bias[:, :, :, 2]))
        torch.testing.assert_close(bias[:, :, :, 2], torch.zeros(1, 2, 1))
        self.assertEqual(get_alibi_slopes(3).numel(), 3)

    def test_every_position_mode_matches_kv_cache(self):
        torch.manual_seed(3)
        input_ids = torch.randint(0, 32, (2, 6))
        for position_encoding in SUPPORTED_POSITION_ENCODINGS:
            with self.subTest(position_encoding=position_encoding):
                config = GPTConfig(
                    vocab_size=32,
                    block_size=16,
                    n_layer=2,
                    n_head=4,
                    n_embd=32,
                    dropout=0.0,
                    position_encoding=position_encoding,
                )
                model = MiniGPT(config).eval()
                with torch.no_grad():
                    full_logits, _ = model(input_ids)
                    chunks = []
                    cache = None
                    for token_index in range(input_ids.size(1)):
                        logits, cache = model.forward_with_cache(
                            input_ids[:, token_index : token_index + 1],
                            cache,
                        )
                        chunks.append(logits)
                torch.testing.assert_close(
                    torch.cat(chunks, dim=1),
                    full_logits,
                    rtol=1e-5,
                    atol=1e-5,
                )


class NormalizationTests(unittest.TestCase):
    def test_rmsnorm_matches_definition(self):
        layer = RMSNorm(3, eps=1e-6)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        expected = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(layer(x), expected)

    def test_scalenorm_sets_vector_norm_to_learned_scale(self):
        layer = ScaleNorm(4)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output = layer(x)
        torch.testing.assert_close(output.norm(dim=-1), layer.scale.detach().view(1))

    def test_norm_factory_and_model_backward(self):
        for norm_type in ("layernorm", "rmsnorm", "scalenorm", "none"):
            with self.subTest(norm_type=norm_type):
                layer = build_norm(8, norm_type, eps=1e-5, bias=True)
                self.assertEqual(tuple(layer(torch.randn(2, 3, 8)).shape), (2, 3, 8))
                model = MiniGPT(
                    GPTConfig(
                        vocab_size=16,
                        n_layer=1,
                        n_head=2,
                        n_embd=8,
                        dropout=0.0,
                        norm_type=norm_type,
                    )
                )
                _, loss = model(
                    torch.randint(0, 16, (2, 4)),
                    torch.randint(0, 16, (2, 4)),
                )
                assert loss is not None
                loss.backward()


class ActivationAndMLPTests(unittest.TestCase):
    def test_all_plain_activations_preserve_shape_and_are_finite(self):
        values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        for name in SUPPORTED_ACTIVATIONS:
            with self.subTest(name=name):
                output = build_activation(name)(values)
                self.assertEqual(output.shape, values.shape)
                self.assertTrue(torch.isfinite(output).all())

        torch.testing.assert_close(
            build_activation("relu_squared")(values),
            torch.tensor([0.0, 0.0, 0.0, 0.25, 4.0]),
        )

    def test_dense_and_gated_mlps_preserve_hidden_shape(self):
        for mlp_type in SUPPORTED_MLP_TYPES:
            with self.subTest(mlp_type=mlp_type):
                config = GPTConfig(
                    vocab_size=16,
                    n_layer=1,
                    n_head=2,
                    n_embd=16,
                    dropout=0.0,
                    mlp_type=mlp_type,
                )
                output = build_mlp(config)(torch.randn(2, 5, config.n_embd))
                self.assertEqual(tuple(output.shape), (2, 5, config.n_embd))
                self.assertGreater(config.intermediate_size, 0)

    def test_default_architecture_keeps_historical_parameter_names(self):
        model = MiniGPT(GPTConfig(vocab_size=32, n_embd=16, n_head=2, n_layer=1))
        state = model.state_dict()
        self.assertIn("position_embedding.weight", state)
        self.assertIn("blocks.0.ln_1.weight", state)
        self.assertIn("blocks.0.ln_1.bias", state)
        self.assertIn("blocks.0.mlp.net.0.weight", state)
        self.assertIn("blocks.0.mlp.net.2.weight", state)


if __name__ == "__main__":
    unittest.main()
