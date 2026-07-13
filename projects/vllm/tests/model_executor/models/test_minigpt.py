import unittest

import torch

from vllm.model_executor.models.minigpt import (
    ScaleNorm,
    SinusoidalPositionEmbedding,
    _build_activation,
    _get_alibi_slopes,
    _map_checkpoint_name,
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


if __name__ == "__main__":
    unittest.main()
