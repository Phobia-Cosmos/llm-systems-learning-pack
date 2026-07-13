import unittest

from vllm.model_executor.models.minigpt import _map_checkpoint_name


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


if __name__ == "__main__":
    unittest.main()
