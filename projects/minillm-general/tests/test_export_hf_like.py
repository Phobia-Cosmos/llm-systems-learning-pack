from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import export_hf_like
from minillm import GPTConfig, MiniGPT
from minillm.tokenizer import CharTokenizer
from minillm.tokenizer_registry import tokenizer_to_checkpoint_payload


class HFLikeExportTests(unittest.TestCase):
    def test_gqa_export_records_canonical_heads_dtype_and_compact_weights(self):
        tokenizer = CharTokenizer.from_text("abcabc")
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=8,
            n_layer=1,
            n_head=4,
            n_embd=8,
            num_key_value_heads=2,
            dropout=0.0,
            position_encoding="rope",
        )
        model = MiniGPT(config)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "checkpoint.pt"
            output_path = root / "export"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "tokenizer_type": "char",
                    "tokenizer": tokenizer_to_checkpoint_payload("char", tokenizer),
                },
                checkpoint_path,
            )

            argv = [
                "export_hf_like.py",
                "--checkpoint",
                str(checkpoint_path),
                "--out-dir",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv):
                export_hf_like.main()

            exported_config = json.loads(
                (output_path / "config.json").read_text(encoding="utf-8")
            )
            exported_state = torch.load(
                output_path / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )

        self.assertEqual(exported_config["num_attention_heads"], 4)
        self.assertEqual(exported_config["num_key_value_heads"], 2)
        self.assertEqual(exported_config["torch_dtype"], "float32")
        self.assertEqual(
            tuple(exported_state["blocks.0.attn.c_attn.weight"].shape),
            (16, 8),
        )


if __name__ == "__main__":
    unittest.main()
