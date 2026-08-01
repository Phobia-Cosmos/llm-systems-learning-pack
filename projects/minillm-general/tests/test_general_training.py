import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from minillm import GPTConfig, MiniGPT
from train_general import (
    PackedTokens,
    checkpoint_rank_state,
    learning_rate,
    nats_per_byte,
    resume_config_matches,
)


class DistributedCheckpointTests(unittest.TestCase):
    @staticmethod
    def state(rank):
        return {
            "rank": rank,
            "torch_rng_state": torch.tensor([rank], dtype=torch.uint8),
            "cuda_rng_state": torch.tensor([rank + 1], dtype=torch.uint8),
            "batch_generator_state": torch.tensor([rank + 2], dtype=torch.uint8),
        }

    def test_matching_world_size_restores_each_rank_state(self):
        checkpoint = {"rank_states": [self.state(rank) for rank in range(4)]}
        for rank in range(4):
            self.assertIs(checkpoint_rank_state(checkpoint, rank, 4), checkpoint["rank_states"][rank])

    def test_world_size_change_preserves_rank_zero_and_reseeds_new_ranks(self):
        primary = self.state(0)
        checkpoint = {
            "rank_states": [primary],
            "torch_rng_state": primary["torch_rng_state"],
            "cuda_rng_state": [primary["cuda_rng_state"]],
            "batch_generator_state": primary["batch_generator_state"],
        }
        self.assertEqual(checkpoint_rank_state(checkpoint, 0, 4)["rank"], 0)
        self.assertIsNone(checkpoint_rank_state(checkpoint, 1, 4))

    def test_legacy_checkpoint_state_is_supported(self):
        primary = self.state(0)
        checkpoint = {
            "torch_rng_state": primary["torch_rng_state"],
            "cuda_rng_state": [primary["cuda_rng_state"]],
            "batch_generator_state": primary["batch_generator_state"],
        }
        restored = checkpoint_rank_state(checkpoint, 0, 1)
        self.assertIsNotNone(restored)
        torch.testing.assert_close(restored["cuda_rng_state"], primary["cuda_rng_state"])


class TokenDrivenScheduleTests(unittest.TestCase):
    @staticmethod
    def args():
        return SimpleNamespace(
            learning_rate=3e-4,
            min_lr_ratio=0.1,
            warmup_steps=500,
            warmup_start_lr_ratio=0.0,
            schedule_start_step=0,
            schedule_end_step=305_176,
            max_steps=305_176,
            schedule_reference_tokens_per_step=32_768,
        )

    def test_world_size_change_keeps_lr_at_the_same_token(self):
        args = self.args()
        tokens = 5_900_468_224
        self.assertEqual(
            learning_rate(180_068, args, tokens),
            learning_rate(99_999, args, tokens),
        )

    def test_four_times_larger_update_advances_four_reference_steps(self):
        args = self.args()
        start_tokens = 1_000 * 32_768
        after_four_reference_steps = start_tokens + 4 * 32_768
        self.assertEqual(
            learning_rate(1_001, args, after_four_reference_steps),
            learning_rate(1_004, args, 1_004 * 32_768),
        )


class GeneralAttentionTests(unittest.TestCase):
    def test_sdpa_causal_mask_uses_boolean_storage(self):
        model = MiniGPT(
            GPTConfig(
                vocab_size=32,
                block_size=32,
                n_layer=1,
                n_head=2,
                n_embd=16,
                use_sdpa=True,
            )
        )
        self.assertEqual(model.blocks[0].attn.causal_mask.dtype, torch.bool)
        self.assertEqual(model.blocks[0].attn.causal_mask.numel(), 0)
        expected = torch.tril(torch.ones(8, 8, dtype=torch.bool)).view(1, 1, 8, 8)
        torch.testing.assert_close(
            model.blocks[0].attn.causal_mask_slice(0, 8, 8, torch.device("cpu")),
            expected,
        )

    def test_qk_norm_sdpa_matches_incremental_cache(self):
        torch.manual_seed(7)
        model = MiniGPT(
            GPTConfig(
                vocab_size=64,
                block_size=16,
                n_layer=2,
                n_head=4,
                num_key_value_heads=2,
                n_embd=32,
                intermediate_size=64,
                dropout=0.0,
                bias=False,
                position_encoding="rope",
                norm_type="rmsnorm",
                mlp_type="swiglu",
                activation="silu",
                qk_norm=True,
                use_sdpa=True,
            )
        ).eval()
        tokens = torch.randint(0, 64, (2, 8))
        full_logits, _ = model(tokens)
        cached_logits = []
        cache = None
        for index in range(tokens.size(1)):
            logits, cache = model.forward_with_cache(tokens[:, index : index + 1], cache)
            cached_logits.append(logits)
        torch.testing.assert_close(full_logits, torch.cat(cached_logits, dim=1), atol=2e-5, rtol=2e-5)

    def test_qk_norm_sdpa_backward_is_finite(self):
        model = MiniGPT(
            GPTConfig(
                vocab_size=32,
                block_size=8,
                n_layer=1,
                n_head=2,
                num_key_value_heads=1,
                n_embd=16,
                intermediate_size=32,
                dropout=0.0,
                bias=False,
                position_encoding="rope",
                norm_type="rmsnorm",
                mlp_type="swiglu",
                qk_norm=True,
                use_sdpa=True,
            )
        )
        tokens = torch.randint(0, 32, (2, 8))
        _, loss = model(tokens, tokens.roll(-1, dims=1))
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_gradient_checkpointing_matches_regular_training_forward(self):
        torch.manual_seed(11)
        model = MiniGPT(
            GPTConfig(
                vocab_size=32,
                block_size=8,
                n_layer=2,
                n_head=2,
                n_embd=16,
                dropout=0.0,
            )
        ).train()
        tokens = torch.randint(0, 32, (2, 8))
        regular, _ = model(tokens)
        model.set_gradient_checkpointing(True)
        recomputed, loss = model(tokens, tokens)
        torch.testing.assert_close(regular, recomputed)
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_training_accepts_noncontiguous_targets(self):
        model = MiniGPT(GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16))
        windows = torch.randint(0, 32, (2, 9))
        inputs = windows[:, :-1]
        targets = windows[:, 1:]
        self.assertFalse(targets.is_contiguous())
        _, loss = model(inputs, targets)
        self.assertIsNotNone(loss)
        loss.backward()


class PackedDatasetTests(unittest.TestCase):
    def test_packed_tokens_batch_supports_uint16_mmap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.bin"
            np.asarray(np.arange(64, dtype=np.uint16)).tofile(path)
            dataset = PackedTokens(path, block_size=8)
            inputs, targets = dataset.batch(3, torch.Generator().manual_seed(1), torch.device("cpu"))
            self.assertEqual(inputs.shape, (3, 8))
            self.assertEqual(targets.shape, (3, 8))
            self.assertEqual(inputs.dtype, torch.int64)
            torch.testing.assert_close(targets, inputs + 1)

    def test_contiguous_batch_layout_preserves_shifted_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.bin"
            np.asarray(np.arange(256, dtype=np.uint16)).tofile(path)
            dataset = PackedTokens(path, block_size=8)
            inputs, targets = dataset.batch(
                4,
                torch.Generator().manual_seed(1),
                torch.device("cpu"),
                layout="contiguous",
            )
            self.assertEqual(inputs.shape, (4, 8))
            torch.testing.assert_close(targets, inputs + 1)
            self.assertEqual(inputs[1, 0].item(), targets[0, -1].item() + 1)

    def test_fixed_record_batches_never_cross_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.bin"
            records = np.asarray(
                [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=np.uint16
            )
            records.tofile(path)
            dataset = PackedTokens(path, block_size=4, record_length=5)
            inputs, targets = dataset.batch(
                32, torch.Generator().manual_seed(1), torch.device("cpu"), layout="records"
            )
            self.assertTrue(set(inputs[:, 0].tolist()) <= {10, 20})
            torch.testing.assert_close(targets, inputs + 1)

    def test_fixed_record_dataset_rejects_misalignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.bin"
            np.asarray(np.arange(11), dtype=np.uint16).tofile(path)
            with self.assertRaisesRegex(ValueError, "not aligned"):
                PackedTokens(path, block_size=4, record_length=5)

    def test_prepare_long_context_dataset_is_aligned_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_dir = root / "tokenizer-source"
            train_source = root / "train.jsonl"
            holdout_source = root / "validation.jsonl"
            train_source.write_text(
                "".join(
                    json.dumps(
                        {
                            "source": "fixture",
                            "text": (f"TRAIN_ONLY document-{index} 中文 English code " * 80),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for index in range(400)
                ),
                encoding="utf-8",
            )
            holdout_source.write_text(
                "".join(
                    json.dumps(
                        {
                            "source": "fixture",
                            "text": (f"HOLDOUT_ONLY document-{index} 中文 English code " * 80),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for index in range(400)
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_packed_dataset.py",
                    "--input", str(train_source),
                    "--output-dir", str(tokenizer_dir),
                    "--vocab-size", "512",
                    "--tokenizer-max-documents", "100",
                    "--validation-fraction", "0.1",
                    "--test-fraction", "0.1",
                    "--min-chars", "8",
                ],
                check=True,
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )
            targets = root / "targets.json"
            targets.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "sequence_length": 31,
                        "holdout_test_fraction": 0.5,
                        "target_tokens_by_source_and_split": {
                            "fixture": {"train": 256, "validation": 256, "test": 256}
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifests = []
            for name in ("long-a", "long-b"):
                output = root / name
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/prepare_long_context_dataset.py",
                        "--train-input", str(train_source),
                        "--holdout-input", str(holdout_source),
                        "--output-dir", str(output),
                        "--tokenizer", str(tokenizer_dir / "tokenizer.json"),
                        "--targets", str(targets),
                    ],
                    check=True,
                    cwd=Path(__file__).parents[1],
                    capture_output=True,
                    text=True,
                )
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                manifests.append(manifest)
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["record_length"], 32)
                self.assertEqual(
                    manifest["split_policy"]["name"],
                    "preserve_train_partition_permanent_holdout_v1",
                )
                self.assertEqual(
                    {entry["role"] for entry in manifest["inputs"]},
                    {"train", "holdout"},
                )
                for split in ("train", "validation", "test"):
                    self.assertGreater(manifest["splits"][split]["records"], 0)
                    self.assertEqual(manifest["splits"][split]["tokens"] % 32, 0)
                    self.assertGreater(manifest["splits"][split]["utf8_bytes"], 0)

                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))

                def decode_split(split):
                    records = np.memmap(
                        output / f"{split}.bin", mode="r", dtype=np.uint16
                    ).reshape(-1, 32)
                    return "\n".join(
                        tokenizer.decode(record.tolist(), skip_special_tokens=True)
                        for record in records
                    )

                self.assertIn("TRAIN_ONLY", decode_split("train"))
                self.assertNotIn("HOLDOUT_ONLY", decode_split("train"))
                for split in ("validation", "test"):
                    decoded = decode_split(split)
                    self.assertIn("HOLDOUT_ONLY", decoded)
                    self.assertNotIn("TRAIN_ONLY", decoded)
            self.assertEqual(manifests[0], manifests[1])

            duplicate_holdout = root / "duplicate-validation.jsonl"
            duplicate_holdout.write_text(
                train_source.read_text(encoding="utf-8").splitlines(keepends=True)[0]
                + holdout_source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            duplicate_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_long_context_dataset.py",
                    "--train-input", str(train_source),
                    "--holdout-input", str(duplicate_holdout),
                    "--output-dir", str(root / "duplicate-output"),
                    "--tokenizer", str(tokenizer_dir / "tokenizer.json"),
                    "--targets", str(targets),
                ],
                check=False,
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(duplicate_result.returncode, 0)
            self.assertIn("permanent holdout violation", duplicate_result.stderr)
            self.assertFalse((root / "duplicate-output").exists())

            incomplete_targets = root / "incomplete-targets.json"
            incomplete_targets.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "sequence_length": 31,
                        "holdout_test_fraction": 0.5,
                        "target_tokens_by_source_and_split": {
                            "fixture": {"train": 256, "validation": 256}
                        },
                    }
                ),
                encoding="utf-8",
            )
            incomplete_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_long_context_dataset.py",
                    "--train-input", str(train_source),
                    "--holdout-input", str(holdout_source),
                    "--output-dir", str(root / "incomplete-output"),
                    "--tokenizer", str(tokenizer_dir / "tokenizer.json"),
                    "--targets", str(incomplete_targets),
                ],
                check=False,
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(incomplete_result.returncode, 0)
            self.assertIn("must define exactly", incomplete_result.stderr)
            self.assertFalse((root / "incomplete-output").exists())

    def test_prepare_packed_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            documents = [
                {"text": f"通用模型训练样本文档 {index}，包含中文 English and code tokens."}
                for index in range(200)
            ]
            documents.append(documents[0])
            source.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in documents),
                encoding="utf-8",
            )
            output = root / "packed"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_packed_dataset.py",
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--vocab-size",
                    "512",
                    "--tokenizer-max-documents",
                    "100",
                    "--validation-fraction",
                    "0.1",
                    "--test-fraction",
                    "0.1",
                    "--min-chars",
                    "8",
                    "--batch-documents",
                    "16",
                ],
                check=True,
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dtype"], "uint16")
            self.assertEqual(sum(split["documents"] for split in manifest["splits"].values()), 200)
            self.assertGreater(
                sum(split["utf8_bytes"] for split in manifest["splits"].values()),
                0,
            )
            from tokenizers import Tokenizer, pre_tokenizers

            trained_tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
            vocabulary = trained_tokenizer.get_vocab()
            self.assertTrue(
                all(symbol in vocabulary for symbol in pre_tokenizers.ByteLevel.alphabet())
            )
            for split in ("train", "validation", "test"):
                packed = np.memmap(output / f"{split}.bin", mode="r", dtype=np.uint16)
                self.assertEqual(len(packed), manifest["splits"][split]["tokens"])
                self.assertGreater(len(packed), 0)

            continued_output = root / "packed-continued"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_packed_dataset.py",
                    "--input",
                    str(source),
                    "--output-dir",
                    str(continued_output),
                    "--tokenizer",
                    str(output / "tokenizer.json"),
                    "--validation-fraction",
                    "0.1",
                    "--test-fraction",
                    "0.1",
                    "--min-chars",
                    "8",
                ],
                check=True,
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )
            continued_manifest = json.loads(
                (continued_output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                (continued_output / "tokenizer.json").read_bytes(),
                (output / "tokenizer.json").read_bytes(),
            )
            self.assertEqual(continued_manifest["vocab_size"], manifest["vocab_size"])
            self.assertIsNotNone(continued_manifest["tokenizer_source"])


class LearningRateTests(unittest.TestCase):
    def test_stage_local_schedule_does_not_jump_to_old_peak(self):
        args = SimpleNamespace(
            learning_rate=3e-5,
            min_lr_ratio=0.1,
            warmup_steps=0,
            warmup_start_lr_ratio=0.0,
            schedule_start_step=20_000,
            schedule_end_step=60_000,
            max_steps=60_000,
        )
        self.assertAlmostEqual(learning_rate(20_000, args), 3e-5)
        self.assertAlmostEqual(learning_rate(59_999, args), 3e-6, places=10)

    def test_context_extension_requires_rope_and_only_changes_block_size(self):
        base = {
            "block_size": 2048,
            "position_encoding": "rope",
            "n_layer": 22,
            "n_embd": 768,
        }
        extended = {**base, "block_size": 8192}
        self.assertTrue(resume_config_matches(base, extended, True))
        self.assertFalse(resume_config_matches(base, extended, False))
        self.assertFalse(
            resume_config_matches(base, {**extended, "n_layer": 24}, True)
        )
        self.assertFalse(
            resume_config_matches(
                {**base, "position_encoding": "learned"},
                {**extended, "position_encoding": "learned"},
                True,
            )
        )

    def test_nats_per_byte_normalizes_tokenizer_dependent_loss(self):
        self.assertAlmostEqual(nats_per_byte(2.0, 300, 200), 3.0)
        self.assertIsNone(nats_per_byte(2.0, 0, 200))


if __name__ == "__main__":
    unittest.main()
