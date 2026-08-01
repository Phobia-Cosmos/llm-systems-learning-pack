from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from minillm import GPTConfig, MiniGPT
from scripts.evaluate_long_context import (
    atomic_write_json,
    load_evaluation_bundle,
    position_ranges,
    sha256_file,
    verify_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


class LongContextFixture:
    def __init__(self, root: Path):
        self.root = root
        self.tokenizer_path = root / "tokenizer.json"
        vocab = {
            "<unk>": 0,
            "<|pad|>": 1,
            "<|eos|>": 2,
            "This": 3,
            "is": 4,
            "ordinary": 5,
            "background": 6,
            "text": 7,
            "with": 8,
            "no": 9,
            "secret": 10,
            "answer": 11,
            "Continue": 12,
            "reading": 13,
            "the": 14,
            "surrounding": 15,
            "document": 16,
            "carefully": 17,
            "Remember": 18,
            "this": 19,
            "fact": 20,
            "passkey": 21,
            "314159": 22,
            "Question": 23,
            "What": 24,
            "Answer": 25,
            ".": 26,
            ":": 27,
            "?": 28,
            "other": 29,
            "token": 30,
            "value": 31,
        }
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(self.tokenizer_path))
        self.tokenizer_sha256 = sha256_file(self.tokenizer_path)

        self.long_dataset = root / "long"
        self._write_fixed_dataset(self.long_dataset, sequence_length=32, records=3)
        self.regression_dataset = root / "regression"
        self._write_packed_dataset(self.regression_dataset, tokens=102)
        regression_manifest = self.regression_dataset / "manifest.json"
        self.regression_manifest_sha256 = sha256_file(regression_manifest)

        torch.manual_seed(17)
        config = GPTConfig(
            vocab_size=len(vocab),
            block_size=48,
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
            activation="silu",
            use_sdpa=True,
        )
        model = MiniGPT(config)
        self.checkpoint_path = root / "checkpoint.pt"
        torch.save(
            {
                "schema_version": 1,
                "step": 7,
                "tokens_processed": 12345,
                "dataset_manifest_sha256": self.regression_manifest_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "config": config.__dict__,
                "model": model.state_dict(),
                "args": {"sequence_length": 16},
            },
            self.checkpoint_path,
        )
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)

    def _copy_tokenizer(self, directory: Path) -> None:
        (directory / "tokenizer.json").write_bytes(self.tokenizer_path.read_bytes())

    def _write_fixed_dataset(
        self,
        directory: Path,
        *,
        sequence_length: int,
        records: int,
    ) -> None:
        directory.mkdir()
        self._copy_tokenizer(directory)
        record_length = sequence_length + 1
        rng = np.random.default_rng(101)
        split_metadata = {}
        for split in ("train", "validation", "test"):
            path = directory / f"{split}.bin"
            values = rng.integers(
                0,
                32,
                size=(records, record_length),
                dtype=np.uint16,
            )
            values.tofile(path)
            split_metadata[split] = {
                "records": records,
                "tokens": int(values.size),
                "utf8_bytes": 1,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": 2,
            "dtype": "uint16",
            "layout": "fixed_records",
            "sequence_length": sequence_length,
            "record_length": record_length,
            "vocab_size": 32,
            "tokenizer_source": {
                "path": str(self.tokenizer_path),
                "sha256": self.tokenizer_sha256,
            },
            "splits": split_metadata,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_packed_dataset(self, directory: Path, *, tokens: int) -> None:
        directory.mkdir()
        self._copy_tokenizer(directory)
        rng = np.random.default_rng(202)
        values = rng.integers(0, 32, size=tokens, dtype=np.uint16)
        split_metadata = {}
        for split in ("train", "validation", "test"):
            path = directory / f"{split}.bin"
            values.tofile(path)
            split_metadata[split] = {
                "documents": 1,
                "tokens": int(values.size),
                "utf8_bytes": 1,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": 1,
            "dtype": "uint16",
            "vocab_size": 32,
            "tokenizer_source": {
                "path": str(self.tokenizer_path),
                "sha256": self.tokenizer_sha256,
            },
            "splits": split_metadata,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )


class LongContextEvaluationEndToEndTests(unittest.TestCase):
    def test_cpu_cli_writes_complete_atomic_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LongContextFixture(Path(temporary))
            output = Path(temporary) / "results" / "long.json"
            command = [
                sys.executable,
                "scripts/evaluate_long_context.py",
                "--checkpoint",
                str(fixture.checkpoint_path),
                "--tokenizer",
                str(fixture.tokenizer_path),
                "--expected-checkpoint-sha256",
                fixture.checkpoint_sha256,
                "--long-dataset-dir",
                str(fixture.long_dataset),
                "--expected-long-manifest-sha256",
                sha256_file(fixture.long_dataset / "manifest.json"),
                "--test-records",
                "2",
                "--test-batch-size",
                "2",
                "--position-segments",
                "7",
                "--loss-chunk-size",
                "5",
                "--regression-dataset-dir",
                str(fixture.regression_dataset),
                "--expected-regression-manifest-sha256",
                fixture.regression_manifest_sha256,
                "--regression-sequence-length",
                "16",
                "--regression-records",
                "2",
                "--passkey-lengths",
                "32",
                "--passkey-depths",
                "0.1,0.9",
                "--parity-lengths",
                "8,17",
                "--parity-chunk-size",
                "4",
                "--minimum-argmax-match",
                "1.0",
                "--parity-atol",
                "0.00002",
                "--parity-rtol",
                "0.00002",
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(str(output), completed.stdout)
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(
                payload["checkpoint"]["sha256"],
                fixture.checkpoint_sha256,
            )
            self.assertEqual(
                payload["tokenizer"]["sha256"],
                fixture.tokenizer_sha256,
            )
            fixed = payload["fixed_record_test"]
            self.assertEqual(fixed["evaluated_records"], 2)
            self.assertEqual(fixed["sequence_length"], 32)
            self.assertEqual(fixed["evaluated_tokens"], 64)
            self.assertTrue(np.isfinite(fixed["overall_loss"]))
            self.assertEqual(
                fixed["position_loss_segments"][0]["logit_position_start"],
                0,
            )
            self.assertEqual(
                fixed["position_loss_segments"][-1][
                    "logit_position_end_exclusive"
                ],
                32,
            )
            self.assertTrue(fixed["position_coverage"]["complete"])
            self.assertEqual(
                payload["regression_loss"]["evaluated_tokens"],
                32,
            )
            self.assertEqual(len(payload["passkey_retrieval"]["cases"]), 2)
            for case in payload["passkey_retrieval"]["cases"]:
                self.assertEqual(case["total_tokens"], 32)
                self.assertTrue(np.isfinite(case["target_nll"]))
                self.assertIn("teacher_forced_token_accuracy", case)
                self.assertIn("greedy_exact", case)
            self.assertTrue(payload["cache_parity"]["all_passed"])
            for case in payload["cache_parity"]["cases"]:
                self.assertTrue(case["dynamic_vs_full"]["passed"])
                self.assertTrue(case["static_vs_full"]["passed"])
                self.assertTrue(case["static_vs_dynamic"]["passed"])
                self.assertTrue(case["full_forward_consistency_passed"])
                self.assertEqual(
                    case["dynamic_vs_full"]["final_cache_length"],
                    case["sequence_length"],
                )
                self.assertEqual(
                    case["static_vs_full"]["final_cache_length"],
                    case["sequence_length"],
                )
            self.assertEqual(list(output.parent.glob("*.tmp")), [])
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])


class LongContextEvaluationIntegrityTests(unittest.TestCase):
    def test_checkpoint_rejects_wrong_expected_hash_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LongContextFixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
                load_evaluation_bundle(
                    fixture.checkpoint_path,
                    fixture.tokenizer_path,
                    expected_checkpoint_sha256="0" * 64,
                    device="cpu",
                    dtype_name="auto",
                )

            other_tokenizer = Path(temporary) / "other-tokenizer.json"
            other_tokenizer.write_text(
                fixture.tokenizer_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "tokenizer SHA-256"):
                load_evaluation_bundle(
                    fixture.checkpoint_path,
                    other_tokenizer,
                    expected_checkpoint_sha256=fixture.checkpoint_sha256,
                    device="cpu",
                    dtype_name="auto",
                )

    def test_dataset_rejects_tampered_split_and_manifest_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LongContextFixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                verify_dataset(
                    fixture.long_dataset,
                    split="test",
                    tokenizer_sha256=fixture.tokenizer_sha256,
                    expected_manifest_sha256="f" * 64,
                    require_fixed_records=True,
                )

            with (fixture.long_dataset / "test.bin").open("ab") as handle:
                handle.write(b"\x00\x00")
            with self.assertRaisesRegex(ValueError, "split 'test' SHA-256 mismatch"):
                verify_dataset(
                    fixture.long_dataset,
                    split="test",
                    tokenizer_sha256=fixture.tokenizer_sha256,
                    require_fixed_records=True,
                )

    def test_position_ranges_cover_tail_and_atomic_write_replaces(self):
        self.assertEqual(
            position_ranges(10, 3),
            [(0, 3), (3, 6), (6, 10)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            output.write_text('{"old": true}\n', encoding="utf-8")
            atomic_write_json(output, {"new": True})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"new": True},
            )
            self.assertEqual(list(Path(temporary).glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
