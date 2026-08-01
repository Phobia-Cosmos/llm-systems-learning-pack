import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.plan_long_context_stage import build_plan


ROOT = Path(__file__).parents[1]


class LongContextPlanTests(unittest.TestCase):
    def test_plan_preserves_global_batch_and_rounds_milestones(self):
        plan = build_plan(
            base_step=100,
            base_tokens_processed=3_276_800,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            world_size=4,
            stage_tokens=1_000_000_000,
            milestone_tokens=[0, 100_000_000, 1_000_000_000],
        )
        self.assertEqual(plan["tokens_per_rank_optimizer_step"], 32_768)
        self.assertEqual(plan["tokens_per_optimizer_step"], 131_072)
        self.assertEqual(plan["world_size"], 4)
        self.assertEqual(plan["stage_start_step"], 100)
        self.assertEqual(plan["stage_end_step"], 7_730)
        self.assertEqual(plan["milestones"][1]["target_step"], 863)
        self.assertGreaterEqual(
            plan["milestones"][1]["actual_additional_tokens"],
            100_000_000,
        )

    def test_plan_rejects_changed_per_rank_batch(self):
        with self.assertRaisesRegex(ValueError, "32,768 tokens/rank/update"):
            build_plan(
                base_step=0,
                base_tokens_processed=0,
                micro_batch_size=4,
                gradient_accumulation_steps=2,
                world_size=4,
                stage_tokens=100,
                milestone_tokens=[0, 100],
            )

    def test_cli_validates_dataset_checkpoint_and_capacity_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            tokenizer = dataset / "tokenizer.json"
            tokenizer.write_text('{"fixture": true}\n', encoding="utf-8")
            tokenizer_sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
            split_metadata = {}
            for split in ("train", "validation", "test"):
                payload = dataset / f"{split}.bin"
                np.zeros(8193, dtype=np.uint16).tofile(payload)
                split_metadata[split] = {
                    "path": payload.name,
                    "records": 1,
                    "tokens": 8193,
                    "bytes": payload.stat().st_size,
                    "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                }
            manifest = {
                "schema_version": 2,
                "dtype": "uint16",
                "layout": "fixed_records",
                "sequence_length": 8192,
                "record_length": 8193,
                "split_policy": {
                    "name": "preserve_train_partition_permanent_holdout_v1",
                    "cross_role_duplicate_policy": "error",
                },
                "validation": {
                    "all_split_targets_met": True,
                    "all_splits_nonempty": True,
                    "cross_role_document_overlap": 0,
                },
                "tokenizer_source": {"sha256": tokenizer_sha},
                "splits": split_metadata,
            }
            (dataset / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            checkpoint = root / "base.pt"
            torch.save(
                {
                    "step": 10,
                    "tokens_processed": 327_680,
                    "tokenizer_sha256": tokenizer_sha,
                    "config": {
                        "position_encoding": "rope",
                        "block_size": 8192,
                    },
                },
                checkpoint,
            )
            capacity = root / "capacity.json"
            capacity.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "world_size": 4,
                        "results": [
                            {
                                "status": "success",
                                "micro_batch_size": 2,
                                "gradient_accumulation_steps": 2,
                                "world_size": 4,
                                "tokens_per_rank_optimizer_step": 32768,
                                "tokens_per_optimizer_step": 131072,
                                "steady_steps": 10,
                                "median_tokens_per_second": 42_000,
                            }
                        ],
                        "recommended": {
                            "micro_batch_size": 2,
                            "gradient_accumulation_steps": 2,
                            "world_size": 4,
                            "tokens_per_optimizer_step": 131072,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "plan.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/plan_long_context_stage.py",
                    "--base-checkpoint",
                    str(checkpoint),
                    "--dataset-dir",
                    str(dataset),
                    "--capacity-report",
                    str(capacity),
                    "--micro-batch-size",
                    "2",
                    "--gradient-accumulation-steps",
                    "2",
                    "--world-size",
                    "4",
                    "--stage-tokens",
                    "262144",
                    "--milestone-tokens",
                    "0",
                    "262144",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(plan["stage_start_step"], 10)
            self.assertEqual(plan["schema_version"], 2)
            self.assertEqual(plan["world_size"], 4)
            self.assertEqual(plan["tokens_per_optimizer_step"], 131_072)
            self.assertEqual(plan["stage_end_step"], 12)
            self.assertEqual(plan["dataset"]["tokenizer_sha256"], tokenizer_sha)
            self.assertEqual(plan["capacity"]["selected"]["steady_steps"], 10)


class LongContextShellTests(unittest.TestCase):
    def test_long_context_shell_scripts_parse(self):
        for script in (
            "scripts/run_long_context_training.sh",
            "scripts/run_long_context_capacity_matrix.sh",
            "scripts/train_long_context_with_benchmarks.sh",
        ):
            subprocess.run(
                ["bash", "-n", script],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        launcher = (ROOT / "scripts/run_long_context_training.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("--standalone", launcher)
        self.assertIn('--master_addr="$MASTER_ADDR"', launcher)
        self.assertIn('--master_port="$MASTER_PORT"', launcher)
        self.assertIn('--local_addr="$MASTER_ADDR"', launcher)

    def test_long_context_controller_reuses_valid_results_and_summarizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "out"
            dataset = root / "long"
            regression = root / "regression"
            dataset.mkdir()
            regression.mkdir()
            (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
            (dataset / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (regression / "manifest.json").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "base.pt"
            checkpoint.write_bytes(b"fixture")
            plan = {
                "schema_version": 2,
                "base_checkpoint": {"path": str(checkpoint), "step": 10},
                "dataset": {"path": str(dataset), "manifest_sha256": "a" * 64},
                "capacity": {
                    "selected": {
                        "micro_batch_size": 2,
                        "gradient_accumulation_steps": 2,
                        "world_size": 4,
                        "tokens_per_optimizer_step": 131072,
                    }
                },
                "world_size": 4,
                "stage_start_step": 10,
                "stage_end_step": 12,
                "milestones": [
                    {
                        "requested_additional_tokens": 0,
                        "target_step": 10,
                        "actual_additional_tokens": 0,
                        "cumulative_tokens_processed": 327_680,
                    }
                ],
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            long_dir = out / "benchmarks" / "long-context-trajectory"
            capability_dir = out / "benchmarks" / "long-context-capability"
            long_dir.mkdir(parents=True)
            capability_dir.mkdir(parents=True)
            (long_dir / "step-00000010.json").write_text(
                json.dumps(
                    {
                        "checkpoint": {"tokens_processed": 327_680},
                        "fixed_record_test": {
                            "overall_loss": 3.0,
                            "position_coverage": {"complete": True},
                            "position_loss_segments": [{"loss": 3.2}],
                        },
                        "regression_loss": {"overall_loss": 2.9},
                        "passkey_retrieval": {
                            "aggregate": {
                                "mean_teacher_forced_token_accuracy": 0.1,
                                "greedy_exact_fraction": 0.0,
                            }
                        },
                        "cache_parity": {"all_passed": True},
                    }
                ),
                encoding="utf-8",
            )
            tasks = {
                task: {"metrics": {"label": {"accuracy_normalized": 0.25}}}
                for task in ("ceval", "cmmlu", "arc_easy", "arc_challenge", "hellaswag")
            }
            (capability_dir / "step-00000010.json").write_text(
                json.dumps({"tasks": tasks}),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "OUT_ROOT": str(out),
                    "STAGE_PLAN": str(plan_path),
                    "REGRESSION_DATASET_DIR": str(regression),
                    "EVAL_MAX_ATTEMPTS": "1",
                    "EVAL_RETRY_SECONDS": "0",
                }
            )

            result = subprocess.run(
                ["bash", "scripts/train_long_context_with_benchmarks.sh"],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(
                (out / "benchmarks" / "long-context-summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["milestones"][0]["long_evaluation"]["long_loss"], 3.0)
            self.assertTrue(
                summary["milestones"][0]["long_evaluation"]["cache_parity_passed"]
            )
            self.assertEqual(
                (out / "160m-openbpe-32k-8k" / "milestones" / "tokenizer.json").resolve(),
                (dataset / "tokenizer.json").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
