import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
CONTROLLER = ROOT / "scripts" / "train_new_model_with_benchmarks.sh"
VALID_BENCHMARK = json.dumps(
    {
        "schema_version": 1,
        "tasks": {
            task: {"metrics": {"label": {"accuracy_normalized": 0.25}}}
            for task in ("ceval", "cmmlu", "arc_easy", "arc_challenge", "hellaswag")
        },
    },
    separators=(",", ":"),
)


class TrainingControllerTests(unittest.TestCase):
    def make_run(self, root: Path, steps: tuple[int, ...]) -> Path:
        run = root / "160m-openbpe-32k-4k"
        run.mkdir(parents=True)
        (run / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (root / "160-train.log").write_text("", encoding="utf-8")
        for step in steps:
            (run / f"step-{step:08d}.pt").write_bytes(f"checkpoint-{step}".encode())
        return run

    def write_command(self, path: Path, body: str) -> Path:
        path.write_text("#!/usr/bin/env bash\nset -u\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o755)
        return path

    def controller_env(
        self,
        out_root: Path,
        benchmark_command: Path,
        milestones: tuple[int, ...],
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "OUT_ROOT": str(out_root),
                "MILESTONES": " ".join(str(step) for step in milestones),
                "FINAL_SCHEDULE_STEP": str(max(milestones)),
                "BENCHMARK_COMMAND": str(benchmark_command),
                "BENCHMARK_MAX_ATTEMPTS": "2",
                "BENCHMARK_RETRY_SECONDS": "0",
                "CONTROLLER_PYTHON": sys.executable,
                "SUMMARY_COMMAND": "/bin/true",
            }
        )
        return env

    def run_controller(self, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(CONTROLLER)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_failed_benchmark_does_not_stop_later_milestone_and_restart_recovers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            run = self.make_run(out_root, (1, 2))
            calls = root / "calls.log"
            first_command = self.write_command(
                root / "benchmark-first.sh",
                f"""
                printf '%s\\n' "$(basename "$1")" >> {calls!s}
                if [[ "$(basename "$1")" == "step-00000001.pt" ]]; then
                  exit 42
                fi
                printf '%s\\n' '{VALID_BENCHMARK}' > "$2"
                """,
            )
            env = self.controller_env(out_root, first_command, (1, 2))

            first = self.run_controller(env)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["step-00000001.pt", "step-00000001.pt", "step-00000002.pt"],
            )
            failed_output = out_root / "benchmarks" / "trajectory" / "step-00000001.json"
            successful_output = out_root / "benchmarks" / "trajectory" / "step-00000002.json"
            failure_record = Path(f"{failed_output}.failed")
            self.assertFalse(failed_output.exists())
            self.assertEqual(
                successful_output.read_text(encoding="utf-8"),
                f"{VALID_BENCHMARK}\n",
            )
            self.assertIn("last_status=42", failure_record.read_text(encoding="utf-8"))
            self.assertTrue((run / "milestones" / "step-00000001.pt").exists())
            self.assertTrue((run / "milestones" / "step-00000002.pt").exists())
            self.assertIn("training controller will continue", first.stderr)

            calls.write_text("", encoding="utf-8")
            recovery_command = self.write_command(
                root / "benchmark-recovery.sh",
                f"""
                printf '%s\\n' "$(basename "$1")" >> {calls!s}
                printf '%s\\n' '{VALID_BENCHMARK}' > "$2"
                """,
            )
            env["BENCHMARK_COMMAND"] = str(recovery_command)

            second = self.run_controller(env)

            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["step-00000001.pt"],
            )
            self.assertEqual(
                failed_output.read_text(encoding="utf-8"),
                f"{VALID_BENCHMARK}\n",
            )
            self.assertFalse(failure_record.exists())
            events = (out_root / "benchmarks" / "benchmark-events.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("status=failed kind=trajectory step=1", events)
            self.assertIn("status=recovered kind=trajectory step=1", events)

    def test_success_exit_with_invalid_json_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            self.make_run(out_root, (1,))
            calls = root / "calls.log"
            invalid_command = self.write_command(
                root / "benchmark-invalid.sh",
                f"""
                printf 'attempt\\n' >> {calls!s}
                printf 'not-json\\n' > "$2"
                """,
            )
            env = self.controller_env(out_root, invalid_command, (1,))
            output = out_root / "benchmarks" / "trajectory" / "step-00000001.json"
            output.parent.mkdir(parents=True)
            output.write_text("previous-partial-output\n", encoding="utf-8")

            result = self.run_controller(env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), ["attempt", "attempt"])
            failure_record = Path(f"{output}.failed")
            self.assertFalse(output.exists())
            quarantined = list(output.parent.glob(f"{output.name}.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                quarantined[0].read_text(encoding="utf-8"),
                "previous-partial-output\n",
            )
            failure = failure_record.read_text(encoding="utf-8")
            self.assertIn("last_status=65", failure)
            self.assertIn(
                "last_reason=command_succeeded_but_output_was_empty_or_invalid_json",
                failure,
            )

    def test_token_milestone_uses_checkpoint_tokens_and_records_actual_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_root = root / "out"
            run = self.make_run(out_root, ())
            checkpoint = run / "step-00000007.pt"
            torch.save({"step": 7, "tokens_processed": 100}, checkpoint)
            (run / "latest.pt").symlink_to(checkpoint.name)
            benchmark = self.write_command(
                root / "benchmark.sh",
                f"printf '%s\\n' '{VALID_BENCHMARK}' > \"$2\"",
            )
            env = os.environ.copy()
            env.update(
                {
                    "OUT_ROOT": str(out_root),
                    "MILESTONES": "",
                    "MILESTONE_TOKENS": "50",
                    "REFERENCE_TOKENS_PER_STEP": "32",
                    "BENCHMARK_COMMAND": str(benchmark),
                    "BENCHMARK_MAX_ATTEMPTS": "1",
                    "BENCHMARK_RETRY_SECONDS": "0",
                    "CONTROLLER_PYTHON": sys.executable,
                    "SUMMARY_COMMAND": "/bin/true",
                }
            )

            result = self.run_controller(env)

            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(
                (run / "milestones" / "target-tokens-000000000050.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(record["actual_step"], 7)
            self.assertEqual(record["actual_tokens"], 100)
            self.assertTrue((run / "milestones" / "step-00000007.pt").exists())
            self.assertTrue(
                (out_root / "benchmarks" / "trajectory" / "step-00000007.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
