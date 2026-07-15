from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from miniagent.config import BenchmarkConfig, RunSpec
from miniagent.runner import client_command, server_command


class RunnerCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BenchmarkConfig(num_prompts=32)
        self.workload = Path("/tmp/workload.jsonl")

    def test_both_servers_disable_prefix_cache_and_match_budgets(self) -> None:
        vllm = server_command("vllm", self.config)
        sglang = server_command("sglang", self.config)
        self.assertIn("--no-enable-prefix-caching", vllm)
        self.assertIn("--disable-radix-cache", sglang)
        self.assertEqual(vllm[vllm.index("--max-num-batched-tokens") + 1], "2048")
        self.assertEqual(sglang[sglang.index("--max-prefill-tokens") + 1], "2048")
        self.assertEqual(vllm[vllm.index("--gpu-memory-utilization") + 1], "0.7")
        self.assertEqual(sglang[sglang.index("--mem-fraction-static") + 1], "0.7")

    def test_measured_client_keeps_raw_samples_and_p95(self) -> None:
        spec = RunSpec("sglang", 8, 2, 43)
        with tempfile.TemporaryDirectory() as directory:
            command = client_command(
                spec,
                self.config,
                self.workload,
                result_dir=Path(directory),
            )
        self.assertIn("--save-detailed", command)
        self.assertEqual(command[command.index("--metric-percentiles") + 1], "50,90,95,99")
        self.assertEqual(command[command.index("--max-concurrency") + 1], "8")
        self.assertIn("engine=sglang", command)

    def test_warmup_is_separate_and_not_saved(self) -> None:
        spec = RunSpec("vllm", 32, 1, 42)
        command = client_command(spec, self.config, self.workload, warmup=True)
        self.assertNotIn("--save-result", command)
        self.assertEqual(command[command.index("--num-prompts") + 1], "32")
        self.assertEqual(command[command.index("--num-warmups") + 1], "0")


if __name__ == "__main__":
    unittest.main()
