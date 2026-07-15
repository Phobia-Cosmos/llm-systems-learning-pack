from __future__ import annotations

import unittest

from miniagent.config import (
    BenchmarkConfig,
    balanced_concurrency_order,
    experiment_matrix,
)


class ConfigTest(unittest.TestCase):
    def test_balanced_order_rotates(self) -> None:
        self.assertEqual(balanced_concurrency_order((1, 8, 32), 1), (1, 8, 32))
        self.assertEqual(balanced_concurrency_order((1, 8, 32), 2), (8, 32, 1))
        self.assertEqual(balanced_concurrency_order((1, 8, 32), 3), (32, 1, 8))

    def test_matrix_pairs_engine_seed_and_order(self) -> None:
        config = BenchmarkConfig(repeats=2, num_prompts=32)
        specs = experiment_matrix(config)
        self.assertEqual(len(specs), 12)
        self.assertEqual(
            [item.run_key for item in specs[:6]],
            [
                "vllm-c1-r1",
                "vllm-c8-r1",
                "vllm-c32-r1",
                "vllm-c8-r2",
                "vllm-c32-r2",
                "vllm-c1-r2",
            ],
        )
        self.assertEqual(specs[0].seed, 42)
        self.assertEqual(specs[3].seed, 43)

    def test_rejects_incomplete_saturation_run(self) -> None:
        config = BenchmarkConfig(num_prompts=8)
        with self.assertRaisesRegex(ValueError, "maximum concurrency"):
            config.validate()


if __name__ == "__main__":
    unittest.main()

