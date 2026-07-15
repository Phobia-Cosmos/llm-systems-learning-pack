import json
from pathlib import Path
import tempfile
import unittest

from miniagent.analysis import (
    analyze_run,
    linear_percentile,
    normalize_metrics,
    summarize_raw_result,
)


def _raw(ttfts, duration=2.0):
    count = len(ttfts)
    return {
        "ttfts": ttfts,
        "itls": [[0.01, 0.02] for _ in ttfts],
        "input_lens": [4] * count,
        "output_lens": [3] * count,
        "completed": count,
        "failed": 0,
        "duration": duration,
        "total_input_tokens": 4 * count,
        "total_output_tokens": 3 * count,
    }


class PercentileTests(unittest.TestCase):
    def test_exact_linear_percentile(self):
        self.assertEqual(linear_percentile([0, 10], 50), 5)
        self.assertAlmostEqual(linear_percentile([1, 2, 3, 4], 95), 3.85)
        self.assertIsNone(linear_percentile([], 95))

    def test_invalid_percentile(self):
        with self.assertRaises(ValueError):
            linear_percentile([1], 101)


class MetricsTests(unittest.TestCase):
    def test_direct_summary_keys(self):
        result = normalize_metrics(
            {
                "samples": [],
                "summary": {
                    "max_waiting": 7,
                    "max_running": 8,
                    "queue_time_mean_seconds": 0.025,
                    "queue_time_delta_count": 96,
                    "preemptions_delta": 2,
                    "retractions_delta": 3,
                },
            }
        )
        self.assertEqual(result["max_waiting"], 7)
        self.assertEqual(result["max_running"], 8)
        self.assertEqual(result["queue_time_mean_seconds"], 0.025)
        self.assertEqual(result["preemptions_delta"], 2)

    def test_nested_prometheus_layout(self):
        result = normalize_metrics(
            {
                "summary": {
                    "gauges": {
                        "vllm:num_requests_waiting": {"max": 12},
                        "vllm:num_requests_running": {"max": 32},
                    },
                    "histograms": {
                        "vllm:request_queue_time_seconds": {
                            "sum_delta": 4.8,
                            "count_delta": 96,
                        }
                    },
                    "counters": {"vllm:num_preemptions": {"delta": 1}},
                }
            }
        )
        self.assertEqual(result["max_waiting"], 12)
        self.assertEqual(result["max_running"], 32)
        self.assertAlmostEqual(result["queue_time_mean_seconds"], 0.05)
        self.assertEqual(result["queue_time_delta_count"], 96)
        self.assertEqual(result["preemptions_delta"], 1)


class RawSummaryTests(unittest.TestCase):
    def test_reconstructs_e2e_and_tpot(self):
        summary, samples = summarize_raw_result(
            _raw([0.1, 0.2]),
            expected_requests=2,
            expected_input_len=4,
            expected_output_len=3,
        )
        self.assertTrue(summary["validation"]["valid"])
        self.assertEqual(samples["e2e_ms"], [130.0, 230.0])
        self.assertEqual(samples["tpot_ms"], [15.0, 15.0])
        self.assertAlmostEqual(summary["stats"]["ttft_ms"]["p95"], 195.0)

    def test_validation_detects_failures_and_wrong_lengths(self):
        raw = _raw([0.1])
        raw["failed"] = 1
        raw["output_lens"] = [2]
        summary, _ = summarize_raw_result(raw, expected_output_len=3)
        self.assertFalse(summary["validation"]["valid"])
        self.assertTrue(any("failed=1" in item for item in summary["validation"]["errors"]))
        self.assertTrue(any("output_len" in item for item in summary["validation"]["errors"]))


class AnalyzeRunTests(unittest.TestCase):
    def test_pools_requests_and_weights_throughput(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = []
            for repeat, (ttfts, duration, waiting, queue_mean) in enumerate(
                (
                    ([0.1, 0.2], 2.0, 0, 0.0),
                    ([0.3, 0.4], 4.0, 3, 0.1),
                ),
                start=1,
            ):
                raw_path = root / f"raw{repeat}.json"
                metrics_path = root / f"metrics{repeat}.json"
                raw_path.write_text(json.dumps(_raw(ttfts, duration)), encoding="utf-8")
                metrics_path.write_text(
                    json.dumps(
                        {
                            "summary": {
                                "max_waiting": waiting,
                                "max_running": 8,
                                "queue_time_mean_seconds": queue_mean,
                                "queue_time_delta_count": 2,
                                "preemptions_delta": repeat - 1,
                                "retractions_delta": 0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                runs.append(
                    {
                        "engine": "vllm",
                        "concurrency": 8,
                        "repeat": repeat,
                        "raw_path": raw_path.name,
                        "metrics_path": metrics_path.name,
                    }
                )
            manifest = {
                "run_id": "synthetic",
                "benchmark": {"num_prompts": 2, "input_len": 4, "output_len": 3},
                "runs": runs,
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            first = analyze_run(root)
            second = analyze_run(root)

            self.assertEqual(first, second)
            self.assertTrue(first["validation"]["valid"])
            group = first["groups"][0]
            self.assertAlmostEqual(group["stats"]["ttft_ms"]["p95"], 385.0)
            self.assertEqual(group["repeat_p95_ttft_ms"]["median"], 295.0)
            self.assertEqual(group["repeat_p95_ttft_ms"]["min"], 195.0)
            self.assertEqual(group["repeat_p95_ttft_ms"]["max"], 395.0)
            # 14 tokens in each run, divided by 2 + 4 seconds.  This is a
            # duration-weighted aggregate, not the mean of run throughputs.
            self.assertAlmostEqual(group["throughput"]["total_tokens_per_second"], 28 / 6)
            self.assertEqual(group["metrics"]["max_waiting"], 3)
            self.assertAlmostEqual(group["metrics"]["queue_time_mean_seconds"], 0.05)
            self.assertEqual(group["metrics"]["preemptions_delta"], 1)
            self.assertEqual(len(first["evidence"]), 5)


if __name__ == "__main__":
    unittest.main()
