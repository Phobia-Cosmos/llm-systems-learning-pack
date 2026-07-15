from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest

from miniagent.prometheus import (
    PrometheusSampler,
    parse_prometheus_text,
    summarize_samples,
)


class ParsePrometheusTextTests(unittest.TestCase):
    def test_aggregates_labels_and_preserves_histogram_boundaries(self) -> None:
        exposition = r'''
# HELP vllm:num_requests_waiting Requests waiting in the scheduler.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="qwen",worker="0"} 2
vllm:num_requests_waiting{model_name="qwen",worker="1"} 3
vllm:num_requests_waiting{model_name="bad"} NaN
vllm:request_queue_time_seconds_bucket{model="a",le="0.1"} 1
vllm:request_queue_time_seconds_bucket{model="b",le="0.1"} 2
vllm:request_queue_time_seconds_bucket{model="a",le="+Inf"} 2
vllm:request_queue_time_seconds_bucket{model="b",le="+Inf"} 2
vllm:request_queue_time_seconds_sum{model="a"} 0.25
vllm:request_queue_time_seconds_sum{model="b"} 0.75
vllm:request_queue_time_seconds_count{model="a"} 2
vllm:request_queue_time_seconds_count{model="b"} 2
unrelated_metric 99
'''
        parsed = parse_prometheus_text(
            exposition,
            {
                "vllm:num_requests_waiting",
                "vllm:request_queue_time_seconds",
            },
        )

        self.assertEqual(parsed.values["vllm:num_requests_waiting"], 5.0)
        self.assertEqual(
            parsed.values["vllm:request_queue_time_seconds_sum"], 1.0
        )
        self.assertEqual(
            parsed.values["vllm:request_queue_time_seconds_count"], 4.0
        )
        self.assertEqual(
            parsed.buckets["vllm:request_queue_time_seconds"],
            {"0.1": 3.0, "+Inf": 4.0},
        )
        self.assertNotIn("unrelated_metric", parsed.values)

    def test_counter_collector_name_selects_total_wire_name(self) -> None:
        parsed = parse_prometheus_text(
            "requests_processed_total{replica=\"a\"} 7\n"
            "requests_processed_total{replica=\"b\"} 8\n",
            {"requests_processed"},
        )

        self.assertEqual(parsed.values, {"requests_processed_total": 15.0})

    def test_malformed_and_nonfinite_samples_are_ignored(self) -> None:
        parsed = parse_prometheus_text(
            b"good 2\nbad NaN\nalso_bad +Inf\nmissing_value\n{broken 3\n"
        )

        self.assertEqual(parsed.values, {"good": 2.0})
        self.assertEqual(parsed.buckets, {})


class SummarizeSamplesTests(unittest.TestCase):
    def test_gauge_counter_and_histogram_deltas(self) -> None:
        samples = [
            {
                "timestamp_unix": 100.0,
                "elapsed_seconds": 0.0,
                "values": {
                    "queue": 1.0,
                    "completed_total": 10.0,
                    "latency_seconds_count": 2.0,
                    "latency_seconds_sum": 1.0,
                },
                "buckets": {"latency_seconds": {"0.1": 1.0, "+Inf": 2.0}},
            },
            {
                "timestamp_unix": 101.0,
                "elapsed_seconds": 1.0,
                "values": {
                    "queue": 5.0,
                    "completed_total": 14.0,
                    "latency_seconds_count": 5.0,
                    "latency_seconds_sum": 2.2,
                },
                "buckets": {"latency_seconds": {"0.1": 2.0, "+Inf": 5.0}},
            },
            {
                "timestamp_unix": 102.0,
                "elapsed_seconds": 2.0,
                "values": {
                    "queue": 2.0,
                    "completed_total": 17.0,
                    "latency_seconds_count": 9.0,
                    "latency_seconds_sum": 4.5,
                },
                "buckets": {"latency_seconds": {"0.1": 4.0, "+Inf": 9.0}},
            },
        ]

        summary = summarize_samples(
            samples,
            {
                "queue": "gauge",
                "completed": "counter",
                "latency_seconds": "histogram",
                "missing": "gauge",
            },
        )

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["duration_seconds"], 2.0)
        self.assertEqual(summary["metrics"]["queue"]["max"], 5.0)
        self.assertAlmostEqual(summary["metrics"]["queue"]["mean"], 8.0 / 3.0)
        self.assertEqual(summary["metrics"]["completed"]["delta"], 7.0)
        histogram = summary["metrics"]["latency_seconds"]
        self.assertEqual(histogram["count_delta"], 7.0)
        self.assertEqual(histogram["sum_delta"], 3.5)
        self.assertEqual(histogram["delta_mean"], 0.5)
        self.assertEqual(histogram["bucket_deltas"], {"0.1": 3.0, "+Inf": 7.0})
        self.assertIsNone(summary["metrics"]["missing"]["max"])

    def test_counter_reset_is_reported_without_hiding_raw_delta(self) -> None:
        summary = summarize_samples(
            [
                {"values": {"jobs_total": 10.0}, "buckets": {}},
                {"values": {"jobs_total": 2.0}, "buckets": {}},
            ],
            {"jobs_total": "counter"},
        )

        counter = summary["metrics"]["jobs_total"]
        self.assertEqual(counter["delta"], -8.0)
        self.assertTrue(counter["reset_detected"])


class PrometheusSamplerTests(unittest.TestCase):
    def test_background_poll_stop_and_json_persistence(self) -> None:
        reached_two_calls = threading.Event()
        call_lock = threading.Lock()
        calls = 0

        def fetcher(_url: str, _timeout: float) -> str:
            nonlocal calls
            with call_lock:
                calls += 1
                current = calls
            if current >= 2:
                reached_two_calls.set()
            return (
                f"queue {current}\n"
                f"completed_total {10 + current}\n"
                f"latency_seconds_count {current}\n"
                f"latency_seconds_sum {current * 0.25}\n"
            )

        spec = {
            "queue": "gauge",
            "completed": "counter",
            "latency_seconds": "histogram",
        }
        sampler = PrometheusSampler(
            "http://127.0.0.1:9999/metrics",
            interval_seconds=0.005,
            selected_metrics=set(spec),
            metric_spec=spec,
            timeout_seconds=0.1,
            fetcher=fetcher,
        )
        sampler.start()
        self.assertTrue(reached_two_calls.wait(timeout=1.0))
        samples = sampler.stop()

        self.assertGreaterEqual(len(samples), 3)  # Includes the final scrape.
        self.assertGreaterEqual(sampler.summary()["metrics"]["queue"]["max"], 3.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "metrics.json"
            payload = sampler.write_json(destination)
            on_disk = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"], on_disk["summary"])
        self.assertEqual(len(payload["samples"]), len(samples))

    def test_scrape_error_is_recorded_and_can_be_raised(self) -> None:
        def broken_fetcher(_url: str, _timeout: float) -> str:
            raise OSError("endpoint unavailable")

        sampler = PrometheusSampler(
            "http://example.invalid/metrics",
            selected_metrics={"queue"},
            fetcher=broken_fetcher,
        )
        self.assertIsNone(sampler.sample_once())
        self.assertEqual(sampler.errors[0]["type"], "OSError")
        with self.assertRaisesRegex(OSError, "endpoint unavailable"):
            sampler.sample_once(raise_on_error=True)


if __name__ == "__main__":
    unittest.main()
