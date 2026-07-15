from pathlib import Path
import tempfile
import unittest

from miniagent.report import generate_report, render_report


def _group(engine, concurrency, ttft_p95, tpot_p95, waiting, queue_mean):
    return {
        "group_id": f"{engine}-c{concurrency}",
        "engine": engine,
        "concurrency": concurrency,
        "repetitions": 3,
        "request_samples": 288,
        "stats": {
            "ttft_ms": {"p50": ttft_p95 / 2, "p95": ttft_p95, "p99": ttft_p95 * 1.2},
            "tpot_ms": {"p95": tpot_p95},
        },
        "repeat_p95_ttft_ms": {
            "median": ttft_p95,
            "min": ttft_p95 - 1,
            "max": ttft_p95 + 1,
        },
        "throughput": {"output_tokens_per_second": 123.45},
        "metrics": {
            "runs_with_metrics": 3,
            "max_waiting": waiting,
            "max_running": concurrency,
            "queue_time_mean_seconds": queue_mean,
            "preemptions_delta": 0,
            "retractions_delta": 0,
        },
    }


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "context": {"run_id": "test-run", "benchmark": {"output_len": 64}},
            "groups": [
                _group("vllm", 1, 10, 2, 0, 0),
                _group("vllm", 8, 30, 3, 4, 0.01),
                _group("vllm", 32, 25, 4, 0, 0),
                _group("sglang", 1, 12, 2, 0, 0),
                _group("sglang", 8, 35, 4, 0, 0),
            ],
            "evidence": [
                {
                    "id": "RAW-VLLM-C1-R1",
                    "kind": "client_raw",
                    "engine": "vllm",
                    "concurrency": 1,
                    "repeat": 1,
                    "path": "raw/example.json",
                    "sha256": "abc123",
                }
            ],
            "validation": {"valid": True, "errors": [], "warnings": []},
        }

    def test_report_separates_evidence_and_inference(self):
        report = render_report(self.summary)
        self.assertIn("pooled p95 TTFT", report)
        self.assertIn("重复 p95 中位数 [min, max]", report)
        self.assertIn("服务端直接观测到排队", report)
        self.assertIn("架构推断", report)
        self.assertIn("这一阶没有 p95 TTFT 退化", report)
        self.assertIn("RAW-VLLM-C1-R1", report)
        self.assertIn("abc123", report)
        self.assertIn("不能断言它是唯一原因", report)

    def test_generate_report_writes_identical_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.md"
            report = generate_report(self.summary, path)
            self.assertEqual(path.read_text(encoding="utf-8"), report)

    def test_micro_queue_is_not_claimed_as_material_cause(self):
        summary = {
            "context": {"run_id": "micro-queue"},
            "groups": [
                _group("vllm", 1, 10, 2, 0, 0.00001),
                _group("vllm", 8, 30, 4, 3, 0.0005),
            ],
            "evidence": [],
            "validation": {"valid": True, "errors": [], "warnings": []},
        }
        report = render_report(summary)
        self.assertIn("仅观测到轻微/短暂排队", report)
        self.assertNotIn("排队至少是 TTFT 退化的一个贡献因素", report)


if __name__ == "__main__":
    unittest.main()
