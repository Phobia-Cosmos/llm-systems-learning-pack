#!/home/undefined/Disk/python-envs/sglang/bin/python
"""CPU-only unit tests for pressure-matrix parsing and JSON statistics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_pressure_bench as bench


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)


class ParameterTests(unittest.TestCase):
    def test_grid_parser_sorts_and_deduplicates(self) -> None:
        batches = bench._parse_positive_int_csv("8, 1,4,8", "--grid-batches")
        contexts = bench._parse_positive_int_csv("128,32", "--grid-contexts")
        cases = bench._grid_cases(batches, contexts, 3)
        self.assertEqual(batches, [1, 4, 8])
        self.assertEqual(
            [(case.requests, case.prefill_tokens, case.decode_tokens) for case in cases],
            [
                (1, 32, 3),
                (4, 32, 3),
                (8, 32, 3),
                (1, 128, 3),
                (4, 128, 3),
                (8, 128, 3),
            ],
        )

    def test_grid_parser_rejects_non_positive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, ">= 1"):
            bench._parse_positive_int_csv("1,0,4", "--grid-batches")

    def test_temperature_accepts_auto_zero_and_positive_values(self) -> None:
        self.assertIsNone(bench._parse_temperature("auto"))
        self.assertEqual(bench._parse_temperature("0"), 0.0)
        self.assertEqual(bench._parse_temperature("0.7"), 0.7)
        with self.assertRaisesRegex(Exception, "finite"):
            bench._parse_temperature("nan")

    def test_prompt_file_plain_text_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = root / "prompts.txt"
            plain.write_text("first\n\nsecond\n", encoding="utf-8")
            jsonl = root / "prompts.jsonl"
            jsonl.write_text(
                "\n".join(
                    (
                        json.dumps("alpha"),
                        json.dumps({"prompt": "beta"}),
                        json.dumps({"text": "gamma"}),
                    )
                ),
                encoding="utf-8",
            )
            self.assertEqual(bench._load_prompt_file(plain), ["first", "second"])
            self.assertEqual(
                bench._load_prompt_file(jsonl), ["alpha", "beta", "gamma"]
            )


class PromptStatisticsTests(unittest.TestCase):
    def test_natural_text_is_truncated_but_never_padded(self) -> None:
        case = bench.BenchCase("text", requests=2, prefill_tokens=4, decode_tokens=1)
        prepared = bench._prepare_text_prompts(
            case,
            tokenizer=CharacterTokenizer(),
            source_texts=["ab", "abcdef"],
            source="unit-test",
            length_policy="truncate",
        )
        self.assertEqual(prepared.token_ids, [[ord("a"), ord("b")], [97, 98, 99, 100]])
        self.assertEqual(prepared.metadata["prompt_tokens"]["mean"], 3.0)
        self.assertEqual(prepared.metadata["truncated_requests"], 1)
        self.assertEqual(prepared.metadata["padding"], "none")
        self.assertEqual(prepared.metadata["chars_per_token"]["mean"], 1.0)
        self.assertEqual(prepared.metadata["bytes_per_token"]["mean"], 1.0)

    def test_repeat_truncate_reaches_controlled_token_cap(self) -> None:
        case = bench.BenchCase("text", requests=1, prefill_tokens=9, decode_tokens=1)
        prepared = bench._prepare_text_prompts(
            case,
            tokenizer=CharacterTokenizer(),
            source_texts=["abc"],
            source="unit-test",
            length_policy="repeat-truncate",
        )
        self.assertEqual(len(prepared.token_ids[0]), 9)


class SummaryTests(unittest.TestCase):
    def test_actual_prefill_cache_and_phase_batches(self) -> None:
        case = bench.BenchCase("variable", requests=2, prefill_tokens=10, decode_tokens=3)
        steps = [
            bench.StepSample("prefill", 10, 2, 2, 2.0, 2.1, False),
            bench.StepSample("decode", 2, 2, 4, 1.0, 1.1, True),
            bench.StepSample("decode", 2, 2, 4, 1.0, 1.1, True),
        ]
        summary = bench._summarize_case(
            case,
            wall_ms=10.0,
            outputs=[[11, 12, 13], [21, 22, 23]],
            prompt_lengths=[4, 6],
            steps=steps,
            arrival_ms={0: [2.0, 4.0, 6.0], 1: [2.0, 4.0, 6.0]},
            record_steps=False,
        )
        self.assertEqual(summary["input_tokens"], 10)
        self.assertEqual(summary["nominal_input_tokens"], 20)
        self.assertEqual(summary["prompt"]["total_prefill_tokens"], 10)
        self.assertEqual(summary["cache"]["final_logical_tokens_total"], 14)
        self.assertEqual(
            summary["cache"]["final_logical_length_per_request"]["max"], 8.0
        )
        self.assertEqual(summary["batch"]["effective_prefill"]["mean"], 2.0)
        self.assertEqual(summary["batch"]["effective_decode"]["mean"], 2.0)
        self.assertEqual(len(summary["semantic_regression"]["output_token_sha256"]), 64)
        self.assertIsNone(summary["semantic_regression"]["output_ids"])

        full_ids = bench._summarize_case(
            case,
            wall_ms=10.0,
            outputs=[[11, 12, 13], [21, 22, 23]],
            prompt_lengths=[4, 6],
            steps=steps,
            arrival_ms={0: [2.0, 4.0, 6.0], 1: [2.0, 4.0, 6.0]},
            record_steps=False,
            record_output_ids=True,
        )
        self.assertEqual(
            full_ids["semantic_regression"]["output_ids"],
            [[11, 12, 13], [21, 22, 23]],
        )

    def test_saturation_analysis_marks_first_low_gain_transition(self) -> None:
        def record(batch: int, throughput: float) -> dict[str, object]:
            return {
                "case": {
                    "requests": batch,
                    "prefill_tokens": 32,
                    "decode_tokens": 8,
                },
                "metrics": {
                    "input_tokens": batch * 32,
                    "throughput": {"output_tokens_per_s_wall": throughput},
                    "cache": {"final_logical_tokens_total": batch * 39},
                    "batch": {
                        "effective_prefill": {"mean": float(batch)},
                        "effective_decode": {"mean": float(batch)},
                    },
                },
            }

        analysis = bench._analyze_saturation(
            [record(1, 100.0), record(2, 170.0), record(4, 175.0)],
            metric="output_tokens_per_s_wall",
            threshold=0.05,
        )
        group = analysis["groups"][0]
        self.assertEqual(group["first_low_gain_batch"], 4)
        self.assertFalse(group["transitions"][0]["below_threshold"])
        self.assertTrue(group["transitions"][1]["below_threshold"])


if __name__ == "__main__":
    unittest.main()
