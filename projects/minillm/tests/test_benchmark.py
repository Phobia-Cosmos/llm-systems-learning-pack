from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from minillm import GPTConfig, MiniGPT
from minillm.benchmark import (
    BenchmarkSettings,
    VariantSpec,
    _build_model,
    _perplexity,
    _sync,
    common_parameter_names,
    component_variant_specs,
    make_fixed_batches,
    make_sequential_eval_batches,
    run_component_benchmark,
    write_benchmark_outputs,
)
from scripts.benchmark_components import format_run_summary


class ComponentBenchmarkTests(unittest.TestCase):
    def test_cli_summary_distinguishes_variants_from_per_seed_rows(self):
        payload = {
            "aggregates": [{} for _ in range(15)],
            "results": [{} for _ in range(45)],
        }

        self.assertEqual(
            format_run_summary(payload),
            "wrote 15 variants / 45 result rows",
        )

    def test_variant_matrix_matches_learning_plan(self):
        specs = component_variant_specs()
        self.assertEqual(len(specs), 15)
        self.assertEqual(
            [spec.variant_id for spec in specs],
            [
                "position/learned",
                "position/sinusoidal",
                "position/rope",
                "position/alibi",
                "position/none",
                "norm/layernorm",
                "norm/rmsnorm",
                "norm/scalenorm",
                "mlp/dense",
                "mlp/swiglu",
                "mlp/geglu",
                "mlp/reglu",
                "attention/mha",
                "attention/gqa",
                "attention/mqa",
            ],
        )
        effective_activations = {
            spec.name: spec.overrides["activation"]
            for spec in specs
            if spec.suite == "mlp"
        }
        self.assertEqual(
            effective_activations,
            {"dense": "gelu", "swiglu": "silu", "geglu": "gelu", "reglu": "relu"},
        )
        attention_kv_heads = {
            spec.name: spec.overrides["num_key_value_heads"]
            for spec in specs
            if spec.suite == "attention"
        }
        self.assertEqual(attention_kv_heads, {"mha": 4, "gqa": 2, "mqa": 1})

    def test_attention_matrix_adapts_to_query_heads_and_rejects_prime_count(self):
        specs = component_variant_specs(("attention",), n_head=12)
        self.assertEqual(
            [spec.overrides["num_key_value_heads"] for spec in specs],
            [12, 6, 1],
        )
        with self.assertRaisesRegex(ValueError, "divisor strictly between 1 and n_head"):
            component_variant_specs(("attention",), n_head=5)
        # Other suites do not need a distinct GQA configuration.
        self.assertEqual(len(component_variant_specs(("position",), n_head=5)), 5)

    def test_fixed_batches_are_seed_reproducible(self):
        data = torch.arange(64)
        first = make_fixed_batches(data, block_size=8, batch_size=3, num_batches=2, seed=7)
        second = make_fixed_batches(data, block_size=8, batch_size=3, num_batches=2, seed=7)
        for (first_x, first_y), (second_x, second_y) in zip(first, second):
            torch.testing.assert_close(first_x, second_x)
            torch.testing.assert_close(first_y, second_y)
            torch.testing.assert_close(first_x[:, 1:], first_y[:, :-1])

    def test_sequential_eval_covers_each_target_once(self):
        data = torch.arange(11)
        batches = make_sequential_eval_batches(data, block_size=4)
        targets = torch.cat([y.flatten() for _, y in batches])
        torch.testing.assert_close(targets, torch.arange(1, 11))
        self.assertEqual(sum(y.numel() for _, y in batches), len(data) - 1)

    def test_mps_sync_branch_and_perplexity(self):
        with patch("torch.mps.synchronize") as synchronize:
            _sync(torch.device("mps"))
        synchronize.assert_called_once_with()
        self.assertAlmostEqual(_perplexity(math.log(4.0)), 4.0)
        with self.assertRaises(FloatingPointError):
            _perplexity(float("nan"))

    def test_one_cached_token_does_not_compute_unused_logits(self):
        model = MiniGPT(
            GPTConfig(
                vocab_size=8,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                position_encoding="rope",
            )
        ).eval()
        prompt = torch.tensor([[1, 2, 3]])
        with patch.object(model, "forward_with_cache", wraps=model.forward_with_cache) as forward:
            model.generate_with_kv_cache(prompt, max_new_tokens=1, greedy=True)
        self.assertEqual(forward.call_count, 1)

    def test_common_parameters_ignore_variant_rng_layout(self):
        settings = BenchmarkSettings(
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_size=8,
            model_seeds=(17,),
        )
        learned, rope = component_variant_specs(("position",))[0:3:2]
        learned_model = _build_model(settings, 16, learned, model_seed=17)
        rope_model = _build_model(settings, 16, rope, model_seed=17)
        learned_parameters = dict(learned_model.named_parameters())
        rope_parameters = dict(rope_model.named_parameters())
        names = common_parameter_names(learned_model, rope_model)
        self.assertIn("token_embedding.weight", names)
        self.assertIn("blocks.0.attn.c_attn.weight", names)
        for name in names:
            torch.testing.assert_close(learned_parameters[name], rope_parameters[name])

    def test_smoke_run_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "corpus.txt"
            data_path.write_text(("猫吃鱼。狗吃肉。\n" * 30), encoding="utf-8")
            settings = BenchmarkSettings(
                data_path=str(data_path),
                max_steps=1,
                batch_size=2,
                block_size=8,
                eval_batches=1,
                n_layer=1,
                n_head=2,
                n_embd=8,
                prompt="猫吃",
                max_new_tokens=2,
                generation_repeats=1,
                torch_threads=1,
                model_seeds=(11,),
            )
            spec = VariantSpec(
                suite="position",
                name="rope",
                overrides={
                    "position_encoding": "rope",
                    "norm_type": "layernorm",
                    "mlp_type": "dense",
                    "activation": "gelu",
                },
            )
            payload = run_component_benchmark(settings, specs=[spec])
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(len(payload["aggregates"]), 1)
            self.assertEqual(len(payload["results"]), 1)
            result = payload["results"][0]
            self.assertTrue(result["cache_matches_full"])
            self.assertEqual(len(result["completion_token_ids"]), 2)
            self.assertEqual(result["trained_tokens"], 16)
            self.assertEqual(result["model_seed"], 11)
            self.assertEqual(result["num_query_heads"], 2)
            self.assertEqual(result["num_key_value_heads"], 2)
            self.assertEqual(result["kv_cache_elements_per_token_per_layer"], 16)
            self.assertEqual(result["kv_cache_bytes_per_token_per_layer"], 64)
            self.assertEqual(result["kv_cache_compression_ratio_vs_mha"], 1.0)
            self.assertEqual(result["resolved_config"]["num_key_value_heads"], 2)
            self.assertEqual(len(payload["data_sha256"]), 64)
            self.assertEqual(len(payload["schedule_sha256"]), 64)

            repeated_payload = run_component_benchmark(settings, specs=[spec])
            self.assertEqual(payload["run_id"], repeated_payload["run_id"])
            deterministic_fields = (
                "initial_train_loss",
                "final_train_loss",
                "initial_val_loss",
                "final_val_loss",
                "completion_token_ids",
                "max_gradient_norm",
                "clipped_steps",
            )
            for field in deterministic_fields:
                self.assertEqual(payload["results"][0][field], repeated_payload["results"][0][field])

            json_path = root / "result.json"
            csv_path = root / "result.csv"
            markdown_path = root / "result.md"
            write_benchmark_outputs(
                payload,
                json_path=json_path,
                csv_path=csv_path,
                markdown_path=markdown_path,
            )
            self.assertEqual(json.loads(json_path.read_text())["schema_version"], 2)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["variant_id"], "position/rope")
            self.assertEqual(json.loads(rows[0]["resolved_config"])["num_key_value_heads"], 2)
            self.assertIn("Fairness contract", markdown_path.read_text(encoding="utf-8"))

    def test_attention_smoke_reports_parameter_and_kv_cache_savings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "corpus.txt"
            data_path.write_text(("猫吃鱼。狗吃肉。\n" * 30), encoding="utf-8")
            settings = BenchmarkSettings(
                data_path=str(data_path),
                max_steps=1,
                batch_size=2,
                block_size=8,
                eval_batches=1,
                n_layer=1,
                n_head=4,
                n_embd=8,
                prompt="猫吃",
                max_new_tokens=1,
                generation_repeats=1,
                torch_threads=1,
                model_seeds=(23,),
            )

            payload = run_component_benchmark(settings, suites=("attention",))
            rows = payload["results"]
            self.assertEqual(
                [row["variant_id"] for row in rows],
                ["attention/mha", "attention/gqa", "attention/mqa"],
            )
            self.assertEqual([row["num_key_value_heads"] for row in rows], [4, 2, 1])
            self.assertEqual(
                [row["kv_cache_elements_per_token_per_layer"] for row in rows],
                [16, 8, 4],
            )
            self.assertEqual(
                [row["kv_cache_bytes_per_token_per_layer"] for row in rows],
                [64, 32, 16],
            )
            self.assertEqual(
                [row["kv_cache_compression_ratio_vs_mha"] for row in rows],
                [1.0, 2.0, 4.0],
            )
            self.assertGreater(rows[0]["parameter_count"], rows[1]["parameter_count"])
            self.assertGreater(rows[1]["parameter_count"], rows[2]["parameter_count"])
            self.assertTrue(all(row["cache_matches_full"] for row in rows))


if __name__ == "__main__":
    unittest.main()
