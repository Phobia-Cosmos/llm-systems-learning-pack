from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from minillm import GPTConfig, MiniGPT
from minillm.inference_baseline import (
    NanoVLLMTorchBaseline,
    TimerSettings,
    run_inference_baseline,
    stage_frequency,
    tensor_metadata,
)


class InferenceBaselineTests(unittest.TestCase):
    def test_tensor_metadata_exposes_stride_and_alignment(self):
        tensor = torch.empty(2, 3, 8).transpose(0, 1)
        metadata = tensor_metadata(tensor)
        self.assertEqual(metadata["shape"], [3, 2, 8])
        self.assertEqual(metadata["stride"], list(tensor.stride()))
        self.assertFalse(metadata["is_contiguous"])
        self.assertEqual(metadata["bytes"], tensor.numel() * tensor.element_size())
        self.assertEqual(metadata["data_ptr"] % metadata["pointer_alignment_bytes"], 0)
        self.assertGreaterEqual(metadata["pointer_alignment_bytes"], metadata["all_row_starts_alignment_bytes"])
        self.assertTrue(metadata["last_dim_multiples"]["8"])

    def test_row_alignment_accounts_for_stride(self):
        storage = torch.empty(4 * 21 + 64, dtype=torch.float32)
        offset = (-storage.data_ptr() // storage.element_size()) % 64
        tensor = storage[offset : offset + 4 * 21].view(4, 21)
        metadata = tensor_metadata(tensor)
        self.assertEqual(metadata["pointer_mod_bytes"]["256"], 0)
        self.assertEqual(metadata["all_row_starts_alignment_bytes"], 4)

    def test_nano_fallback_frequency_includes_layer_and_sequence_loops(self):
        frequency = stage_frequency(
            "softmax",
            "nano_vllm_torch",
            "decode",
            num_layers=2,
            batch_size=8,
            generated_tokens=16,
        )
        self.assertEqual(frequency["logical_stage_calls_per_pass"], 16)
        self.assertEqual(frequency["model_passes_for_generation"], 15)
        self.assertEqual(frequency["logical_stage_calls_for_generation"], 240)

    def test_kv_pair_frequency_counts_two_primary_operations(self):
        frequency = stage_frequency(
            "nano_paged_kv_gather",
            "nano_vllm_torch",
            "decode",
            num_layers=2,
            batch_size=3,
            generated_tokens=4,
        )
        self.assertEqual(frequency["logical_stage_calls_per_pass"], 6)
        self.assertEqual(frequency["primary_ops_per_stage_call"], 2)
        self.assertEqual(frequency["primary_ops_per_pass"], 12)

    def test_flat_paged_path_matches_native_cache_path(self):
        torch.manual_seed(7)
        config = GPTConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            position_encoding="rope",
        )
        model = MiniGPT(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

        with torch.inference_mode():
            native_prefill, past_key_values = model.forward_with_cache(input_ids)
            next_ids_2d = native_prefill[:, -1, :].argmax(dim=-1, keepdim=True)
            native_decode, _ = model.forward_with_cache(next_ids_2d, past_key_values)

            nano = NanoVLLMTorchBaseline(
                model,
                batch_size=input_ids.shape[0],
                max_context_length=input_ids.shape[1],
                page_size=8,
            )
            nano_prefill = nano.prefill(input_ids)
            nano_decode = nano.decode(next_ids_2d[:, 0], input_ids.shape[1])

        torch.testing.assert_close(native_prefill[:, -1, :], nano_prefill, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(native_decode[:, 0, :], nano_decode, atol=1e-5, rtol=1e-5)

    def test_gqa_flat_paged_path_matches_native_and_keeps_compact_kv_cache(self):
        torch.manual_seed(11)
        config = GPTConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=4,
            num_key_value_heads=2,
            n_embd=16,
            dropout=0.0,
            position_encoding="rope",
        )
        model = MiniGPT(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        batch_size, context_length = input_ids.shape
        page_size = 8

        with torch.inference_mode():
            native_prefill, past_key_values = model.forward_with_cache(input_ids)
            next_ids_2d = native_prefill[:, -1, :].argmax(dim=-1, keepdim=True)
            native_decode, present_key_values = model.forward_with_cache(
                next_ids_2d, past_key_values
            )

            nano = NanoVLLMTorchBaseline(
                model,
                batch_size=batch_size,
                max_context_length=context_length,
                page_size=page_size,
            )
            nano_prefill, prefill_trace = nano.prefill(input_ids, trace=True)
            nano_decode, decode_trace = nano.decode(
                next_ids_2d[:, 0], context_length, trace=True
            )

        torch.testing.assert_close(
            native_prefill[:, -1, :], nano_prefill, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            native_decode[:, 0, :], nano_decode, atol=1e-5, rtol=1e-5
        )

        head_dim = config.n_embd // config.n_head
        compact_native_shape = (
            batch_size,
            config.num_key_value_heads,
            context_length,
            head_dim,
        )
        for key, value in past_key_values:
            self.assertEqual(tuple(key.shape), compact_native_shape)
            self.assertEqual(tuple(value.shape), compact_native_shape)
        for key, value in present_key_values:
            self.assertEqual(
                tuple(key.shape),
                (batch_size, config.num_key_value_heads, context_length + 1, head_dim),
            )
            self.assertEqual(tuple(value.shape), tuple(key.shape))

        total_pages = batch_size * nano.pages_per_sequence
        compact_paged_shape = (
            total_pages,
            page_size,
            config.num_key_value_heads,
            head_dim,
        )
        full_head_cache_numel = total_pages * page_size * config.n_head * head_dim
        for key_cache, value_cache in zip(nano.key_caches, nano.value_caches):
            self.assertEqual(tuple(key_cache.shape), compact_paged_shape)
            self.assertEqual(tuple(value_cache.shape), compact_paged_shape)
            self.assertEqual(
                key_cache.numel(),
                full_head_cache_numel
                * config.num_key_value_heads
                // config.n_head,
            )
        self.assertEqual(prefill_trace["k_raw"].shape[1], config.num_key_value_heads)
        self.assertEqual(prefill_trace["first_key"].shape[1], config.n_head)
        self.assertEqual(decode_trace["key_cache"].shape[-2], config.num_key_value_heads)
        self.assertEqual(decode_trace["first_key"].shape[1], config.n_head)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_multiple_batch_sizes_reuse_model_within_dtype(self):
        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "checkpoints"
            / "minillm-rope.pt"
        )
        zero_timing = {
            "inner_loops": 1,
            "samples": 1,
            "gpu_ms": {key: 0.001 for key in ("median", "p90", "mean", "min", "max", "stddev")},
            "synchronized_wall_ms": {
                key: 0.001 for key in ("median", "p90", "mean", "min", "max", "stddev")
            },
        }
        zero_profile = {
            "bmm_count": 0,
            "clone_count": 0,
            "contiguous_count": 0,
            "copy_count": 0,
            "einsum_count": 0,
            "matmul_count": 0,
            "mm_count": 0,
            "implicit_materialization": False,
            "operators": [],
        }
        with (
            patch("minillm.inference_baseline.benchmark_cuda_operation", return_value=zero_timing),
            patch("minillm.inference_baseline.profile_cuda_operation", return_value=zero_profile),
        ):
            payload = run_inference_baseline(
                checkpoint,
                "LLM 是",
                batch_sizes=(1, 2),
                dtypes=(torch.float32,),
                timer_settings=TimerSettings(warmup=0, samples=1),
                project_root=Path(__file__).resolve().parents[3],
            )
        self.assertEqual([item["batch_size"] for item in payload["workloads"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
