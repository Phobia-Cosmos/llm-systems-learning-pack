from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from .config import GPTConfig
from .model import MiniGPT
from .tokenizer_registry import tokenizer_from_checkpoint


ALIGNMENTS = (256, 128, 64, 32, 16, 8, 4, 2, 1)
LINEAR_BACKEND = "PyTorch aten::linear -> cuBLAS/cuBLASLt"
MATMUL_BACKEND = "PyTorch aten::matmul/bmm -> cuBLAS"
EINSUM_BACKEND = "PyTorch aten::einsum -> bmm/cuBLAS"
CUDA_BACKEND = "PyTorch eager CUDA"


@dataclass(frozen=True)
class TimerSettings:
    warmup: int = 10
    samples: int = 30
    target_sample_ms: float = 3.0
    max_inner_loops: int = 512


@dataclass(frozen=True)
class StageCase:
    name: str
    runtime: str
    phase: str
    category: str
    backend: str
    operation: Callable[[], Any]
    inputs: dict[str, torch.Tensor]
    notes: str = ""
    estimated_flops: int | None = None


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _pointer_alignment(pointer: int) -> int:
    if pointer == 0:
        return 0
    return next(alignment for alignment in ALIGNMENTS if pointer % alignment == 0)


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    pointer = tensor.data_ptr() if tensor.numel() else 0
    stride = list(tensor.stride())
    element_size = tensor.element_size()
    active_outer_stride_bytes = [
        stride_value * element_size
        for size, stride_value in zip(tensor.shape[:-1], stride[:-1])
        if size > 1
    ]
    row_start_alignment = 0
    if pointer:
        row_start_alignment = next(
            alignment
            for alignment in ALIGNMENTS
            if pointer % alignment == 0
            and all(value % alignment == 0 for value in active_outer_stride_bytes)
        )
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
        "device": str(tensor.device),
        "stride": stride,
        "is_contiguous": tensor.is_contiguous(),
        "storage_offset": tensor.storage_offset(),
        "numel": tensor.numel(),
        "element_size_bytes": element_size,
        "bytes": tensor.numel() * element_size,
        "data_ptr": pointer,
        "pointer_alignment_bytes": _pointer_alignment(pointer),
        "all_row_starts_alignment_bytes": row_start_alignment,
        "pointer_mod_bytes": {
            str(alignment): pointer % alignment if pointer else None
            for alignment in (16, 32, 64, 128, 256)
        },
        "last_dim_stride": stride[-1] if stride else None,
        "last_dim_multiples": {
            str(multiple): bool(tensor.ndim and tensor.shape[-1] % multiple == 0)
            for multiple in (8, 16, 32)
        },
        "row_stride_bytes": stride[-2] * element_size if tensor.ndim >= 2 else None,
        "active_outer_stride_bytes": active_outer_stride_bytes,
        "last_dimension_bytes": tensor.shape[-1] * element_size if tensor.ndim else None,
    }


def _named_tensor_metadata(value: Any, prefix: str = "output") -> dict[str, dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        return {prefix: tensor_metadata(value)}
    if isinstance(value, (tuple, list)):
        result: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(value):
            result.update(_named_tensor_metadata(item, f"{prefix}.{index}"))
        return result
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            result.update(_named_tensor_metadata(item, f"{prefix}.{name}"))
        return result
    return {}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.pstdev(values),
    }


def _cuda_elapsed_ms(operation: Callable[[], Any], inner_loops: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner_loops):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def benchmark_cuda_operation(
    operation: Callable[[], Any],
    settings: TimerSettings,
) -> dict[str, Any]:
    for _ in range(settings.warmup):
        operation()
    torch.cuda.synchronize()

    inner_loops = 1
    while inner_loops < settings.max_inner_loops:
        elapsed_ms = _cuda_elapsed_ms(operation, inner_loops)
        if elapsed_ms >= settings.target_sample_ms:
            break
        inner_loops = min(settings.max_inner_loops, inner_loops * 2)

    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(settings.samples):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start_event.record()
        for _ in range(inner_loops):
            operation()
        end_event.record()
        end_event.synchronize()
        wall_end = time.perf_counter_ns()
        gpu_samples.append(start_event.elapsed_time(end_event) / inner_loops)
        wall_samples.append((wall_end - wall_start) / 1_000_000 / inner_loops)

    return {
        "inner_loops": inner_loops,
        "samples": settings.samples,
        "gpu_ms": _summary(gpu_samples),
        "synchronized_wall_ms": _summary(wall_samples),
    }


def profile_cuda_operation(operation: Callable[[], Any]) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        operation()
    torch.cuda.synchronize()

    operator_aliases = {
        "aten::bmm": "bmm",
        "aten::clone": "clone",
        "aten::contiguous": "contiguous",
        "aten::copy_": "copy",
        "aten::einsum": "einsum",
        "aten::matmul": "matmul",
        "aten::mm": "mm",
    }
    operators = []
    counts = {alias + "_count": 0 for alias in operator_aliases.values()}
    for event in profiler.key_averages(group_by_input_shape=True):
        if event.key not in operator_aliases:
            continue
        counts[operator_aliases[event.key] + "_count"] += event.count
        operators.append(
            {
                "operator": event.key,
                "count": event.count,
                "self_cpu_time_us": event.self_cpu_time_total,
                "self_cuda_time_us": event.self_device_time_total,
                "input_shapes": event.input_shapes,
            }
        )
    return {
        **counts,
        "implicit_materialization": bool(
            counts["clone_count"] or counts["contiguous_count"]
        ),
        "operators": operators,
    }


def stage_frequency(
    stage: str,
    runtime: str,
    phase: str,
    *,
    num_layers: int,
    batch_size: int,
    generated_tokens: int,
) -> dict[str, Any]:
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"unsupported phase: {phase}")
    if runtime not in {"minillm", "nano_vllm_torch"}:
        raise ValueError(f"unsupported runtime: {runtime}")

    model_level = {"embedding", "final_norm", "lm_head", "full_prefill", "full_decode_step"}
    per_sequence_nano = {
        "qk_matmul",
        "causal_mask",
        "softmax",
        "attention_value_matmul",
    }

    if stage in model_level:
        logical_calls = 1
        scope = "per model pass"
    elif runtime == "nano_vllm_torch" and stage in per_sequence_nano:
        logical_calls = num_layers * batch_size
        scope = "per layer and sequence (PyTorch fallback loop)"
    elif runtime == "nano_vllm_torch" and stage == "nano_paged_kv_gather":
        logical_calls = num_layers * batch_size if phase == "decode" else 0
        scope = "one logical K/V pair per layer and sequence"
    else:
        logical_calls = num_layers
        scope = "per layer"

    primary_ops_per_call = 2 if stage in {
        "native_kv_concat",
        "nano_paged_kv_store",
        "nano_paged_kv_gather",
    } else 1
    model_passes = 1 if phase == "prefill" else max(generated_tokens - 1, 0)
    return {
        "scope": scope,
        "logical_stage_calls_per_pass": logical_calls,
        "primary_ops_per_stage_call": primary_ops_per_call,
        "primary_ops_per_pass": logical_calls * primary_ops_per_call,
        "model_passes_for_generation": model_passes,
        "logical_stage_calls_for_generation": logical_calls * model_passes,
        "generation_convention": (
            "prefill produces token 1; N generated tokens require N-1 decode model passes"
        ),
    }


def _linear_flops(input_tensor: torch.Tensor, weight: torch.Tensor) -> int:
    rows = input_tensor.numel() // input_tensor.shape[-1]
    return 2 * rows * weight.shape[0] * weight.shape[1]


def _attention_flops(query: torch.Tensor, key: torch.Tensor) -> int:
    if query.ndim == 4:
        batch, heads, query_len, head_dim = query.shape
        key_len = key.shape[-2]
    elif query.ndim == 3:
        query_len, heads, head_dim = query.shape
        batch = 1
        key_len = key.shape[0]
    else:
        raise ValueError("attention tensors must be [B,H,Q,D] or [Q,H,D]")
    return 2 * batch * heads * query_len * key_len * head_dim


def _case(
    name: str,
    runtime: str,
    phase: str,
    category: str,
    backend: str,
    operation: Callable[[], Any],
    inputs: dict[str, torch.Tensor],
    *,
    notes: str = "",
    estimated_flops: int | None = None,
) -> StageCase:
    return StageCase(
        name=name,
        runtime=runtime,
        phase=phase,
        category=category,
        backend=backend,
        operation=operation,
        inputs=inputs,
        notes=notes,
        estimated_flops=estimated_flops,
    )


def _native_layer_trace(
    model: MiniGPT,
    input_ids: torch.Tensor,
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> dict[str, Any]:
    batch_size, query_len = input_ids.shape
    past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
    positions = torch.arange(past_len, past_len + query_len, device=input_ids.device)
    hidden = model.token_embedding(input_ids)
    if model.position_embedding is not None:
        hidden = hidden + model.position_embedding(positions).to(hidden.dtype)
    hidden = model.drop(hidden)

    block = model.blocks[0]
    if block.attn.c_attn is None or not hasattr(block.mlp, "net"):
        raise ValueError("the current baseline expects fused QKV and the dense MiniLLM MLP")
    attention_norm = block.ln_1(hidden)
    q_flat, k_flat, v_flat = block.attn.project_qkv(attention_norm)
    qkv = torch.cat((q_flat, k_flat, v_flat), dim=-1)
    q_raw = q_flat.view(
        batch_size, query_len, block.attn.n_head, block.attn.head_dim
    ).transpose(1, 2)
    k_raw = k_flat.view(
        batch_size,
        query_len,
        block.attn.num_key_value_heads,
        block.attn.head_dim,
    ).transpose(1, 2)
    v_raw = v_flat.view(
        batch_size,
        query_len,
        block.attn.num_key_value_heads,
        block.attn.head_dim,
    ).transpose(1, 2)
    query, key_new = block.attn.position_encoding.apply_qk(q_raw, k_raw, positions)

    if past_key_values is None:
        compact_key, compact_value = key_new, v_raw
    else:
        compact_key = torch.cat((past_key_values[0][0], key_new), dim=2)
        compact_value = torch.cat((past_key_values[0][1], v_raw), dim=2)
    key = block.attn.expand_kv_heads(compact_key)
    value = block.attn.expand_kv_heads(compact_value)
    total_len = compact_key.shape[2]
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
    position_bias = block.attn.position_encoding.attention_bias(
        positions,
        torch.arange(total_len, device=input_ids.device),
        dtype=scores.dtype,
        device=scores.device,
    )
    if position_bias is not None:
        scores = scores + position_bias
    mask = block.attn.causal_mask[:, :, past_len : past_len + query_len, :total_len]
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))
    probabilities = F.softmax(masked_scores, dim=-1)
    attention_values = probabilities @ value
    merged_heads = attention_values.transpose(1, 2).contiguous().view(
        batch_size, query_len, model.config.n_embd
    )
    attention_projection = block.attn.c_proj(merged_heads)
    attention_residual = hidden + attention_projection
    mlp_norm = block.ln_2(attention_residual)
    mlp_fc1 = block.mlp.net[0](mlp_norm)
    mlp_activation = block.mlp.net[1](mlp_fc1)
    mlp_fc2 = block.mlp.net[2](mlp_activation)
    layer_output = attention_residual + mlp_fc2

    present = [(compact_key, compact_value)]
    hidden_after_layers = layer_output
    for layer_index, next_block in enumerate(model.blocks[1:], start=1):
        layer_past = None if past_key_values is None else past_key_values[layer_index]
        hidden_after_layers, layer_present = next_block.forward_with_cache(
            hidden_after_layers, positions, layer_past
        )
        present.append(layer_present)
    pre_final_norm = hidden_after_layers
    final_norm = model.ln_f(pre_final_norm)
    logits = model.lm_head(final_norm)
    return {
        "input_ids": input_ids,
        "positions": positions,
        "hidden": hidden,
        "attention_norm": attention_norm,
        "qkv": qkv,
        "q_raw": q_raw,
        "k_raw": k_raw,
        "v_raw": v_raw,
        "query": query,
        "key_new": key_new,
        "compact_key": compact_key,
        "compact_value": compact_value,
        "key": key,
        "value": value,
        "scores": scores,
        "mask": mask,
        "masked_scores": masked_scores,
        "probabilities": probabilities,
        "attention_values": attention_values,
        "merged_heads": merged_heads,
        "attention_projection": attention_projection,
        "attention_residual": attention_residual,
        "mlp_norm": mlp_norm,
        "mlp_fc1": mlp_fc1,
        "mlp_activation": mlp_activation,
        "mlp_fc2": mlp_fc2,
        "layer_output": layer_output,
        "pre_final_norm": pre_final_norm,
        "final_norm": final_norm,
        "logits": logits,
        "present": present,
    }


def _native_cases(
    model: MiniGPT,
    trace: dict[str, Any],
    phase: str,
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> list[StageCase]:
    block = model.blocks[0]
    runtime = "minillm"
    cases = [
        _case(
            "embedding", runtime, phase, "lookup", CUDA_BACKEND,
            lambda: model.token_embedding(trace["input_ids"]),
            {"input_ids": trace["input_ids"], "weight": model.token_embedding.weight},
        ),
        _case(
            "attention_norm", runtime, phase, "normalization", CUDA_BACKEND,
            lambda: block.ln_1(trace["hidden"]),
            {"hidden_states": trace["hidden"]},
        ),
        _case(
            "qkv_linear", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["attention_norm"], block.attn.c_attn.weight, block.attn.c_attn.bias),
            {"hidden_states": trace["attention_norm"], "weight": block.attn.c_attn.weight},
            estimated_flops=_linear_flops(trace["attention_norm"], block.attn.c_attn.weight),
        ),
        _case(
            "rope", runtime, phase, "position", CUDA_BACKEND,
            lambda: block.attn.position_encoding.apply_qk(
                trace["q_raw"], trace["k_raw"], trace["positions"]
            ),
            {"q_raw": trace["q_raw"], "k_raw": trace["k_raw"], "positions": trace["positions"]},
            notes="RoPE converts the strided fused-QKV views into contiguous Q/K tensors.",
        ),
        _case(
            "qk_matmul", runtime, phase, "matmul", MATMUL_BACKEND,
            lambda: (trace["query"] @ trace["key"].transpose(-2, -1))
            / math.sqrt(block.attn.head_dim),
            {"query": trace["query"], "key": trace["key"]},
            estimated_flops=_attention_flops(trace["query"], trace["key"]),
        ),
        _case(
            "causal_mask", runtime, phase, "elementwise", CUDA_BACKEND,
            lambda: trace["scores"].masked_fill(trace["mask"] == 0, float("-inf")),
            {"scores": trace["scores"], "mask": trace["mask"]},
        ),
        _case(
            "softmax", runtime, phase, "reduction", CUDA_BACKEND,
            lambda: F.softmax(trace["masked_scores"], dim=-1),
            {"scores": trace["masked_scores"]},
        ),
        _case(
            "attention_value_matmul", runtime, phase, "matmul", MATMUL_BACKEND,
            lambda: trace["probabilities"] @ trace["value"],
            {"probabilities": trace["probabilities"], "value": trace["value"]},
            estimated_flops=_attention_flops(trace["query"], trace["key"]),
        ),
        _case(
            "head_merge", runtime, phase, "layout", CUDA_BACKEND,
            lambda: trace["attention_values"].transpose(1, 2).contiguous().view(
                *trace["hidden"].shape
            ),
            {"attention_values": trace["attention_values"]},
        ),
        _case(
            "output_projection", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["merged_heads"], block.attn.c_proj.weight, block.attn.c_proj.bias),
            {"hidden_states": trace["merged_heads"], "weight": block.attn.c_proj.weight},
            estimated_flops=_linear_flops(trace["merged_heads"], block.attn.c_proj.weight),
        ),
        _case(
            "mlp_norm", runtime, phase, "normalization", CUDA_BACKEND,
            lambda: block.ln_2(trace["attention_residual"]),
            {"hidden_states": trace["attention_residual"]},
        ),
        _case(
            "mlp_fc1", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["mlp_norm"], block.mlp.net[0].weight, block.mlp.net[0].bias),
            {"hidden_states": trace["mlp_norm"], "weight": block.mlp.net[0].weight},
            estimated_flops=_linear_flops(trace["mlp_norm"], block.mlp.net[0].weight),
        ),
        _case(
            "gelu", runtime, phase, "activation", CUDA_BACKEND,
            lambda: F.gelu(trace["mlp_fc1"]),
            {"hidden_states": trace["mlp_fc1"]},
        ),
        _case(
            "mlp_fc2", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["mlp_activation"], block.mlp.net[2].weight, block.mlp.net[2].bias),
            {"hidden_states": trace["mlp_activation"], "weight": block.mlp.net[2].weight},
            estimated_flops=_linear_flops(trace["mlp_activation"], block.mlp.net[2].weight),
        ),
        _case(
            "final_norm", runtime, phase, "normalization", CUDA_BACKEND,
            lambda: model.ln_f(trace["pre_final_norm"]),
            {"hidden_states": trace["pre_final_norm"]},
        ),
        _case(
            "lm_head", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["final_norm"], model.lm_head.weight),
            {"hidden_states": trace["final_norm"], "weight": model.lm_head.weight},
            estimated_flops=_linear_flops(trace["final_norm"], model.lm_head.weight),
        ),
    ]
    if phase == "decode":
        assert past_key_values is not None
        cases.insert(
            4,
            _case(
                "native_kv_concat", runtime, phase, "kv_cache", CUDA_BACKEND,
                lambda: (
                    torch.cat((past_key_values[0][0], trace["key_new"]), dim=2),
                    torch.cat((past_key_values[0][1], trace["v_raw"]), dim=2),
                ),
                {
                    "past_key": past_key_values[0][0],
                    "new_key": trace["key_new"],
                    "past_value": past_key_values[0][1],
                    "new_value": trace["v_raw"],
                },
                notes="The teaching cache reallocates and copies both complete K and V histories each step.",
            ),
        )
        cases.append(
            _case(
                "full_decode_step", runtime, phase, "full_model", "PyTorch eager mixed",
                lambda: model.forward_with_cache(trace["input_ids"], past_key_values)[0],
                {"input_ids": trace["input_ids"], "past_key": past_key_values[0][0]},
            )
        )
    else:
        cases.append(
            _case(
                "full_prefill", runtime, phase, "full_model", "PyTorch eager mixed",
                lambda: model.forward_with_cache(trace["input_ids"])[0],
                {"input_ids": trace["input_ids"]},
            )
        )
    return cases


def _flat_rope(
    model: MiniGPT,
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = model.blocks[0].attn.rotary
    if rotary is None:
        return query, key
    cos = rotary.cos.index_select(0, positions).unsqueeze(1).to(query.dtype)
    sin = rotary.sin.index_select(0, positions).unsqueeze(1).to(query.dtype)

    def apply(tensor: torch.Tensor) -> torch.Tensor:
        first, second = torch.chunk(tensor.float(), 2, dim=-1)
        result = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        return result.to(tensor.dtype)

    return apply(query), apply(key)


def _paged_store_pair(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache.view(-1, key_cache.shape[-2], key_cache.shape[-1]).index_copy_(0, slot_mapping, key)
    value_cache.view(-1, value_cache.shape[-2], value_cache.shape[-1]).index_copy_(
        0, slot_mapping, value
    )
    return key_cache, value_cache


def _paged_gather_pair(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_size = key_cache.shape[1]
    page_count = math.ceil(sequence_length / page_size)
    block_ids = block_table[:page_count].long()

    def gather(cache: torch.Tensor) -> torch.Tensor:
        return cache.index_select(0, block_ids).reshape(
            -1, cache.shape[-2], cache.shape[-1]
        )[:sequence_length]

    return gather(key_cache), gather(value_cache)


def _flat_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start: int,
    scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = torch.einsum("qhd,khd->hqk", query, key).float() * scale
    query_positions = torch.arange(
        query_start, query_start + query.shape[0], device=query.device
    )
    key_positions = torch.arange(key.shape[0], device=query.device)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    masked_scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(masked_scores, dim=-1).to(query.dtype)
    output = torch.einsum("hqk,khd->qhd", probabilities, value)
    return output, {
        "scores": scores,
        "mask": mask.unsqueeze(0),
        "masked_scores": masked_scores,
        "probabilities": probabilities,
    }


class NanoVLLMTorchBaseline:
    """PyTorch fallback with nano-vLLM's flattened tokens and paged KV layout."""

    def __init__(
        self,
        model: MiniGPT,
        batch_size: int,
        max_context_length: int,
        page_size: int,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.page_size = page_size
        attention = model.blocks[0].attn
        self.heads = attention.n_head
        self.num_key_value_heads = attention.num_key_value_heads
        self.head_dim = attention.head_dim
        self.pages_per_sequence = math.ceil((max_context_length + 1) / page_size)
        total_pages = batch_size * self.pages_per_sequence
        device = model.token_embedding.weight.device
        dtype = model.token_embedding.weight.dtype
        self.block_tables = torch.arange(
            total_pages, device=device, dtype=torch.int32
        ).view(batch_size, self.pages_per_sequence)
        cache_shape = (
            total_pages,
            page_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        self.key_caches = [
            torch.empty(cache_shape, device=device, dtype=dtype)
            for _ in range(model.config.n_layer)
        ]
        self.value_caches = [
            torch.empty(cache_shape, device=device, dtype=dtype)
            for _ in range(model.config.n_layer)
        ]
        self._slot_mapping_cache: dict[tuple[int, int], torch.Tensor] = {}

    def slot_mapping(self, query_length: int, position_start: int) -> torch.Tensor:
        cache_key = (query_length, position_start)
        cached = self._slot_mapping_cache.get(cache_key)
        if cached is not None:
            return cached

        positions = torch.arange(
            position_start,
            position_start + query_length,
            device=self.block_tables.device,
            dtype=torch.long,
        ).expand(self.batch_size, -1)
        page_indices = torch.div(positions, self.page_size, rounding_mode="floor")
        block_ids = self.block_tables.long().gather(1, page_indices)
        slots = (block_ids * self.page_size + positions.remainder(self.page_size)).reshape(-1)
        self._slot_mapping_cache[cache_key] = slots
        return slots

    def prefill(self, input_ids: torch.Tensor, *, trace: bool = False):
        if input_ids.ndim != 2 or input_ids.shape[0] != self.batch_size:
            raise ValueError("prefill input_ids must have shape [batch_size, sequence_length]")
        sequence_length = input_ids.shape[1]
        flat_ids = input_ids.reshape(-1)
        positions = torch.arange(sequence_length, device=input_ids.device).repeat(self.batch_size)
        slots = self.slot_mapping(sequence_length, 0)
        return self._forward(
            flat_ids, positions, slots, sequence_length, is_prefill=True, trace=trace
        )

    def decode(self, input_ids: torch.Tensor, context_length: int, *, trace: bool = False):
        if input_ids.shape != (self.batch_size,):
            raise ValueError("decode input_ids must have shape [batch_size]")
        positions = torch.full(
            (self.batch_size,), context_length, device=input_ids.device, dtype=torch.long
        )
        slots = self.slot_mapping(1, context_length)
        return self._forward(
            input_ids, positions, slots, context_length, is_prefill=False, trace=trace
        )

    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        context_length: int,
        *,
        is_prefill: bool,
        trace: bool,
    ):
        hidden = self.model.token_embedding(input_ids)
        if self.model.position_embedding is not None:
            hidden = hidden + self.model.position_embedding(positions).to(hidden.dtype)
        initial_hidden = hidden
        layer_trace: dict[str, Any] | None = None

        for layer_index, block in enumerate(self.model.blocks):
            if block.attn.c_attn is None or not hasattr(block.mlp, "net"):
                raise ValueError("the current baseline expects fused QKV and a dense MLP")
            attention_norm = block.ln_1(hidden)
            query_flat, key_flat, value_flat = block.attn.project_qkv(attention_norm)
            qkv = torch.cat((query_flat, key_flat, value_flat), dim=-1)
            query_raw = query_flat.view(-1, self.heads, self.head_dim)
            key_raw = key_flat.view(-1, self.num_key_value_heads, self.head_dim)
            value = value_flat.view(-1, self.num_key_value_heads, self.head_dim)
            query, key = _flat_rope(self.model, query_raw, key_raw, positions)
            _paged_store_pair(
                key,
                value,
                self.key_caches[layer_index],
                self.value_caches[layer_index],
                slots,
            )

            outputs = []
            first_attention_trace = None
            if is_prefill:
                expanded_key = block.attn.expand_kv_heads(key)
                expanded_value = block.attn.expand_kv_heads(value)
                for sequence_index in range(self.batch_size):
                    start = sequence_index * context_length
                    end = start + context_length
                    output, attention_trace = _flat_attention(
                        query[start:end],
                        expanded_key[start:end],
                        expanded_value[start:end],
                        0,
                        self.head_dim**-0.5,
                    )
                    outputs.append(output)
                    if first_attention_trace is None:
                        first_attention_trace = attention_trace
            else:
                sequence_length = context_length + 1
                for sequence_index in range(self.batch_size):
                    gathered_key, gathered_value = _paged_gather_pair(
                        self.key_caches[layer_index],
                        self.value_caches[layer_index],
                        self.block_tables[sequence_index],
                        sequence_length,
                    )
                    expanded_key = block.attn.expand_kv_heads(gathered_key)
                    expanded_value = block.attn.expand_kv_heads(gathered_value)
                    output, attention_trace = _flat_attention(
                        query[sequence_index : sequence_index + 1],
                        expanded_key,
                        expanded_value,
                        sequence_length - 1,
                        self.head_dim**-0.5,
                    )
                    outputs.append(output)
                    if first_attention_trace is None:
                        first_attention_trace = attention_trace

            attention_values = torch.cat(outputs, dim=0)
            merged_heads = attention_values.flatten(1, -1)
            attention_projection = block.attn.c_proj(merged_heads)
            attention_residual = hidden + attention_projection
            mlp_norm = block.ln_2(attention_residual)
            mlp_fc1 = block.mlp.net[0](mlp_norm)
            mlp_activation = block.mlp.net[1](mlp_fc1)
            mlp_fc2 = block.mlp.net[2](mlp_activation)
            hidden = attention_residual + mlp_fc2

            if layer_index == 0 and trace:
                if is_prefill:
                    first_key = expanded_key[:context_length]
                    first_value = expanded_value[:context_length]
                else:
                    first_key, first_value = _paged_gather_pair(
                        self.key_caches[0], self.value_caches[0], self.block_tables[0],
                        context_length + 1,
                    )
                    first_key = block.attn.expand_kv_heads(first_key)
                    first_value = block.attn.expand_kv_heads(first_value)
                layer_trace = {
                    "input_ids": input_ids,
                    "positions": positions,
                    "hidden": initial_hidden,
                    "attention_norm": attention_norm,
                    "qkv": qkv,
                    "q_raw": query_raw,
                    "k_raw": key_raw,
                    "v_raw": value,
                    "query": query,
                    "key": key,
                    "first_query": query[:context_length] if is_prefill else query[:1],
                    "first_key": first_key,
                    "first_value": first_value,
                    "scores": first_attention_trace["scores"],
                    "mask": first_attention_trace["mask"],
                    "masked_scores": first_attention_trace["masked_scores"],
                    "probabilities": first_attention_trace["probabilities"],
                    "attention_outputs": outputs,
                    "attention_values": attention_values,
                    "merged_heads": merged_heads,
                    "attention_projection": attention_projection,
                    "attention_residual": attention_residual,
                    "mlp_norm": mlp_norm,
                    "mlp_fc1": mlp_fc1,
                    "mlp_activation": mlp_activation,
                    "mlp_fc2": mlp_fc2,
                    "slot_mapping": slots,
                    "key_cache": self.key_caches[0],
                    "value_cache": self.value_caches[0],
                    "block_table": self.block_tables[0],
                }

        pre_final_norm = hidden
        final_norm = self.model.ln_f(pre_final_norm)
        if is_prefill:
            last_indices = (
                torch.arange(1, self.batch_size + 1, device=input_ids.device) * context_length - 1
            )
            head_input = final_norm.index_select(0, last_indices).contiguous()
        else:
            head_input = final_norm
        logits = self.model.lm_head(head_input)
        if trace:
            assert layer_trace is not None
            layer_trace["pre_final_norm"] = pre_final_norm
            layer_trace["final_norm"] = final_norm
            layer_trace["head_input"] = head_input
            layer_trace["logits"] = logits
            return logits, layer_trace
        return logits


def _nano_cases(
    model: MiniGPT,
    executor: NanoVLLMTorchBaseline,
    trace: dict[str, Any],
    phase: str,
    input_ids_2d: torch.Tensor,
    next_ids: torch.Tensor,
    context_length: int,
) -> list[StageCase]:
    block = model.blocks[0]
    runtime = "nano_vllm_torch"
    q = trace["first_query"]
    k = trace["first_key"]
    v = trace["first_value"]
    cases = [
        _case(
            "embedding", runtime, phase, "lookup", CUDA_BACKEND,
            lambda: model.token_embedding(trace["input_ids"]),
            {"input_ids": trace["input_ids"], "weight": model.token_embedding.weight},
        ),
        _case(
            "attention_norm", runtime, phase, "normalization", CUDA_BACKEND,
            lambda: block.ln_1(trace["hidden"]),
            {"hidden_states": trace["hidden"]},
        ),
        _case(
            "qkv_linear", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["attention_norm"], block.attn.c_attn.weight, block.attn.c_attn.bias),
            {"hidden_states": trace["attention_norm"], "weight": block.attn.c_attn.weight},
            estimated_flops=_linear_flops(trace["attention_norm"], block.attn.c_attn.weight),
        ),
        _case(
            "rope", runtime, phase, "position", CUDA_BACKEND,
            lambda: _flat_rope(model, trace["q_raw"], trace["k_raw"], trace["positions"]),
            {"q_raw": trace["q_raw"], "k_raw": trace["k_raw"], "positions": trace["positions"]},
            notes="Uses nano-vLLM's flattened [tokens, heads, head_dim] convention.",
        ),
        _case(
            "nano_paged_kv_store", runtime, phase, "kv_cache", CUDA_BACKEND,
            lambda: _paged_store_pair(
                trace["key"], trace["v_raw"], trace["key_cache"], trace["value_cache"],
                trace["slot_mapping"],
            ),
            {
                "key": trace["key"], "value": trace["v_raw"],
                "key_cache": trace["key_cache"], "slot_mapping": trace["slot_mapping"],
            },
            notes="PyTorch index_copy_ baseline; nano-vLLM production path replaces this with Triton.",
        ),
        _case(
            "qk_matmul", runtime, phase, "matmul", EINSUM_BACKEND,
            lambda: torch.einsum("qhd,khd->hqk", q, k).float()
            * (block.attn.head_dim**-0.5),
            {"query": q, "key": k},
            notes="One sequence invocation; fallback loops over batch sequences.",
            estimated_flops=_attention_flops(q, k),
        ),
        _case(
            "causal_mask", runtime, phase, "elementwise", CUDA_BACKEND,
            lambda: trace["scores"].masked_fill(
                ~trace["mask"], torch.finfo(trace["scores"].dtype).min
            ),
            {"scores": trace["scores"], "mask": trace["mask"]},
        ),
        _case(
            "softmax", runtime, phase, "reduction", CUDA_BACKEND,
            lambda: F.softmax(trace["masked_scores"], dim=-1).to(q.dtype),
            {"scores": trace["masked_scores"]},
            notes="nano-vLLM's PyTorch fallback promotes attention scores to FP32.",
        ),
        _case(
            "attention_value_matmul", runtime, phase, "matmul", EINSUM_BACKEND,
            lambda: torch.einsum("hqk,khd->qhd", trace["probabilities"], v),
            {"probabilities": trace["probabilities"], "value": v},
            estimated_flops=_attention_flops(q, k),
        ),
        _case(
            "attention_output_concat", runtime, phase, "layout", CUDA_BACKEND,
            lambda: torch.cat(trace["attention_outputs"], dim=0),
            {
                f"sequence_{index}": output
                for index, output in enumerate(trace["attention_outputs"])
            },
            notes="nano-vLLM's PyTorch fallback concatenates one attention result per sequence.",
        ),
        _case(
            "head_merge", runtime, phase, "layout", CUDA_BACKEND,
            lambda: trace["attention_values"].flatten(1, -1),
            {"attention_values": trace["attention_values"]},
        ),
        _case(
            "output_projection", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["merged_heads"], block.attn.c_proj.weight, block.attn.c_proj.bias),
            {"hidden_states": trace["merged_heads"], "weight": block.attn.c_proj.weight},
            estimated_flops=_linear_flops(trace["merged_heads"], block.attn.c_proj.weight),
        ),
        _case(
            "mlp_norm", runtime, phase, "normalization", CUDA_BACKEND,
            lambda: block.ln_2(trace["attention_residual"]),
            {"hidden_states": trace["attention_residual"]},
        ),
        _case(
            "mlp_fc1", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["mlp_norm"], block.mlp.net[0].weight, block.mlp.net[0].bias),
            {"hidden_states": trace["mlp_norm"], "weight": block.mlp.net[0].weight},
            estimated_flops=_linear_flops(trace["mlp_norm"], block.mlp.net[0].weight),
        ),
        _case(
            "gelu", runtime, phase, "activation", CUDA_BACKEND,
            lambda: F.gelu(trace["mlp_fc1"]),
            {"hidden_states": trace["mlp_fc1"]},
        ),
        _case(
            "mlp_fc2", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["mlp_activation"], block.mlp.net[2].weight, block.mlp.net[2].bias),
            {"hidden_states": trace["mlp_activation"], "weight": block.mlp.net[2].weight},
            estimated_flops=_linear_flops(trace["mlp_activation"], block.mlp.net[2].weight),
        ),
        _case(
            "final_norm", runtime, phase, "normalization", CUDA_BACKEND,
            lambda: model.ln_f(trace["pre_final_norm"]),
            {"hidden_states": trace["pre_final_norm"]},
        ),
        _case(
            "lm_head", runtime, phase, "linear", LINEAR_BACKEND,
            lambda: F.linear(trace["head_input"], model.lm_head.weight),
            {"hidden_states": trace["head_input"], "weight": model.lm_head.weight},
            estimated_flops=_linear_flops(trace["head_input"], model.lm_head.weight),
            notes="Prefill projects only each sequence's last token, matching nano-vLLM serving.",
        ),
    ]
    if phase == "decode":
        cases.insert(
            5,
            _case(
                "nano_paged_kv_gather", runtime, phase, "kv_cache", CUDA_BACKEND,
                lambda: _paged_gather_pair(
                    trace["key_cache"], trace["value_cache"], trace["block_table"],
                    context_length + 1,
                ),
                {
                    "key_cache": trace["key_cache"], "value_cache": trace["value_cache"],
                    "block_table": trace["block_table"],
                },
                notes="One sequence's K/V pair; fallback executes this per layer and sequence.",
            ),
        )
        cases.append(
            _case(
                "full_decode_step", runtime, phase, "full_model", "PyTorch fallback mixed",
                lambda: executor.decode(next_ids, context_length),
                {"input_ids": next_ids, "key_cache": trace["key_cache"]},
            )
        )
    else:
        cases.append(
            _case(
                "full_prefill", runtime, phase, "full_model", "PyTorch fallback mixed",
                lambda: executor.prefill(input_ids_2d),
                {"input_ids": input_ids_2d.reshape(-1), "key_cache": trace["key_cache"]},
            )
        )
    return cases


def _canonical_layouts(
    workload_id: str,
    runtime: str,
    phase: str,
    trace: dict[str, Any],
) -> list[dict[str, Any]]:
    names = (
        "input_ids",
        "hidden",
        "qkv",
        "q_raw",
        "k_raw",
        "v_raw",
        "query",
        "key",
        "scores",
        "probabilities",
        "key_cache",
        "slot_mapping",
        "block_table",
        "head_input",
        "logits",
    )
    layouts = []
    for name in names:
        tensor = trace.get(name)
        if isinstance(tensor, torch.Tensor):
            layouts.append(
                {
                    "workload_id": workload_id,
                    "runtime": runtime,
                    "phase": phase,
                    "name": name,
                    **tensor_metadata(tensor),
                }
            )
    return layouts


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_summary(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn": torch.backends.cudnn.version(),
        "pid": os.getpid(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        environment["gpu"] = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        }
        try:
            environment["preferred_blas_library"] = str(
                torch.backends.cuda.preferred_blas_library()
            )
        except (AttributeError, RuntimeError):
            environment["preferred_blas_library"] = "cublas (PyTorch default)"
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,pstate,memory.total",
                    "--format=csv,noheader,nounits",
                    f"--id={index}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            driver, pstate, memory_mib = (
                part.strip() for part in completed.stdout.strip().split(",")
            )
            environment["nvidia_driver"] = driver
            environment["initial_gpu_pstate"] = pstate
            environment["nvidia_smi_memory_mib"] = int(memory_mib)
        except (OSError, ValueError, subprocess.CalledProcessError):
            environment["nvidia_driver"] = None
    return environment


def _load_model_and_tokenizer(checkpoint_path: Path, device: torch.device, dtype: torch.dtype):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device=device, dtype=dtype)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    return model, tokenizer, checkpoint


def _validation_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    absolute = (reference_float - candidate_float).abs()
    denominator = reference_float.abs().clamp_min(1e-6)
    return {
        "max_absolute_error": absolute.max().item(),
        "mean_absolute_error": absolute.mean().item(),
        "max_relative_error": (absolute / denominator).max().item(),
        "argmax_match_fraction": (
            reference_float.argmax(dim=-1) == candidate_float.argmax(dim=-1)
        ).float().mean().item(),
    }


def run_inference_baseline(
    checkpoint_path: str | Path,
    prompt: str,
    *,
    batch_sizes: Iterable[int] = (1, 8),
    dtypes: Iterable[torch.dtype] = (torch.float32, torch.float16),
    generated_tokens: int = 16,
    page_size: int = 256,
    timer_settings: TimerSettings = TimerSettings(),
    device: str | torch.device = "cuda",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the timed inference baseline requires a CUDA device")
    checkpoint_path = Path(checkpoint_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else checkpoint_path.parents[4]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    all_results: list[dict[str, Any]] = []
    all_layouts: list[dict[str, Any]] = []
    workloads: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    resolved_config: dict[str, Any] | None = None
    prompt_ids: list[int] | None = None

    for dtype in dtypes:
        model, tokenizer, checkpoint = _load_model_and_tokenizer(checkpoint_path, device, dtype)
        resolved_config = asdict(model.config)
        encoded = tokenizer.encode(prompt)
        if not encoded:
            raise ValueError("the prompt encodes to zero tokens")
        if len(encoded) + 1 > model.config.block_size:
            raise ValueError(
                f"prompt has {len(encoded)} tokens; one decode step exceeds block_size={model.config.block_size}"
            )
        prompt_ids = encoded

        for batch_size in batch_sizes:
            input_ids = torch.tensor(encoded, device=device, dtype=torch.long).repeat(batch_size, 1)
            context_length = input_ids.shape[1]
            workload_id = f"b{batch_size}_t{context_length}_{_dtype_name(dtype)}"
            workloads.append(
                {
                    "workload_id": workload_id,
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "prefill_minillm_input_shape": [batch_size, context_length, model.config.n_embd],
                    "prefill_nano_input_shape": [batch_size * context_length, model.config.n_embd],
                    "decode_minillm_input_shape": [batch_size, 1, model.config.n_embd],
                    "decode_nano_input_shape": [batch_size, model.config.n_embd],
                    "dtype": _dtype_name(dtype),
                    "generated_tokens": generated_tokens,
                    "decode_model_passes": max(generated_tokens - 1, 0),
                }
            )

            with torch.inference_mode():
                native_prefill_logits, past_key_values = model.forward_with_cache(input_ids)
                next_ids_2d = native_prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                next_ids = next_ids_2d[:, 0]
                native_decode_logits, _ = model.forward_with_cache(next_ids_2d, past_key_values)
                native_prefill_trace = _native_layer_trace(model, input_ids, None)
                native_decode_trace = _native_layer_trace(model, next_ids_2d, past_key_values)

                nano = NanoVLLMTorchBaseline(model, batch_size, context_length, page_size)
                nano_prefill_logits, nano_prefill_trace = nano.prefill(input_ids, trace=True)
                nano_decode_logits, nano_decode_trace = nano.decode(
                    next_ids, context_length, trace=True
                )

                validations.extend(
                    [
                        {
                            "workload_id": workload_id,
                            "phase": "prefill_last_token_logits",
                            **_validation_metrics(
                                native_prefill_logits[:, -1, :], nano_prefill_logits
                            ),
                        },
                        {
                            "workload_id": workload_id,
                            "phase": "decode_logits",
                            **_validation_metrics(
                                native_decode_logits[:, 0, :], nano_decode_logits
                            ),
                        },
                    ]
                )

                all_layouts.extend(
                    _canonical_layouts(workload_id, "minillm", "prefill", native_prefill_trace)
                )
                all_layouts.extend(
                    _canonical_layouts(workload_id, "minillm", "decode", native_decode_trace)
                )
                all_layouts.extend(
                    _canonical_layouts(
                        workload_id, "nano_vllm_torch", "prefill", nano_prefill_trace
                    )
                )
                all_layouts.extend(
                    _canonical_layouts(
                        workload_id, "nano_vllm_torch", "decode", nano_decode_trace
                    )
                )

                cases = []
                cases.extend(_native_cases(model, native_prefill_trace, "prefill", None))
                cases.extend(
                    _native_cases(model, native_decode_trace, "decode", past_key_values)
                )
                cases.extend(
                    _nano_cases(
                        model, nano, nano_prefill_trace, "prefill", input_ids, next_ids,
                        context_length,
                    )
                )
                cases.extend(
                    _nano_cases(
                        model, nano, nano_decode_trace, "decode", input_ids, next_ids,
                        context_length,
                    )
                )

                for case in cases:
                    output = case.operation()
                    timing = benchmark_cuda_operation(case.operation, timer_settings)
                    profiler = (
                        profile_cuda_operation(case.operation)
                        if case.category in {"linear", "matmul", "layout", "kv_cache"}
                        else None
                    )
                    frequency = stage_frequency(
                        case.name,
                        case.runtime,
                        case.phase,
                        num_layers=model.config.n_layer,
                        batch_size=batch_size,
                        generated_tokens=generated_tokens,
                    )
                    calls_per_pass = frequency["logical_stage_calls_per_pass"]
                    median_gpu_ms = timing["gpu_ms"]["median"]
                    result = {
                        "workload_id": workload_id,
                        "runtime": case.runtime,
                        "phase": case.phase,
                        "stage": case.name,
                        "category": case.category,
                        "backend": case.backend,
                        "notes": case.notes,
                        "estimated_flops_per_stage_call": case.estimated_flops,
                        "inputs": {
                            name: tensor_metadata(tensor) for name, tensor in case.inputs.items()
                        },
                        "outputs": _named_tensor_metadata(output),
                        "frequency": frequency,
                        "timing": timing,
                        "profiler": profiler,
                        "estimated_gpu_ms_per_model_pass": median_gpu_ms * calls_per_pass,
                    }
                    if case.category == "full_model":
                        tokens = (
                            batch_size * context_length if case.phase == "prefill" else batch_size
                        )
                        result["throughput_tokens_per_second"] = (
                            tokens / (median_gpu_ms / 1000) if median_gpu_ms else None
                        )
                    all_results.append(result)

            del nano
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    assert resolved_config is not None and prompt_ids is not None
    return {
        "schema_version": 1,
        "benchmark": "MiniLLM/nano-vLLM PyTorch-cuBLAS inference shape baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "timer": "CUDA Events plus synchronized host wall clock",
            "warmup": timer_settings.warmup,
            "samples": timer_settings.samples,
            "target_sample_ms": timer_settings.target_sample_ms,
            "max_inner_loops": timer_settings.max_inner_loops,
            "excluded": ["FlashAttention", "Triton attention", "CUDA Graph", "torch.compile"],
            "linear_backend_contract": LINEAR_BACKEND,
            "matmul_backend_contract": MATMUL_BACKEND,
            "einsum_backend_contract": EINSUM_BACKEND,
            "profiler": (
                "one post-timing PyTorch profiler invocation for linear, matmul, layout, and KV-cache stages"
            ),
        },
        "environment": environment_summary(device),
        "git": _git_info(root),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "training_args": checkpoint.get("args", {}),
        },
        "model_config": resolved_config,
        "prompt": {"text": prompt, "token_ids": prompt_ids, "token_count": len(prompt_ids)},
        "page_size": page_size,
        "workloads": workloads,
        "validations": validations,
        "layouts": all_layouts,
        "results": all_results,
    }


def _shape_text(metadata_by_name: dict[str, dict[str, Any]]) -> str:
    return "; ".join(
        f"{name}={metadata['shape']}" for name, metadata in metadata_by_name.items()
    )


def _metadata_text(
    metadata_by_name: dict[str, dict[str, Any]],
    field: str,
) -> str:
    return "; ".join(
        f"{name}={metadata[field]}" for name, metadata in metadata_by_name.items()
    )


def write_baseline_outputs(
    payload: dict[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    markdown_path = Path(markdown_path)
    for path in (json_path, csv_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_fields = [
        "workload_id", "runtime", "phase", "stage", "category", "backend",
        "input_shapes", "input_dtypes", "input_strides", "input_pointer_alignment_bytes",
        "input_all_row_starts_alignment_bytes", "output_shapes", "gpu_median_ms", "gpu_p90_ms",
        "wall_median_ms", "calls_per_pass", "primary_ops_per_pass",
        "calls_for_generation",
        "profiler_clone_count", "profiler_contiguous_count", "profiler_copy_count",
        "profiler_bmm_count", "implicit_materialization",
        "estimated_gpu_ms_per_model_pass", "estimated_flops_per_stage_call",
        "throughput_tokens_per_second", "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for result in payload["results"]:
            writer.writerow(
                {
                    "workload_id": result["workload_id"],
                    "runtime": result["runtime"],
                    "phase": result["phase"],
                    "stage": result["stage"],
                    "category": result["category"],
                    "backend": result["backend"],
                    "input_shapes": _shape_text(result["inputs"]),
                    "input_dtypes": _metadata_text(result["inputs"], "dtype"),
                    "input_strides": _metadata_text(result["inputs"], "stride"),
                    "input_pointer_alignment_bytes": _metadata_text(
                        result["inputs"], "pointer_alignment_bytes"
                    ),
                    "input_all_row_starts_alignment_bytes": _metadata_text(
                        result["inputs"], "all_row_starts_alignment_bytes"
                    ),
                    "output_shapes": _shape_text(result["outputs"]),
                    "gpu_median_ms": result["timing"]["gpu_ms"]["median"],
                    "gpu_p90_ms": result["timing"]["gpu_ms"]["p90"],
                    "wall_median_ms": result["timing"]["synchronized_wall_ms"]["median"],
                    "calls_per_pass": result["frequency"]["logical_stage_calls_per_pass"],
                    "primary_ops_per_pass": result["frequency"]["primary_ops_per_pass"],
                    "calls_for_generation": result["frequency"][
                        "logical_stage_calls_for_generation"
                    ],
                    "profiler_clone_count": (
                        result["profiler"]["clone_count"] if result["profiler"] else None
                    ),
                    "profiler_contiguous_count": (
                        result["profiler"]["contiguous_count"] if result["profiler"] else None
                    ),
                    "profiler_copy_count": (
                        result["profiler"]["copy_count"] if result["profiler"] else None
                    ),
                    "profiler_bmm_count": (
                        result["profiler"]["bmm_count"] if result["profiler"] else None
                    ),
                    "implicit_materialization": (
                        result["profiler"]["implicit_materialization"]
                        if result["profiler"] else None
                    ),
                    "estimated_gpu_ms_per_model_pass": result[
                        "estimated_gpu_ms_per_model_pass"
                    ],
                    "estimated_flops_per_stage_call": result[
                        "estimated_flops_per_stage_call"
                    ],
                    "throughput_tokens_per_second": result.get(
                        "throughput_tokens_per_second"
                    ),
                    "notes": result["notes"],
                }
            )

    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")


def _render_markdown(payload: dict[str, Any]) -> str:
    environment = payload["environment"]
    config = payload["model_config"]
    num_key_value_heads = config.get("num_key_value_heads") or config["n_head"]
    lines = [
        "# MiniLLM / nano-vLLM PyTorch-cuBLAS inference baseline",
        "",
        f"Generated: `{payload['timestamp']}`",
        "",
        "## Scope and method",
        "",
        "This is an eager PyTorch baseline on the real MiniLLM checkpoint and the tensor layouts used by MiniLLM and nano-vLLM. Linear layers use `F.linear` (cuBLAS/cuBLASLt); MiniLLM QK/PV use `matmul`, while the nano-vLLM fallback uses its real `einsum` equations; both lower to batched GEMM/cuBLAS for these shapes. FlashAttention, Triton attention, CUDA Graph, and `torch.compile` are intentionally excluded so later optimized kernels have a stable reference.",
        "",
        f"Timing uses {payload['method']['warmup']} warmups and {payload['method']['samples']} CUDA Event samples. The adaptive inner loop targets {payload['method']['target_sample_ms']} ms per sample. Synchronized wall time includes Python dispatch and launch/synchronization overhead; GPU Event time covers work on the CUDA stream.",
        "",
        "For `N` generated tokens, prefill produces token 1 and the model executes `N-1` decode passes. Component contribution is `median time for one measured stage invocation x logical calls per pass`.",
        "",
        "## Environment",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| GPU | {environment.get('gpu', {}).get('name', 'n/a')} |",
        f"| Compute capability | {environment.get('gpu', {}).get('compute_capability', 'n/a')} |",
        f"| PyTorch / CUDA | {environment['torch']} / {environment['cuda_runtime']} |",
        f"| NVIDIA driver | {environment.get('nvidia_driver', 'n/a')} |",
        f"| BLAS | {environment.get('preferred_blas_library', 'n/a')} |",
        f"| TF32 | matmul={environment['allow_tf32_matmul']}, cuDNN={environment['allow_tf32_cudnn']} |",
        f"| Checkpoint SHA-256 | `{payload['checkpoint']['sha256']}` |",
        f"| Git | `{payload['git']['commit']}`; dirty={payload['git']['dirty']} |",
        "",
        "## Model and workloads",
        "",
        f"Model: `L={config['n_layer']}, C={config['n_embd']}, H={config['n_head']}, Hkv={num_key_value_heads}, D={config['n_embd'] // config['n_head']}, V={config['vocab_size']}, block_size={config['block_size']}`. Prompt token count: `{payload['prompt']['token_count']}`. nano-vLLM KV page size: `{payload['page_size']}`.",
        f"The decode row measures the first incremental pass: one new token is processed while attention reads the prompt plus that token. The call count for a {payload['workloads'][0]['generated_tokens']}-token generation is exact, but later decode passes have progressively longer KV lengths and are not assigned the first-step latency.",
        "",
        "| Workload | MiniLLM prefill | nano prefill | MiniLLM decode | nano decode | dtype |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for workload in payload["workloads"]:
        lines.append(
            f"| {workload['workload_id']} | `{workload['prefill_minillm_input_shape']}` | "
            f"`{workload['prefill_nano_input_shape']}` | `{workload['decode_minillm_input_shape']}` | "
            f"`{workload['decode_nano_input_shape']}` | {workload['dtype']} |"
        )

    lines.extend(["", "## Full-pass latency", "", "| Workload | Runtime | Phase | GPU median (ms) | p90 (ms) | Wall median (ms) | Throughput token/s |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]) 
    full_results = [result for result in payload["results"] if result["category"] == "full_model"]
    for result in full_results:
        lines.append(
            f"| {result['workload_id']} | {result['runtime']} | {result['phase']} | "
            f"{result['timing']['gpu_ms']['median']:.6f} | {result['timing']['gpu_ms']['p90']:.6f} | "
            f"{result['timing']['synchronized_wall_ms']['median']:.6f} | "
            f"{result.get('throughput_tokens_per_second', 0):.1f} |"
        )

    full_lookup = {
        (result["workload_id"], result["runtime"], result["phase"]): result
        for result in full_results
    }
    workload_lookup = {
        (workload["batch_size"], workload["dtype"]): workload["workload_id"]
        for workload in payload["workloads"]
    }
    scaling_keys = {
        (batch_size, dtype_name)
        for batch_size in (1, 8)
        for dtype_name in ("float32", "float16")
    }
    if scaling_keys.issubset(workload_lookup):
        lines.extend(
            [
                "",
                "### Scaling observations",
                "",
                "`B=1 -> B=8 throughput scaling` is `8 x latency(B=1) / latency(B=8)`. An ideal eightfold throughput increase is 8.0x. `FP16 / FP32 latency` below 1.0 means FP16 is faster.",
                "",
                "| Runtime | Phase | FP32 B1->B8 throughput | FP16 B1->B8 throughput | B1 FP16/FP32 latency | B8 FP16/FP32 latency |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for runtime in ("minillm", "nano_vllm_torch"):
            for phase in ("prefill", "decode"):
                fp32_b1 = full_lookup[
                    (workload_lookup[(1, "float32")], runtime, phase)
                ]["timing"]["gpu_ms"]["median"]
                fp32_b8 = full_lookup[
                    (workload_lookup[(8, "float32")], runtime, phase)
                ]["timing"]["gpu_ms"]["median"]
                fp16_b1 = full_lookup[
                    (workload_lookup[(1, "float16")], runtime, phase)
                ]["timing"]["gpu_ms"]["median"]
                fp16_b8 = full_lookup[
                    (workload_lookup[(8, "float16")], runtime, phase)
                ]["timing"]["gpu_ms"]["median"]
                lines.append(
                    f"| {runtime} | {phase} | {8 * fp32_b1 / fp32_b8:.3f}x | "
                    f"{8 * fp16_b1 / fp16_b8:.3f}x | {fp16_b1 / fp32_b1:.3f}x | "
                    f"{fp16_b8 / fp32_b8:.3f}x |"
                )

    lines.extend(["", "## Component hotspots", ""])
    grouping_keys = []
    for result in payload["results"]:
        key = (result["workload_id"], result["runtime"], result["phase"])
        if key not in grouping_keys:
            grouping_keys.append(key)
    lines.extend(
        [
            "The ranking multiplies isolated median stage latency by the logical calls in one model pass. It identifies optimization candidates; it is not a claim that these isolated rows sum exactly to full-model latency.",
            "",
            "| Workload | Runtime/phase | Top 1 | Top 2 | Top 3 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for workload_id, runtime, phase in grouping_keys:
        ranked = [
            result for result in payload["results"]
            if result["workload_id"] == workload_id
            and result["runtime"] == runtime
            and result["phase"] == phase
            and result["category"] != "full_model"
        ]
        ranked.sort(key=lambda item: item["estimated_gpu_ms_per_model_pass"], reverse=True)
        top = [
            f"{result['stage']} ({result['estimated_gpu_ms_per_model_pass'] * 1000:.1f} us/pass)"
            for result in ranked[:3]
        ]
        lines.append(
            f"| {workload_id} | {runtime}/{phase} | {top[0]} | {top[1]} | {top[2]} |"
        )
    lines.append("")
    for workload_id, runtime, phase in grouping_keys:
        components = [
            result for result in payload["results"]
            if result["workload_id"] == workload_id
            and result["runtime"] == runtime
            and result["phase"] == phase
            and result["category"] != "full_model"
        ]
        components.sort(key=lambda item: item["estimated_gpu_ms_per_model_pass"], reverse=True)
        lines.extend(
            [
                f"### {workload_id}: {runtime} {phase}",
                "",
                "| Stage | GPU median/call (us) | p90 (us) | Calls/pass | Calls/generation | Estimated/pass (us) | Input shape(s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for result in components:
            lines.append(
                f"| {result['stage']} | {result['timing']['gpu_ms']['median'] * 1000:.3f} | "
                f"{result['timing']['gpu_ms']['p90'] * 1000:.3f} | "
                f"{result['frequency']['logical_stage_calls_per_pass']} | "
                f"{result['frequency']['logical_stage_calls_for_generation']} | "
                f"{result['estimated_gpu_ms_per_model_pass'] * 1000:.3f} | "
                f"`{_shape_text(result['inputs'])}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Canonical tensor layouts",
            "",
            "The pointer alignment is the maximum power-of-two alignment observed for that concrete allocation. `all-row alignment` additionally requires every active outer stride to preserve that alignment, so it is the guarantee for all last-dimension row starts. Both values are diagnostic for this run, not permanent allocator guarantees.",
            "",
            "| Workload | Runtime/phase | Tensor | Shape | Stride | dtype | Contiguous | Ptr / all-row alignment | mod 128 / 256 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for layout in payload["layouts"]:
        if layout["name"] not in {
            "input_ids", "hidden", "q_raw", "k_raw", "v_raw", "query", "key",
            "scores", "key_cache", "slot_mapping", "head_input", "logits",
        }:
            continue
        pointer_mod = layout["pointer_mod_bytes"]
        lines.append(
            f"| {layout['workload_id']} | {layout['runtime']}/{layout['phase']} | {layout['name']} | "
            f"`{layout['shape']}` | `{layout['stride']}` | {layout['dtype']} | "
            f"{layout['is_contiguous']} | {layout['pointer_alignment_bytes']} / "
            f"{layout['all_row_starts_alignment_bytes']} B | "
            f"{pointer_mod['128']} / {pointer_mod['256']} |"
        )

    lines.extend(
        [
            "",
            "## Numerical layout parity",
            "",
            "nano-vLLM serving returns only the last prefill logit per sequence, while MiniLLM returns all positions. These checks compare equivalent last-token/decode logits.",
            "",
            "| Workload | Phase | Max abs error | Mean abs error | Argmax match |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for validation in payload["validations"]:
        lines.append(
            f"| {validation['workload_id']} | {validation['phase']} | "
            f"{validation['max_absolute_error']:.6g} | {validation['mean_absolute_error']:.6g} | "
            f"{validation['argmax_match_fraction']:.3f} |"
        )

    materializations = [
        result for result in payload["results"]
        if result.get("profiler") and result["profiler"]["implicit_materialization"]
    ]
    lines.extend(
        [
            "",
            "## Profiler materialization audit",
            "",
            "Each linear, matmul, layout, and KV-cache stage gets one post-timing PyTorch profiler invocation. `clone` or `contiguous` means the eager operator materialized a new layout before or during the measured operation; profiler overhead is not part of CUDA Event timing.",
            "",
            "| Workload | Runtime/phase | Stage | clone | contiguous | copy | bmm | Evidence |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in materializations:
        profiler = result["profiler"]
        evidence = "; ".join(
            f"{operator['operator']} {operator['input_shapes']}"
            for operator in profiler["operators"]
            if operator["operator"] in {"aten::clone", "aten::contiguous"}
        )
        lines.append(
            f"| {result['workload_id']} | {result['runtime']}/{result['phase']} | "
            f"{result['stage']} | {profiler['clone_count']} | {profiler['contiguous_count']} | "
            f"{profiler['copy_count']} | {profiler['bmm_count']} | `{evidence}` |"
        )
    if not materializations:
        lines.append("| all | all | none observed | 0 | 0 | 0 | 0 | n/a |")

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- This checkpoint is intentionally tiny. Decode GEMMs have very small `M` and are commonly launch/latency bound rather than Tensor Core throughput bound.",
            "- nano-vLLM's PyTorch attention is a correctness fallback that loops over sequences and gathers paged KV. FlashAttention/Triton/CUDA Graph results must be measured separately, not inferred from this table.",
            "- Stage sums are attribution estimates. They do not include every residual/add/cast/allocation and therefore need not equal full-pass latency.",
            "- Decode timing is for the first incremental step at the reported context. Attention and cache costs increase as more tokens are appended.",
            "- `F.linear` and batched matmul dispatch through PyTorch's CUDA BLAS integration; exact cuBLAS versus cuBLASLt algorithm selection is internal and can change with shape, dtype, and PyTorch version.",
            "",
        ]
    )
    return "\n".join(lines)
