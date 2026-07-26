#!/usr/bin/env python3
"""Offline TorchAO INT8 weight-only smoke benchmark for local teaching models.

The script intentionally quantizes in memory and never overwrites the source
checkpoint.  It compares the original model with INT8 weight-only quantization
for every ``nn.Linear`` except the tied ``lm_head`` and emits one JSON object.

Examples:

    /home/undefined/Disk/python-envs/sglang/bin/python compress_models.py \
        --model minillm

    /home/undefined/Disk/python-envs/sglang/bin/python compress_models.py \
        --model qwen --iterations 20 --corpus-tokens 512 \
        --output results/qwen_int8.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parent
AI_ROOT = PROJECTS_DIR.parent
DEFAULT_MINILLM_CHECKPOINT = (
    PROJECTS_DIR / "minillm" / "artifacts" / "checkpoints" / "minillm-rope.pt"
)
DEFAULT_QWEN_PATH = AI_ROOT / ".model_cache" / "huggingface" / "Qwen3-0.6B"
DEFAULT_CORPUS = PROJECTS_DIR / "minillm" / "data" / "teaching_corpus.txt"
DEFAULT_PROMPT = "请用一句话解释模型量化。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a local MiniLLM or Qwen model before and after TorchAO "
            "INT8 weight-only quantization. All model access is offline."
        )
    )
    parser.add_argument("--model", required=True, choices=("minillm", "qwen"))
    parser.add_argument(
        "--minillm-checkpoint",
        type=Path,
        default=DEFAULT_MINILLM_CHECKPOINT,
    )
    parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--corpus-tokens",
        type=int,
        default=512,
        help="Maximum teaching-corpus tokens used for NLL/PPL (at least 2).",
    )
    parser.add_argument(
        "--eval-context-length",
        type=int,
        default=None,
        help=(
            "Context per PPL chunk. Defaults to MiniLLM block_size or 512 for Qwen."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="auto means float32 for MiniLLM and bfloat16 for Qwen.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file. The same JSON is always printed to stdout.",
    )
    args = parser.parse_args()

    if args.corpus_tokens < 2:
        parser.error("--corpus-tokens must be at least 2")
    if args.eval_context_length is not None and args.eval_context_length < 1:
        parser.error("--eval-context-length must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    return args


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_dtype(model_kind: str, requested: str) -> torch.dtype:
    if requested == "auto":
        return torch.float32 if model_kind == "minillm" else torch.bfloat16
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]


def current_cuda_allocated(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.memory_allocated(device))


def _flattened_payload_bytes(value: Any, seen: set[int]) -> int:
    """Count storage tensors nested inside a TorchAO wrapper tensor.

    AffineQuantizedTensor advertises the logical floating dtype and shape, so
    ``numel * element_size`` would incorrectly count it as an FP tensor. Its
    ``tensor_impl`` instead exposes the physical INT data, scales, and zero
    points through ``__tensor_flatten__``.
    """

    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    tensor_impl = getattr(value, "tensor_impl", None)
    if tensor_impl is not None:
        return _flattened_payload_bytes(tensor_impl, seen)

    flatten = getattr(value, "__tensor_flatten__", None)
    if flatten is not None and not isinstance(value, torch.Tensor):
        names, _metadata = flatten()
        return sum(
            _flattened_payload_bytes(getattr(value, name), seen)
            for name in names
            if getattr(value, name, None) is not None
        )

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return 0


def parameter_stats(model: torch.nn.Module) -> dict[str, int]:
    parameters = list(model.named_parameters(remove_duplicate=True))
    return {
        "unique_parameter_tensors": len(parameters),
        "logical_parameter_numel": int(sum(p.numel() for _, p in parameters)),
        "physical_parameter_payload_bytes": int(
            sum(_flattened_payload_bytes(p, set()) for _, p in parameters)
        ),
        "torchao_quantized_parameter_tensors": sum(
            type(p).__module__.startswith("torchao") for _, p in parameters
        ),
    }


def source_checkpoint_bytes(model_kind: str, source: Path) -> tuple[int, list[str]]:
    if model_kind == "minillm":
        return source.stat().st_size, [str(source)]

    files = sorted(source.glob("*.safetensors"))
    if not files:
        files = sorted(source.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"No model weight files found under {source}")
    return sum(path.stat().st_size for path in files), [str(path) for path in files]


def load_minillm(
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    project_root = PROJECTS_DIR / "minillm"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from minillm import GPTConfig, MiniGPT
    from minillm.tokenizer_registry import tokenizer_from_checkpoint

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device=device, dtype=dtype)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    metadata = {
        "architecture": type(model).__name__,
        "vocab_size": config.vocab_size,
        "max_context_length": config.block_size,
        "checkpoint_config": checkpoint["config"],
    }
    return model, tokenizer, metadata


def load_qwen(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Qwen mode requires transformers; use the registered sglang environment."
        ) from exc

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    metadata = {
        "architecture": type(model).__name__,
        "vocab_size": int(config.vocab_size),
        "max_context_length": int(config.max_position_embeddings),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "tie_word_embeddings": bool(config.tie_word_embeddings),
        "checkpoint_dtype": str(getattr(config, "torch_dtype", None)),
    }
    return model, tokenizer, metadata


def encode_text(tokenizer: Any, text: str, model_kind: str) -> list[int]:
    if model_kind == "minillm":
        return [int(token_id) for token_id in tokenizer.encode(text)]
    return [
        int(token_id)
        for token_id in tokenizer.encode(text, add_special_tokens=False)
    ]


def forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    model_kind: str,
) -> torch.Tensor:
    if model_kind == "minillm":
        output = model(input_ids)
        return output[0]
    return model(input_ids=input_ids, use_cache=True).logits


def forward_logits_without_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    model_kind: str,
) -> torch.Tensor:
    if model_kind == "minillm":
        return model(input_ids)[0]
    return model(input_ids=input_ids, use_cache=False).logits


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_forward(
    forward: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
    input_tokens: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            forward()
        synchronize(device)

        elapsed_ms: list[float] = []
        for _ in range(iterations):
            synchronize(device)
            started = time.perf_counter_ns()
            forward()
            synchronize(device)
            elapsed_ms.append((time.perf_counter_ns() - started) / 1e6)

    median_ms = statistics.median(elapsed_ms)
    return {
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "input_tokens": input_tokens,
        "latency_ms_min": min(elapsed_ms),
        "latency_ms_median": median_ms,
        "latency_ms_p95": percentile(elapsed_ms, 0.95),
        "prefill_input_tokens_per_second_at_median": input_tokens / (median_ms / 1e3),
    }


def teaching_corpus_metrics(
    model: torch.nn.Module,
    token_ids: list[int],
    *,
    model_kind: str,
    context_length: int,
    device: torch.device,
) -> dict[str, float | int | None]:
    total_nll = 0.0
    predicted_tokens = 0
    chunks = 0

    with torch.inference_mode():
        for start in range(0, len(token_ids) - 1, context_length):
            window = token_ids[start : start + context_length + 1]
            if len(window) < 2:
                continue
            input_ids = torch.tensor(
                [window[:-1]],
                dtype=torch.long,
                device=device,
            )
            targets = torch.tensor(
                window[1:],
                dtype=torch.long,
                device=device,
            )
            logits = forward_logits_without_cache(model, input_ids, model_kind)
            loss_sum = F.cross_entropy(
                logits[0].float(),
                targets,
                reduction="sum",
            )
            total_nll += float(loss_sum)
            predicted_tokens += targets.numel()
            chunks += 1

    if predicted_tokens == 0:
        raise ValueError("Teaching corpus produced no next-token targets")
    mean_nll = total_nll / predicted_tokens
    perplexity = math.exp(mean_nll) if mean_nll < 700 else None
    return {
        "source_tokens": len(token_ids),
        "predicted_tokens": predicted_tokens,
        "context_length_per_chunk": context_length,
        "chunks": chunks,
        "mean_next_token_nll": mean_nll,
        "perplexity": perplexity,
    }


def logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.float().cpu()
    candidate = candidate.float().cpu()
    absolute = (reference - candidate).abs()
    reference_top1 = reference.argmax(dim=-1)
    candidate_top1 = candidate.argmax(dim=-1)
    return {
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "top1_match_fraction_all_prompt_positions": float(
            (reference_top1 == candidate_top1).float().mean()
        ),
        "next_token_top1_match": bool(
            reference_top1[0, -1] == candidate_top1[0, -1]
        ),
        "baseline_next_token_id": int(reference_top1[0, -1]),
        "quantized_next_token_id": int(candidate_top1[0, -1]),
    }


def quantization_targets(model: torch.nn.Module) -> dict[str, Any]:
    names: list[str] = []
    weight_numel = 0
    weight_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            names.append(name)
            weight_numel += module.weight.numel()
            weight_bytes += module.weight.numel() * module.weight.element_size()
    return {
        "linear_module_count": len(names),
        "logical_weight_numel": int(weight_numel),
        "baseline_logical_weight_bytes": int(weight_bytes),
        "excluded_modules": ["lm_head"],
        "module_names": names,
    }


def quantize_int8_weight_only(model: torch.nn.Module) -> None:
    try:
        from torchao.quantization import int8_weight_only, quantize_
    except ImportError as exc:
        raise RuntimeError(
            "TorchAO is not installed in this Python environment. "
            "Use /home/undefined/Disk/python-envs/sglang/bin/python."
        ) from exc

    quantize_(
        model,
        int8_weight_only(),
        filter_fn=lambda module, fqn: (
            isinstance(module, torch.nn.Linear) and fqn != "lm_head"
        ),
    )


def relative_change(after: float, before: float) -> float | None:
    if before == 0:
        return None
    return after / before - 1.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    dtype = resolve_dtype(args.model, args.dtype)

    source = (
        args.minillm_checkpoint.resolve()
        if args.model == "minillm"
        else args.qwen_path.resolve()
    )
    corpus_path = args.corpus.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)

    checkpoint_bytes, weight_files = source_checkpoint_bytes(args.model, source)
    if args.model == "minillm":
        model, tokenizer, model_metadata = load_minillm(source, device, dtype)
    else:
        model, tokenizer, model_metadata = load_qwen(source, device, dtype)

    prompt = args.prompt.replace("\\n", "\n")
    prompt_ids = encode_text(tokenizer, prompt, args.model)
    max_context_length = int(model_metadata["max_context_length"])
    if len(prompt_ids) > max_context_length:
        prompt_ids = prompt_ids[:max_context_length]
    if not prompt_ids:
        raise ValueError("Prompt encoded to zero tokens")
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    corpus_text = corpus_path.read_text(encoding="utf-8")
    corpus_ids = encode_text(tokenizer, corpus_text, args.model)[: args.corpus_tokens]
    eval_context_length = args.eval_context_length
    if eval_context_length is None:
        eval_context_length = (
            max_context_length if args.model == "minillm" else min(512, max_context_length)
        )
    eval_context_length = min(eval_context_length, max_context_length)

    forward = lambda: forward_logits(model, prompt_tensor, args.model)
    baseline_size = parameter_stats(model)
    baseline_cuda_bytes = current_cuda_allocated(device)
    baseline_speed = benchmark_forward(
        forward,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
        input_tokens=len(prompt_ids),
    )
    with torch.inference_mode():
        reference_logits = forward().detach().float().cpu()
    baseline_corpus = teaching_corpus_metrics(
        model,
        corpus_ids,
        model_kind=args.model,
        context_length=eval_context_length,
        device=device,
    )

    targets = quantization_targets(model)
    quantization_started = time.perf_counter()
    quantize_int8_weight_only(model)
    quantization_seconds = time.perf_counter() - quantization_started
    gc.collect()

    quantized_size = parameter_stats(model)
    quantized_cuda_bytes = current_cuda_allocated(device)
    quantized_speed = benchmark_forward(
        forward,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
        input_tokens=len(prompt_ids),
    )
    with torch.inference_mode():
        quantized_logits = forward().detach().float().cpu()
    quantized_corpus = teaching_corpus_metrics(
        model,
        corpus_ids,
        model_kind=args.model,
        context_length=eval_context_length,
        device=device,
    )

    baseline_payload = baseline_size["physical_parameter_payload_bytes"]
    quantized_payload = quantized_size["physical_parameter_payload_bytes"]
    baseline_latency = float(baseline_speed["latency_ms_median"])
    quantized_latency = float(quantized_speed["latency_ms_median"])
    baseline_ppl = baseline_corpus["perplexity"]
    quantized_ppl = quantized_corpus["perplexity"]

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "torchao_int8_weight_only",
        "model": args.model,
        "source": {
            "path": str(source),
            "weight_files": weight_files,
            "checkpoint_weight_file_bytes": checkpoint_bytes,
        },
        "teaching_corpus": str(corpus_path),
        "prompt": prompt,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchao": package_version("torchao"),
            "transformers": package_version("transformers"),
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "torch_cpu_threads": torch.get_num_threads(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "model_metadata": model_metadata,
        "quantization": {
            "method": "TorchAO Int8WeightOnlyConfig",
            "weight_bits": 8,
            "activation_quantized": False,
            "in_memory_only": True,
            "quantization_seconds": quantization_seconds,
            "targets": targets,
        },
        "baseline": {
            "model_size": baseline_size,
            "cuda_memory_allocated_bytes": baseline_cuda_bytes,
            "speed": baseline_speed,
            "teaching_corpus_quality": baseline_corpus,
        },
        "quantized": {
            "model_size": quantized_size,
            "cuda_memory_allocated_bytes": quantized_cuda_bytes,
            "speed": quantized_speed,
            "teaching_corpus_quality": quantized_corpus,
        },
        "comparison": {
            "logits": logit_metrics(reference_logits, quantized_logits),
            "physical_parameter_payload_reduction_fraction": (
                1.0 - quantized_payload / baseline_payload
            ),
            "median_latency_change_fraction": relative_change(
                quantized_latency, baseline_latency
            ),
            "median_latency_ratio_quantized_over_baseline": (
                quantized_latency / baseline_latency
            ),
            "teaching_corpus_mean_nll_delta": (
                float(quantized_corpus["mean_next_token_nll"])
                - float(baseline_corpus["mean_next_token_nll"])
            ),
            "teaching_corpus_perplexity_ratio": (
                quantized_ppl / baseline_ppl
                if isinstance(quantized_ppl, float)
                and isinstance(baseline_ppl, float)
                and baseline_ppl != 0
                else None
            ),
        },
        "notes": [
            "The source checkpoint is never modified; quantization exists only in this process.",
            "Checkpoint file bytes and loaded unique parameter payload differ when tied weights are duplicated in safetensors.",
            "A small smoke corpus and prompt are useful for regression checks, not a complete model-quality evaluation.",
            "INT8 can reduce parameter payload yet be slower when dequantization or launch overhead dominates.",
        ],
    }
    return result


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
        exit_code = 0
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "error",
            "model": args.model,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 1

    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
