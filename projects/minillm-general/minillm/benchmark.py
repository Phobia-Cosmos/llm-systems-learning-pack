from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .config import GPTConfig
from .model import MiniGPT
from .tokenizer import CharTokenizer


BENCHMARK_SCHEMA_VERSION = 2
SUPPORTED_BENCHMARK_SUITES = ("position", "norm", "mlp", "attention")


@dataclass(frozen=True)
class VariantSpec:
    suite: str
    name: str
    overrides: dict[str, object]

    @property
    def variant_id(self) -> str:
        return f"{self.suite}/{self.name}"


@dataclass(frozen=True)
class BenchmarkSettings:
    data_path: str = "data/tiny_corpus.txt"
    device: str = "cpu"
    max_steps: int = 100
    batch_size: int = 8
    block_size: int = 32
    eval_batches: int = 4
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 32
    learning_rate: float = 3e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    model_seeds: tuple[int, ...] = (1337, 1338, 1339)
    data_seed: int = 2026
    prompt: str = "LLM 是"
    max_new_tokens: int = 8
    generation_repeats: int = 10
    torch_threads: int = 1

    def validate(self) -> None:
        positive_ints = {
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "eval_batches": self.eval_batches,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "max_new_tokens": self.max_new_tokens,
            "generation_repeats": self.generation_repeats,
            "torch_threads": self.torch_threads,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.weight_decay:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not self.model_seeds:
            raise ValueError("model_seeds must contain at least one seed")
        if len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("model_seeds must not contain duplicates")


def _gqa_num_key_value_heads(n_head: int) -> int:
    """Pick a distinct, valid GQA head count, preferring two query heads per KV head."""

    proper_divisors = [
        candidate for candidate in range(n_head - 1, 1, -1) if n_head % candidate == 0
    ]
    if not proper_divisors:
        raise ValueError(
            "the attention benchmark requires n_head to have a divisor strictly between "
            f"1 and n_head so MHA, GQA, and MQA are distinct; got n_head={n_head}"
        )
    return proper_divisors[0]


def component_variant_specs(
    suites: Iterable[str] = SUPPORTED_BENCHMARK_SUITES,
    *,
    n_head: int = 4,
) -> list[VariantSpec]:
    requested = tuple(suites)
    unknown = sorted(set(requested) - set(SUPPORTED_BENCHMARK_SUITES))
    if unknown:
        raise ValueError(f"unsupported benchmark suites: {', '.join(unknown)}")

    specs: list[VariantSpec] = []
    if "position" in requested:
        for name in ("learned", "sinusoidal", "rope", "alibi", "none"):
            specs.append(
                VariantSpec(
                    suite="position",
                    name=name,
                    overrides={
                        "position_encoding": name,
                        "norm_type": "layernorm",
                        "mlp_type": "dense",
                        "activation": "gelu",
                    },
                )
            )
    if "norm" in requested:
        for name in ("layernorm", "rmsnorm", "scalenorm"):
            specs.append(
                VariantSpec(
                    suite="norm",
                    name=name,
                    overrides={
                        "position_encoding": "rope",
                        "norm_type": name,
                        "mlp_type": "dense",
                        "activation": "gelu",
                    },
                )
            )
    if "mlp" in requested:
        effective_activations = {
            "dense": "gelu",
            "swiglu": "silu",
            "geglu": "gelu",
            "reglu": "relu",
        }
        for name in ("dense", "swiglu", "geglu", "reglu"):
            specs.append(
                VariantSpec(
                    suite="mlp",
                    name=name,
                    overrides={
                        "position_encoding": "rope",
                        "norm_type": "rmsnorm",
                        "mlp_type": name,
                        "activation": effective_activations[name],
                    },
                )
            )
    if "attention" in requested:
        gqa_heads = _gqa_num_key_value_heads(n_head)
        for name, num_key_value_heads in (
            ("mha", n_head),
            ("gqa", gqa_heads),
            ("mqa", 1),
        ):
            specs.append(
                VariantSpec(
                    suite="attention",
                    name=name,
                    overrides={
                        "position_encoding": "rope",
                        "norm_type": "rmsnorm",
                        "mlp_type": "dense",
                        "activation": "gelu",
                        "num_key_value_heads": num_key_value_heads,
                    },
                )
            )
    return specs


def make_fixed_batches(
    data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if len(data) < block_size + 1:
        raise ValueError("dataset split is too small for the configured block_size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_batches):
        starts = torch.randint(
            0,
            len(data) - block_size,
            (batch_size,),
            generator=generator,
        )
        x = torch.stack([data[start : start + block_size] for start in starts])
        y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
        batches.append((x, y))
    return tuple(batches)


def make_sequential_eval_batches(
    data: torch.Tensor,
    *,
    block_size: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Cover each target token exactly once using contiguous, non-overlapping windows."""

    if len(data) < 2:
        raise ValueError("evaluation split must contain at least two tokens")
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(data) - 1, block_size):
        window_size = min(block_size, len(data) - 1 - start)
        x = data[start : start + window_size].unsqueeze(0)
        y = data[start + 1 : start + window_size + 1].unsqueeze(0)
        batches.append((x, y))
    return tuple(batches)


def _build_config(settings: BenchmarkSettings, vocab_size: int, spec: VariantSpec) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": vocab_size,
        "block_size": settings.block_size,
        "n_layer": settings.n_layer,
        "n_head": settings.n_head,
        "n_embd": settings.n_embd,
        "dropout": 0.0,
        "bias": True,
    }
    values.update(spec.overrides)
    return GPTConfig(**values)


def _build_model(
    settings: BenchmarkSettings,
    vocab_size: int,
    spec: VariantSpec,
    *,
    model_seed: int,
) -> MiniGPT:
    torch.manual_seed(model_seed)
    model = MiniGPT(_build_config(settings, vocab_size, spec))
    # MiniGPT's normal initialization uses the global RNG in module traversal
    # order. A learned position table or gated MLP would otherwise shift every
    # later common tensor. Reinitialize random matrices from a stable seed per
    # semantic parameter name; deterministic norm scales/biases retain their
    # architecture defaults. Common names and shapes are now bit-identical
    # even when a variant adds or removes unrelated parameters.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim < 2:
                continue
            digest = hashlib.sha256(f"{model_seed}:{name}".encode("utf-8")).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.02
            parameter.copy_(values.to(dtype=parameter.dtype, device=parameter.device))
    return model


def common_parameter_names(left: MiniGPT, right: MiniGPT) -> tuple[str, ...]:
    left_parameters = dict(left.named_parameters())
    right_parameters = dict(right.named_parameters())
    return tuple(
        name
        for name in sorted(left_parameters.keys() & right_parameters.keys())
        if left_parameters[name].shape == right_parameters[name].shape
    )


@torch.no_grad()
def _fixed_loss(
    model: MiniGPT,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for x_cpu, y_cpu in batches:
        _, loss = model(x_cpu.to(device), y_cpu.to(device))
        if loss is None:
            raise RuntimeError("benchmark evaluation expected a loss")
        token_count = y_cpu.numel()
        total_nll += float(loss.item()) * token_count
        total_tokens += token_count
    return total_nll / total_tokens


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def _time_generation(
    model: MiniGPT,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    repeats: int,
    use_kv_cache: bool,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    generate = model.generate_with_kv_cache if use_kv_cache else model.generate
    # One unmeasured call avoids charging lazy kernel/setup work to one variant.
    output = generate(prompt_ids, max_new_tokens=max_new_tokens, greedy=True)
    _sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        output = generate(prompt_ids, max_new_tokens=max_new_tokens, greedy=True)
    _sync(device)
    elapsed = max(time.perf_counter() - started, 1e-12)
    return output, repeats * max_new_tokens / elapsed


def _perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        raise FloatingPointError("perplexity requires a finite loss")
    if loss >= math.log(sys.float_info.max):
        raise OverflowError("loss is too large to represent perplexity as a finite float")
    return math.exp(loss)


def run_variant(
    *,
    settings: BenchmarkSettings,
    spec: VariantSpec,
    tokenizer: CharTokenizer,
    train_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    train_eval_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    val_eval_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    model_seed: int,
) -> dict[str, object]:
    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")

    model = _build_model(
        settings,
        tokenizer.vocab_size,
        spec,
        model_seed=model_seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=(settings.adam_beta1, settings.adam_beta2),
        eps=settings.adam_eps,
        weight_decay=settings.weight_decay,
    )

    # Warm the same forward/backward/AdamW path used below, then restore the
    # exact initial parameters and zero optimizer moments. This keeps lazy
    # allocation/setup outside the timed region without changing training.
    initial_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    warm_x, warm_y = train_batches[0]
    model.train()
    _, warm_loss = model(warm_x.to(device), warm_y.to(device))
    if warm_loss is None:
        raise RuntimeError("benchmark warmup expected a loss")
    optimizer.zero_grad(set_to_none=True)
    warm_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    _sync(device)
    model.load_state_dict(initial_state)
    del initial_state
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    initial_train_loss = _fixed_loss(model, train_eval_batches, device)
    initial_val_loss = _fixed_loss(model, val_eval_batches, device)

    model.train()
    max_gradient_norm = 0.0
    clipped_steps = 0
    _sync(device)
    train_started = time.perf_counter()
    for x_cpu, y_cpu in train_batches:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("benchmark training expected a loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm for {spec.variant_id}")
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        clipped_steps += int(gradient_norm > 1.0)
        optimizer.step()
    _sync(device)
    train_seconds = max(time.perf_counter() - train_started, 1e-12)

    final_train_loss = _fixed_loss(model, train_eval_batches, device)
    final_val_loss = _fixed_loss(model, val_eval_batches, device)

    prompt_token_ids = tokenizer.encode(settings.prompt)
    if not prompt_token_ids:
        raise ValueError("benchmark prompt must encode to at least one token")
    if len(prompt_token_ids) + settings.max_new_tokens > settings.block_size:
        raise ValueError("prompt token count + max_new_tokens must not exceed block_size")
    prompt = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    model.eval()
    full_output, full_tokens_per_second = _time_generation(
        model,
        prompt,
        max_new_tokens=settings.max_new_tokens,
        repeats=settings.generation_repeats,
        use_kv_cache=False,
        device=device,
    )
    cached_output, cached_tokens_per_second = _time_generation(
        model,
        prompt,
        max_new_tokens=settings.max_new_tokens,
        repeats=settings.generation_repeats,
        use_kv_cache=True,
        device=device,
    )
    cache_matches_full = torch.equal(full_output, cached_output)
    if not cache_matches_full:
        raise AssertionError(f"KV-cache generation diverged for {spec.variant_id}")

    output_ids = full_output[0].detach().cpu().tolist()
    completion_ids = output_ids[len(prompt_token_ids) :]
    config = model.config
    parameter = next(model.parameters())
    dtype = str(parameter.dtype).removeprefix("torch.")
    head_dim = config.n_embd // config.n_head
    kv_cache_elements_per_token_per_layer = 2 * config.num_key_value_heads * head_dim
    kv_cache_bytes_per_token_per_layer = (
        kv_cache_elements_per_token_per_layer * parameter.element_size()
    )
    mha_kv_cache_elements = 2 * config.n_head * head_dim
    trained_tokens = len(train_batches) * settings.batch_size * settings.block_size
    peak_cuda_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    return {
        "suite": spec.suite,
        "variant": spec.name,
        "variant_id": spec.variant_id,
        "model_seed": model_seed,
        "position_encoding": config.position_encoding,
        "norm_type": config.norm_type,
        "mlp_type": config.mlp_type,
        "activation": config.activation,
        "intermediate_size": config.intermediate_size,
        "num_query_heads": config.n_head,
        "num_key_value_heads": config.num_key_value_heads,
        "parameter_count": model.parameter_count(),
        "resolved_config": asdict(config),
        "dtype": dtype,
        "kv_cache_elements_per_token_per_layer": kv_cache_elements_per_token_per_layer,
        "kv_cache_bytes_per_token_per_layer": kv_cache_bytes_per_token_per_layer,
        "kv_cache_compression_ratio_vs_mha": (
            mha_kv_cache_elements / kv_cache_elements_per_token_per_layer
        ),
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "final_val_perplexity": _perplexity(final_val_loss),
        "val_loss_change": final_val_loss - initial_val_loss,
        "trained_tokens": trained_tokens,
        "train_seconds": train_seconds,
        "train_tokens_per_second": trained_tokens / train_seconds,
        "max_gradient_norm": max_gradient_norm,
        "clipped_steps": clipped_steps,
        "full_generation_tokens_per_second": full_tokens_per_second,
        "kv_cache_generation_tokens_per_second": cached_tokens_per_second,
        "peak_cuda_memory_bytes": peak_cuda_memory,
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_ids,
        "completion_text": tokenizer.decode(completion_ids),
        "cache_matches_full": cache_matches_full,
    }


def _git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty(cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _benchmark_code_sha256() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "benchmark.py",
        "config.py",
        "model.py",
        "position.py",
        "rope.py",
        "norm.py",
        "mlp.py",
        "activations.py",
        "tokenizer.py",
    ):
        path = package_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_component_benchmark(
    settings: BenchmarkSettings,
    *,
    suites: Iterable[str] = SUPPORTED_BENCHMARK_SUITES,
    specs: Sequence[VariantSpec] | None = None,
) -> dict[str, object]:
    settings.validate()
    torch.set_num_threads(settings.torch_threads)
    data_path = Path(settings.data_path).resolve()
    text = data_path.read_text(encoding="utf-8")
    raw_split = max(1, int(len(text) * 0.9))
    train_text = text[:raw_split]
    val_text = text[raw_split:]
    tokenizer = CharTokenizer.from_text(train_text)
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    train_batches = make_fixed_batches(
        train_data,
        block_size=settings.block_size,
        batch_size=settings.batch_size,
        num_batches=settings.max_steps,
        seed=settings.data_seed,
    )
    train_eval_batches = make_fixed_batches(
        train_data,
        block_size=settings.block_size,
        batch_size=settings.batch_size,
        num_batches=settings.eval_batches,
        seed=settings.data_seed + 1,
    )
    val_eval_batches = make_sequential_eval_batches(
        val_data,
        block_size=settings.block_size,
    )

    selected_specs = (
        list(specs)
        if specs is not None
        else component_variant_specs(suites, n_head=settings.n_head)
    )
    if not selected_specs:
        raise ValueError("at least one benchmark variant is required")

    results: list[dict[str, object]] = []
    result_cache: dict[tuple[int, str], dict[str, object]] = {}
    spec_order = {spec.variant_id: index for index, spec in enumerate(selected_specs)}
    seed_order = {seed: index for index, seed in enumerate(settings.model_seeds)}
    for seed_index, model_seed in enumerate(settings.model_seeds):
        offset = seed_index % len(selected_specs)
        execution_specs = selected_specs[offset:] + selected_specs[:offset]
        for spec in execution_specs:
            config_key = json.dumps(
                asdict(_build_config(settings, tokenizer.vocab_size, spec)),
                sort_keys=True,
                separators=(",", ":"),
            )
            cache_key = (model_seed, config_key)
            cached = result_cache.get(cache_key)
            if cached is None:
                cached = run_variant(
                    settings=settings,
                    spec=spec,
                    tokenizer=tokenizer,
                    train_batches=train_batches,
                    train_eval_batches=train_eval_batches,
                    val_eval_batches=val_eval_batches,
                    model_seed=model_seed,
                )
                result_cache[cache_key] = cached
            row = dict(cached)
            row.update(suite=spec.suite, variant=spec.name, variant_id=spec.variant_id)
            results.append(row)
    results.sort(key=lambda row: (seed_order[int(row["model_seed"])], spec_order[str(row["variant_id"])]))

    data_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    tokenizer_sha256 = hashlib.sha256(
        json.dumps(tokenizer.itos, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    schedule_payload = [
        [[x.tolist(), y.tolist()] for x, y in batches]
        for batches in (train_batches, train_eval_batches, val_eval_batches)
    ]
    schedule_sha256 = hashlib.sha256(
        json.dumps(schedule_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    aggregates = aggregate_results(results, selected_specs)
    code_location = Path(__file__).resolve().parent
    commit = _git_commit(code_location)
    git_dirty = _git_dirty(code_location)
    code_sha256 = _benchmark_code_sha256()
    run_identity = {
        "settings": asdict(settings),
        "variants": [
            {
                "variant_id": spec.variant_id,
                "resolved_config": asdict(_build_config(settings, tokenizer.vocab_size, spec)),
            }
            for spec in selected_specs
        ],
        "data_sha256": data_sha256,
        "schedule_sha256": schedule_sha256,
        "git_commit": commit,
        "git_dirty": git_dirty,
        "code_sha256": code_sha256,
        "torch_version": torch.__version__,
        "device": settings.device,
        "dtype": "float32",
    }
    run_id = hashlib.sha256(
        json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": git_dirty,
        "code_sha256": code_sha256,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "settings": asdict(settings),
        "fairness_contract": {
            "tokenizer": "one CharTokenizer fitted only on the 90% training text",
            "data_split": "one deterministic 90/10 contiguous raw-text split",
            "batches": "identical pre-generated train/eval batches for every variant",
            "validation": "every validation target token evaluated exactly once in sequential windows",
            "initialization": "stable per-name parameter seeds make common name-and-shape parameters bit-identical",
            "training_budget": "same optimizer, learning rate, update count, and token count",
            "dropout": 0.0,
            "generation": "same prompt, greedy decoder, output length, and repeat count",
        },
        "data_sha256": data_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "schedule_sha256": schedule_sha256,
        "train_token_count": len(train_data),
        "val_token_count": len(val_data),
        "train_eval_target_count": sum(y.numel() for _, y in train_eval_batches),
        "val_eval_target_count": sum(y.numel() for _, y in val_eval_batches),
        "vocab_size": tokenizer.vocab_size,
        "device_info": _device_info(torch.device(settings.device)),
        "results": results,
        "aggregates": aggregates,
    }


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_results(
    rows: Sequence[dict[str, object]],
    specs: Sequence[VariantSpec],
) -> list[dict[str, object]]:
    preferred_baselines = {
        "position": "learned",
        "norm": "layernorm",
        "mlp": "dense",
        "attention": "mha",
    }
    available_by_suite: dict[str, list[str]] = {}
    for spec in specs:
        available_by_suite.setdefault(spec.suite, []).append(spec.name)
    baselines = {
        suite: preferred_baselines[suite]
        if preferred_baselines[suite] in names
        else names[0]
        for suite, names in available_by_suite.items()
    }
    aggregates: list[dict[str, object]] = []
    for spec in specs:
        variant_rows = [row for row in rows if row["variant_id"] == spec.variant_id]
        if not variant_rows:
            continue
        baseline_id = f"{spec.suite}/{baselines[spec.suite]}"
        baseline_by_seed = {
            int(row["model_seed"]): row for row in rows if row["variant_id"] == baseline_id
        }
        final_losses = [float(row["final_val_loss"]) for row in variant_rows]
        final_train_losses = [float(row["final_train_loss"]) for row in variant_rows]
        perplexities = [float(row["final_val_perplexity"]) for row in variant_rows]
        train_rates = [float(row["train_tokens_per_second"]) for row in variant_rows]
        cache_rates = [float(row["kv_cache_generation_tokens_per_second"]) for row in variant_rows]
        paired_deltas = [
            float(row["final_val_loss"])
            - float(baseline_by_seed[int(row["model_seed"])]["final_val_loss"])
            for row in variant_rows
        ]
        loss_mean, loss_std = _mean_std(final_losses)
        train_loss_mean, train_loss_std = _mean_std(final_train_losses)
        ppl_mean, ppl_std = _mean_std(perplexities)
        train_mean, train_std = _mean_std(train_rates)
        cache_mean, cache_std = _mean_std(cache_rates)
        delta_mean, delta_std = _mean_std(paired_deltas)
        aggregates.append(
            {
                "suite": spec.suite,
                "variant": spec.name,
                "variant_id": spec.variant_id,
                "seed_count": len(variant_rows),
                "parameter_count": int(variant_rows[0]["parameter_count"]),
                "intermediate_size": int(variant_rows[0]["intermediate_size"]),
                "num_query_heads": int(variant_rows[0]["num_query_heads"]),
                "num_key_value_heads": int(variant_rows[0]["num_key_value_heads"]),
                "kv_cache_elements_per_token_per_layer": int(
                    variant_rows[0]["kv_cache_elements_per_token_per_layer"]
                ),
                "kv_cache_bytes_per_token_per_layer": int(
                    variant_rows[0]["kv_cache_bytes_per_token_per_layer"]
                ),
                "kv_cache_compression_ratio_vs_mha": float(
                    variant_rows[0]["kv_cache_compression_ratio_vs_mha"]
                ),
                "resolved_config": variant_rows[0]["resolved_config"],
                "final_val_loss_mean": loss_mean,
                "final_val_loss_std": loss_std,
                "final_train_loss_mean": train_loss_mean,
                "final_train_loss_std": train_loss_std,
                "final_val_perplexity_mean": ppl_mean,
                "final_val_perplexity_std": ppl_std,
                "paired_val_loss_delta_vs_baseline_mean": delta_mean,
                "paired_val_loss_delta_vs_baseline_std": delta_std,
                "train_tokens_per_second_mean": train_mean,
                "train_tokens_per_second_std": train_std,
                "kv_cache_generation_tokens_per_second_mean": cache_mean,
                "kv_cache_generation_tokens_per_second_std": cache_std,
                "all_cache_matches_full": all(bool(row["cache_matches_full"]) for row in variant_rows),
                "example_model_seed": int(variant_rows[0]["model_seed"]),
                "example_completion_token_ids": variant_rows[0]["completion_token_ids"],
                "example_completion_text": variant_rows[0]["completion_text"],
            }
        )
    return aggregates


def _device_info(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "type": device.type,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["name"] = torch.cuda.get_device_name(device)
        info["capability"] = list(torch.cuda.get_device_capability(device))
    elif device.type == "mps":
        info["name"] = "Apple Metal Performance Shaders"
    else:
        info["name"] = platform.processor() or platform.machine()
    return info


CSV_FIELDS = (
    "suite",
    "variant",
    "variant_id",
    "model_seed",
    "position_encoding",
    "norm_type",
    "mlp_type",
    "activation",
    "intermediate_size",
    "num_query_heads",
    "num_key_value_heads",
    "parameter_count",
    "resolved_config",
    "dtype",
    "kv_cache_elements_per_token_per_layer",
    "kv_cache_bytes_per_token_per_layer",
    "kv_cache_compression_ratio_vs_mha",
    "initial_train_loss",
    "final_train_loss",
    "initial_val_loss",
    "final_val_loss",
    "final_val_perplexity",
    "val_loss_change",
    "trained_tokens",
    "train_seconds",
    "train_tokens_per_second",
    "max_gradient_norm",
    "clipped_steps",
    "full_generation_tokens_per_second",
    "kv_cache_generation_tokens_per_second",
    "peak_cuda_memory_bytes",
    "prompt_token_ids",
    "completion_token_ids",
    "completion_text",
    "cache_matches_full",
)


def write_benchmark_outputs(
    payload: dict[str, object],
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
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = payload["results"]
    if not isinstance(rows, list):
        raise TypeError("benchmark payload results must be a list")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["resolved_config"] = json.dumps(
                csv_row["resolved_config"], sort_keys=True, separators=(",", ":")
            )
            csv_row["prompt_token_ids"] = json.dumps(csv_row["prompt_token_ids"])
            csv_row["completion_token_ids"] = json.dumps(csv_row["completion_token_ids"])
            writer.writerow({field: csv_row.get(field) for field in CSV_FIELDS})

    settings = payload["settings"]
    if not isinstance(settings, dict):
        raise TypeError("benchmark payload settings must be a dict")
    lines = [
        "# MiniLLM component benchmark",
        "",
        "This is a controlled teaching benchmark, not a claim about large-model quality.",
        "",
        "## Fairness contract",
        "",
        f"- Data: `{settings['data_path']}`",
        f"- Updates per variant: {settings['max_steps']}",
        f"- Batch/sequence: {settings['batch_size']} × {settings['block_size']}",
        f"- Model: {settings['n_layer']} layer(s), {settings['n_head']} head(s), hidden size {settings['n_embd']}",
        f"- Seeds: model={settings['model_seeds']}, data={settings['data_seed']}",
        "- Every variant receives the same pre-generated batches; common semantic parameter names use identical initial values.",
        "- CPU peak Torch memory is reported as blank because PyTorch has no reliable allocator peak counter for CPU.",
        "",
    ]
    aggregate_rows = payload.get("aggregates", [])
    if not isinstance(aggregate_rows, list):
        raise TypeError("benchmark payload aggregates must be a list")
    for suite in SUPPORTED_BENCHMARK_SUITES:
        suite_rows = [row for row in aggregate_rows if row["suite"] == suite]
        if not suite_rows:
            continue
        lines.extend(
            [
                f"## {suite}",
                "",
                "| variant | Q/KV heads | seeds | params | KV elems/token/layer | KV bytes/token/layer | KV compression vs MHA | final train loss | final val loss | paired Δ vs baseline | val ppl | train tok/s | KV tok/s | example ids | cache parity |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in suite_rows:
            lines.append(
                "| {variant} | {num_query_heads}/{num_key_value_heads} | {seed_count} | {parameter_count:,} | "
                "{kv_cache_elements_per_token_per_layer:,} | {kv_cache_bytes_per_token_per_layer:,} | "
                "{kv_cache_compression_ratio_vs_mha:.1f}× | "
                "{final_train_loss_mean:.4f} ± {final_train_loss_std:.4f} | "
                "{final_val_loss_mean:.4f} ± {final_val_loss_std:.4f} | "
                "{paired_val_loss_delta_vs_baseline_mean:+.4f} ± {paired_val_loss_delta_vs_baseline_std:.4f} | "
                "{final_val_perplexity_mean:.2f} | {train_tokens_per_second_mean:.1f} | "
                "{kv_cache_generation_tokens_per_second_mean:.1f} | `{example_ids}` | {all_cache_matches_full} |".format(
                    example_ids=json.dumps(row["example_completion_token_ids"]),
                    **row,
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "Loss and throughput from a tiny corpus and tiny CPU model are useful for regression and learning, but they do not predict the ranking of production-scale LLMs. Timing should only be compared within the same machine/run, and generation text is recorded as a regression artifact rather than a quality score.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def environment_summary() -> str:
    return f"python={sys.version.split()[0]} torch={torch.__version__}"
