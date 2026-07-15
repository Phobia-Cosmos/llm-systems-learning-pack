"""Configuration and experiment-matrix construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fully explicit inputs for a Qwen serving comparison.

    Defaults describe the canonical experiment. Smoke runs override only the
    matrix size and leave the fairness-sensitive server settings unchanged.
    """

    model_path: Path = Path(
        "/home/undefined/Disk/cache/models/huggingface/Qwen3-0.6B"
    )
    served_model_name: str = "qwen3-0.6b"
    engines: tuple[str, ...] = ("vllm", "sglang")
    concurrencies: tuple[int, ...] = (1, 8, 32)
    repeats: int = 3
    num_prompts: int = 256
    input_tokens: int = 1024
    output_tokens: int = 128
    seed: int = 42
    warmup_requests: int = 8
    max_model_len: int = 2048
    max_running_requests: int = 32
    max_batched_tokens: int = 2048
    vllm_gpu_memory_fraction: float = 0.70
    # The two fractions are not definitionally identical, but the same 70%
    # resource ceiling is explicit and fits this local GPU. Fairness is also
    # checked from KV usage and zero retractions/preemptions at runtime.
    sglang_memory_fraction: float = 0.70
    host: str = "127.0.0.1"
    vllm_port: int = 18000
    sglang_port: int = 18001
    server_start_timeout_seconds: float = 600.0
    server_stop_timeout_seconds: float = 30.0
    client_timeout_seconds: float = 1800.0
    metrics_interval_seconds: float = 0.10
    gpu_interval_seconds: float = 0.50
    vllm_python: Path = Path("/home/undefined/Disk/python-envs/vllm/bin/python")
    vllm_cli: Path = Path("/home/undefined/Disk/python-envs/vllm/bin/vllm")
    sglang_python: Path = Path("/home/undefined/Disk/python-envs/sglang/bin/python")
    cuda_home: Path = Path("/usr/local/cuda-13.0")
    hf_home: Path = Path("/home/undefined/Disk/cache/models/huggingface")
    flashinfer_workspace: Path = Path(
        "/home/undefined/Disk/cache/flashinfer-system-cuda-release"
    )
    output_root: Path = PROJECT_ROOT / "artifacts" / "runs"

    def validate(self) -> None:
        if not self.model_path.exists():
            raise ValueError(f"model path does not exist: {self.model_path}")
        for executable in (self.vllm_python, self.vllm_cli, self.sglang_python):
            if not executable.exists():
                raise ValueError(f"required executable does not exist: {executable}")
        if not self.engines or any(e not in {"vllm", "sglang"} for e in self.engines):
            raise ValueError("engines must be a non-empty subset of vllm,sglang")
        if not self.concurrencies or any(c <= 0 for c in self.concurrencies):
            raise ValueError("concurrencies must contain positive integers")
        if max(self.concurrencies) > self.max_running_requests:
            raise ValueError("concurrency exceeds max_running_requests")
        if self.repeats <= 0 or self.num_prompts <= 0:
            raise ValueError("repeats and num_prompts must be positive")
        if self.num_prompts < max(self.concurrencies):
            raise ValueError("num_prompts must be at least the maximum concurrency")
        if self.input_tokens <= 0 or self.output_tokens <= 1:
            raise ValueError("input_tokens must be positive and output_tokens > 1")
        if self.input_tokens + self.output_tokens > self.max_model_len:
            raise ValueError("input_tokens + output_tokens exceeds max_model_len")
        if self.max_batched_tokens < self.input_tokens:
            raise ValueError("max_batched_tokens must fit at least one prompt")
        for name, fraction in (
            ("vllm_gpu_memory_fraction", self.vllm_gpu_memory_fraction),
            ("sglang_memory_fraction", self.sglang_memory_fraction),
        ):
            if not 0.0 < fraction < 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.metrics_interval_seconds <= 0 or self.gpu_interval_seconds <= 0:
            raise ValueError("sampling intervals must be positive")

    def port_for(self, engine: str) -> int:
        if engine == "vllm":
            return self.vllm_port
        if engine == "sglang":
            return self.sglang_port
        raise ValueError(f"unsupported engine: {engine}")

    def public_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            return value

        return encode(asdict(self))

    def with_overrides(self, **kwargs: Any) -> "BenchmarkConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RunSpec:
    engine: str
    concurrency: int
    repeat: int
    seed: int

    @property
    def run_key(self) -> str:
        return f"{self.engine}-c{self.concurrency}-r{self.repeat}"


def balanced_concurrency_order(
    concurrencies: Iterable[int], repeat: int
) -> tuple[int, ...]:
    values = tuple(concurrencies)
    if not values:
        return ()
    shift = (repeat - 1) % len(values)
    return values[shift:] + values[:shift]


def experiment_matrix(config: BenchmarkConfig) -> list[RunSpec]:
    """Return paired, cyclically balanced run order for each engine."""

    specs: list[RunSpec] = []
    for engine in config.engines:
        for repeat in range(1, config.repeats + 1):
            for concurrency in balanced_concurrency_order(
                config.concurrencies, repeat
            ):
                specs.append(
                    RunSpec(
                        engine=engine,
                        concurrency=concurrency,
                        repeat=repeat,
                        seed=config.seed + repeat - 1,
                    )
                )
    return specs
