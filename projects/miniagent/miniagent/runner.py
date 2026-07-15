"""Process orchestration for the reproducible Qwen serving experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Iterable

from .config import BenchmarkConfig, RunSpec, experiment_matrix
from .workload import generate_workload, sha256_file, verify_workload


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_run_id() -> str:
    return datetime.now().astimezone().strftime("qwen-serve-%Y%m%d-%H%M%S")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prepend_env(env: dict[str, str], key: str, entries: Iterable[str]) -> None:
    old = env.get(key, "")
    values = [item for item in entries if item]
    if old:
        values.append(old)
    env[key] = os.pathsep.join(values)


def server_environment(engine: str, config: BenchmarkConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_HOME": str(config.cuda_home),
            "HF_HOME": str(config.hf_home),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    _prepend_env(env, "PATH", [str(config.cuda_home / "bin")])
    library_paths = [str(config.cuda_home / "lib64")]
    if engine == "vllm":
        env.update(
            {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_USE_FLASHINFER_SAMPLER": "1",
            }
        )
    elif engine == "sglang":
        env.update(
            {
                "FLASHINFER_WORKSPACE_BASE": str(config.flashinfer_workspace),
                "FLASHINFER_NVCC": str(config.cuda_home / "bin" / "nvcc"),
            }
        )
        site_packages = config.sglang_python.parent.parent / "lib" / "python3.12" / "site-packages"
        library_paths.extend(
            str(path)
            for path in sorted((site_packages / "nvidia").glob("*/lib"))
        )
    else:
        raise ValueError(f"unsupported engine: {engine}")
    _prepend_env(env, "LD_LIBRARY_PATH", library_paths)
    return env


def server_command(engine: str, config: BenchmarkConfig) -> list[str]:
    common_model = str(config.model_path)
    port = str(config.port_for(engine))
    if engine == "vllm":
        return [
            str(config.vllm_cli),
            "serve",
            common_model,
            "--host",
            config.host,
            "--port",
            port,
            "--served-model-name",
            config.served_model_name,
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(config.max_model_len),
            "--gpu-memory-utilization",
            str(config.vllm_gpu_memory_fraction),
            "--max-num-seqs",
            str(config.max_running_requests),
            "--max-num-batched-tokens",
            str(config.max_batched_tokens),
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--stream-interval",
            "1",
            "--generation-config",
            "vllm",
            "--seed",
            str(config.seed),
        ]
    if engine == "sglang":
        return [
            str(config.sglang_python),
            "-m",
            "sglang.launch_server",
            "--model-path",
            common_model,
            "--host",
            config.host,
            "--port",
            port,
            "--served-model-name",
            config.served_model_name,
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--context-length",
            str(config.max_model_len),
            "--mem-fraction-static",
            str(config.sglang_memory_fraction),
            "--max-running-requests",
            str(config.max_running_requests),
            "--chunked-prefill-size",
            str(config.max_batched_tokens),
            "--max-prefill-tokens",
            str(config.max_batched_tokens),
            "--attention-backend",
            "flashinfer",
            "--sampling-backend",
            "flashinfer",
            "--cuda-graph-max-bs",
            str(config.max_running_requests),
            "--cuda-graph-bs",
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
            "--disable-radix-cache",
            "--stream-interval",
            "1",
            "--random-seed",
            str(config.seed),
            "--enable-metrics",
        ]
    raise ValueError(f"unsupported engine: {engine}")


def client_command(
    spec: RunSpec,
    config: BenchmarkConfig,
    workload_path: Path,
    *,
    result_dir: Path | None = None,
    warmup: bool = False,
) -> list[str]:
    num_prompts = (
        max(config.warmup_requests, spec.concurrency)
        if warmup
        else config.num_prompts
    )
    command = [
        str(config.vllm_cli),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://{config.host}:{config.port_for(spec.engine)}",
        "--endpoint",
        "/v1/completions",
        "--model",
        str(config.model_path),
        "--served-model-name",
        config.served_model_name,
        "--tokenizer",
        str(config.model_path),
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        str(workload_path),
        "--sharegpt-output-len",
        str(config.output_tokens),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(spec.concurrency),
        "--num-warmups",
        "0",
        "--seed",
        str(spec.seed),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,95,99",
        "--ready-check-timeout-sec",
        "0",
        "--disable-tqdm",
        "--request-id-prefix",
        f"{spec.run_key}-{'warm-' if warmup else ''}",
    ]
    if not warmup:
        if result_dir is None:
            raise ValueError("result_dir is required for a measured command")
        command.extend(
            [
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(result_dir),
                "--result-filename",
                f"{spec.run_key}.json",
                "--metadata",
                f"engine={spec.engine}",
                f"concurrency={spec.concurrency}",
                f"repeat={spec.repeat}",
                f"input_len={config.input_tokens}",
                f"output_len={config.output_tokens}",
                "workload=sharegpt_exact_tokens",
            ]
        )
    return command


def _run_text(command: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "output": completed.stdout.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": repr(error)}


def collect_environment(config: BenchmarkConfig) -> dict[str, Any]:
    version_code = (
        "import json,platform; "
        "d={'python':platform.python_version()}; "
        "import torch; d.update(torch=__import__('torch').__version__,cuda=torch.version.cuda); "
        "print(json.dumps(d,sort_keys=True))"
    )
    package_code = (
        "import json,platform; d={'python':platform.python_version()}; "
        "import torch; d.update(torch=torch.__version__,cuda=torch.version.cuda); "
        "\ntry:\n import vllm; d['vllm']=vllm.__version__\nexcept Exception: pass"
        "\ntry:\n import sglang; d['sglang']=getattr(sglang,'__version__','unknown')\nexcept Exception: pass"
        "\nprint(json.dumps(d,sort_keys=True))"
    )
    del version_code  # package_code covers the desired values.
    environment: dict[str, Any] = {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "vllm_environment": _run_text(
            [str(config.vllm_python), "-c", package_code], timeout=60
        ),
        "sglang_environment": _run_text(
            [str(config.sglang_python), "-c", package_code], timeout=60
        ),
        "gpu": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
        "cuda_compiler": _run_text([str(config.cuda_home / "bin" / "nvcc"), "--version"]),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "git_status_at_start": _run_text(["git", "status", "--short"]),
    }
    model_files = []
    for path in sorted(config.model_path.iterdir()):
        if path.is_file() and path.suffix in {".json", ".safetensors", ".model"}:
            model_files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    environment["model_files"] = model_files
    return environment


@dataclass
class ServerProcess:
    engine: str
    process: subprocess.Popen[str]
    log_handle: IO[str]
    log_path: Path
    command: list[str]


def _tail(path: Path, limit: int = 6000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-limit:]
    except OSError:
        return ""


def start_server(engine: str, config: BenchmarkConfig, log_path: Path) -> ServerProcess:
    command = server_command(engine, config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    handle.write(f"# started_at={utc_now()}\n# command={json.dumps(command)}\n")
    handle.flush()
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=server_environment(engine, config),
        start_new_session=True,
    )
    return ServerProcess(engine, process, handle, log_path, command)


def wait_until_ready(server: ServerProcess, config: BenchmarkConfig) -> None:
    deadline = time.monotonic() + config.server_start_timeout_seconds
    urls = [
        f"http://{config.host}:{config.port_for(server.engine)}/health",
        f"http://{config.host}:{config.port_for(server.engine)}/v1/models",
    ]
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            server.log_handle.flush()
            raise RuntimeError(
                f"{server.engine} server exited with {server.process.returncode}\n"
                f"{_tail(server.log_path)}"
            )
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if 200 <= response.status < 300:
                        return
            except (OSError, urllib.error.URLError) as error:
                last_error = repr(error)
        time.sleep(1.0)
    raise TimeoutError(
        f"{server.engine} was not ready in {config.server_start_timeout_seconds}s: "
        f"{last_error}\n{_tail(server.log_path)}"
    )


def stop_server(server: ServerProcess, timeout: float) -> None:
    try:
        if server.process.poll() is None:
            os.killpg(server.process.pid, signal.SIGTERM)
            try:
                server.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(server.process.pid, signal.SIGKILL)
                server.process.wait(timeout=10)
    finally:
        server.log_handle.write(f"\n# stopped_at={utc_now()}\n")
        server.log_handle.close()


GPU_FIELDS = (
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "clocks.current.sm",
    "clocks.current.memory",
    "temperature.gpu",
    "pstate",
)


class GpuSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        return self.samples

    def _loop(self) -> None:
        while not self._stop.is_set():
            command = [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_FIELDS)}",
                "--format=csv,noheader,nounits",
            ]
            observed_at = time.time()
            result = _run_text(command, timeout=5)
            output = result.get("output", "")
            if result.get("returncode") == 0 and output:
                row = next(csv.reader([output.splitlines()[0]], skipinitialspace=True))
                if len(row) == len(GPU_FIELDS):
                    sample: dict[str, Any] = {"observed_at": observed_at}
                    for key, raw in zip(GPU_FIELDS, row):
                        raw = raw.strip()
                        try:
                            value: Any = float(raw)
                        except ValueError:
                            value = raw
                        sample[key] = value
                    self.samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, float]:
        mapping = {
            "utilization.gpu": "gpu_utilization_percent",
            "utilization.memory": "gpu_memory_utilization_percent",
            "memory.used": "gpu_memory_used_mib",
            "power.draw": "gpu_power_watts",
            "temperature.gpu": "gpu_temperature_c",
        }
        result: dict[str, float] = {}
        for raw_key, output_key in mapping.items():
            values = [
                float(item[raw_key])
                for item in self.samples
                if isinstance(item.get(raw_key), (float, int))
                and math.isfinite(float(item[raw_key]))
            ]
            if values:
                result[f"{output_key}_mean"] = sum(values) / len(values)
                result[f"{output_key}_max"] = max(values)
        result["gpu_sample_count"] = float(len(self.samples))
        return result


def _validate_result(path: Path, config: BenchmarkConfig) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    completed = int(data.get("completed", -1))
    failed = int(data.get("failed", -1))
    ttfts = data.get("ttfts")
    input_lens = data.get("input_lens")
    output_lens = data.get("output_lens")
    errors: list[str] = []
    if completed != config.num_prompts:
        errors.append(f"completed={completed}, expected={config.num_prompts}")
    if failed != 0:
        errors.append(f"failed={failed}, expected=0")
    if not isinstance(ttfts, list) or len(ttfts) != config.num_prompts:
        errors.append("detailed ttfts are missing or incomplete")
    if not isinstance(input_lens, list) or any(
        int(value) != config.input_tokens for value in input_lens
    ):
        errors.append("input_lens are not all the configured exact length")
    if not isinstance(output_lens, list) or any(
        int(value) != config.output_tokens for value in output_lens
    ):
        errors.append("output_lens are not all the configured exact length")
    if errors:
        raise RuntimeError(f"invalid result {path}: {'; '.join(errors)}")
    return {
        "completed": completed,
        "failed": failed,
        "ttft_samples": len(ttfts),
        "input_len_min": min(input_lens),
        "input_len_max": max(input_lens),
        "output_len_min": min(output_lens),
        "output_len_max": max(output_lens),
        "raw_sha256": sha256_file(path),
    }


def _run_client(
    command: list[str],
    log_path: Path,
    config: BenchmarkConfig,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# started_at={utc_now()}\n# command={json.dumps(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_environment("vllm", config),
            timeout=config.client_timeout_seconds,
            check=False,
        )
        handle.write(
            f"\n# finished_at={utc_now()}\n# elapsed_seconds={time.monotonic()-started:.6f}"
            f"\n# returncode={completed.returncode}\n"
        )
    if completed.returncode:
        raise RuntimeError(
            f"benchmark client failed with {completed.returncode}: {command}\n"
            f"{_tail(log_path)}"
        )
    return completed


def _metrics_capture(
    engine: str, metrics_url: str, interval_seconds: float
) -> Any:
    from .prometheus import (
        PrometheusSampler,
        SGLANG_METRIC_SPEC,
        VLLM_METRIC_SPEC,
    )

    metric_spec = VLLM_METRIC_SPEC if engine == "vllm" else SGLANG_METRIC_SPEC
    return PrometheusSampler(
        metrics_url,
        interval_seconds=interval_seconds,
        selected_metrics=metric_spec,
        metric_spec=metric_spec,
    )


def _save_combined_metrics(
    sampler: Any,
    gpu_sampler: GpuSampler,
    path: Path,
) -> dict[str, Any]:
    payload = sampler.write_json(path)
    if payload is None:
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["gpu_samples"] = gpu_sampler.samples
    summary = payload.setdefault("summary", {})
    summary.update(gpu_sampler.summary())
    atomic_write_json(path, payload)
    return payload


def dry_run_plan(config: BenchmarkConfig, run_dir: Path) -> dict[str, Any]:
    workload_path = run_dir / "workload" / "qwen_exact_tokens.json"
    raw_dir = run_dir / "raw"
    return {
        "run_dir": str(run_dir),
        "config": config.public_dict(),
        "servers": {
            engine: {
                "command": server_command(engine, config),
                "environment_overrides": {
                    key: value
                    for key, value in server_environment(engine, config).items()
                    if os.environ.get(key) != value
                },
            }
            for engine in config.engines
        },
        "runs": [
            {
                "run_key": spec.run_key,
                "warmup_command": client_command(
                    spec, config, workload_path, warmup=True
                ),
                "measured_command": client_command(
                    spec, config, workload_path, result_dir=raw_dir
                ),
            }
            for spec in experiment_matrix(config)
        ],
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
) -> Path | dict[str, Any]:
    config.validate()
    run_id = run_id or canonical_run_id()
    run_dir = config.output_root / run_id
    if dry_run:
        return dry_run_plan(config, run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")

    for directory in ("raw", "logs", "metrics", "server", "workload"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": run_id,
        "status": "initializing",
        "created_at": utc_now(),
        "config": config.public_dict(),
        "protocol": {
            "client": "vllm bench serve",
            "api": "OpenAI-compatible streaming /v1/completions",
            "load_model": "closed-loop saturation; request-rate=inf with client max-concurrency",
            "prefix_cache": "disabled on both engines",
            "sampling": "temperature=0, top_p=1, ignore_eos=true",
            "percentiles": [50, 90, 95, 99],
            "warmup": "separate unmeasured run before each measured cell",
            "metrics_note": "ready-check is disabled during measured commands; deltas cover measured requests only",
        },
        "runs": [],
    }
    atomic_write_json(manifest_path, manifest)
    print(f"[{utc_now()}] collecting environment", flush=True)
    manifest["environment"] = collect_environment(config)

    workload_path = run_dir / "workload" / "qwen_exact_tokens.json"
    print(f"[{utc_now()}] generating exact-token workload", flush=True)
    workload = generate_workload(
        workload_path,
        config.vllm_python,
        config.model_path,
        config.input_tokens,
        config.output_tokens,
        config.num_prompts,
    )
    workload["tokenizer_verification"] = {
        "vllm_environment": verify_workload(
            workload_path,
            config.vllm_python,
            config.model_path,
            config.input_tokens,
        ),
        "sglang_environment": verify_workload(
            workload_path,
            config.sglang_python,
            config.model_path,
            config.input_tokens,
        ),
    }
    manifest["workload"] = workload
    manifest["status"] = "running"
    atomic_write_json(manifest_path, manifest)

    specs = experiment_matrix(config)
    try:
        for engine in config.engines:
            engine_specs = [spec for spec in specs if spec.engine == engine]
            server_log = run_dir / "server" / f"{engine}.log"
            print(f"[{utc_now()}] starting {engine} server", flush=True)
            server = start_server(engine, config, server_log)
            try:
                wait_until_ready(server, config)
                manifest.setdefault("servers", {})[engine] = {
                    "command": server.command,
                    "log_path": str(server_log.relative_to(run_dir)),
                    "ready_at": utc_now(),
                }
                atomic_write_json(manifest_path, manifest)
                for spec in engine_specs:
                    raw_path = run_dir / "raw" / f"{spec.run_key}.json"
                    metrics_path = run_dir / "metrics" / f"{spec.run_key}.json"
                    warm_log = run_dir / "logs" / f"{spec.run_key}-warmup.log"
                    client_log = run_dir / "logs" / f"{spec.run_key}.log"
                    warm_command = client_command(
                        spec, config, workload_path, warmup=True
                    )
                    measured_command = client_command(
                        spec,
                        config,
                        workload_path,
                        result_dir=run_dir / "raw",
                    )
                    run_record: dict[str, Any] = {
                        "run_key": spec.run_key,
                        "engine": spec.engine,
                        "source_format": "vllm_bench_serve_detailed_json",
                        "concurrency": spec.concurrency,
                        "repeat": spec.repeat,
                        "seed": spec.seed,
                        "status": "warming",
                        "warmup_command": warm_command,
                        "command": measured_command,
                        "raw_path": str(raw_path.relative_to(run_dir)),
                        "metrics_path": str(metrics_path.relative_to(run_dir)),
                        "client_log_path": str(client_log.relative_to(run_dir)),
                    }
                    manifest["runs"].append(run_record)
                    atomic_write_json(manifest_path, manifest)
                    print(
                        f"[{utc_now()}] {spec.run_key}: warmup "
                        f"({max(config.warmup_requests, spec.concurrency)} requests)",
                        flush=True,
                    )
                    _run_client(warm_command, warm_log, config)
                    run_record["status"] = "measuring"
                    run_record["measured_started_at"] = utc_now()
                    atomic_write_json(manifest_path, manifest)
                    print(
                        f"[{utc_now()}] {spec.run_key}: measuring "
                        f"({config.num_prompts} requests)",
                        flush=True,
                    )
                    metrics = _metrics_capture(
                        engine,
                        f"http://{config.host}:{config.port_for(engine)}/metrics",
                        config.metrics_interval_seconds,
                    )
                    gpu = GpuSampler(config.gpu_interval_seconds)
                    metrics.start()
                    gpu.start()
                    try:
                        _run_client(measured_command, client_log, config)
                    finally:
                        gpu.stop()
                        metrics.stop()
                    metrics_payload = _save_combined_metrics(metrics, gpu, metrics_path)
                    validation = _validate_result(raw_path, config)
                    run_record.update(
                        {
                            "status": "completed",
                            "measured_finished_at": utc_now(),
                            "validation": validation,
                            "metrics_summary": metrics_payload.get("summary", {}),
                        }
                    )
                    atomic_write_json(manifest_path, manifest)
            finally:
                print(f"[{utc_now()}] stopping {engine} server", flush=True)
                stop_server(server, config.server_stop_timeout_seconds)
                if "servers" in manifest and engine in manifest["servers"]:
                    manifest["servers"][engine]["stopped_at"] = utc_now()
                    manifest["servers"][engine]["returncode"] = server.process.returncode
                atomic_write_json(manifest_path, manifest)
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = repr(error)
        atomic_write_json(manifest_path, manifest)
        raise

    from .analysis import analyze_run, write_summary
    from .report import generate_report

    summary = analyze_run(run_dir, manifest)
    write_summary(summary, run_dir / "summary.json")
    report_path = run_dir / "report.md"
    generate_report(summary, report_path)
    reports_dir = config.output_root.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    generate_report(summary, reports_dir / f"{run_id}.md")
    return run_dir
