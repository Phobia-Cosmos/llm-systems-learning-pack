from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable

import torch

from operators import OperatorCase, select_cases
from triton_ops import fused_bias_silu_eligibility, triton_fused_bias_silu

# PyTorch dtype 数量会随版本和 backend 增长，不能依赖一个固定“40 种”的数字；它包含 bool、各宽度整数、
# FP16/BF16/FP32/FP64、complex、float8、量化 dtype 等，而且并非每个算子/backend 都支持全部 dtype。
# 本 benchmark 的公式包含 exp/sigmoid/norm/attention，当前只选择三种对 LLM 有代表性、且所有 30 个 case
# 都有共同语义的浮点 dtype。扩展整数/FP8/complex 时应新增适用算子组与各自 reference/tolerance，而非只往字典加名字。
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def output_tensors(output: object) -> tuple[torch.Tensor, ...]:
    """把单 Tensor 或纯 Tensor tuple/list 统一成 tuple，便于通用正确性比较。"""

    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, (tuple, list)):
        # topk/max 等算子可能同时返回 values 和 indices，模型 forward 也可能返回 logits/loss，因此需要支持
        # 多输出。若 output 中混有 None、Python 标量或其他对象，过滤后的 tensors 长度就会小于 output；
        # 这里不静默忽略它们，而是明确报错，防止 reference/teaching 的非 Tensor 辅助语义漏比。
        tensors = tuple(item for item in output if isinstance(item, torch.Tensor))
        if len(tensors) != len(output):
            raise TypeError("operator outputs must contain only tensors")
        return tensors
    raise TypeError(f"unsupported output type: {type(output)!r}")


def compare_outputs(
    expected: object,
    actual: object,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float, str]:
    expected_tensors = output_tensors(expected)
    actual_tensors = output_tensors(actual)
    if len(expected_tensors) != len(actual_tensors):
        return False, math.inf, "output arity differs"
    maximum_error = 0.0
    # zip(..., strict=True) 要求两个 iterable 恰好等长，否则抛 ValueError，而普通 zip 会静默截到较短者。
    # 前面已检查长度，这里 strict=True 是第二道保护，避免未来重构后漏比某个输出。
    for expected_tensor, actual_tensor in zip(expected_tensors, actual_tensors, strict=True):
        if expected_tensor.dtype.is_floating_point:
            # reduction 后得到 0-dim Tensor；.item() 把其中一个标量转换成 Python number，便于写 JSON/比较。
            # CUDA Tensor 的 .item() 会触发 device→host 取值/同步，所以只用于计时区外的 correctness，不能放进 kernel benchmark。
            error = torch.max(torch.abs(expected_tensor.float() - actual_tensor.float())).item()
            maximum_error = max(maximum_error, error)
        # assert_close 同时检查 shape、dtype/device 兼容性与每个数值是否满足
        # |actual-expected| <= atol + rtol*|expected|；整数/索引要求精确相等。失败时截取首行作为简短报告。
        try:
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)
        except AssertionError as exc:
            return False, maximum_error, str(exc).splitlines()[0]
    return True, maximum_error, "ok"


def time_callable(
    function: Callable[..., object],
    inputs: tuple[object, ...],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    inner: int,
) -> tuple[float, float, list[float]]:
    with torch.inference_mode():
        for _ in range(warmup):
            function(*inputs)
        synchronize(device)
        samples_us: list[float] = []
        if device.type == "cuda":
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(inner):
                    function(*inputs)
                end.record()
                end.synchronize()
                samples_us.append(start.elapsed_time(end) * 1000.0 / inner)
        else:
            for _ in range(repeats):
                start_time = time.perf_counter_ns()
                for _ in range(inner):
                    function(*inputs)
                elapsed_ns = time.perf_counter_ns() - start_time
                samples_us.append(elapsed_ns / 1000.0 / inner)
    return statistics.median(samples_us), percentile(samples_us, 0.95), samples_us


def build_variant(
    case: OperatorCase,
    variant: str,
    inputs: tuple[object, ...],
) -> tuple[Callable[..., object], str]:
    if variant == "reference":
        return case.pytorch_reference, "PyTorch reference"
    if variant == "teaching":
        return case.teaching_python, "Python composition"
    if variant == "compiled":
        # torch.compile 不是固定把 Python 翻译成“GPU C”：Dynamo 捕获 Tensor 计算图，Inductor 按 device
        # 为 CPU 生成 C++/向量化代码，或为 CUDA 生成 Triton/CUDA 等后端代码。fullgraph=True 要求整个
        # teaching function 被捕获为一张图；若出现不支持的 Python 导致 graph break 就报错，避免悄悄混入 eager 段。
        compiled = torch.compile(case.teaching_python, fullgraph=True)
        # 是的，compiled 仍是 callable；第一次传入真实 inputs 会执行 Dynamo capture、后端编译并返回结果。
        # 这里故意忽略这次结果，因为只用它提前完成编译，JIT latency 不进入后面的 steady-state 计时。
        compiled(*inputs)  # Compilation is intentionally outside the timed region.
        return compiled, "torch.compile teaching function"
    if variant == "triton":
        if case.name != "fused_bias_silu":
            raise ValueError("the current Triton teaching kernel only implements fused_bias_silu")
        x, bias = inputs
        if not isinstance(x, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise TypeError("fused_bias_silu inputs must be tensors")
        eligibility = fused_bias_silu_eligibility(x, bias)
        if not eligibility.supported:
            raise ValueError(f"Triton variant unavailable: {eligibility.reason}")
        
        triton_fused_bias_silu(x, bias)  # JIT compilation is outside the timed region.
        return triton_fused_bias_silu, "Triton JIT kernel"
    raise ValueError(f"unknown variant: {variant}")

def adjusted_tolerance(case: OperatorCase, dtype: torch.dtype) -> tuple[float, float]:
    """根据实际 dtype 放宽 case 的最低绝对/相对误差阈值。

    FP16/BF16 的 fraction 位更少，同一公式因 reduction 顺序、融合和 FP32 中间计算不同会产生更大的合理舍入差。
    max(case tolerance, dtype floor) 保留个别算子原本更宽的要求，同时避免给低精度使用不现实的 1e-5；
    FP32 继续使用每个 OperatorCase 自己的 atol/rtol。容差只决定 correctness，不会改变算子输出。
    """

    if dtype == torch.float16:
        return max(case.atol, 2e-2), max(case.rtol, 2e-2)
    if dtype == torch.bfloat16:
        return max(case.atol, 5e-2), max(case.rtol, 5e-2)
    return case.atol, case.rtol

def run_cases(
    cases: Iterable[OperatorCase],
    variants: tuple[str, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    profile: str,
    warmup: int,
    repeats: int,
    inner: int,
    check_only: bool,
) -> list[dict[str, object]]:
    """对选中的 case/variant 执行正确性检查，并可选择计时。

    函数签名中单独的 `*` 表示其后的 device/dtype/profile 等必须用关键字传入，例如
    run_cases(cases, variants, device=...)；它不是可变参数。调用算子时的 function(*inputs) 是另一种语法，
    表示把参数 tuple 解包成独立 positional arguments。
    """

    results: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        # 固定 seed 让同一 case 每次生成相同随机输入，reference/teaching 和多次运行可复现；加 case_index
        # 让不同算子不都拿到完全相同随机序列。torch.manual_seed 同时设置 CPU 及当前可见 CUDA RNG 的默认 seed。
        torch.manual_seed(20260716 + case_index)
        inputs = case.make_inputs(device, dtype, profile)
        # 这是 forward microbenchmark，不测训练；inference_mode 关闭 autograd graph、grad metadata 和部分
        # version-counter 开销，使各 variant 都在相同推理语义下运行。若要比较 backward，必须另建
        # forward+backward harness，保留 requires_grad，不能沿用这里的 inference_mode 数字。
        with torch.inference_mode():
            expected = case.pytorch_reference(*inputs)

        tolerance = adjusted_tolerance(case, dtype)

        reference_median: float | None = None
        case_rows: list[dict[str, object]] = []
        for variant in variants:
            function, implementation = build_variant(case, variant, inputs)
            with torch.inference_mode():
                actual = function(*inputs)
            correct, max_abs_error, detail = compare_outputs(
                expected,
                actual,
                atol=tolerance[0],
                rtol=tolerance[1],
            )
            if not correct:
                raise AssertionError(f"{case.name}/{variant} failed correctness: {detail}")
            median_us = None
            p95_us = None
            samples: list[float] = []
            if not check_only:
                median_us, p95_us, samples = time_callable(
                    function,
                    inputs,
                    device=device,
                    warmup=warmup,
                    repeats=repeats,
                    inner=inner,
                )
                if variant == "reference":
                    reference_median = median_us
            row: dict[str, object] = {
                "operator": case.name,
                "family": case.family,
                "variant": variant,
                "implementation": implementation,
                "description": case.description,
                "device": str(device),
                "dtype": str(dtype).removeprefix("torch."),
                "profile": profile,
                "correct": correct,
                "max_abs_error": max_abs_error,
                "median_us": median_us,
                "p95_us": p95_us,
                "samples_us": samples,
                "speedup_vs_reference": None,
            }
            case_rows.append(row)
        if not check_only and reference_median is not None:
            for row in case_rows:
                median = row["median_us"]
                if isinstance(median, float) and median > 0:
                    row["speedup_vs_reference"] = reference_median / median
        results.extend(case_rows)
        status = "PASS" if all(bool(row["correct"]) for row in case_rows) else "FAIL"
        timings = ""
        if not check_only:
            timings = " ".join(
                f"{row['variant']}={float(row['median_us']):.2f}us"
                for row in case_rows
                if isinstance(row["median_us"], float)
            )
        print(f"{case.name:<22} {status} {timings}".rstrip())
        del inputs, expected
    return results


def environment(device: torch.device) -> dict[str, object]:
    """记录解释和复现实验所需的软件/设备元数据。"""

    # python 是解释器版本，不代表 CPU；pytorch 是框架 build 版本；cuda_runtime 是该 PyTorch build
    # 编译/绑定的 CUDA runtime 版本（CPU-only build 时为 None）；device 是本次 Tensor 实际选择的 cpu/cuda:N。
    # 只有 device=cuda 时下面才增加真实 GPU 名称与硬件 Compute Capability；软件版本和硬件字段是不同层次。
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        # dict.update 会新增不存在的 key，并覆盖已经存在的同名 key；这里 gpu/compute_capability 原先不存在，
        # 所以只是追加两项。若未来重复使用同名 key，后传入值会覆盖旧值。
        payload.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return payload


def write_results(path: Path, metadata: dict[str, object], results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    csv_path = path.with_suffix(".csv")
    json_path.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [key for key in results[0].keys() if key != "samples_us"] if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: value for key, value in result.items() if key != "samples_us"})
    print(json_path)
    print(csv_path)


def parse_args() -> argparse.Namespace:
    """定义命令行接口；这些参数只控制选择/计时，不改变算子数学语义。"""

    parser = argparse.ArgumentParser(description="Compare PyTorch references with readable Python operator implementations.")
    parser.add_argument("--operators", default="all", help="Comma-separated operator names or 'all'.")
    # reference=PyTorch 官方 baseline；teaching=未编译的可读 Python Tensor 组合；compiled=把同一 teaching
    # callable 交给 torch.compile/Inductor；triton=显式 Triton JIT kernel（当前仅 fused_bias_silu）。
    parser.add_argument("--variants", default="reference,teaching", help="Comma-separated: reference,teaching,compiled,triton.")
    # auto 优先选择可用 CUDA，否则 CPU；cuda 在本项目指 NVIDIA GPU backend。profile 是输入 shape 档位，
    # 不是 profiler 工具：smoke 小而覆盖边界，llm 使用代表性大 shape。warmup 是不计时预热次数；repeats
    # 是独立 timing sample 数；inner 是每个 sample 内重复调用次数，最后除以 inner 以摊薄 Event/时钟开销；
    # check-only 只做正确性、不采样时间；output 是结果 stem，会同时写 .json 与 .csv。
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="auto prefers CUDA when available; otherwise CPU.")
    parser.add_argument("--dtype", default="float32", choices=tuple(DTYPES))
    parser.add_argument("--profile", default="smoke", choices=("smoke", "llm"), help="Input-shape profile, not a profiler tool.")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed calls before measurement.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of independent timing samples.")
    parser.add_argument("--inner", type=int, default=10, help="Calls per timing sample.")
    parser.add_argument("--check-only", action="store_true", help="Run correctness without timing.")
    parser.add_argument("--output", type=Path, default=None, help="Output stem; writes .json and .csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.warmup, args.repeats, args.inner) < 1:
        raise ValueError("warmup, repeats, and inner must be positive")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build cannot access CUDA")
    # 可以直接传 "cpu"：torch.device("cpu") 是合法 device 对象；后续 input factory 会在 CPU 创建 Tensor，
    # synchronize/time_callable 也会自动使用 perf_counter 而不是 CUDA Event。若传 "cuda"，前面的可用性检查
    # 保证不会在 CPU-only PyTorch 中走到这里。
    device = torch.device(device_name)
    dtype = DTYPES[args.dtype]
    variants = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    if not variants:
        raise ValueError("at least one variant is required")
    if "reference" not in variants and not args.check_only:
        raise ValueError("timed comparisons require the reference variant")
    cases = select_cases(args.operators)
    metadata = environment(device)
    metadata.update(
        {
            "profile": args.profile,
            "dtype": args.dtype,
            "variants": list(variants),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "inner": args.inner,
            "compile_and_jit_time_included": False,
        }
    )
    print(json.dumps(metadata, ensure_ascii=False))
    results = run_cases(
        cases,
        variants,
        device=device,
        dtype=dtype,
        profile=args.profile,
        warmup=args.warmup,
        repeats=args.repeats,
        inner=args.inner,
        check_only=args.check_only,
    )
    if args.output is not None:
        write_results(args.output, metadata, results)


if __name__ == "__main__":
    main()
