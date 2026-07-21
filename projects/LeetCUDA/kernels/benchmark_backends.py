from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

import kernels.torch_compile_backend as torch_compile_backend
import kernels.triton_backend as triton_backend


@dataclass
class BenchmarkCase:
    eager: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]
    triton: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]
    torch_compile: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]


def _time(function: Callable, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _activation_case(name: str, shape: tuple[int, int]) -> BenchmarkCase:
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    references = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "elu": F.elu,
        "gelu": lambda value: F.gelu(value, approximate="tanh"),
        "swish": F.silu,
        "hardswish": F.hardswish,
        "hardshrink": F.hardshrink,
    }
    return BenchmarkCase(
        eager=lambda: references[name](x),
        triton=lambda: getattr(triton_backend, name)(x),
        torch_compile=lambda: getattr(torch_compile_backend, name)(x),
    )


def make_case(name: str) -> BenchmarkCase:
    shape = (4096, 1024)
    if name in {"relu", "sigmoid", "elu", "gelu", "swish", "hardswish", "hardshrink"}:
        return _activation_case(name, shape)
    if name == "elementwise_add":
        a = torch.randn(shape, device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        return BenchmarkCase(lambda: a + b, lambda: triton_backend.elementwise_add(a, b), lambda: torch_compile_backend.elementwise_add(a, b))
    if name == "dot_product":
        a = torch.randn(4 * 1024 * 1024, device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        return BenchmarkCase(lambda: (a.float() * b.float()).sum(), lambda: triton_backend.dot_product(a, b), lambda: torch_compile_backend.dot_product(a, b))
    if name == "gemv":
        matrix = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        vector = torch.randn(4096, device="cuda", dtype=torch.float16)
        return BenchmarkCase(lambda: matrix @ vector, lambda: triton_backend.gemv(matrix, vector), lambda: torch_compile_backend.gemv(matrix, vector))
    if name == "gemm":
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        return BenchmarkCase(lambda: a @ b, lambda: triton_backend.gemm(a, b), lambda: torch_compile_backend.gemm(a, b))
    if name == "softmax":
        x = torch.randn(4096, 1024, device="cuda", dtype=torch.float16)
        return BenchmarkCase(lambda: torch.softmax(x, -1), lambda: triton_backend.softmax(x), lambda: torch_compile_backend.softmax(x))
    if name == "layer_norm":
        x = torch.randn(4096, 1024, device="cuda", dtype=torch.float16)
        return BenchmarkCase(lambda: F.layer_norm(x, (1024,)), lambda: triton_backend.layer_norm(x), lambda: torch_compile_backend.layer_norm(x))
    if name == "layer_norm_affine":
        x = torch.randn(4096, 1024, device="cuda", dtype=torch.float16)
        weight = torch.randn(1024, device="cuda", dtype=torch.float16)
        bias = torch.randn(1024, device="cuda", dtype=torch.float16)
        return BenchmarkCase(lambda: F.layer_norm(x, (1024,), weight, bias), lambda: triton_backend.layer_norm_affine(x, weight, bias), lambda: torch_compile_backend.layer_norm_affine(x, weight, bias))
    if name == "layer_norm_backward":
        x = torch.randn(4096, 1024, device="cuda", dtype=torch.float16)
        weight = torch.randn(1024, device="cuda", dtype=torch.float16)
        grad_output = torch.randn_like(x)
        mean = x.float().mean(dim=-1)
        rstd = torch.rsqrt(x.float().var(dim=-1, correction=0) + 1e-5)

        def eager_backward():
            normalized = (x.float() - mean[:, None]) * rstd[:, None]
            weighted_grad = grad_output.float() * weight.float()
            projection = (normalized * weighted_grad).mean(dim=-1, keepdim=True)
            mean_gradient = weighted_grad.mean(dim=-1, keepdim=True)
            grad_input = ((weighted_grad - normalized * projection - mean_gradient) * rstd[:, None]).half()
            grad_weight = (grad_output.float() * normalized).sum(dim=0).half()
            grad_bias = grad_output.float().sum(dim=0).half()
            return grad_input, grad_weight, grad_bias

        args = (grad_output, x, weight, mean, rstd)
        return BenchmarkCase(eager_backward, lambda: triton_backend.layer_norm_backward(*args), lambda: torch_compile_backend.layer_norm_backward(*args))
    if name == "rms_norm":
        x = torch.randn(4096, 1024, device="cuda", dtype=torch.float16)
        return BenchmarkCase(lambda: F.rms_norm(x, (1024,), eps=1e-5), lambda: triton_backend.rms_norm(x), lambda: torch_compile_backend.rms_norm(x))
    if name == "reduce_sum":
        x = torch.randn(4 * 1024 * 1024, device="cuda", dtype=torch.float16)
        return BenchmarkCase(lambda: x.float().sum(), lambda: triton_backend.reduce_sum(x), lambda: torch_compile_backend.reduce_sum(x))
    if name == "embedding":
        weight = torch.randn(65536, 1024, device="cuda", dtype=torch.float16)
        indices = torch.randint(0, weight.shape[0], (4096,), device="cuda")
        return BenchmarkCase(lambda: F.embedding(indices, weight), lambda: triton_backend.embedding(indices, weight), lambda: torch_compile_backend.embedding(indices, weight))
    if name == "matrix_transpose":
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
        return BenchmarkCase(lambda: x.T.contiguous(), lambda: triton_backend.matrix_transpose(x), lambda: torch_compile_backend.matrix_transpose(x))
    if name == "rope":
        x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)

        def eager_rope():
            pairs = x.reshape(4096, 512, 2)
            positions = torch.arange(4096, device=x.device)[:, None]
            dimensions = torch.arange(512, device=x.device)[None, :]
            angles = positions * torch.exp(-math.log(10000.0) * 2.0 * dimensions / 1024)
            return torch.stack((pairs[..., 0] * angles.cos() - pairs[..., 1] * angles.sin(), pairs[..., 0] * angles.sin() + pairs[..., 1] * angles.cos()), -1).reshape_as(x)

        return BenchmarkCase(eager_rope, lambda: triton_backend.rope(x), lambda: torch_compile_backend.rope(x))
    if name == "histogram":
        values = torch.randint(0, 256, (4 * 1024 * 1024,), device="cuda", dtype=torch.int32)
        return BenchmarkCase(lambda: torch.bincount(values, minlength=256), lambda: triton_backend.histogram(values, 256), lambda: torch_compile_backend.histogram(values, 256))
    if name == "flash_attention":
        query = torch.randn(2, 16, 1024, 64, device="cuda", dtype=torch.float16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        def eager_attention():
            scores = query @ key.transpose(-2, -1) * (1.0 / math.sqrt(64))
            return torch.softmax(scores, -1) @ value

        return BenchmarkCase(eager_attention, lambda: triton_backend.flash_attention(query, key, value), lambda: torch_compile_backend.flash_attention(query, key, value))
    if name == "merge_attention_states":
        prefix_output = torch.randn(4096, 16, 128, device="cuda", dtype=torch.float16)
        suffix_output = torch.randn_like(prefix_output)
        prefix_lse = torch.randn(16, 4096, device="cuda")
        suffix_lse = torch.randn_like(prefix_lse)

        def eager_merge():
            output_lse = torch.logaddexp(prefix_lse, suffix_lse)
            return (prefix_output * torch.exp(prefix_lse - output_lse).T[..., None] + suffix_output * torch.exp(suffix_lse - output_lse).T[..., None]).half()

        args = (prefix_output, prefix_lse, suffix_output, suffix_lse)
        return BenchmarkCase(eager_merge, lambda: triton_backend.merge_attention_states(*args), lambda: torch_compile_backend.merge_attention_states(*args))
    if name == "nms":
        from torchvision.ops import nms as eager_nms

        boxes = torch.rand(1024, 4, device="cuda")
        boxes = torch.cat((torch.minimum(boxes[:, :2], boxes[:, 2:]), torch.maximum(boxes[:, :2], boxes[:, 2:])), 1).contiguous()
        scores = torch.rand(1024, device="cuda")
        return BenchmarkCase(lambda: eager_nms(boxes, scores, 0.5), lambda: triton_backend.nms(boxes, scores, 0.5), lambda: torch_compile_backend.nms(boxes, scores, 0.5))
    raise KeyError(name)


OPERATORS = [
    "elementwise_add", "relu", "sigmoid", "elu", "gelu", "swish", "hardswish", "hardshrink",
    "dot_product", "gemv", "gemm", "softmax", "layer_norm", "layer_norm_affine", "layer_norm_backward",
    "rms_norm", "reduce_sum", "embedding",
    "matrix_transpose", "rope", "histogram", "flash_attention", "merge_attention_states", "nms",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=["all", *OPERATORS], default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    selected = OPERATORS if args.operator == "all" else [args.operator]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'operator':<25} {'eager/ms':>12} {'triton/ms':>12} {'compile/ms':>12} {'T/eager':>10} {'C/eager':>10}")
    for name in selected:
        case = make_case(name)
        eager_ms = _time(case.eager, args.warmup, args.iterations)
        triton_ms = _time(case.triton, args.warmup, args.iterations)
        compile_ms = _time(case.torch_compile, args.warmup, args.iterations)
        print(f"{name:<25} {eager_ms:12.5f} {triton_ms:12.5f} {compile_ms:12.5f} {eager_ms/triton_ms:10.2f} {eager_ms/compile_ms:10.2f}")


if __name__ == "__main__":
    main()
