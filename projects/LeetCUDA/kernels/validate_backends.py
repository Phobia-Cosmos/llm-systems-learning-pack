from __future__ import annotations

import torch
import torch.nn.functional as F

import kernels.torch_compile_backend as torch_compile_backend
import kernels.triton_backend as triton_backend


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol=1e-3, rtol=1e-3) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(f"PASS {name:<42} shape={tuple(actual.shape)!s:<18} dtype={actual.dtype}")


def validate(backend_name: str, backend) -> None:
    torch.manual_seed(0)
    x = torch.randn(257, device="cuda")
    y = torch.randn_like(x)
    assert_close(f"{backend_name}/elementwise_add", backend.elementwise_add(x, y), x + y)
    activations = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "elu": F.elu,
        "gelu": lambda value: F.gelu(value, approximate="tanh"),
        "swish": F.silu,
        "hardswish": F.hardswish,
        "hardshrink": F.hardshrink,
    }
    for name, reference in activations.items():
        assert_close(f"{backend_name}/{name}", getattr(backend, name)(x), reference(x), atol=2e-3, rtol=2e-3)

    a = torch.randn(200_003, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    assert_close(f"{backend_name}/dot_product", backend.dot_product(a, b), torch.dot(a.float(), b.float()), atol=1e-1, rtol=2e-3)
    matrix = torch.randn(33, 65, device="cuda", dtype=torch.float16)
    vector = torch.randn(65, device="cuda", dtype=torch.float16)
    assert_close(f"{backend_name}/gemv", backend.gemv(matrix, vector), matrix @ vector, atol=8e-2, rtol=2e-2)
    right = torch.randn(65, 47, device="cuda", dtype=torch.float16)
    assert_close(f"{backend_name}/gemm", backend.gemm(matrix, right), matrix @ right, atol=1e-1, rtol=2e-2)

    normalized = torch.randn(17, 65, device="cuda")
    assert_close(f"{backend_name}/softmax", backend.softmax(normalized), torch.softmax(normalized, -1))
    assert_close(f"{backend_name}/layer_norm", backend.layer_norm(normalized, 1.2, -0.3), F.layer_norm(normalized, (65,)) * 1.2 - 0.3)
    assert_close(f"{backend_name}/rms_norm", backend.rms_norm(normalized, 1.2), F.rms_norm(normalized, (65,), eps=1e-5) * 1.2)
    assert_close(f"{backend_name}/reduce_sum", backend.reduce_sum(normalized), normalized.sum())
    integers = torch.randint(-4, 5, (10003,), device="cuda", dtype=torch.int8)
    assert_close(f"{backend_name}/reduce_sum_int8", backend.reduce_sum(integers), torch.sum(integers, dtype=torch.int32))
    for dtype_name in ("float8_e4m3fn", "float8_e5m2"):
        dtype = getattr(torch, dtype_name, None)
        if dtype is not None:
            float8_values = torch.randn(4099, device="cuda").to(dtype)
            assert_close(f"{backend_name}/reduce_sum_{dtype_name}", backend.reduce_sum(float8_values), float8_values.float().sum(), atol=1e-2, rtol=1e-2)

    training_input = torch.randn(23, 65, device="cuda", dtype=torch.float16, requires_grad=True)
    training_weight = torch.randn(65, device="cuda", dtype=torch.float16, requires_grad=True)
    training_bias = torch.randn(65, device="cuda", dtype=torch.float16, requires_grad=True)
    grad_output = torch.randn_like(training_input)
    training_output = backend.layer_norm_affine(training_input, training_weight, training_bias)
    reference_input = training_input.detach().clone().requires_grad_()
    reference_weight = training_weight.detach().clone().requires_grad_()
    reference_bias = training_bias.detach().clone().requires_grad_()
    reference_output = F.layer_norm(reference_input, (65,), reference_weight, reference_bias, 1e-5)
    assert_close(f"{backend_name}/layer_norm_affine", training_output, reference_output, atol=3e-3, rtol=3e-3)
    training_output.backward(grad_output)
    reference_output.backward(grad_output)
    assert_close(f"{backend_name}/layer_norm_affine_dx", training_input.grad, reference_input.grad, atol=5e-3, rtol=5e-3)
    assert_close(f"{backend_name}/layer_norm_affine_dw", training_weight.grad, reference_weight.grad, atol=1e-2, rtol=1e-2)
    assert_close(f"{backend_name}/layer_norm_affine_db", training_bias.grad, reference_bias.grad, atol=1e-2, rtol=1e-2)
    mean = reference_input.detach().float().mean(dim=-1)
    rstd = torch.rsqrt(reference_input.detach().float().var(dim=-1, correction=0) + 1e-5)
    explicit_dx, explicit_dw, explicit_db = backend.layer_norm_backward(grad_output, reference_input.detach(), reference_weight.detach(), mean, rstd)
    assert_close(f"{backend_name}/layer_norm_backward_dx", explicit_dx, reference_input.grad, atol=5e-3, rtol=5e-3)
    assert_close(f"{backend_name}/layer_norm_backward_dw", explicit_dw, reference_weight.grad, atol=1e-2, rtol=1e-2)
    assert_close(f"{backend_name}/layer_norm_backward_db", explicit_db, reference_bias.grad, atol=1e-2, rtol=1e-2)

    indices = torch.tensor([3, 1, 4], device="cuda", dtype=torch.int64)
    weight = torch.randn(8, 17, device="cuda")
    assert_close(f"{backend_name}/embedding", backend.embedding(indices, weight), F.embedding(indices, weight))
    assert_close(f"{backend_name}/matrix_transpose", backend.matrix_transpose(normalized), normalized.T.contiguous())
    rope_input = torch.randn(11, 32, device="cuda")
    pairs = rope_input.view(11, 16, 2)
    positions = torch.arange(11, device="cuda")[:, None]
    dimensions = torch.arange(16, device="cuda")[None, :]
    angles = positions * (10000.0 ** (-2 * dimensions / 32))
    rope_reference = torch.stack((pairs[..., 0] * angles.cos() - pairs[..., 1] * angles.sin(), pairs[..., 0] * angles.sin() + pairs[..., 1] * angles.cos()), -1).flatten(1)
    assert_close(f"{backend_name}/rope", backend.rope(rope_input), rope_reference)
    values = torch.tensor([0, 1, 1, 4, 4, 4], device="cuda", dtype=torch.int32)
    assert_close(f"{backend_name}/histogram", backend.histogram(values, 5), torch.bincount(values, minlength=5).to(torch.int32))

    query = torch.randn(2, 3, 77, 64, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    for causal in (False, True):
        reference = F.scaled_dot_product_attention(query, key, value, is_causal=causal)
        assert_close(f"{backend_name}/flash_attention/causal={causal}", backend.flash_attention(query, key, value, causal=causal), reference, atol=6e-2, rtol=3e-2)

    tokens, heads, head_dim = 13, 4, 32
    prefix_output = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.float16)
    suffix_output = torch.randn_like(prefix_output)
    prefix_lse = torch.randn(heads, tokens, device="cuda")
    suffix_lse = torch.randn_like(prefix_lse)
    expected_lse = torch.logaddexp(prefix_lse, suffix_lse)
    expected_output = (prefix_output.float() * torch.exp(prefix_lse - expected_lse).T[..., None] + suffix_output.float() * torch.exp(suffix_lse - expected_lse).T[..., None]).half()
    output, output_lse = backend.merge_attention_states(prefix_output, prefix_lse, suffix_output, suffix_lse, return_lse=True)
    assert_close(f"{backend_name}/merge_attention_states", output, expected_output, atol=3e-3, rtol=3e-3)
    assert_close(f"{backend_name}/merge_attention_states_lse", output_lse, expected_lse)

    try:
        from torchvision.ops import nms as reference_nms
    except (ImportError, RuntimeError):
        print(f"SKIP {backend_name}/nms: torchvision unavailable")
    else:
        boxes = torch.rand(100, 4, device="cuda")
        boxes = torch.cat((torch.minimum(boxes[:, :2], boxes[:, 2:]), torch.maximum(boxes[:, :2], boxes[:, 2:])), 1).contiguous()
        scores = torch.rand(100, device="cuda")
        assert_close(f"{backend_name}/nms", backend.nms(boxes, scores, 0.5), reference_nms(boxes, scores, 0.5))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    validate("triton", triton_backend)
    validate("torch_compile", torch_compile_backend)
    print("All accelerated backend checks passed.")


if __name__ == "__main__":
    main()
