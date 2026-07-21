from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(params=["triton", "torch_compile"])
def backend(request):
    if request.param == "triton":
        import kernels.triton_backend as implementation
    else:
        import kernels.torch_compile_backend as implementation
    return implementation


def assert_close(actual, expected, *, atol=1e-3, rtol=1e-3):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_elementwise_and_activations(backend):
    torch.manual_seed(0)
    x = torch.randn(257, device="cuda")
    y = torch.randn_like(x)
    assert_close(backend.elementwise_add(x, y), x + y)
    references = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "elu": F.elu,
        "gelu": lambda value: F.gelu(value, approximate="tanh"),
        "swish": F.silu,
        "hardswish": F.hardswish,
        "hardshrink": F.hardshrink,
    }
    for name, reference in references.items():
        assert_close(getattr(backend, name)(x), reference(x), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_linear_algebra(backend, dtype):
    torch.manual_seed(1)
    a = torch.randn(4099, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    assert_close(backend.dot_product(a, b), torch.dot(a.float(), b.float()), atol=5e-2, rtol=2e-2)

    matrix = torch.randn(33, 65, device="cuda", dtype=dtype)
    vector = torch.randn(65, device="cuda", dtype=dtype)
    assert_close(backend.gemv(matrix, vector), matrix @ vector, atol=8e-2, rtol=2e-2)

    right = torch.randn(65, 47, device="cuda", dtype=dtype)
    assert_close(backend.gemm(matrix, right), matrix @ right, atol=1e-1, rtol=2e-2)


def test_normalization_and_reduction(backend):
    torch.manual_seed(2)
    x = torch.randn(17, 65, device="cuda")
    assert_close(backend.softmax(x), torch.softmax(x, dim=-1))
    assert_close(
        backend.layer_norm(x, 1.2, -0.3),
        F.layer_norm(x, (65,)) * 1.2 - 0.3,
    )
    assert_close(backend.rms_norm(x, 1.2), F.rms_norm(x, (65,), eps=1e-5) * 1.2)
    assert_close(backend.reduce_sum(x), x.sum())

    integers = torch.randint(-4, 5, (10003,), device="cuda", dtype=torch.int8)
    assert_close(backend.reduce_sum(integers), torch.sum(integers, dtype=torch.int32))
    for dtype_name in ("float8_e4m3fn", "float8_e5m2"):
        dtype = getattr(torch, dtype_name, None)
        if dtype is not None:
            float8_values = torch.randn(4099, device="cuda").to(dtype)
            assert_close(backend.reduce_sum(float8_values), float8_values.float().sum(), atol=1e-2, rtol=1e-2)


def test_layer_norm_affine_backward(backend):
    torch.manual_seed(7)
    x = torch.randn(23, 65, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(65, device="cuda", dtype=torch.float16, requires_grad=True)
    bias = torch.randn(65, device="cuda", dtype=torch.float16, requires_grad=True)
    grad_output = torch.randn_like(x)
    actual = backend.layer_norm_affine(x, weight, bias)

    reference_x = x.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()
    expected = F.layer_norm(reference_x, (65,), reference_weight, reference_bias, 1e-5)
    assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    actual.backward(grad_output)
    expected.backward(grad_output)
    assert_close(x.grad, reference_x.grad, atol=5e-3, rtol=5e-3)
    assert_close(weight.grad, reference_weight.grad, atol=1e-2, rtol=1e-2)
    assert_close(bias.grad, reference_bias.grad, atol=1e-2, rtol=1e-2)

    mean = reference_x.detach().float().mean(dim=-1)
    rstd = torch.rsqrt(reference_x.detach().float().var(dim=-1, correction=0) + 1e-5)
    grad_input, grad_weight, grad_bias = backend.layer_norm_backward(
        grad_output,
        reference_x.detach(),
        reference_weight.detach(),
        mean,
        rstd,
    )
    assert_close(grad_input, reference_x.grad, atol=5e-3, rtol=5e-3)
    assert_close(grad_weight, reference_weight.grad, atol=1e-2, rtol=1e-2)
    assert_close(grad_bias, reference_bias.grad, atol=1e-2, rtol=1e-2)


def test_indexing_and_layout(backend):
    torch.manual_seed(3)
    indices = torch.tensor([3, 1, 4], device="cuda", dtype=torch.int64)
    weight = torch.randn(8, 17, device="cuda")
    assert_close(backend.embedding(indices, weight), F.embedding(indices, weight))

    matrix = torch.randn(17, 65, device="cuda")
    assert_close(backend.matrix_transpose(matrix), matrix.T.contiguous())

    x = torch.randn(11, 32, device="cuda")
    pairs = x.view(11, 16, 2)
    positions = torch.arange(11, device="cuda")[:, None]
    pair_indices = torch.arange(16, device="cuda")[None, :]
    angles = positions * (10000.0 ** (-2 * pair_indices / 32))
    expected = torch.stack(
        (
            pairs[..., 0] * angles.cos() - pairs[..., 1] * angles.sin(),
            pairs[..., 0] * angles.sin() + pairs[..., 1] * angles.cos(),
        ),
        dim=-1,
    ).flatten(1)
    assert_close(backend.rope(x), expected)

    values = torch.tensor([0, 1, 1, 4, 4, 4], device="cuda", dtype=torch.int32)
    assert_close(
        backend.histogram(values, num_bins=5),
        torch.bincount(values, minlength=5).to(torch.int32),
    )


@pytest.mark.parametrize("causal", [False, True])
def test_attention(backend, causal):
    torch.manual_seed(4)
    query = torch.randn(2, 3, 77, 64, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = F.scaled_dot_product_attention(query, key, value, is_causal=causal)
    assert_close(
        backend.flash_attention(query, key, value, causal=causal),
        expected,
        atol=6e-2,
        rtol=3e-2,
    )


def test_merge_attention_states(backend):
    torch.manual_seed(5)
    tokens, heads, head_dim = 13, 4, 32
    prefix_output = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.float16)
    suffix_output = torch.randn_like(prefix_output)
    prefix_lse = torch.randn(heads, tokens, device="cuda")
    suffix_lse = torch.randn_like(prefix_lse)
    expected_lse = torch.logaddexp(prefix_lse, suffix_lse)
    expected = (
        prefix_output.float() * torch.exp(prefix_lse - expected_lse).T[..., None]
        + suffix_output.float() * torch.exp(suffix_lse - expected_lse).T[..., None]
    ).half()
    actual, actual_lse = backend.merge_attention_states(
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        return_lse=True,
    )
    assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    assert_close(actual_lse, expected_lse)


def test_nms(backend):
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(6)
    boxes = torch.rand(100, 4, device="cuda")
    boxes = torch.cat(
        (torch.minimum(boxes[:, :2], boxes[:, 2:]), torch.maximum(boxes[:, :2], boxes[:, 2:])),
        dim=1,
    ).contiguous()
    scores = torch.rand(100, device="cuda")
    expected = torchvision.ops.nms(boxes, scores, 0.5)
    assert_close(backend.nms(boxes, scores, 0.5), expected)
