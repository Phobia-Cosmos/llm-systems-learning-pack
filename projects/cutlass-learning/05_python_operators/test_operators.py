from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from benchmark import adjusted_tolerance, compare_outputs
from operators import CASES
from triton_ops import fused_bias_silu_dispatch, fused_bias_silu_eligibility, triton_fused_bias_silu


class PythonOperatorTests(unittest.TestCase):
    def test_all_teaching_operators_match_cpu_references(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        for index, case in enumerate(CASES):
            with self.subTest(operator=case.name):
                torch.manual_seed(20260716 + index)
                inputs = case.make_inputs(device, dtype, "smoke")
                with torch.inference_mode():
                    expected = case.pytorch_reference(*inputs)
                    actual = case.teaching_python(*inputs)
                atol, rtol = adjusted_tolerance(case, dtype)
                correct, _error, detail = compare_outputs(expected, actual, atol=atol, rtol=rtol)
                self.assertTrue(correct, detail)

    def test_triton_dispatch_falls_back_on_cpu(self) -> None:
        x = torch.randn(4, 8)
        bias = torch.randn(8)
        eligibility = fused_bias_silu_eligibility(x, bias)
        self.assertFalse(eligibility.supported)
        torch.testing.assert_close(fused_bias_silu_dispatch(x, bias), F.silu(x + bias))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_triton_fused_bias_silu_matches_pytorch(self) -> None:
        x = torch.randn(129, 257, device="cuda", dtype=torch.float16)
        bias = torch.randn(257, device="cuda", dtype=torch.float16)
        eligibility = fused_bias_silu_eligibility(x, bias)
        self.assertTrue(eligibility.supported, eligibility.reason)
        expected = F.silu(x + bias)
        actual = triton_fused_bias_silu(x, bias)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_triton_dispatch_falls_back_for_autograd(self) -> None:
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        bias = torch.randn(8, device="cuda", requires_grad=True)
        eligibility = fused_bias_silu_eligibility(x, bias)
        self.assertFalse(eligibility.supported)
        output = fused_bias_silu_dispatch(x, bias)
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)


if __name__ == "__main__":
    unittest.main()
