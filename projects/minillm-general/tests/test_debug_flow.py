from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from minillm import GPTConfig, MiniGPT
from minillm.data import get_batch
from minillm.debug import split_qkv_parameters, trace_forward
from export_hf_like import pack_separate_qkv_for_export


class DebugTraceTests(unittest.TestCase):
    def make_model(self, position_encoding: str = "rope", fused_qkv: bool = True) -> MiniGPT:
        torch.manual_seed(17)
        return MiniGPT(
            GPTConfig(
                vocab_size=11,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=8,
                fused_qkv=fused_qkv,
                dropout=0.0,
                position_encoding=position_encoding,
            )
        )

    def test_trace_reconstructs_forward_qkv_mask_softmax_and_loss(self):
        input_ids = torch.tensor([[5, 8, 3, 10, 2]], dtype=torch.long)
        targets = torch.tensor([[8, 3, 10, 2, 1]], dtype=torch.long)

        for position_encoding in ("learned", "rope"):
            for fused_qkv in (True, False):
                with self.subTest(position_encoding=position_encoding, fused_qkv=fused_qkv):
                    model = self.make_model(position_encoding, fused_qkv).eval()
                    trace = trace_forward(model, input_ids, targets)

                    self.assertTrue(all(trace["checks"].values()))
                    self.assertIs(model.lm_head.weight, model.token_embedding.weight)
                    block = trace["blocks"][0]
                    self.assertEqual(tuple(block["q_flat"].shape), (1, 5, 8))
                    self.assertEqual(tuple(block["q_heads"].shape), (1, 2, 5, 4))
                    self.assertEqual(tuple(block["attention_weights"].shape), (1, 2, 5, 5))

                    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
                    self.assertEqual(torch.count_nonzero(block["attention_weights"][..., future]).item(), 0)
                    torch.testing.assert_close(
                        block["attention_weights"].sum(dim=-1),
                        torch.ones(1, 2, 5),
                    )

                    manual_loss = -F.log_softmax(trace["logits"], dim=-1).gather(
                        dim=-1,
                        index=targets.unsqueeze(-1),
                    ).mean()
                    self.assertAlmostEqual(float(manual_loss.item()), trace["loss"], places=6)

                    ln_1 = block["ln_1"]
                    qkv_parameters = split_qkv_parameters(model.blocks[0].attn)
                    for name in ("q", "k", "v"):
                        weight, bias = qkv_parameters[name]
                        expected = F.linear(ln_1, weight.detach().cpu(), None if bias is None else bias.detach().cpu())
                        torch.testing.assert_close(expected, block[f"{name}_flat"])

    def test_causal_prefix_logits_ignore_changed_future_tokens(self):
        model = self.make_model("rope").eval()
        first = torch.tensor([[5, 8, 3, 10, 2]], dtype=torch.long)
        second = torch.tensor([[5, 8, 3, 6, 7]], dtype=torch.long)

        with torch.no_grad():
            first_logits, _ = model(first)
            second_logits, _ = model(second)

        torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])
        self.assertFalse(torch.allclose(first_logits[:, 3:], second_logits[:, 3:]))

    def test_backward_and_optimizer_change_parameters_and_fixed_sample_qkv(self):
        model = self.make_model("rope")
        input_ids = torch.tensor([[5, 8, 3, 10, 2]], dtype=torch.long)
        targets = torch.tensor([[8, 3, 10, 2, 1]], dtype=torch.long)

        model.eval()
        before_trace = trace_forward(model, input_ids, targets)
        parameter = model.blocks[0].attn.c_attn.weight
        before_parameter = parameter.detach().clone()

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(input_ids, targets)
        self.assertIsNotNone(loss)
        loss.backward()

        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.norm().item(), 0.0)
        torch.testing.assert_close(parameter.detach(), before_parameter)

        optimizer.step()
        self.assertFalse(torch.equal(parameter.detach(), before_parameter))

        model.eval()
        after_trace = trace_forward(model, input_ids, targets)
        for name in ("q_flat", "k_flat", "v_flat"):
            self.assertGreater(
                (after_trace["blocks"][0][name] - before_trace["blocks"][0][name]).norm().item(),
                0.0,
            )

    def test_greedy_first_token_kv_cache_and_checkpoint_roundtrip(self):
        model = self.make_model("rope").eval()
        prompt = torch.tensor([[5, 8, 3]], dtype=torch.long)

        with torch.no_grad():
            logits, _ = model(prompt)
            expected_first = logits[:, -1].argmax(dim=-1)
            ordinary = model.generate(prompt.clone(), max_new_tokens=3, greedy=True)
            cached = model.generate_with_kv_cache(prompt.clone(), max_new_tokens=3, greedy=True)

        torch.testing.assert_close(ordinary[:, 3], expected_first)
        torch.testing.assert_close(cached, ordinary)
        self.assertTrue(((ordinary >= 0) & (ordinary < model.config.vocab_size)).all())

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            restored = self.make_model("rope").eval()
            restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            with torch.no_grad():
                restored_logits, _ = restored(prompt)
            torch.testing.assert_close(restored_logits, logits)

    def test_smallest_valid_batch_has_one_shifted_window(self):
        data = torch.arange(5, dtype=torch.long)
        x, y = get_batch(data, block_size=4, batch_size=2, device=torch.device("cpu"))

        torch.testing.assert_close(x, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
        torch.testing.assert_close(y, torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]))

    def test_separate_qkv_export_packs_without_changing_logits(self):
        model = self.make_model("rope", fused_qkv=False).eval()
        input_ids = torch.tensor([[5, 8, 3, 10, 2]], dtype=torch.long)
        with torch.no_grad():
            expected_logits, _ = model(input_ids)

        packed_model, packed_config = pack_separate_qkv_for_export(model, model.config)
        with torch.no_grad():
            packed_logits, _ = packed_model(input_ids)

        self.assertTrue(packed_config.fused_qkv)
        self.assertIsNotNone(packed_model.blocks[0].attn.c_attn)
        self.assertIsNone(packed_model.blocks[0].attn.q_proj)
        torch.testing.assert_close(packed_logits, expected_logits)


if __name__ == "__main__":
    unittest.main()
