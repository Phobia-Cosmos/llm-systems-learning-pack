import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


def scheduler_config(policy: str = "prefill_first", num_blocks: int = 8):
    return SimpleNamespace(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=num_blocks,
        scheduling_policy=policy,
    )


class SamplerTests(unittest.TestCase):

    def test_zero_temperature_is_represented_by_params_and_sequence(self):
        params = SamplingParams(temperature=0, max_tokens=4)
        seq = Sequence([1, 2], params)

        self.assertTrue(params.is_greedy)
        self.assertTrue(seq.is_greedy)
        self.assertEqual(seq.temperature, 0)

    def test_negative_and_non_finite_temperatures_are_rejected(self):
        for temperature in (-0.1, float("inf"), float("nan")):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    SamplingParams(temperature=temperature)

    def test_all_greedy_equals_argmax_and_skips_random_path(self):
        sampler = Sampler()
        logits = torch.tensor([[0.1, 3.0, 2.0], [7.0, -1.0, 4.0]])

        with patch.object(
            sampler,
            "_sample_random",
            side_effect=AssertionError("random path must not run"),
        ):
            actual = sampler(logits, temperatures=None, greedy_mask=True)

        torch.testing.assert_close(actual, logits.argmax(dim=-1))

    def test_model_runner_all_greedy_skips_temperature_transfer(self):
        runner = object.__new__(ModelRunner)
        seqs = [
            Sequence([1], SamplingParams(temperature=0)),
            Sequence([2], SamplingParams(temperature=0)),
        ]

        with patch(
            "nanovllm.engine.model_runner.torch.tensor",
            side_effect=AssertionError("temperature tensor must not be created"),
        ):
            temperatures, greedy_mask = runner.prepare_sample(seqs)

        self.assertIsNone(temperatures)
        self.assertIs(greedy_mask, True)

    def test_positive_temperature_uses_stochastic_path(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        temperatures = torch.tensor([0.7])
        expected = torch.tensor([1])

        with patch.object(sampler, "_sample_random", return_value=expected) as random_sample:
            actual = sampler(logits, temperatures)

        torch.testing.assert_close(actual, expected)
        random_sample.assert_called_once()

    def test_random_implementation_can_select_more_than_argmax(self):
        torch.manual_seed(7)
        logits = torch.zeros(2048, 2)
        temperatures = torch.ones(2048)

        tokens = Sampler._sample_random_impl(logits, temperatures)

        self.assertEqual(set(tokens.tolist()), {0, 1})

    def test_mixed_batch_only_sends_positive_temperature_rows_to_random_path(self):
        sampler = Sampler()
        logits = torch.tensor(
            [
                [0.0, 5.0, 1.0],
                [4.0, 3.0, 2.0],
                [1.0, 2.0, 8.0],
            ]
        )
        temperatures = torch.tensor([0.0, 0.8, 0.0])
        mixed_mask = torch.tensor([True, False, True])

        with patch.object(
            sampler, "_sample_random", return_value=torch.tensor([2])
        ) as random_sample:
            actual = sampler(logits, temperatures, mixed_mask)

        torch.testing.assert_close(actual, torch.tensor([1, 2, 2]))
        sampled_logits, sampled_temperatures = random_sample.call_args.args
        torch.testing.assert_close(sampled_logits, logits[1:2])
        torch.testing.assert_close(sampled_temperatures, temperatures[1:2])


class SchedulerPolicyTests(unittest.TestCase):

    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 256

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    @staticmethod
    def add_waiting_and_running(scheduler: Scheduler):
        waiting = Sequence([1, 2, 3])
        running = Sequence([4, 5])
        running.status = SequenceStatus.RUNNING
        running.is_prefill = False
        scheduler.block_manager.allocate(running, num_cached_blocks=0)
        scheduler.waiting.append(waiting)
        scheduler.running.append(running)
        return waiting, running

    def test_prefill_first_preserves_existing_phase_choice(self):
        scheduler = Scheduler(scheduler_config("prefill_first"))
        waiting, _ = self.add_waiting_and_running(scheduler)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.scheduling_policy, "prefill_first")
        self.assertEqual(snapshot.prefill_batches, 1)
        self.assertEqual(snapshot.decode_batches, 0)
        self.assertEqual(snapshot.prefill_tokens, 3)

    def test_decode_first_selects_running_sequence_before_waiting_prompt(self):
        scheduler = Scheduler(scheduler_config("decode-first"))
        waiting, running = self.add_waiting_and_running(scheduler)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])
        self.assertEqual(list(scheduler.waiting), [waiting])
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.scheduling_policy, "decode_first")
        self.assertEqual(snapshot.prefill_batches, 0)
        self.assertEqual(snapshot.decode_batches, 1)
        self.assertEqual(snapshot.decode_tokens, 1)

    def test_missing_policy_field_defaults_to_prefill_first(self):
        config = scheduler_config()
        del config.scheduling_policy
        scheduler = Scheduler(config)
        waiting, _ = self.add_waiting_and_running(scheduler)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])

    def test_preemption_and_actual_recompute_are_counted(self):
        scheduler = Scheduler(scheduler_config())
        seq = Sequence([10, 11, 12])
        seq.status = SequenceStatus.RUNNING
        scheduler.block_manager.allocate(seq, num_cached_blocks=0)
        scheduler.running.append(seq)

        scheduler.preempt(scheduler.running.pop())
        before_recompute = scheduler.snapshot()
        self.assertEqual(before_recompute.preemptions, 1)
        self.assertEqual(before_recompute.recompute_sequences, 0)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.total_batches, 1)
        self.assertEqual(snapshot.prefill_batches, 1)
        self.assertEqual(snapshot.preemptions, 1)
        self.assertEqual(snapshot.recompute_sequences, 1)
        self.assertEqual(snapshot.recompute_batches, 1)
        self.assertEqual(snapshot.recomputed_tokens, 3)


if __name__ == "__main__":
    unittest.main()
