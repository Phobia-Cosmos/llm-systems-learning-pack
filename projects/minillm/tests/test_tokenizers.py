from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from minillm import CharTokenizer
from minillm.tokenizer_registry import tokenizer_from_checkpoint
from minillm.tokenizer_variants import HFByteBPETokenizer, HFTokenizerAdapter, SentencePieceTokenizer


ROOT = Path(__file__).resolve().parents[1]


class CharTokenizerTests(unittest.TestCase):
    def test_common_interface_and_legacy_checkpoint(self):
        tokenizer = CharTokenizer.from_text("ab中")

        self.assertEqual(tokenizer.decode(tokenizer.encode("a中")), "a中")
        self.assertIsNone(tokenizer.pad_token_id)
        self.assertEqual(tokenizer.token_to_id("a"), tokenizer.stoi["a"])
        self.assertEqual(tokenizer.id_to_token(tokenizer.stoi["中"]), "中")

        ragged = tokenizer.batch_encode(["a", "ab"], padding=False)
        self.assertEqual(ragged.attention_mask, [[1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "no pad token"):
            tokenizer.batch_encode(["a", "ab"], padding=True)

        restored = tokenizer_from_checkpoint({"tokenizer": tokenizer.to_dict()})
        self.assertIsInstance(restored, CharTokenizer)
        self.assertEqual(restored.encode("ab"), tokenizer.encode("ab"))

    def test_plain_text_chat_template(self):
        tokenizer = CharTokenizer.from_text("系统用户助手: 你好\n")
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "你好"}, {"role": "user", "content": "你好"}],
            add_generation_prompt=True,
        )
        self.assertEqual(prompt, "系统: 你好\n用户: 你好\n助手:")


class ByteBPETokenizerTests(unittest.TestCase):
    def test_special_tokens_batch_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "corpus.txt"
            corpus.write_text("用户: 你好\n助手: 你好\nembedding token cache", encoding="utf-8")
            tokenizer = HFByteBPETokenizer.train([corpus], vocab_size=320)

            text = "用户: embedding 🚀"
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertIsNotNone(tokenizer.bos_token_id)
            self.assertIsNotNone(tokenizer.eos_token_id)
            self.assertIsNotNone(tokenizer.pad_token_id)

            batch = tokenizer.batch_encode(["a", "abc"], padding=True)
            self.assertEqual(len(batch.input_ids[0]), len(batch.input_ids[1]))
            self.assertEqual(batch.attention_mask[0][-1], 0)

            restored = HFByteBPETokenizer.from_dict(tokenizer.to_dict())
            self.assertEqual(restored.encode(text), tokenizer.encode(text))


class HFTokenizerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer_path = ROOT / "tokenizer_variants" / "byte_bpe"

    def test_hf_adapter_checkpoint_is_self_contained(self):
        tokenizer = HFTokenizerAdapter.from_pretrained(self.tokenizer_path)
        tokenizer.tokenizer.padding_side = "left"
        tokenizer.tokenizer.truncation_side = "left"
        text = "用户: 什么是 embedding?\n助手:"

        restored = HFTokenizerAdapter.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.encode(text), tokenizer.encode(text))
        self.assertEqual(restored.decode(restored.encode(text)), text)
        self.assertEqual(restored.pad_token_id, tokenizer.pad_token_id)
        self.assertEqual(restored.padding_side, "left")
        self.assertEqual(restored.truncation_side, "left")

        padded = restored.batch_encode(["a", "abc"], padding=True)
        self.assertEqual(padded.attention_mask[0][0], 0)

    def test_chat_template_batch_and_standard_save(self):
        tokenizer = HFTokenizerAdapter.from_pretrained(self.tokenizer_path)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "解释 KV cache。"}],
            add_generation_prompt=True,
        )
        self.assertIn("<|user|>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))

        batch = tokenizer.batch_encode(["a", "longer"], padding=True, max_length=12)
        self.assertEqual([len(row) for row in batch.input_ids], [12, 12])
        self.assertEqual([len(row) for row in batch.attention_mask], [12, 12])

        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir, model_max_length=256)
            loaded = AutoTokenizer.from_pretrained(temp_dir, use_fast=True)
            self.assertEqual(loaded.model_max_length, 256)
            self.assertEqual(loaded.decode(loaded.encode("hello", add_special_tokens=False)), "hello")


class SentencePieceTokenizerTests(unittest.TestCase):
    def test_bpe_and_unigram_roundtrip_checkpoint_and_save(self):
        for model_type in ("bpe", "unigram"):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as temp_dir:
                corpus = Path(temp_dir) / "corpus.txt"
                corpus.write_text(
                    "用户: 你好 embedding 🚀\n助手: 你好\n" * 20,
                    encoding="utf-8",
                )
                tokenizer = SentencePieceTokenizer.train(
                    [corpus],
                    model_type=model_type,
                    vocab_size=320,
                )
                text = "用户: 未见字符 🚀"
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), text)
                self.assertIsNotNone(tokenizer.pad_token_id)
                self.assertIsNotNone(tokenizer.token_to_id("<|user|>"))

                restored = SentencePieceTokenizer.from_dict(tokenizer.to_dict())
                self.assertEqual(restored.encode(text), tokenizer.encode(text))

                output_dir = Path(temp_dir) / "saved"
                tokenizer.save_pretrained(output_dir)
                self.assertTrue((output_dir / "tokenizer.model").is_file())
                self.assertTrue((output_dir / "tokenizer.vocab").is_file())


if __name__ == "__main__":
    unittest.main()
