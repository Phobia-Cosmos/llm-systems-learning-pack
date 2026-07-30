import json
import subprocess
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path

from tokenizers import pre_tokenizers, trainers


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_tokenizer_candidates import PROBES, SPECIAL_TOKENS, new_tokenizer  # noqa: E402


class TokenizerLabTests(unittest.TestCase):
    def test_modern_byte_bpe_is_byte_complete_and_lossless(self):
        tokenizer = new_tokenizer()
        training_documents = list(PROBES) * 20 + [
            f"训练样本 {index} English code_{index} = value + {index}; العربية हिन्दी"
            for index in range(200)
        ]
        tokenizer.train_from_iterator(
            training_documents,
            trainer=trainers.BpeTrainer(
                vocab_size=1024,
                min_frequency=1,
                special_tokens=list(SPECIAL_TOKENS),
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                max_token_length=64,
                show_progress=False,
            ),
        )
        self.assertEqual(
            set(pre_tokenizers.ByteLevel.alphabet()) - set(tokenizer.get_vocab()),
            set(),
        )
        unk_id = tokenizer.token_to_id("<|unk|>")
        for probe in PROBES:
            encoding = tokenizer.encode(probe, add_special_tokens=False)
            self.assertNotIn(unk_id, encoding.ids)
            self.assertEqual(
                tokenizer.decode(encoding.ids, skip_special_tokens=False),
                unicodedata.normalize("NFC", probe),
            )

    def test_balanced_corpus_builder_writes_manifest_and_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps(
                        {"text": f"文档 {index} contains enough English and multilingual tokenizer material."},
                        ensure_ascii=False,
                    )
                    + "\n"
                    for index in range(500)
                ),
                encoding="utf-8",
            )
            config = {
                "schema_version": 1,
                "normalization": "NFC",
                "minimum_characters": 32,
                "maximum_characters_per_document": 1024,
                "validation_modulus": 10,
                "validation_buckets": 2,
                "sources": [
                    {
                        "name": "fixture",
                        "domain": "fixture-domain",
                        "paths": [str(source)],
                        "text_fields": ["text"],
                        "target_train_characters": 5000,
                        "sample_modulus": 1,
                        "sample_buckets": 1,
                    }
                ],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "corpus"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "prepare_tokenizer_lab_corpus.py"),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(manifest["sources"]["fixture"]["train_characters"], 5000)
            self.assertGreater(manifest["sources"]["fixture"]["validation_documents"], 0)
            self.assertGreater((output / "train.jsonl").stat().st_size, 0)
            self.assertGreater((output / "validation.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
