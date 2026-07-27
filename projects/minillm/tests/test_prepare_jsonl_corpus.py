from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_jsonl_corpus import (
    PrepareConfig,
    choose_split,
    normalize_text,
    prepare_corpus,
)


class PrepareJsonlCorpusTests(unittest.TestCase):
    def test_normalization_and_group_split_are_stable(self):
        self.assertEqual(normalize_text("  A\r\nＢ  ", "NFC"), "A\nＢ")
        self.assertEqual(normalize_text("  A\r\nＢ  ", "NFKC"), "A\nB")
        first = choose_split(
            "document-42",
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        second = choose_split(
            "document-42",
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "validation", "test"})

    def test_streaming_dedup_split_manifest_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            records = [
                {"text": f"文档 {index:03d} 的可重复训练内容。", "source": f"source-{index // 2}"}
                for index in range(100)
            ]
            lines = [json.dumps(record, ensure_ascii=False) for record in records]
            lines.extend(
                [
                    json.dumps(records[0], ensure_ascii=False),
                    "{not-json",
                    json.dumps({"other": "missing text"}, ensure_ascii=False),
                    json.dumps({"text": 123}, ensure_ascii=False),
                    json.dumps({"text": "短"}, ensure_ascii=False),
                ]
            )
            source.write_text("\n".join(lines) + "\n", encoding="utf-8")

            def run(output_name: str):
                return prepare_corpus(
                    PrepareConfig(
                        inputs=(source,),
                        output_dir=root / output_name,
                        group_field="source",
                        validation_fraction=0.2,
                        test_fraction=0.2,
                        min_chars=4,
                        write_jsonl=True,
                    )
                )

            first = run("prepared-a")
            second = run("prepared-b")

            stats = first["stats"]
            self.assertEqual(stats["lines_seen"], 105)
            self.assertEqual(stats["accepted"], 100)
            self.assertEqual(stats["duplicate"], 1)
            self.assertEqual(stats["invalid_json"], 1)
            self.assertEqual(stats["missing_text"], 1)
            self.assertEqual(stats["non_string_text"], 1)
            self.assertEqual(stats["too_short"], 1)
            self.assertEqual(
                sum(item["records"] for item in stats["splits"].values()),
                100,
            )

            for split in ("train", "validation", "test"):
                first_text = (root / "prepared-a" / f"{split}.txt").read_bytes()
                second_text = (root / "prepared-b" / f"{split}.txt").read_bytes()
                self.assertEqual(first_text, second_text)
                self.assertEqual(
                    first["outputs"][f"{split}.txt"]["sha256"],
                    second["outputs"][f"{split}.txt"]["sha256"],
                )

            split_by_source: dict[str, str] = {}
            for split in ("train", "validation", "test"):
                with (root / "prepared-a" / f"{split}.jsonl").open(encoding="utf-8") as handle:
                    texts = [json.loads(line)["text"] for line in handle]
                for text in texts:
                    index = int(text.split()[1])
                    source_id = f"source-{index // 2}"
                    previous = split_by_source.setdefault(source_id, split)
                    self.assertEqual(previous, split)

            manifest = json.loads(
                (root / "prepared-a" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["algorithm"]["tokenizer_training_input"], "train.txt only")
            self.assertEqual(manifest["stats"]["accepted"], 100)

    def test_existing_output_and_invalid_fractions_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text('{"text":"足够长的文本"}\n', encoding="utf-8")
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_corpus(
                    PrepareConfig(inputs=(source,), output_dir=existing, min_chars=1)
                )
            with self.assertRaises(ValueError):
                prepare_corpus(
                    PrepareConfig(
                        inputs=(source,),
                        output_dir=root / "invalid",
                        validation_fraction=0.6,
                        test_fraction=0.4,
                        min_chars=1,
                    )
                )


if __name__ == "__main__":
    unittest.main()
