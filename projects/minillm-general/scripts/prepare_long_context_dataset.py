#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
from array import array
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack coherent fixed-length long-context records.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument("--batch-documents", type=int, default=128)
    return parser.parse_args()


class RecordWriter:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("wb")
        self.sha256 = hashlib.sha256()
        self.records = 0
        self.tokens = 0
        self.utf8_bytes = 0

    def write(self, token_ids: list[int], utf8_bytes: int) -> None:
        values = array("H", token_ids)
        if os.sys.byteorder != "little":
            values.byteswap()
        payload = values.tobytes()
        self.handle.write(payload)
        self.sha256.update(payload)
        self.records += 1
        self.tokens += len(token_ids)
        self.utf8_bytes += utf8_bytes

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def choose_split(digest: bytes, validation_fraction: float, test_fraction: float) -> str:
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + validation_fraction:
        return "validation"
    return "train"


def documents(paths: list[Path]):
    seen: set[bytes] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                text = payload.get("text")
                source = payload.get("source")
                if not isinstance(text, str) or not isinstance(source, str):
                    continue
                text = unicodedata.normalize(
                    "NFC",
                    text.replace("\r\n", "\n").replace("\r", "\n").strip(),
                )
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                yield source, text, digest


def main() -> None:
    args = parse_args()
    paths = [Path(value).expanduser().resolve() for value in args.input]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("one or more long-context inputs do not exist")
    if not 0 <= args.validation_fraction < 1 or not 0 <= args.test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("validation + test fractions must be less than one")

    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    targets_path = Path(args.targets).expanduser().resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    targets_payload = json.loads(targets_path.read_text(encoding="utf-8"))
    sequence_length = int(targets_payload["sequence_length"])
    record_length = sequence_length + 1
    train_targets = {
        source: int(tokens)
        for source, tokens in targets_payload["target_train_tokens_by_source"].items()
    }
    train_fraction = 1.0 - args.validation_fraction - args.test_fraction
    split_targets = {
        source: {
            "train": tokens,
            "validation": math.ceil(tokens * args.validation_fraction / train_fraction),
            "test": math.ceil(tokens * args.test_fraction / train_fraction),
        }
        for source, tokens in train_targets.items()
    }
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not define <|eos|>")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    writers = {split: RecordWriter(temporary_dir / f"{split}.bin") for split in SPLITS}
    source_stats = defaultdict(
        lambda: {
            "documents": 0,
            "long_documents": 0,
            "records_by_split": {split: 0 for split in SPLITS},
            "tokens_by_split": {split: 0 for split in SPLITS},
        }
    )
    pending = []

    def flush() -> None:
        if not pending:
            return
        encodings = tokenizer.encode_batch([text for _, text, _ in pending], add_special_tokens=False)
        for (source, text, digest), encoding in zip(pending, encodings):
            stats = source_stats[source]
            stats["documents"] += 1
            token_ids = encoding.ids + [eos_id]
            if len(token_ids) < record_length:
                continue
            stats["long_documents"] += 1
            split = choose_split(digest, args.validation_fraction, args.test_fraction)
            if stats["tokens_by_split"][split] >= split_targets[source][split]:
                continue
            available_records = len(token_ids) // record_length
            remainder = len(token_ids) - available_records * record_length
            offset = int.from_bytes(digest[8:16], "big") % (remainder + 1)
            for record_index in range(available_records):
                if stats["tokens_by_split"][split] >= split_targets[source][split]:
                    break
                start = offset + record_index * record_length
                record = token_ids[start : start + record_length]
                record_utf8_bytes = len(
                    tokenizer.decode(record, skip_special_tokens=True).encode("utf-8")
                )
                writers[split].write(record, record_utf8_bytes)
                stats["records_by_split"][split] += 1
                stats["tokens_by_split"][split] += record_length
        pending.clear()

    try:
        for source, text, digest in documents(paths):
            if source not in train_targets:
                continue
            if all(
                source_stats[source]["tokens_by_split"][split]
                >= split_targets[source][split]
                for split in SPLITS
            ):
                continue
            pending.append((source, text, digest))
            if len(pending) >= args.batch_documents:
                flush()
        flush()
        for writer in writers.values():
            writer.close()
        shortages = {
            source: {
                split: target - source_stats[source]["tokens_by_split"][split]
                for split, target in targets.items()
                if source_stats[source]["tokens_by_split"][split] < target
            }
            for source, targets in split_targets.items()
        }
        shortages = {source: values for source, values in shortages.items() if values}
        if shortages:
            raise RuntimeError(f"insufficient long-context records: {shortages}")
        shutil.copy2(tokenizer_path, temporary_dir / "tokenizer.json")
        manifest = {
            "schema_version": 1,
            "dtype": "uint16",
            "layout": "fixed_records",
            "sequence_length": sequence_length,
            "record_length": record_length,
            "target_tokens_by_source_and_split": split_targets,
            "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_source": {
                "path": str(tokenizer_path),
                "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            },
            "targets_source": {
                "path": str(targets_path),
                "sha256": hashlib.sha256(targets_path.read_bytes()).hexdigest(),
            },
            "inputs": [{"path": str(path), "bytes": path.stat().st_size} for path in paths],
            "splits": {
                split: {
                    "records": writer.records,
                    "tokens": writer.tokens,
                    "utf8_bytes": writer.utf8_bytes,
                    "path": writer.path.name,
                    "bytes": writer.path.stat().st_size,
                    "sha256": writer.sha256.hexdigest(),
                }
                for split, writer in writers.items()
            },
            "sources": dict(source_stats),
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
    except BaseException:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.handle.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
