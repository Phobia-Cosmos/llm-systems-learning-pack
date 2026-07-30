#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


SPECIAL_TOKENS = ("<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>")
SPLITS = ("train", "validation", "test")


@dataclass
class SplitStats:
    documents: int = 0
    tokens: int = 0
    utf8_bytes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream JSONL into a deduplicated packed-token dataset.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="JSONL or Parquet input; repeat for shards.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Reuse an existing tokenizer.json. Required for checkpoint-compatible continued pretraining.",
    )
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--tokenizer-max-documents", type=int, default=2_000_000)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=32)
    parser.add_argument("--batch-documents", type=int, default=256)
    return parser.parse_args()


def raw_documents(path: Path, text_field: str) -> Iterator[str]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet input requires pyarrow") from exc
        parquet = pq.ParquetFile(path)
        if text_field not in parquet.schema.names:
            raise ValueError(f"missing text field {text_field!r} in {path}")
        for batch in parquet.iter_batches(batch_size=8192, columns=[text_field]):
            for text in batch.column(0).to_pylist():
                if isinstance(text, str):
                    yield text
        return

    if path.suffix.lower() not in (".jsonl", ".json"):
        raise ValueError(f"unsupported input format for {path}; expected JSONL or Parquet")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            text = payload.get(text_field)
            if isinstance(text, str):
                yield text


def normalized_documents(paths: list[Path], text_field: str, min_chars: int) -> Iterator[tuple[str, bytes]]:
    seen: set[bytes] = set()
    for path in paths:
        for text in raw_documents(path, text_field):
            text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n").strip())
            if len(text) < min_chars:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            if digest in seen:
                continue
            seen.add(digest)
            yield text, digest


def choose_split(digest: bytes, validation_fraction: float, test_fraction: float) -> str:
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + validation_fraction:
        return "validation"
    return "train"


def build_tokenizer(
    paths: list[Path],
    text_field: str,
    min_chars: int,
    vocab_size: int,
    max_documents: int,
    validation_fraction: float,
    test_fraction: float,
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        # Reserve every byte symbol even when the tokenizer-training sample
        # does not contain it. Without this, unseen scripts can emit <|unk|>
        # and fail lossless UTF-8 round trips despite using ByteLevel.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def training_documents() -> Iterator[str]:
        accepted = 0
        for text, digest in normalized_documents(paths, text_field, min_chars):
            if choose_split(digest, validation_fraction, test_fraction) != "train":
                continue
            yield text
            accepted += 1
            if max_documents > 0 and accepted >= max_documents:
                break

    tokenizer.train_from_iterator(training_documents(), trainer=trainer)
    if tokenizer.get_vocab_size() > 65535:
        raise ValueError("vocab must fit uint16; choose --vocab-size <= 65535")
    return tokenizer


class TokenWriter:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("wb")
        self.sha256 = hashlib.sha256()
        self.tokens = 0

    def write(self, token_ids: list[int]) -> None:
        values = array("H", token_ids)
        if values.itemsize != 2:
            raise RuntimeError("native unsigned short is not 16-bit")
        if os.sys.byteorder != "little":
            values.byteswap()
        payload = values.tobytes()
        self.handle.write(payload)
        self.sha256.update(payload)
        self.tokens += len(token_ids)

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def main() -> None:
    args = parse_args()
    paths = [Path(value).expanduser().resolve() for value in args.input]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    if not 0 <= args.validation_fraction < 1 or not 0 <= args.test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("validation + test fractions must be less than one")
    if args.tokenizer is None and not 260 <= args.vocab_size <= 65535:
        raise ValueError("--vocab-size must be between 260 and 65535")
    if args.batch_documents <= 0:
        raise ValueError("--batch-documents must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))

    try:
        tokenizer_source: Path | None = None
        if args.tokenizer:
            tokenizer_source = Path(args.tokenizer).expanduser().resolve()
            if not tokenizer_source.is_file():
                raise FileNotFoundError(f"tokenizer not found: {tokenizer_source}")
            tokenizer = Tokenizer.from_file(str(tokenizer_source))
        else:
            tokenizer = build_tokenizer(
                paths,
                args.text_field,
                args.min_chars,
                args.vocab_size,
                args.tokenizer_max_documents,
                args.validation_fraction,
                args.test_fraction,
            )
        if tokenizer.get_vocab_size() > 65535:
            raise ValueError("tokenizer vocab must fit uint16")
        tokenizer.save(str(temporary_dir / "tokenizer.json"))
        eos_id = tokenizer.token_to_id("<|eos|>")
        if eos_id is None:
            raise RuntimeError("tokenizer did not create an EOS token")

        writers = {name: TokenWriter(temporary_dir / f"{name}.bin") for name in SPLITS}
        stats = {name: SplitStats() for name in SPLITS}
        pending: list[tuple[str, str]] = []

        def flush_pending() -> None:
            if not pending:
                return
            encodings = tokenizer.encode_batch([text for _, text in pending], add_special_tokens=False)
            for (split, text), encoding in zip(pending, encodings):
                token_ids = encoding.ids + [eos_id]
                writers[split].write(token_ids)
                stats[split].documents += 1
                stats[split].tokens += len(token_ids)
                stats[split].utf8_bytes += len(text.encode("utf-8"))
            pending.clear()

        for text, digest in normalized_documents(paths, args.text_field, args.min_chars):
            split = choose_split(digest, args.validation_fraction, args.test_fraction)
            pending.append((split, text))
            if len(pending) >= args.batch_documents:
                flush_pending()
        flush_pending()
        for writer in writers.values():
            writer.close()

        manifest = {
            "schema_version": 1,
            "dtype": "uint16",
            "vocab_size": tokenizer.get_vocab_size(),
            "special_token_ids": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
            "inputs": [{"path": str(path), "bytes": path.stat().st_size} for path in paths],
            "tokenizer_source": (
                {
                    "path": str(tokenizer_source),
                    "sha256": hashlib.sha256(tokenizer_source.read_bytes()).hexdigest(),
                }
                if tokenizer_source is not None
                else None
            ),
            "config": vars(args),
            "splits": {
                name: {
                    **asdict(stats[name]),
                    "path": f"{name}.bin",
                    "bytes": writers[name].path.stat().st_size,
                    "sha256": writers[name].sha256.hexdigest(),
                }
                for name in SPLITS
            },
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
