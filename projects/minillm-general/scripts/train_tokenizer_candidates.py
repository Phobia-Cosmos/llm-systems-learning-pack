#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Iterator

# Hugging Face tokenizers uses Rayon and otherwise detects every host core.
# Keep login-node experiments bounded and reproducible.
os.environ.setdefault("RAYON_NUM_THREADS", "8")

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


SPECIAL_TOKENS = (
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|repo_name|>",
    "<|file_sep|>",
)

# This follows the public Llama 3/tiktoken design: contractions, letter runs,
# 1-3 digit groups, punctuation/newline runs, and whitespace are segmented
# before byte-level BPE learns merges.
PRETOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

PROBES = (
    "中文 English العربية हिन्दी 日本語 한국어 Русский español",
    "def f(x: list[int]) -> int:\n    return sum(v * v for v in x)\n",
    "∂ℒ/∂W = Xᵀ(Ŷ − Y), 0xDEADBEEF, 2026-07-29T19:45:00+08:00",
    "Emoji: 👩🏽‍💻🚀🧠; URL: https://例子.测试/a?q=hello%20world#片段",
    "Syriac: ܐܒܓ; NKo: ߒߞߏ; controls: A\x00B\x04C\x0bD\ra",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reproducible byte-complete tokenizer candidates.")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        help="NAME:VOCAB_SIZE; repeat for multiple candidates (default: openbpe-32k and openbpe-48k).",
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-token-length", type=int, default=64)
    return parser.parse_args()


def parse_candidates(values: list[str] | None) -> list[tuple[str, int]]:
    values = values or ["openbpe-32k:32768", "openbpe-48k:49152"]
    candidates: list[tuple[str, int]] = []
    for value in values:
        try:
            name, raw_size = value.rsplit(":", 1)
            size = int(raw_size)
        except ValueError as exc:
            raise ValueError(f"invalid candidate {value!r}; expected NAME:VOCAB_SIZE") from exc
        if not name or not 512 <= size <= 65535:
            raise ValueError(f"invalid candidate {value!r}; vocab must be in [512, 65535]")
        candidates.append((name, size))
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    return candidates


def training_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            payload = json.loads(line)
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError(f"missing text at {path}:{line_number}")
            yield text


def new_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKEN_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_candidate(
    *,
    name: str,
    vocab_size: int,
    train_path: Path,
    corpus_manifest: dict,
    output_root: Path,
    min_frequency: int,
    max_token_length: int,
) -> dict:
    final_dir = output_root / name
    if (final_dir / "manifest.json").is_file():
        print(f"already complete: {final_dir}", flush=True)
        return json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    if final_dir.exists():
        raise FileExistsError(f"candidate output exists without a manifest: {final_dir}")
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=output_root))
    started = time.perf_counter()
    try:
        tokenizer = new_tokenizer()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            max_token_length=max_token_length,
            show_progress=True,
        )
        tokenizer.train_from_iterator(training_texts(train_path), trainer=trainer)
        actual_vocab_size = tokenizer.get_vocab_size()
        if actual_vocab_size != vocab_size:
            raise RuntimeError(f"{name} reached vocab size {actual_vocab_size}, expected {vocab_size}")
        missing_bytes = sorted(set(pre_tokenizers.ByteLevel.alphabet()) - set(tokenizer.get_vocab()))
        if missing_bytes:
            raise RuntimeError(f"{name} is missing {len(missing_bytes)} byte alphabet symbols")
        for probe in PROBES:
            expected = unicodedata.normalize("NFC", probe)
            encoding = tokenizer.encode(probe, add_special_tokens=False)
            decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
            if tokenizer.token_to_id("<|unk|>") in encoding.ids or decoded != expected:
                raise RuntimeError(f"{name} failed byte-complete probe: {probe!r} -> {decoded!r}")

        tokenizer_path = temporary_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        write_json(
            temporary_dir / "tokenizer_config.json",
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 32768,
                "padding_side": "right",
                "truncation_side": "right",
                "clean_up_tokenization_spaces": False,
                "bos_token": "<|bos|>",
                "eos_token": "<|eos|>",
                "unk_token": "<|unk|>",
                "pad_token": "<|pad|>",
                "additional_special_tokens": list(SPECIAL_TOKENS[4:]),
            },
        )
        write_json(
            temporary_dir / "special_tokens_map.json",
            {
                "bos_token": "<|bos|>",
                "eos_token": "<|eos|>",
                "unk_token": "<|unk|>",
                "pad_token": "<|pad|>",
                "additional_special_tokens": list(SPECIAL_TOKENS[4:]),
            },
        )
        elapsed = time.perf_counter() - started
        manifest = {
            "schema_version": 1,
            "name": name,
            "algorithm": "byte-level-bpe",
            "normalization": "NFC",
            "pretoken_pattern": PRETOKEN_PATTERN,
            "vocab_size": actual_vocab_size,
            "min_frequency": min_frequency,
            "max_token_length": max_token_length,
            "special_token_ids": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
            "byte_alphabet_size": len(pre_tokenizers.ByteLevel.alphabet()),
            "corpus_manifest_sha256": hashlib.sha256(
                (json.dumps(corpus_manifest, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            ).hexdigest(),
            "training_seconds": elapsed,
            "tokenizer_json_bytes": tokenizer_path.stat().st_size,
            "tokenizer_json_sha256": sha256(tokenizer_path),
        }
        write_json(temporary_dir / "manifest.json", manifest)
        os.replace(temporary_dir, final_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir).expanduser().resolve()
    train_path = corpus_dir / "train.jsonl"
    corpus_manifest_path = corpus_dir / "manifest.json"
    if not train_path.is_file() or not corpus_manifest_path.is_file():
        raise FileNotFoundError(f"incomplete tokenizer corpus: {corpus_dir}")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    expected_train_hash = corpus_manifest["outputs"]["train"]["sha256"]
    actual_train_hash = sha256(train_path)
    if actual_train_hash != expected_train_hash:
        raise RuntimeError(f"training corpus hash mismatch: {actual_train_hash} != {expected_train_hash}")

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, vocab_size in parse_candidates(args.candidate):
        results.append(
            train_candidate(
                name=name,
                vocab_size=vocab_size,
                train_path=train_path,
                corpus_manifest=corpus_manifest,
                output_root=output_root,
                min_frequency=args.min_frequency,
                max_token_length=args.max_token_length,
            )
        )
    write_json(output_root / "manifest.json", {"schema_version": 1, "candidates": results})


if __name__ == "__main__":
    main()
