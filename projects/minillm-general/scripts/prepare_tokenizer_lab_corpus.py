#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

os.environ.setdefault("OMP_NUM_THREADS", "4")


@dataclass
class SourceStats:
    scanned_documents: int = 0
    invalid_documents: int = 0
    short_documents: int = 0
    duplicate_documents: int = 0
    sampling_rejections: int = 0
    train_documents: int = 0
    train_characters: int = 0
    validation_documents: int = 0
    validation_characters: int = 0


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("w", encoding="utf-8")
        self.sha256 = hashlib.sha256()
        self.bytes = 0

    def write(self, payload: dict) -> None:
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.handle.write(encoded.decode("utf-8"))
        self.sha256.update(encoded)
        self.bytes += len(encoded)

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced, deterministic tokenizer-training corpus.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-shortfall", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path, text_fields: list[str]) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            for field in text_fields:
                value = payload.get(field)
                if isinstance(value, str):
                    yield value
                    break


def iter_parquet(path: Path, text_fields: list[str]) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet sources require pyarrow; use the minillm-eval-py311 environment") from exc
    parquet = pq.ParquetFile(path)
    text_field = next((field for field in text_fields if field in parquet.schema.names), None)
    if text_field is None:
        raise ValueError(f"none of text_fields={text_fields!r} exists in {path}; fields={parquet.schema.names!r}")
    for batch in parquet.iter_batches(batch_size=2048, columns=[text_field]):
        for value in batch.column(0).to_pylist():
            if isinstance(value, str):
                yield value


def iter_documents(path: Path, text_fields: list[str]) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from iter_parquet(path, text_fields)
    elif suffix in {".json", ".jsonl"}:
        yield from iter_jsonl(path, text_fields)
    else:
        raise ValueError(f"unsupported source format: {path}")


def normalize_document(text: str, maximum_characters: int) -> tuple[str, bytes]:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    if maximum_characters > 0 and len(normalized) > maximum_characters:
        span = len(normalized) - maximum_characters + 1
        offset = int.from_bytes(digest[16:24], "big") % span
        normalized = normalized[offset : offset + maximum_characters]
    return normalized, digest


def selected(digest: bytes, modulus: int, buckets: int) -> bool:
    if modulus <= 0 or not 0 < buckets <= modulus:
        raise ValueError(f"invalid deterministic sampling ratio: buckets={buckets}, modulus={modulus}")
    return int.from_bytes(digest[8:16], "big") % modulus < buckets


def is_validation(digest: bytes, modulus: int, buckets: int) -> bool:
    if modulus <= 0 or not 0 <= buckets < modulus:
        raise ValueError(f"invalid validation ratio: buckets={buckets}, modulus={modulus}")
    return int.from_bytes(digest[:8], "big") % modulus < buckets


def expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if any(character in value for character in "*?["):
            matches = sorted(path.parent.glob(path.name))
            paths.extend(matches)
        else:
            paths.append(path)
    return [path.resolve() for path in paths]


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("normalization") != "NFC":
        raise ValueError("tokenizer lab currently requires NFC normalization")
    minimum_characters = int(config["minimum_characters"])
    maximum_characters = int(config["maximum_characters_per_document"])
    validation_modulus = int(config["validation_modulus"])
    validation_buckets = int(config["validation_buckets"])

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    writers = {
        "train": JsonlWriter(temporary_dir / "train.jsonl"),
        "validation": JsonlWriter(temporary_dir / "validation.jsonl"),
    }
    seen: set[bytes] = set()
    all_stats: dict[str, SourceStats] = {}
    input_files: list[dict] = []

    try:
        for source in config["sources"]:
            name = source["name"]
            domain = source["domain"]
            paths = expand_paths(source["paths"])
            if not paths or any(not path.is_file() for path in paths):
                missing = [str(path) for path in paths if not path.is_file()] or source["paths"]
                raise FileNotFoundError(f"missing files for {name}: {missing}")
            for path in paths:
                input_files.append({"source": name, "path": str(path), "bytes": path.stat().st_size})
            target = int(source["target_train_characters"])
            stats = SourceStats()
            all_stats[name] = stats

            for path in paths:
                for raw_text in iter_documents(path, source["text_fields"]):
                    stats.scanned_documents += 1
                    if not isinstance(raw_text, str):
                        stats.invalid_documents += 1
                        continue
                    text, digest = normalize_document(raw_text, maximum_characters)
                    if len(text) < minimum_characters:
                        stats.short_documents += 1
                        continue
                    if digest in seen:
                        stats.duplicate_documents += 1
                        continue
                    if not selected(digest, int(source["sample_modulus"]), int(source["sample_buckets"])):
                        stats.sampling_rejections += 1
                        continue
                    seen.add(digest)
                    split = (
                        "validation"
                        if is_validation(digest, validation_modulus, validation_buckets)
                        else "train"
                    )
                    writers[split].write({"source": name, "domain": domain, "text": text})
                    if split == "train":
                        stats.train_documents += 1
                        stats.train_characters += len(text)
                    else:
                        stats.validation_documents += 1
                        stats.validation_characters += len(text)
                    if stats.train_characters >= target:
                        break
                if stats.train_characters >= target:
                    break

            print(
                f"{name}: train_chars={stats.train_characters:,}/{target:,} "
                f"validation_chars={stats.validation_characters:,} scanned={stats.scanned_documents:,}",
                flush=True,
            )
            if stats.train_characters < target and not args.allow_shortfall:
                raise RuntimeError(f"{name} did not reach target characters: {stats.train_characters} < {target}")

        for writer in writers.values():
            writer.close()
        manifest = {
            "schema_version": 1,
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "normalization": "NFC",
            "inputs": input_files,
            "outputs": {
                split: {
                    "path": writer.path.name,
                    "bytes": writer.bytes,
                    "sha256": writer.sha256.hexdigest(),
                }
                for split, writer in writers.items()
            },
            "sources": {name: asdict(stats) for name, stats in all_stats.items()},
            "unique_documents": len(seen),
        }
        write_manifest(temporary_dir / "manifest.json", manifest)
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
