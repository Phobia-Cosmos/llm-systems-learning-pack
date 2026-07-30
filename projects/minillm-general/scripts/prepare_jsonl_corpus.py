#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class PrepareConfig:
    inputs: tuple[Path, ...]
    output_dir: Path
    text_field: str = "text"
    group_field: str | None = None
    validation_fraction: float = 0.01
    test_fraction: float = 0.01
    min_chars: int = 16
    max_chars: int | None = None
    normalization: str = "NFC"
    max_records: int | None = None
    write_jsonl: bool = False
    strict: bool = False


@dataclass
class SplitStats:
    records: int = 0
    characters: int = 0
    utf8_bytes: int = 0


@dataclass
class CorpusStats:
    lines_seen: int = 0
    accepted: int = 0
    duplicate: int = 0
    invalid_json: int = 0
    missing_text: int = 0
    non_string_text: int = 0
    too_short: int = 0
    too_long: int = 0
    invalid_group: int = 0
    splits: dict[str, SplitStats] = field(
        default_factory=lambda: {name: SplitStats() for name in SPLITS}
    )


class HashedWriter:
    def __init__(self, handle: TextIO):
        self.handle = handle
        self.sha256 = hashlib.sha256()
        self.bytes_written = 0

    def write(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.handle.write(value)
        self.sha256.update(encoded)
        self.bytes_written += len(encoded)

    def manifest(self, path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "bytes": self.bytes_written,
            "sha256": self.sha256.hexdigest(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream JSONL documents into deterministic train/validation/test text files. "
            "Exact duplicates are removed before splitting."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSONL path. Repeat the flag to combine multiple shards.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--group-field",
        default=None,
        help=(
            "Optional document/source id field used as the split key. Records with the same "
            "value stay in one split; otherwise normalized content is the split key."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=16)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Reject longer documents; 0 disables this filter.",
    )
    parser.add_argument("--normalization", choices=("NFC", "NFKC", "none"), default="NFC")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Stop after this many physical JSONL lines; 0 scans all input.",
    )
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help="Also write normalized JSONL with content hashes; text files are always written.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first malformed record instead of counting and skipping it.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PrepareConfig:
    return PrepareConfig(
        inputs=tuple(Path(path).expanduser().resolve() for path in args.input),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        text_field=args.text_field,
        group_field=args.group_field,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        min_chars=args.min_chars,
        max_chars=None if args.max_chars == 0 else args.max_chars,
        normalization=args.normalization,
        max_records=None if args.max_records == 0 else args.max_records,
        write_jsonl=args.write_jsonl,
        strict=args.strict,
    )


def validate_config(config: PrepareConfig) -> None:
    if not config.inputs:
        raise ValueError("at least one --input is required")
    missing = [str(path) for path in config.inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input files do not exist: {missing}")
    if config.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {config.output_dir}; choose a new run directory"
        )
    if config.validation_fraction < 0 or config.test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if config.validation_fraction + config.test_fraction >= 1:
        raise ValueError("validation + test fractions must be less than 1")
    if config.min_chars < 0:
        raise ValueError("--min-chars must be non-negative")
    if config.max_chars is not None and config.max_chars < config.min_chars:
        raise ValueError("--max-chars must be at least --min-chars")
    if config.max_records is not None and config.max_records <= 0:
        raise ValueError("--max-records must be positive when set")
    if not config.text_field:
        raise ValueError("--text-field cannot be empty")


def normalize_text(value: str, normalization: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if normalization != "none":
        value = unicodedata.normalize(normalization, value)
    return value


def content_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def choose_split(
    split_key: str,
    *,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    digest = hashlib.sha256(split_key.encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < test_fraction:
        return "test"
    if unit < test_fraction + validation_fraction:
        return "validation"
    return "train"


def _reject(stats: CorpusStats, field_name: str, config: PrepareConfig, message: str) -> None:
    setattr(stats, field_name, getattr(stats, field_name) + 1)
    if config.strict:
        raise ValueError(message)


def _parse_record(raw_line: bytes, source: Path, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(raw_line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{source}:{line_number}: invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise TypeError(f"{source}:{line_number}: expected a JSON object")
    return record


def _open_split_writers(
    temp_dir: Path,
    *,
    write_jsonl: bool,
) -> tuple[
    dict[str, HashedWriter],
    dict[str, HashedWriter],
    list[TextIO],
]:
    handles: list[TextIO] = []
    text_writers: dict[str, HashedWriter] = {}
    jsonl_writers: dict[str, HashedWriter] = {}
    for split in SPLITS:
        text_handle = (temp_dir / f"{split}.txt").open("w", encoding="utf-8", newline="\n")
        handles.append(text_handle)
        text_writers[split] = HashedWriter(text_handle)
        if write_jsonl:
            jsonl_handle = (temp_dir / f"{split}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            )
            handles.append(jsonl_handle)
            jsonl_writers[split] = HashedWriter(jsonl_handle)
    return text_writers, jsonl_writers, handles


def prepare_corpus(config: PrepareConfig) -> dict[str, Any]:
    validate_config(config)
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.",
            dir=config.output_dir.parent,
        )
    )
    stats = CorpusStats()
    seen_content_hashes: set[bytes] = set()
    input_hashes: dict[str, str] = {}
    text_writers: dict[str, HashedWriter] = {}
    jsonl_writers: dict[str, HashedWriter] = {}
    handles: list[TextIO] = []
    try:
        text_writers, jsonl_writers, handles = _open_split_writers(
            temp_path,
            write_jsonl=config.write_jsonl,
        )
        should_stop = False
        for source in config.inputs:
            source_hash = hashlib.sha256()
            with source.open("rb") as input_file:
                for line_number, raw_line in enumerate(input_file, start=1):
                    source_hash.update(raw_line)
                    stats.lines_seen += 1
                    if not raw_line.strip():
                        _reject(
                            stats,
                            "invalid_json",
                            config,
                            f"{source}:{line_number}: blank line",
                        )
                    else:
                        try:
                            record = _parse_record(raw_line, source, line_number)
                        except (ValueError, TypeError) as exc:
                            _reject(stats, "invalid_json", config, str(exc))
                            record = None
                        if record is not None:
                            if config.text_field not in record:
                                _reject(
                                    stats,
                                    "missing_text",
                                    config,
                                    f"{source}:{line_number}: missing field {config.text_field!r}",
                                )
                            elif not isinstance(record[config.text_field], str):
                                _reject(
                                    stats,
                                    "non_string_text",
                                    config,
                                    f"{source}:{line_number}: text field is not a string",
                                )
                            else:
                                text = normalize_text(
                                    record[config.text_field],
                                    config.normalization,
                                )
                                if len(text) < config.min_chars:
                                    _reject(
                                        stats,
                                        "too_short",
                                        config,
                                        f"{source}:{line_number}: document has {len(text)} characters",
                                    )
                                elif config.max_chars is not None and len(text) > config.max_chars:
                                    _reject(
                                        stats,
                                        "too_long",
                                        config,
                                        f"{source}:{line_number}: document has {len(text)} characters",
                                    )
                                else:
                                    digest = content_digest(text)
                                    if digest in seen_content_hashes:
                                        stats.duplicate += 1
                                    else:
                                        split_key = text
                                        if config.group_field is not None:
                                            group_value = record.get(config.group_field)
                                            if not isinstance(group_value, (str, int)):
                                                _reject(
                                                    stats,
                                                    "invalid_group",
                                                    config,
                                                    (
                                                        f"{source}:{line_number}: group field "
                                                        f"{config.group_field!r} must be str or int"
                                                    ),
                                                )
                                                split_key = ""
                                            else:
                                                split_key = str(group_value)
                                        if split_key:
                                            seen_content_hashes.add(digest)
                                            split = choose_split(
                                                split_key,
                                                validation_fraction=config.validation_fraction,
                                                test_fraction=config.test_fraction,
                                            )
                                            text_writers[split].write(text + "\n\n")
                                            if config.write_jsonl:
                                                jsonl_writers[split].write(
                                                    json.dumps(
                                                        {
                                                            "text": text,
                                                            "sha256": digest.hex(),
                                                        },
                                                        ensure_ascii=False,
                                                        separators=(",", ":"),
                                                    )
                                                    + "\n"
                                                )
                                            encoded_size = len(text.encode("utf-8"))
                                            split_stats = stats.splits[split]
                                            split_stats.records += 1
                                            split_stats.characters += len(text)
                                            split_stats.utf8_bytes += encoded_size
                                            stats.accepted += 1
                    if (
                        config.max_records is not None
                        and stats.lines_seen >= config.max_records
                    ):
                        should_stop = True
                        break
            input_hashes[str(source)] = source_hash.hexdigest()
            if should_stop:
                break
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()

        outputs: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            outputs[f"{split}.txt"] = text_writers[split].manifest(
                config.output_dir / f"{split}.txt"
            )
            if config.write_jsonl:
                outputs[f"{split}.jsonl"] = jsonl_writers[split].manifest(
                    config.output_dir / f"{split}.jsonl"
                )
        manifest = {
            "schema_version": 1,
            "algorithm": {
                "normalization": config.normalization,
                "deduplication": "exact sha256 of normalized text",
                "split": "first 64 bits of sha256(group key) mapped to [0, 1)",
                "document_separator": "\\n\\n",
                "tokenizer_training_input": "train.txt only",
            },
            "config": {
                **asdict(config),
                "inputs": [str(path) for path in config.inputs],
                "output_dir": str(config.output_dir),
            },
            "input_scanned_sha256": input_hashes,
            "input_scan_complete": config.max_records is None,
            "stats": {
                **asdict(stats),
                "splits": {
                    name: asdict(split_stats)
                    for name, split_stats in stats.splits.items()
                },
            },
            "outputs": outputs,
        }
        (temp_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        temp_path.replace(config.output_dir)
        return manifest
    except Exception:
        for handle in handles:
            if not handle.closed:
                handle.close()
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def main() -> None:
    config = config_from_args(parse_args())
    manifest = prepare_corpus(config)
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))
    print(f"wrote deterministic corpus to {config.output_dir}")
    print(f"train tokenizer/model only from {config.output_dir / 'train.txt'}")


if __name__ == "__main__":
    main()
