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
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack coherent fixed-length long-context records.")
    parser.add_argument(
        "--train-input",
        action="append",
        required=True,
        help="JSONL already assigned to the permanent training split. Repeat for shards.",
    )
    parser.add_argument(
        "--holdout-input",
        action="append",
        required=True,
        help=(
            "JSONL already assigned to a permanent holdout. Documents from this role are "
            "deterministically partitioned between validation and test and can never enter train."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--batch-documents", type=int, default=128)
    parser.add_argument(
        "--drop-cross-role-duplicates",
        action="store_true",
        help=(
            "When a normalized document is present in both roles, drop the holdout copy "
            "and record it in the manifest. The default is to fail closed."
        ),
    )
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


def choose_holdout_split(digest: bytes, test_fraction: float) -> str:
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "test" if bucket < test_fraction else "validation"


def documents(paths_by_role: dict[str, list[Path]], inputs: list[dict], *, drop_cross_role_duplicates: bool = False):
    seen_by_role: dict[str, set[bytes]] = {"train": set(), "holdout": set()}
    for role in ("train", "holdout"):
        other_role = "holdout" if role == "train" else "train"
        for path in paths_by_role[role]:
            input_stats = {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": None,
                "documents": 0,
                "duplicate_documents_within_role": 0,
                "duplicate_documents_cross_role_dropped": 0,
            }
            inputs.append(input_stats)
            source_sha256 = hashlib.sha256()
            with path.open("rb") as handle:
                for line_number, line in enumerate(handle, 1):
                    source_sha256.update(line)
                    input_stats["documents"] += 1
                    try:
                        payload = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
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
                    if digest in seen_by_role[other_role]:
                        if drop_cross_role_duplicates and role == "holdout":
                            input_stats["duplicate_documents_cross_role_dropped"] += 1
                            continue
                        raise RuntimeError(
                            "permanent holdout violation: normalized document "
                            f"{digest.hex()} appears in both train and holdout inputs; "
                            f"second occurrence is {path}:{line_number}"
                        )
                    if digest in seen_by_role[role]:
                        input_stats["duplicate_documents_within_role"] += 1
                        continue
                    seen_by_role[role].add(digest)
                    yield role, source, text, digest
            input_stats["sha256"] = source_sha256.hexdigest()


def main() -> None:
    args = parse_args()
    paths_by_role = {
        "train": [Path(value).expanduser().resolve() for value in args.train_input],
        "holdout": [Path(value).expanduser().resolve() for value in args.holdout_input],
    }
    all_paths = [path for paths in paths_by_role.values() for path in paths]
    if any(not path.is_file() for path in all_paths):
        raise FileNotFoundError("one or more long-context inputs do not exist")
    train_paths = set(paths_by_role["train"])
    holdout_paths = set(paths_by_role["holdout"])
    if train_paths & holdout_paths:
        raise ValueError("the same path cannot be both --train-input and --holdout-input")
    if args.batch_documents <= 0:
        raise ValueError("--batch-documents must be positive")

    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    targets_path = Path(args.targets).expanduser().resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    targets_payload = json.loads(targets_path.read_text(encoding="utf-8"))
    if targets_payload.get("schema_version") != 2:
        raise ValueError("long-context targets require schema_version=2")
    sequence_length = int(targets_payload["sequence_length"])
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    record_length = sequence_length + 1
    holdout_test_fraction = float(targets_payload["holdout_test_fraction"])
    if not 0 < holdout_test_fraction < 1:
        raise ValueError("holdout_test_fraction must be in (0, 1)")
    raw_targets = targets_payload["target_tokens_by_source_and_split"]
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ValueError("target_tokens_by_source_and_split must be a non-empty object")
    split_targets = {}
    for source, targets in raw_targets.items():
        if not isinstance(targets, dict) or set(targets) != set(SPLITS):
            raise ValueError(f"targets for {source!r} must define exactly {SPLITS}")
        split_targets[source] = {split: int(targets[split]) for split in SPLITS}
        if any(tokens <= 0 for tokens in split_targets[source].values()):
            raise ValueError(f"all split targets for {source!r} must be positive")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not define <|eos|>")
    if tokenizer.get_vocab_size() > 1 << 16:
        raise RuntimeError("uint16 long-context records require vocab_size <= 65536")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    writers = {split: RecordWriter(temporary_dir / f"{split}.bin") for split in SPLITS}
    source_stats = defaultdict(
        lambda: {
            "documents_by_input_role": {"train": 0, "holdout": 0},
            "long_documents_by_input_role": {"train": 0, "holdout": 0},
            "records_by_split": {split: 0 for split in SPLITS},
            "tokens_by_split": {split: 0 for split in SPLITS},
        }
    )
    inputs = []
    pending = []
    # Records may span several documents. EOS tokens preserve document boundaries while
    # allowing ordinary short documents to contribute to the 8K continuation budget.
    token_buffers: dict[tuple[str, str], list[int]] = {}

    def flush() -> None:
        if not pending:
            return
        encodings = tokenizer.encode_batch(
            [text for _, _, text, _ in pending], add_special_tokens=False
        )
        for (role, source, text, digest), encoding in zip(pending, encodings):
            stats = source_stats[source]
            stats["documents_by_input_role"][role] += 1
            token_ids = encoding.ids + [eos_id]
            split = (
                "train"
                if role == "train"
                else choose_holdout_split(digest, holdout_test_fraction)
            )
            if stats["tokens_by_split"][split] >= split_targets[source][split]:
                continue
            if len(token_ids) >= record_length:
                stats["long_documents_by_input_role"][role] += 1
            buffer_key = (source, split)
            buffer = token_buffers.setdefault(buffer_key, [])
            buffer.extend(token_ids)
            while len(buffer) >= record_length:
                if stats["tokens_by_split"][split] >= split_targets[source][split]:
                    break
                record = buffer[:record_length]
                del buffer[:record_length]
                record_utf8_bytes = len(
                    tokenizer.decode(record, skip_special_tokens=True).encode("utf-8")
                )
                writers[split].write(record, record_utf8_bytes)
                stats["records_by_split"][split] += 1
                stats["tokens_by_split"][split] += record_length
        pending.clear()

    try:
        for role, source, text, digest in documents(
            paths_by_role, inputs, drop_cross_role_duplicates=args.drop_cross_role_duplicates
        ):
            if source not in split_targets:
                continue
            eligible_splits = ("train",) if role == "train" else ("validation", "test")
            if all(
                source_stats[source]["tokens_by_split"][split]
                >= split_targets[source][split]
                for split in eligible_splits
            ):
                continue
            pending.append((role, source, text, digest))
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
        if any(writer.records <= 0 for writer in writers.values()):
            raise RuntimeError("all train/validation/test splits must contain at least one record")
        shutil.copy2(tokenizer_path, temporary_dir / "tokenizer.json")
        manifest = {
            "schema_version": 2,
            "dtype": "uint16",
            "layout": "fixed_records",
            "sequence_length": sequence_length,
            "record_length": record_length,
            "split_policy": {
                "name": "preserve_train_partition_permanent_holdout_v1",
                "train_input_destination": "train only",
                "holdout_input_destinations": ["validation", "test"],
                "holdout_partition": (
                    "first 64 bits of normalized-document sha256 mapped to [0, 1); "
                    f"test when value < {holdout_test_fraction}, otherwise validation"
                ),
                "cross_role_duplicate_policy": "error",
                "verified_nonempty_splits": list(SPLITS),
            },
            "validation": {
                "all_split_targets_met": True,
                "all_splits_nonempty": True,
                "cross_role_document_overlap": 0,
                "cross_role_documents_dropped_from_holdout": sum(
                    int(item["duplicate_documents_cross_role_dropped"])
                    for item in inputs
                ),
            },
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
            "inputs": inputs,
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
