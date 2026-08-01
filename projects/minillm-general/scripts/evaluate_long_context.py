#!/usr/bin/env python3
"""Reproducible long-context evaluation for MiniLLM checkpoints.

The harness intentionally keeps its dependencies small: it evaluates the
native MiniLLM model, fixed-record uint16 datasets, synthetic passkey cases,
and the model's two KV-cache implementations.  Every on-disk artifact is
hashed before use and the final JSON is committed with an atomic rename.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minillm import GPTConfig, MiniGPT  # noqa: E402


HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class VerifiedDataset:
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    tokenizer_sha256: str
    split: str
    split_path: Path
    split_sha256: str
    token_count: int
    record_length: int | None
    sequence_length: int | None


@dataclass(frozen=True)
class EvaluationBundle:
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint: dict[str, Any]
    tokenizer_path: Path
    tokenizer_sha256: str
    tokenizer: Tokenizer
    model: MiniGPT
    device: torch.device
    dtype: torch.dtype


class ParityAccumulator:
    def __init__(self, *, atol: float, rtol: float):
        self.atol = atol
        self.rtol = rtol
        self.max_absolute_error = 0.0
        self.absolute_error_sum = 0.0
        self.value_count = 0
        self.argmax_matches = 0
        self.argmax_count = 0
        self.numerical_violations = 0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(
                f"parity tensors have different shapes: {reference.shape} and {candidate.shape}"
            )
        reference_float = reference.float()
        candidate_float = candidate.float()
        difference = (candidate_float - reference_float).abs()
        self.max_absolute_error = max(
            self.max_absolute_error,
            float(difference.max().item()),
        )
        self.absolute_error_sum += float(difference.double().sum().item())
        self.value_count += difference.numel()
        tolerance = self.atol + self.rtol * reference_float.abs()
        self.numerical_violations += int((difference > tolerance).sum().item())
        reference_argmax = reference.argmax(dim=-1)
        candidate_argmax = candidate.argmax(dim=-1)
        self.argmax_matches += int((reference_argmax == candidate_argmax).sum().item())
        self.argmax_count += reference_argmax.numel()

    def result(self, *, minimum_argmax_match: float) -> dict[str, Any]:
        mean_error = self.absolute_error_sum / max(self.value_count, 1)
        argmax_fraction = self.argmax_matches / max(self.argmax_count, 1)
        numerical_close = self.numerical_violations == 0
        argmax_passed = argmax_fraction >= minimum_argmax_match
        return {
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": mean_error,
            "numerical_values": self.value_count,
            "numerical_violations": self.numerical_violations,
            "numerically_close": numerical_close,
            "argmax_matches": self.argmax_matches,
            "argmax_positions": self.argmax_count,
            "argmax_match_fraction": argmax_fraction,
            "argmax_exact": self.argmax_matches == self.argmax_count,
            "minimum_argmax_match": minimum_argmax_match,
            "passed": numerical_close and argmax_passed,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed-record long-context loss, 4K regression, passkey retrieval, "
            "and native dynamic/static KV-cache parity."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer.json; defaults to the checkpoint directory.",
    )
    parser.add_argument("--long-dataset-dir", required=True)
    parser.add_argument("--long-split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=None,
        help="Optional externally recorded checkpoint digest.",
    )
    parser.add_argument(
        "--expected-long-manifest-sha256",
        default=None,
        help="Optional externally recorded long-dataset manifest digest.",
    )
    parser.add_argument(
        "--test-records",
        type=int,
        default=0,
        help="Number of fixed test records; 0 evaluates every record.",
    )
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument(
        "--position-segments",
        type=int,
        default=8,
        help="Equal position ranges; the final range always ends at sequence_length.",
    )
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        default=256,
        help="Positions converted to float32 at once while computing token NLL.",
    )
    parser.add_argument("--regression-dataset-dir", default=None)
    parser.add_argument("--regression-split", default="test")
    parser.add_argument("--regression-sequence-length", type=int, default=None)
    parser.add_argument("--regression-records", type=int, default=16)
    parser.add_argument(
        "--expected-regression-manifest-sha256",
        default=None,
        help="Optional externally recorded old-dataset manifest digest.",
    )
    parser.add_argument(
        "--passkey-lengths",
        default=None,
        help="Comma-separated total token lengths; defaults to useful lengths up to block_size.",
    )
    parser.add_argument("--passkey-depths", default="0.1,0.5,0.9")
    parser.add_argument("--passkey", default="314159")
    parser.add_argument("--skip-passkey", action="store_true")
    parser.add_argument(
        "--parity-lengths",
        default=None,
        help="Comma-separated lengths; defaults to 32,128,512 clipped to block_size.",
    )
    parser.add_argument("--parity-chunk-size", type=int, default=64)
    parser.add_argument("--skip-cache-parity", action="store_true")
    parser.add_argument("--parity-atol", type=float, default=None)
    parser.add_argument("--parity-rtol", type=float, default=None)
    parser.add_argument("--minimum-argmax-match", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
        help="auto uses BF16 on CUDA and float32 on CPU.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value.lower())
    ):
        raise ValueError(f"{description} must be a 64-character SHA-256 digest")
    return value.lower()


def check_expected_sha256(actual: str, expected: str | None, description: str) -> None:
    if expected is None:
        return
    normalized = require_sha256(expected, f"expected {description}")
    if actual != normalized:
        raise ValueError(
            f"{description} SHA-256 mismatch: expected {normalized}, got {actual}"
        )


def read_json_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value, payload


def resolve_split_path(dataset_dir: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("dataset split path must be a non-empty string")
    path = (dataset_dir / relative_path).resolve()
    try:
        path.relative_to(dataset_dir)
    except ValueError as exc:
        raise ValueError("dataset split path escapes the dataset directory") from exc
    return path


def verify_dataset(
    dataset_dir: str | Path,
    *,
    split: str,
    tokenizer_sha256: str,
    expected_manifest_sha256: str | None = None,
    require_fixed_records: bool = False,
) -> VerifiedDataset:
    directory = Path(dataset_dir).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    tokenizer_path = directory / "tokenizer.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"dataset tokenizer does not exist: {tokenizer_path}")

    manifest, manifest_bytes = read_json_object(manifest_path, "dataset manifest")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    check_expected_sha256(
        manifest_hash,
        expected_manifest_sha256,
        "dataset manifest",
    )
    if manifest.get("dtype") != "uint16":
        raise ValueError("long-context evaluator requires a uint16 dataset")

    actual_dataset_tokenizer_hash = sha256_file(tokenizer_path)
    if actual_dataset_tokenizer_hash != tokenizer_sha256:
        raise ValueError(
            "dataset tokenizer SHA-256 does not match the checkpoint tokenizer: "
            f"{actual_dataset_tokenizer_hash} != {tokenizer_sha256}"
        )
    tokenizer_source = manifest.get("tokenizer_source")
    if tokenizer_source is not None:
        if not isinstance(tokenizer_source, dict):
            raise ValueError("manifest tokenizer_source must be an object or null")
        source_hash = require_sha256(
            tokenizer_source.get("sha256"),
            "manifest tokenizer_source.sha256",
        )
        if source_hash != tokenizer_sha256:
            raise ValueError(
                "manifest tokenizer source does not match the checkpoint tokenizer"
            )

    splits = manifest.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        raise ValueError(f"manifest does not define split {split!r}")
    split_metadata = splits[split]
    if not isinstance(split_metadata, dict):
        raise ValueError(f"manifest split {split!r} must be an object")
    split_path = resolve_split_path(directory, split_metadata.get("path"))
    if not split_path.is_file():
        raise FileNotFoundError(f"dataset split does not exist: {split_path}")
    expected_split_hash = require_sha256(
        split_metadata.get("sha256"),
        f"manifest splits.{split}.sha256",
    )
    actual_split_hash = sha256_file(split_path)
    if actual_split_hash != expected_split_hash:
        raise ValueError(
            f"dataset split {split!r} SHA-256 mismatch: "
            f"expected {expected_split_hash}, got {actual_split_hash}"
        )
    split_bytes = split_path.stat().st_size
    if split_bytes % np.dtype(np.uint16).itemsize != 0:
        raise ValueError(f"dataset split {split!r} is not aligned to uint16")
    metadata_bytes = split_metadata.get("bytes")
    if not isinstance(metadata_bytes, int) or metadata_bytes != split_bytes:
        raise ValueError(
            f"dataset split {split!r} byte count does not match its manifest"
        )
    token_count = split_bytes // np.dtype(np.uint16).itemsize
    metadata_tokens = split_metadata.get("tokens")
    if not isinstance(metadata_tokens, int) or metadata_tokens != token_count:
        raise ValueError(
            f"dataset split {split!r} token count does not match its manifest"
        )

    record_length: int | None = None
    sequence_length: int | None = None
    if manifest.get("layout") == "fixed_records":
        record_length = manifest.get("record_length")
        sequence_length = manifest.get("sequence_length")
        if (
            not isinstance(record_length, int)
            or not isinstance(sequence_length, int)
            or record_length != sequence_length + 1
            or sequence_length <= 0
        ):
            raise ValueError("fixed-record manifest has invalid sequence/record length")
        if token_count % record_length != 0:
            raise ValueError(f"dataset split {split!r} is not record-aligned")
        metadata_records = split_metadata.get("records")
        if (
            not isinstance(metadata_records, int)
            or metadata_records != token_count // record_length
        ):
            raise ValueError(
                f"dataset split {split!r} record count does not match its manifest"
            )
    elif require_fixed_records:
        raise ValueError("long-context test dataset must use layout=fixed_records")

    return VerifiedDataset(
        directory=directory,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        tokenizer_sha256=actual_dataset_tokenizer_hash,
        split=split,
        split_path=split_path,
        split_sha256=actual_split_hash,
        token_count=token_count,
        record_length=record_length,
        sequence_length=sequence_length,
    )


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    result = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]
    if device.type == "cpu" and result == torch.float16:
        raise ValueError("float16 CPU evaluation is not supported")
    return result


def load_evaluation_bundle(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path | None,
    *,
    expected_checkpoint_sha256: str | None,
    device: str | torch.device,
    dtype_name: str,
) -> EvaluationBundle:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_file}")
    checkpoint_hash = sha256_file(checkpoint_file)
    check_expected_sha256(
        checkpoint_hash,
        expected_checkpoint_sha256,
        "checkpoint",
    )
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    for key in ("config", "model", "tokenizer_sha256"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint is missing required field {key!r}")
    embedded_tokenizer_hash = require_sha256(
        checkpoint["tokenizer_sha256"],
        "checkpoint tokenizer_sha256",
    )

    tokenizer_file = (
        Path(tokenizer_path).expanduser().resolve()
        if tokenizer_path is not None
        else checkpoint_file.parent / "tokenizer.json"
    )
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"tokenizer does not exist: {tokenizer_file}")
    tokenizer_hash = sha256_file(tokenizer_file)
    if tokenizer_hash != embedded_tokenizer_hash:
        raise ValueError(
            "tokenizer SHA-256 does not match checkpoint metadata: "
            f"{tokenizer_hash} != {embedded_tokenizer_hash}"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    config = GPTConfig(**checkpoint["config"])
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            "tokenizer vocabulary size does not match checkpoint config: "
            f"{tokenizer.get_vocab_size()} != {config.vocab_size}"
        )

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but torch.cuda.is_available() is false")
    dtype = resolve_dtype(dtype_name, target_device)
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device=target_device, dtype=dtype).eval()
    return EvaluationBundle(
        checkpoint_path=checkpoint_file,
        checkpoint_sha256=checkpoint_hash,
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_file,
        tokenizer_sha256=tokenizer_hash,
        tokenizer=tokenizer,
        model=model,
        device=target_device,
        dtype=dtype,
    )


def autocast_context(bundle: EvaluationBundle):
    if bundle.device.type == "cuda" and bundle.dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=bundle.dtype)
    return contextlib.nullcontext()


def selected_indices(count: int, limit: int) -> list[int]:
    if count <= 0:
        raise ValueError("dataset contains no usable records")
    if limit < 0:
        raise ValueError("record limit must be non-negative")
    if limit == 0 or limit >= count:
        return list(range(count))
    if limit == 1:
        return [0]
    return [
        (index * (count - 1)) // (limit - 1)
        for index in range(limit)
    ]


def batches(values: Sequence[int], batch_size: int) -> Iterator[Sequence[int]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def per_token_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets do not have matching token dimensions")
    if chunk_size <= 0:
        raise ValueError("loss chunk size must be positive")
    pieces = []
    for start in range(0, targets.size(1), chunk_size):
        end = min(start + chunk_size, targets.size(1))
        chunk_logits = logits[:, start:end, :].float()
        chunk_targets = targets[:, start:end]
        nll = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_targets.reshape(-1),
            reduction="none",
        ).view(chunk_targets.shape)
        pieces.append(nll.detach().cpu())
    return torch.cat(pieces, dim=1)


def position_ranges(sequence_length: int, segment_count: int) -> list[tuple[int, int]]:
    if sequence_length <= 0 or segment_count <= 0:
        raise ValueError("sequence length and position segment count must be positive")
    segment_count = min(segment_count, sequence_length)
    boundaries = [
        (index * sequence_length) // segment_count
        for index in range(segment_count + 1)
    ]
    ranges = [
        (boundaries[index], boundaries[index + 1])
        for index in range(segment_count)
        if boundaries[index] < boundaries[index + 1]
    ]
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != sequence_length:
        raise RuntimeError("position segmentation did not cover the full sequence")
    return ranges


@torch.inference_mode()
def evaluate_fixed_record_loss(
    bundle: EvaluationBundle,
    dataset: VerifiedDataset,
    *,
    record_limit: int,
    batch_size: int,
    segment_count: int,
    loss_chunk_size: int,
) -> dict[str, Any]:
    if dataset.record_length is None or dataset.sequence_length is None:
        raise ValueError("fixed-record loss requires a fixed-record dataset")
    if dataset.sequence_length > bundle.model.config.block_size:
        raise ValueError(
            f"dataset sequence length {dataset.sequence_length} exceeds model block_size "
            f"{bundle.model.config.block_size}"
        )
    record_count = dataset.token_count // dataset.record_length
    indices = selected_indices(record_count, record_limit)
    records = np.memmap(
        dataset.split_path,
        mode="r",
        dtype=np.uint16,
        shape=(record_count, dataset.record_length),
    )
    position_sums = torch.zeros(dataset.sequence_length, dtype=torch.float64)
    examples = 0
    for batch_indices in batches(indices, batch_size):
        values = np.asarray(records[list(batch_indices)], dtype=np.int64)
        windows = torch.from_numpy(values).to(bundle.device)
        inputs = windows[:, :-1]
        targets = windows[:, 1:]
        with autocast_context(bundle):
            logits, _ = bundle.model(inputs)
        nll = per_token_nll(logits, targets, chunk_size=loss_chunk_size)
        position_sums += nll.double().sum(dim=0)
        examples += nll.size(0)
        del logits, nll, inputs, targets, windows
    position_means = position_sums / examples
    overall_loss = float(position_means.mean().item())
    segments = []
    for start, end in position_ranges(dataset.sequence_length, segment_count):
        segments.append(
            {
                "logit_position_start": start,
                "logit_position_end_exclusive": end,
                "target_position_start": start + 1,
                "target_position_end_inclusive": end,
                "tokens": examples * (end - start),
                "loss": float(position_means[start:end].mean().item()),
            }
        )
    return {
        "split": dataset.split,
        "available_records": record_count,
        "evaluated_records": examples,
        "sequence_length": dataset.sequence_length,
        "evaluated_tokens": examples * dataset.sequence_length,
        "overall_loss": overall_loss,
        "perplexity": math.exp(overall_loss),
        "position_loss_segments": segments,
        "position_coverage": {
            "start": segments[0]["logit_position_start"],
            "end_exclusive": segments[-1]["logit_position_end_exclusive"],
            "complete": (
                segments[0]["logit_position_start"] == 0
                and segments[-1]["logit_position_end_exclusive"]
                == dataset.sequence_length
            ),
        },
    }


def regression_starts(token_count: int, sequence_length: int, limit: int) -> list[int]:
    if sequence_length <= 0:
        raise ValueError("regression sequence length must be positive")
    window_length = sequence_length + 1
    available = token_count // window_length
    if available <= 0:
        raise ValueError("regression split has no complete evaluation window")
    record_indices = selected_indices(available, limit)
    return [index * window_length for index in record_indices]


@torch.inference_mode()
def evaluate_regression_loss(
    bundle: EvaluationBundle,
    dataset: VerifiedDataset,
    *,
    sequence_length: int,
    record_limit: int,
    loss_chunk_size: int,
) -> dict[str, Any]:
    if sequence_length > bundle.model.config.block_size:
        raise ValueError("regression sequence length exceeds model block_size")
    tokens = np.memmap(dataset.split_path, mode="r", dtype=np.uint16)
    starts = regression_starts(len(tokens), sequence_length, record_limit)
    loss_sum = 0.0
    loss_tokens = 0
    for start in starts:
        values = np.asarray(
            tokens[start : start + sequence_length + 1],
            dtype=np.int64,
        )
        window = torch.from_numpy(values).unsqueeze(0).to(bundle.device)
        with autocast_context(bundle):
            logits, _ = bundle.model(window[:, :-1])
        nll = per_token_nll(
            logits,
            window[:, 1:],
            chunk_size=loss_chunk_size,
        )
        loss_sum += float(nll.double().sum().item())
        loss_tokens += nll.numel()
        del logits, nll, window
    loss = loss_sum / loss_tokens
    return {
        "split": dataset.split,
        "sequence_length": sequence_length,
        "evaluated_records": len(starts),
        "evaluated_tokens": loss_tokens,
        "overall_loss": loss,
        "perplexity": math.exp(loss),
    }


def parse_int_csv(value: str, description: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{description} must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"{description} must contain positive integers")
    return list(dict.fromkeys(result))


def parse_float_csv(value: str, description: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{description} must be comma-separated numbers") from exc
    if not result:
        raise ValueError(f"{description} cannot be empty")
    return list(dict.fromkeys(result))


def default_passkey_lengths(block_size: int) -> list[int]:
    candidates = [length for length in (2048, 4096, 8192) if length <= block_size]
    if not candidates:
        candidates = [block_size]
    elif candidates[-1] != block_size:
        candidates.append(block_size)
    return candidates


def default_parity_lengths(block_size: int) -> list[int]:
    return sorted(set(min(block_size, length) for length in (32, 128, 512)))


def encode_nonempty(tokenizer: Tokenizer, text: str, description: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        raise ValueError(f"{description} encoded to no tokens")
    return ids


def repeated_to_length(values: Sequence[int], length: int) -> list[int]:
    if not values:
        raise ValueError("cannot repeat an empty token sequence")
    if length < 0:
        raise ValueError("repeated token length must be non-negative")
    copies, remainder = divmod(length, len(values))
    return list(values) * copies + list(values[:remainder])


def build_passkey_tokens(
    tokenizer: Tokenizer,
    *,
    total_length: int,
    depth: float,
    passkey: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    if not 0.0 <= depth <= 1.0:
        raise ValueError("passkey depths must be in [0, 1]")
    needle_text = f"\nRemember this fact: the secret passkey is {passkey}.\n"
    query_text = "\nQuestion: What is the secret passkey? Answer:"
    target_text = f" {passkey}"
    filler_text = (
        "This is ordinary background text with no secret answer. "
        "Continue reading the surrounding document carefully. "
    )
    needle_ids = encode_nonempty(tokenizer, needle_text, "passkey needle")
    query_ids = encode_nonempty(tokenizer, query_text, "passkey query")
    target_ids = encode_nonempty(tokenizer, target_text, "passkey target")
    filler_ids = encode_nonempty(tokenizer, filler_text, "passkey filler")
    prompt_length = total_length - len(target_ids)
    filler_length = prompt_length - len(needle_ids) - len(query_ids)
    if filler_length < 0:
        minimum = len(needle_ids) + len(query_ids) + len(target_ids)
        raise ValueError(
            f"passkey length {total_length} is too short; tokenizer requires at least {minimum}"
        )
    prefix_length = int(round(filler_length * depth))
    suffix_length = filler_length - prefix_length
    prompt_ids = (
        repeated_to_length(filler_ids, prefix_length)
        + needle_ids
        + repeated_to_length(filler_ids, suffix_length)
        + query_ids
    )
    if len(prompt_ids) + len(target_ids) != total_length:
        raise RuntimeError("synthetic passkey construction produced the wrong length")
    metadata = {
        "requested_depth": depth,
        "needle_start_token": prefix_length,
        "needle_end_token_exclusive": prefix_length + len(needle_ids),
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "actual_depth": (
            prefix_length / filler_length if filler_length > 0 else 0.0
        ),
        "target_text": target_text,
    }
    return prompt_ids, target_ids, metadata


@torch.inference_mode()
def greedy_full_forward(
    bundle: EvaluationBundle,
    prompt_ids: Sequence[int],
    token_count: int,
) -> list[int]:
    sequence = torch.tensor(
        [list(prompt_ids)],
        dtype=torch.long,
        device=bundle.device,
    )
    predicted = []
    for _ in range(token_count):
        with autocast_context(bundle):
            logits, _ = bundle.model(sequence)
        next_id = int(logits[0, -1].argmax().item())
        predicted.append(next_id)
        sequence = torch.cat(
            (
                sequence,
                torch.tensor([[next_id]], dtype=torch.long, device=bundle.device),
            ),
            dim=1,
        )
        del logits
    return predicted


@torch.inference_mode()
def evaluate_passkey(
    bundle: EvaluationBundle,
    *,
    lengths: Sequence[int],
    depths: Sequence[float],
    passkey: str,
) -> dict[str, Any]:
    cases = []
    for total_length in lengths:
        if total_length > bundle.model.config.block_size:
            raise ValueError(
                f"passkey length {total_length} exceeds model block_size "
                f"{bundle.model.config.block_size}"
            )
        for depth in depths:
            prompt_ids, target_ids, metadata = build_passkey_tokens(
                bundle.tokenizer,
                total_length=total_length,
                depth=depth,
                passkey=passkey,
            )
            combined = prompt_ids + target_ids
            inputs = torch.tensor(
                [combined[:-1]],
                dtype=torch.long,
                device=bundle.device,
            )
            with autocast_context(bundle):
                logits, _ = bundle.model(inputs)
            score_start = len(prompt_ids) - 1
            score_end = score_start + len(target_ids)
            target_logits = logits[:, score_start:score_end, :].float()
            target_tensor = torch.tensor(
                [target_ids],
                dtype=torch.long,
                device=bundle.device,
            )
            token_nll = F.cross_entropy(
                target_logits.reshape(-1, target_logits.size(-1)),
                target_tensor.reshape(-1),
                reduction="none",
            )
            teacher_predictions = target_logits.argmax(dim=-1)
            token_accuracy = float(
                (teacher_predictions == target_tensor).float().mean().item()
            )
            greedy_ids = greedy_full_forward(
                bundle,
                prompt_ids,
                len(target_ids),
            )
            target_nll = float(token_nll.mean().item())
            cases.append(
                {
                    "total_tokens": total_length,
                    **metadata,
                    "target_ids": target_ids,
                    "target_nll": target_nll,
                    "target_perplexity": math.exp(target_nll),
                    "teacher_forced_token_accuracy": token_accuracy,
                    "teacher_forced_exact": bool(
                        torch.equal(teacher_predictions, target_tensor)
                    ),
                    "greedy_ids": greedy_ids,
                    "greedy_text": bundle.tokenizer.decode(
                        greedy_ids,
                        skip_special_tokens=True,
                    ),
                    "greedy_exact": greedy_ids == target_ids,
                }
            )
            del logits, target_logits, target_tensor, token_nll, inputs
    return {
        "passkey": passkey,
        "cases": cases,
        "aggregate": {
            "cases": len(cases),
            "mean_target_nll": sum(case["target_nll"] for case in cases)
            / len(cases),
            "mean_teacher_forced_token_accuracy": sum(
                case["teacher_forced_token_accuracy"] for case in cases
            )
            / len(cases),
            "greedy_exact_cases": sum(case["greedy_exact"] for case in cases),
            "greedy_exact_fraction": sum(case["greedy_exact"] for case in cases)
            / len(cases),
        },
    }


def update_parity_in_position_chunks(
    accumulator: ParityAccumulator,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    chunk_size: int,
) -> None:
    for start in range(0, reference.size(1), chunk_size):
        end = min(start + chunk_size, reference.size(1))
        accumulator.update(
            reference[:, start:end, :],
            candidate[:, start:end, :],
        )


@torch.inference_mode()
def compare_cache_paths(
    bundle: EvaluationBundle,
    tokens: torch.Tensor,
    reference: torch.Tensor,
    *,
    chunk_size: int,
    atol: float,
    rtol: float,
    minimum_argmax_match: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dynamic_full = ParityAccumulator(atol=atol, rtol=rtol)
    static_full = ParityAccumulator(atol=atol, rtol=rtol)
    static_dynamic = ParityAccumulator(atol=atol, rtol=rtol)
    dynamic_cache = None
    with autocast_context(bundle):
        static_cache = bundle.model.allocate_static_kv_cache(
            batch_size=tokens.size(0),
            max_len=tokens.size(1),
            device=bundle.device,
            dtype=bundle.dtype,
        )
    for start in range(0, tokens.size(1), chunk_size):
        end = min(start + chunk_size, tokens.size(1))
        input_chunk = tokens[:, start:end]
        with autocast_context(bundle):
            dynamic_candidate, dynamic_cache = bundle.model.forward_with_cache(
                input_chunk,
                dynamic_cache,
            )
            static_candidate, static_cache = bundle.model.forward_with_static_cache(
                input_chunk,
                static_cache,
            )
        update_parity_in_position_chunks(
            dynamic_full,
            reference[:, start:end, :],
            dynamic_candidate,
            chunk_size,
        )
        update_parity_in_position_chunks(
            static_full,
            reference[:, start:end, :],
            static_candidate,
            chunk_size,
        )
        update_parity_in_position_chunks(
            static_dynamic,
            dynamic_candidate,
            static_candidate,
            chunk_size,
        )
        del dynamic_candidate, static_candidate

    dynamic_result = dynamic_full.result(minimum_argmax_match=minimum_argmax_match)
    dynamic_result.update(
        {
            "cache_mode": "dynamic",
            "chunk_size": chunk_size,
            "final_cache_length": dynamic_cache[0][0].size(2),
        }
    )
    static_result = static_full.result(minimum_argmax_match=minimum_argmax_match)
    static_result.update(
        {
            "cache_mode": "static",
            "chunk_size": chunk_size,
            "final_cache_length": static_cache.length,
        }
    )
    cache_result = static_dynamic.result(minimum_argmax_match=minimum_argmax_match)
    cache_result.update(
        {
            "cache_mode": "static_vs_dynamic",
            "chunk_size": chunk_size,
            "final_cache_length": static_cache.length,
        }
    )
    del dynamic_cache, static_cache
    return dynamic_result, static_result, cache_result


@torch.inference_mode()
def evaluate_cache_parity(
    bundle: EvaluationBundle,
    *,
    lengths: Sequence[int],
    chunk_size: int,
    seed: int,
    atol: float,
    rtol: float,
    minimum_argmax_match: float,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("parity chunk size must be positive")
    cases = []
    generator = torch.Generator().manual_seed(seed)
    for length in lengths:
        if length > bundle.model.config.block_size:
            raise ValueError(
                f"parity length {length} exceeds model block_size "
                f"{bundle.model.config.block_size}"
            )
        tokens = torch.randint(
            0,
            bundle.model.config.vocab_size,
            (1, length),
            generator=generator,
            dtype=torch.long,
        ).to(bundle.device)
        with autocast_context(bundle):
            reference, _ = bundle.model(tokens)
        dynamic, static, static_dynamic = compare_cache_paths(
            bundle,
            tokens,
            reference,
            chunk_size=min(chunk_size, length),
            atol=atol,
            rtol=rtol,
            minimum_argmax_match=minimum_argmax_match,
        )
        cases.append(
            {
                "sequence_length": length,
                "full_forward": {"shape": list(reference.shape)},
                "dynamic_vs_full": dynamic,
                "static_vs_full": static,
                "static_vs_dynamic": static_dynamic,
                "full_forward_consistency_passed": dynamic["passed"] and static["passed"],
                "passed": static_dynamic["passed"],
            }
        )
        del reference, tokens
    return {
        "atol": atol,
        "rtol": rtol,
        "minimum_argmax_match": minimum_argmax_match,
        "cases": cases,
        "all_passed": all(case["passed"] for case in cases),
        "all_full_forward_consistency_passed": all(
            case["full_forward_consistency_passed"] for case in cases
        ),
    }


def dataset_identity(dataset: VerifiedDataset) -> dict[str, Any]:
    return {
        "directory": str(dataset.directory),
        "manifest_sha256": dataset.manifest_sha256,
        "tokenizer_sha256": dataset.tokenizer_sha256,
        "split": dataset.split,
        "split_path": str(dataset.split_path),
        "split_sha256": dataset.split_sha256,
        "split_tokens": dataset.token_count,
        "record_length": dataset.record_length,
        "sequence_length": dataset.sequence_length,
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def validated_options(args: argparse.Namespace, block_size: int, dtype: torch.dtype):
    if args.test_records < 0:
        raise ValueError("--test-records must be non-negative")
    if args.test_batch_size <= 0:
        raise ValueError("--test-batch-size must be positive")
    if args.position_segments <= 0:
        raise ValueError("--position-segments must be positive")
    if args.loss_chunk_size <= 0:
        raise ValueError("--loss-chunk-size must be positive")
    if args.regression_records <= 0:
        raise ValueError("--regression-records must be positive")
    if not 0.0 <= args.minimum_argmax_match <= 1.0:
        raise ValueError("--minimum-argmax-match must be in [0, 1]")
    passkey_lengths = (
        parse_int_csv(args.passkey_lengths, "passkey lengths")
        if args.passkey_lengths
        else default_passkey_lengths(block_size)
    )
    passkey_depths = parse_float_csv(args.passkey_depths, "passkey depths")
    if any(not 0.0 <= depth <= 1.0 for depth in passkey_depths):
        raise ValueError("passkey depths must be in [0, 1]")
    parity_lengths = (
        parse_int_csv(args.parity_lengths, "parity lengths")
        if args.parity_lengths
        else default_parity_lengths(block_size)
    )
    default_tolerance = 0.05 if dtype in (torch.bfloat16, torch.float16) else 2e-5
    atol = default_tolerance if args.parity_atol is None else args.parity_atol
    rtol = default_tolerance if args.parity_rtol is None else args.parity_rtol
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("parity tolerances must be non-negative")
    return passkey_lengths, passkey_depths, parity_lengths, atol, rtol


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_evaluation_bundle(
        args.checkpoint,
        args.tokenizer,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        device=args.device,
        dtype_name=args.dtype,
    )
    (
        passkey_lengths,
        passkey_depths,
        parity_lengths,
        parity_atol,
        parity_rtol,
    ) = validated_options(args, bundle.model.config.block_size, bundle.dtype)
    long_dataset = verify_dataset(
        args.long_dataset_dir,
        split=args.long_split,
        tokenizer_sha256=bundle.tokenizer_sha256,
        expected_manifest_sha256=args.expected_long_manifest_sha256,
        require_fixed_records=True,
    )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)
    fixed_record_result = evaluate_fixed_record_loss(
        bundle,
        long_dataset,
        record_limit=args.test_records,
        batch_size=args.test_batch_size,
        segment_count=args.position_segments,
        loss_chunk_size=args.loss_chunk_size,
    )

    regression_result = None
    regression_identity = None
    if args.regression_dataset_dir is not None:
        regression_dataset = verify_dataset(
            args.regression_dataset_dir,
            split=args.regression_split,
            tokenizer_sha256=bundle.tokenizer_sha256,
            expected_manifest_sha256=args.expected_regression_manifest_sha256,
            require_fixed_records=False,
        )
        regression_sequence_length = args.regression_sequence_length
        if regression_sequence_length is None:
            training_args = bundle.checkpoint.get("args", {})
            if not isinstance(training_args, dict):
                training_args = {}
            regression_sequence_length = int(
                training_args.get(
                    "sequence_length",
                    min(4096, bundle.model.config.block_size),
                )
            )
        regression_result = evaluate_regression_loss(
            bundle,
            regression_dataset,
            sequence_length=regression_sequence_length,
            record_limit=args.regression_records,
            loss_chunk_size=args.loss_chunk_size,
        )
        regression_identity = dataset_identity(regression_dataset)

    passkey_result = None
    if not args.skip_passkey:
        passkey_result = evaluate_passkey(
            bundle,
            lengths=passkey_lengths,
            depths=passkey_depths,
            passkey=args.passkey,
        )

    parity_result = None
    if not args.skip_cache_parity:
        parity_result = evaluate_cache_parity(
            bundle,
            lengths=parity_lengths,
            chunk_size=args.parity_chunk_size,
            seed=args.seed,
            atol=parity_atol,
            rtol=parity_rtol,
            minimum_argmax_match=args.minimum_argmax_match,
        )

    runtime: dict[str, Any] = {
        "device": str(bundle.device),
        "dtype": str(bundle.dtype).removeprefix("torch."),
        "torch_version": torch.__version__,
        "seed": args.seed,
    }
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)
        runtime.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(bundle.device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(bundle.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(bundle.device),
            }
        )
    current_manifest_hash = bundle.checkpoint.get("dataset_manifest_sha256")
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(bundle.checkpoint_path),
            "sha256": bundle.checkpoint_sha256,
            "step": bundle.checkpoint.get("step"),
            "tokens_processed": bundle.checkpoint.get("tokens_processed"),
            "parameters": bundle.model.parameter_count(),
            "block_size": bundle.model.config.block_size,
            "vocab_size": bundle.model.config.vocab_size,
            "dataset_manifest_sha256": current_manifest_hash,
        },
        "tokenizer": {
            "path": str(bundle.tokenizer_path),
            "sha256": bundle.tokenizer_sha256,
            "vocab_size": bundle.tokenizer.get_vocab_size(),
        },
        "long_dataset": {
            **dataset_identity(long_dataset),
            "matches_checkpoint_current_dataset": (
                isinstance(current_manifest_hash, str)
                and current_manifest_hash == long_dataset.manifest_sha256
            ),
        },
        "fixed_record_test": fixed_record_result,
        "regression_dataset": regression_identity,
        "regression_loss": regression_result,
        "passkey_retrieval": passkey_result,
        "cache_parity": parity_result,
        "runtime": runtime,
    }
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    atomic_write_json(args.output, result)
    summary = {
        "output": str(Path(args.output).expanduser().resolve()),
        "long_loss": result["fixed_record_test"]["overall_loss"],
        "cache_parity": (
            result["cache_parity"]["all_passed"]
            if result["cache_parity"] is not None
            else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
