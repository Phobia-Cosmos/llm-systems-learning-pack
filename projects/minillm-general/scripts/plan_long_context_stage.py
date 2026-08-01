#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch


SPLITS = ("train", "validation", "test")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(
    dataset_dir: Path,
    checkpoint: dict,
    *,
    verify_payload_hashes: bool = True,
) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    tokenizer_path = dataset_dir / "tokenizer.json"
    if not manifest_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("long-context dataset requires manifest.json and tokenizer.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {
        "schema_version": 2,
        "dtype": "uint16",
        "layout": "fixed_records",
        "sequence_length": 8192,
        "record_length": 8193,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"long-context manifest {key} must be {value!r}")

    split_summary = {}
    for split in SPLITS:
        metadata = manifest.get("splits", {}).get(split)
        if not isinstance(metadata, dict):
            raise ValueError(f"long-context manifest is missing split {split}")
        records = int(metadata.get("records", 0))
        tokens = int(metadata.get("tokens", 0))
        payload = dataset_dir / str(metadata.get("path", f"{split}.bin"))
        if records <= 0 or tokens <= 0:
            raise ValueError(f"long-context split {split} must be non-empty")
        if tokens != records * 8193:
            raise ValueError(f"long-context split {split} has inconsistent record count")
        if not payload.is_file() or payload.stat().st_size != tokens * 2:
            raise ValueError(f"long-context split {split} payload size does not match manifest")
        if int(metadata.get("bytes", -1)) != payload.stat().st_size:
            raise ValueError(f"long-context split {split} byte count does not match manifest")
        expected_sha256 = metadata.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"long-context split {split} has no valid SHA-256")
        if verify_payload_hashes and sha256_file(payload) != expected_sha256:
            raise ValueError(f"long-context split {split} SHA-256 does not match manifest")
        split_summary[split] = {
            "records": records,
            "tokens": tokens,
            "bytes": payload.stat().st_size,
            "sha256": metadata.get("sha256"),
        }

    split_policy = manifest.get("split_policy", {})
    validation = manifest.get("validation", {})
    if split_policy.get("name") != "preserve_train_partition_permanent_holdout_v1":
        raise ValueError("long-context dataset does not preserve the permanent holdout")
    if split_policy.get("cross_role_duplicate_policy") != "error":
        raise ValueError("long-context dataset must reject train/holdout duplicates")
    if (
        validation.get("all_split_targets_met") is not True
        or validation.get("all_splits_nonempty") is not True
        or int(validation.get("cross_role_document_overlap", -1)) != 0
    ):
        raise ValueError("long-context dataset validation gates are incomplete")

    tokenizer_sha256 = sha256_file(tokenizer_path)
    manifest_tokenizer_sha256 = manifest.get("tokenizer_source", {}).get("sha256")
    if manifest_tokenizer_sha256 != tokenizer_sha256:
        raise ValueError("long-context tokenizer hash does not match its manifest")
    checkpoint_tokenizer_sha256 = checkpoint.get("tokenizer_sha256")
    if checkpoint_tokenizer_sha256 != tokenizer_sha256:
        raise ValueError("base checkpoint tokenizer does not match long-context dataset")

    config = checkpoint.get("config", {})
    if config.get("position_encoding") != "rope":
        raise ValueError("long-context continuation requires a RoPE checkpoint")
    if int(config.get("block_size", 0)) < 8192:
        raise ValueError("base checkpoint model maximum context must be at least 8192")

    return {
        "path": str(dataset_dir),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tokenizer_sha256": tokenizer_sha256,
        "splits": split_summary,
        "split_policy": split_policy,
    }


def validate_capacity(
    report_path: Path,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 2:
        raise ValueError("8K capacity report must use distributed-aware schema version 2")
    selected = next(
        (
            result
            for result in report.get("results", [])
            if int(result.get("micro_batch_size", -1)) == micro_batch_size
            and int(result.get("gradient_accumulation_steps", -1))
            == gradient_accumulation_steps
            and int(result.get("world_size", -1)) == world_size
        ),
        None,
    )
    if selected is None:
        raise ValueError("selected 8K micro/accumulation pair is absent from capacity report")
    if selected.get("status") != "success":
        raise ValueError("selected 8K capacity run did not succeed")
    if int(selected.get("steady_steps", 0)) < 10:
        raise ValueError("selected 8K capacity run has fewer than 10 steady measurements")
    if selected.get("median_tokens_per_second") is None:
        raise ValueError("selected 8K capacity run has no throughput measurement")
    expected_tokens = 8192 * micro_batch_size * gradient_accumulation_steps * world_size
    if int(selected.get("tokens_per_optimizer_step", -1)) != expected_tokens:
        raise ValueError("selected 8K capacity run has an inconsistent global token batch")
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "selected": selected,
        "recommended": report.get("recommended"),
    }


def build_plan(
    base_step: int,
    base_tokens_processed: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    stage_tokens: int,
    milestone_tokens: list[int],
) -> dict:
    if base_step < 0 or base_tokens_processed < 0:
        raise ValueError("base step and token count must be non-negative")
    if micro_batch_size <= 0 or gradient_accumulation_steps <= 0 or world_size <= 0:
        raise ValueError("micro batch, accumulation, and world size must be positive")
    if stage_tokens <= 0:
        raise ValueError("stage token budget must be positive")
    if sorted(set(milestone_tokens)) != milestone_tokens:
        raise ValueError("milestone token offsets must be unique and increasing")
    if not milestone_tokens or milestone_tokens[0] != 0:
        raise ValueError("milestones must include a zero-token baseline")
    if milestone_tokens[-1] != stage_tokens:
        raise ValueError("last milestone must equal the stage token budget")

    tokens_per_rank_step = 8192 * micro_batch_size * gradient_accumulation_steps
    if tokens_per_rank_step != 32768:
        raise ValueError("8K continuation must preserve 32,768 tokens/rank/update")
    tokens_per_step = tokens_per_rank_step * world_size
    stage_steps = math.ceil(stage_tokens / tokens_per_step)
    stage_end_step = base_step + stage_steps
    milestones = []
    seen_steps = set()
    for requested_tokens in milestone_tokens:
        if requested_tokens < 0 or requested_tokens > stage_tokens:
            raise ValueError("milestone token offset is outside the stage budget")
        step_offset = math.ceil(requested_tokens / tokens_per_step)
        target_step = base_step + step_offset
        if target_step in seen_steps:
            raise ValueError("two milestone token offsets round to the same optimizer step")
        seen_steps.add(target_step)
        actual_tokens = step_offset * tokens_per_step
        milestones.append(
            {
                "requested_additional_tokens": requested_tokens,
                "target_step": target_step,
                "actual_additional_tokens": actual_tokens,
                "cumulative_tokens_processed": base_tokens_processed + actual_tokens,
            }
        )
    return {
        "base_step": base_step,
        "base_tokens_processed": base_tokens_processed,
        "world_size": world_size,
        "tokens_per_rank_optimizer_step": tokens_per_rank_step,
        "tokens_per_optimizer_step": tokens_per_step,
        "stage_tokens_requested": stage_tokens,
        "stage_steps": stage_steps,
        "stage_start_step": base_step,
        "stage_end_step": stage_end_step,
        "milestones": milestones,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and plan an 8K continuation stage.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--capacity-report", required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--stage-tokens", type=int, default=1_000_000_000)
    parser.add_argument(
        "--milestone-tokens",
        type=int,
        nargs="+",
        default=[0, 100_000_000, 250_000_000, 500_000_000, 1_000_000_000],
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument(
        "--skip-payload-hashes",
        action="store_true",
        help="Use only after a persisted stage plan has already recorded a full payload check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.base_checkpoint).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    capacity_report = Path(args.capacity_report).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    plan = build_plan(
        base_step=int(checkpoint["step"]),
        base_tokens_processed=int(checkpoint.get("tokens_processed", 0)),
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=args.world_size,
        stage_tokens=args.stage_tokens,
        milestone_tokens=args.milestone_tokens,
    )
    payload = {
        "schema_version": 2,
        "base_checkpoint": {
            "path": str(checkpoint_path),
            "step": int(checkpoint["step"]),
            "tokens_processed": int(checkpoint.get("tokens_processed", 0)),
            "sha256": sha256_file(checkpoint_path) if args.hash_checkpoint else None,
        },
        "dataset": validate_dataset(
            dataset_dir,
            checkpoint,
            verify_payload_hashes=not args.skip_payload_hashes,
        ),
        "capacity": validate_capacity(
            capacity_report,
            args.micro_batch_size,
            args.gradient_accumulation_steps,
            args.world_size,
        ),
        **plan,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
