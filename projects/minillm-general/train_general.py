from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from minillm import GPTConfig, MiniGPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the general-purpose MiniLLM on packed tokens.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint path, or 'auto' for OUT/latest.pt.")
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--batch-layout",
        choices=("random", "contiguous", "records"),
        default="random",
        help="Contiguous groups reduce random reads from network-backed packed files.",
    )
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Tokens per training sample; defaults to --block-size and may be shorter.",
    )
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--warmup-start-lr-ratio",
        type=float,
        default=0.0,
        help="Learning-rate ratio at the start of this schedule stage.",
    )
    parser.add_argument(
        "--schedule-start-step",
        type=int,
        default=0,
        help="Global optimizer step where this stage-local warmup/cosine schedule starts.",
    )
    parser.add_argument(
        "--schedule-end-step",
        type=int,
        default=None,
        help="Global step where this schedule reaches min LR; defaults to --max-steps.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--allow-context-extension",
        action="store_true",
        help="Resume a RoPE checkpoint with only block_size increased.",
    )
    parser.add_argument(
        "--allow-dataset-change",
        action="store_true",
        help="Allow continued pretraining on a different manifest with the same tokenizer.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute block activations during backward to fit larger models.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=0,
        help="Keep only the newest N step checkpoints; 0 keeps all.",
    )
    parser.add_argument(
        "--initial-tokens-processed",
        type=int,
        default=None,
        help="Explicit cumulative token count for resuming a legacy checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build the model and report parameters only.")
    return parser.parse_args()


class PackedTokens:
    def __init__(self, path: Path, block_size: int, record_length: int | None = None):
        self.path = path
        self.array = np.memmap(path, mode="r", dtype=np.uint16)
        if len(self.array) < block_size + 1:
            raise ValueError(f"{path} has too few tokens for block size {block_size}")
        self.block_size = block_size
        self.record_length = record_length
        if record_length is not None:
            if record_length != block_size + 1:
                raise ValueError("fixed record length must equal block_size + 1")
            if len(self.array) % record_length != 0:
                raise ValueError(f"{path} is not aligned to fixed-length records")

    def batch(
        self,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
        layout: str = "random",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layout == "contiguous":
            span = batch_size * (self.block_size + 1)
            if span > len(self.array):
                raise ValueError(f"{self.path} has too few tokens for a contiguous batch")
            start = int(
                torch.randint(
                    0,
                    len(self.array) - span + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            # One contiguous slice minimizes random I/O on network-backed storage.
            values = np.asarray(self.array[start : start + span], dtype=np.int64)
            windows = torch.from_numpy(values.reshape(batch_size, self.block_size + 1))
        elif layout == "records":
            if self.record_length is None:
                raise ValueError("records layout requires a fixed-record dataset manifest")
            record_count = len(self.array) // self.record_length
            record_indices = torch.randint(
                0,
                record_count,
                (batch_size,),
                generator=generator,
            ).numpy()
            records = self.array.reshape(record_count, self.record_length)
            windows = torch.from_numpy(records[record_indices].astype(np.int64, copy=False))
        elif layout == "random":
            starts = torch.randint(
                0,
                len(self.array) - self.block_size,
                (batch_size,),
                generator=generator,
            )
            # PyTorch 2.6 does not implement CPU advanced indexing for UInt16.
            indices = starts.numpy()[:, None] + np.arange(
                self.block_size + 1, dtype=np.int64
            )[None, :]
            windows = torch.from_numpy(self.array[indices].astype(np.int64, copy=False))
        else:
            raise ValueError(f"unknown batch layout: {layout}")
        if device.type == "cuda":
            windows = windows.pin_memory().to(device, non_blocking=True)
        else:
            windows = windows.to(device)
        return windows[:, :-1], windows[:, 1:]


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("general training requires a CUDA GPU")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def learning_rate(step: int, args: argparse.Namespace) -> float:
    schedule_end = args.max_steps if args.schedule_end_step is None else args.schedule_end_step
    relative_step = max(0, step - args.schedule_start_step)
    if relative_step < args.warmup_steps:
        progress = (relative_step + 1) / max(1, args.warmup_steps)
        ratio = args.warmup_start_lr_ratio + (1.0 - args.warmup_start_lr_ratio) * progress
        return args.learning_rate * ratio
    decay_steps = max(1, schedule_end - args.schedule_start_step - args.warmup_steps)
    progress = min(1.0, (relative_step - args.warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.learning_rate * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)


def resume_config_matches(
    checkpoint_config: dict,
    current_config: dict,
    allow_context_extension: bool,
) -> bool:
    if checkpoint_config == current_config:
        return True
    if not allow_context_extension:
        return False
    old_block_size = int(checkpoint_config.get("block_size", 0))
    comparable_checkpoint = dict(checkpoint_config)
    comparable_current = dict(current_config)
    comparable_checkpoint.pop("block_size", None)
    comparable_current.pop("block_size", None)
    return (
        checkpoint_config.get("position_encoding") == "rope"
        and current_config.get("position_encoding") == "rope"
        and int(current_config.get("block_size", 0)) >= old_block_size
        and comparable_checkpoint == comparable_current
    )


def nats_per_byte(loss_per_token: float, tokens: int, utf8_bytes: int) -> float | None:
    if tokens <= 0 or utf8_bytes <= 0:
        return None
    return loss_per_token * tokens / utf8_bytes


def optimizer_for(model: MiniGPT, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    kwargs = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "weight_decay": args.weight_decay,
    }
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        **kwargs,
    )


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_latest_symlink(checkpoint_path: Path, latest_path: Path) -> None:
    temporary = latest_path.with_suffix(latest_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint_path.name)
    os.replace(temporary, latest_path)


def checkpoint_payload(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    args: argparse.Namespace,
    step: int,
    batch_generator: torch.Generator,
    tokens_processed: int,
    dataset_manifest_sha256: str,
    tokenizer_sha256: str,
) -> dict:
    return {
        "schema_version": 1,
        "step": step,
        "tokens_processed": tokens_processed,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "batch_generator_state": batch_generator.get_state(),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: PackedTokens,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 100_000 + rank)
    total = torch.zeros((), device=device)
    for _ in range(args.eval_batches):
        inputs, targets = dataset.batch(
            args.micro_batch_size, generator, device, args.batch_layout
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("model did not return validation loss")
        total += loss.detach()
    count = torch.tensor(float(args.eval_batches), device=device)
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    return (total / count).item()


def main() -> None:
    args = parse_args()
    if args.sequence_length is None:
        args.sequence_length = args.block_size
    if args.schedule_end_step is None:
        args.schedule_end_step = args.max_steps
    if not 0 <= args.warmup_start_lr_ratio <= 1:
        raise ValueError("--warmup-start-lr-ratio must be in [0, 1]")
    if not 0 < args.min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must be in (0, 1]")
    if args.schedule_start_step < 0:
        raise ValueError("--schedule-start-step must be non-negative")
    if args.schedule_end_step <= args.schedule_start_step:
        raise ValueError("--schedule-end-step must be greater than --schedule-start-step")
    if args.max_steps > args.schedule_end_step:
        raise ValueError("--max-steps must not exceed --schedule-end-step")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.keep_checkpoints < 0:
        raise ValueError("--keep-checkpoints must be non-negative")
    if args.initial_tokens_processed is not None and args.initial_tokens_processed < 0:
        raise ValueError("--initial-tokens-processed must be non-negative")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.sequence_length <= 0 or args.sequence_length > args.block_size:
        raise ValueError("--sequence-length must be in [1, block_size]")
    rank, local_rank, world_size, device = distributed_context()
    primary = rank == 0
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = dataset_dir / "manifest.json"
    tokenizer_path = dataset_dir / "tokenizer.json"
    manifest_bytes = manifest_path.read_bytes()
    tokenizer_bytes = tokenizer_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    dataset_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()
    if manifest["dtype"] != "uint16":
        raise ValueError("this trainer currently requires a uint16 packed dataset")

    config = GPTConfig(
        vocab_size=int(manifest["vocab_size"]),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        num_key_value_heads=args.num_key_value_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=False,
        position_encoding="rope",
        rope_theta=1_000_000.0,
        norm_type="rmsnorm",
        norm_eps=1e-6,
        mlp_type="swiglu",
        activation="silu",
        intermediate_size=args.intermediate_size,
        qk_norm=True,
        use_sdpa=True,
        scale_residual_init=True,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = MiniGPT(config).to(device)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    parameter_count = model.parameter_count()
    if primary:
        print(json.dumps({"config": asdict(config), "parameters": parameter_count}, ensure_ascii=False))
    if args.dry_run:
        if world_size > 1:
            dist.destroy_process_group()
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if primary:
        shutil.copy2(tokenizer_path, out_dir / "tokenizer.json")
        (out_dir / "resolved_config.json").write_text(
            json.dumps(
                {
                    "model": asdict(config),
                    "training": vars(args),
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "tokenizer_sha256": tokenizer_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    optimizer = optimizer_for(model, args)
    batch_generator = torch.Generator().manual_seed(args.seed + rank)
    start_step = 0
    tokens_processed_at_start = args.initial_tokens_processed or 0
    resume_path: Path | None = None
    if args.resume:
        resume_path = out_dir / "latest.pt" if args.resume == "auto" else Path(args.resume).expanduser().resolve()
    if resume_path is not None and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint["config"]
        current_config = asdict(config)
        config_matches = resume_config_matches(
            checkpoint_config,
            current_config,
            args.allow_context_extension,
        )
        if not config_matches:
            raise ValueError("resume checkpoint model config does not match current arguments")
        checkpoint_tokenizer_sha256 = checkpoint.get("tokenizer_sha256")
        if (
            checkpoint_tokenizer_sha256 is not None
            and checkpoint_tokenizer_sha256 != tokenizer_sha256
        ):
            raise ValueError("resume checkpoint tokenizer does not match the packed dataset")
        checkpoint_manifest_sha256 = checkpoint.get("dataset_manifest_sha256")
        if (
            checkpoint_manifest_sha256 is not None
            and checkpoint_manifest_sha256 != dataset_manifest_sha256
            and not args.allow_dataset_change
        ):
            raise ValueError(
                "resume checkpoint dataset does not match; use --allow-dataset-change "
                "for an intentional continued-pretraining stage"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if "tokens_processed" in checkpoint:
            tokens_processed_at_start = int(checkpoint["tokens_processed"])
        elif args.initial_tokens_processed is None:
            prior_args = checkpoint.get("args", {})
            tokens_processed_at_start = (
                start_step
                * int(prior_args.get("micro_batch_size", args.micro_batch_size))
                * int(
                    prior_args.get(
                        "sequence_length",
                        prior_args.get("block_size", args.sequence_length),
                    )
                )
                * int(
                    prior_args.get(
                        "gradient_accumulation_steps",
                        args.gradient_accumulation_steps,
                    )
                )
            )
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if primary:
            batch_generator.set_state(checkpoint["batch_generator_state"])
        else:
            batch_generator.manual_seed(args.seed + rank + start_step * world_size)
        if primary:
            print(f"resumed {resume_path} at optimizer step {start_step}")

    record_length = (
        int(manifest["record_length"])
        if manifest.get("layout") == "fixed_records"
        else None
    )
    if args.batch_layout == "records" and record_length is None:
        raise ValueError("--batch-layout records requires manifest layout=fixed_records")
    if record_length is not None and args.batch_layout != "records":
        raise ValueError("fixed-record datasets require --batch-layout records")
    train_data = PackedTokens(
        dataset_dir / "train.bin",
        args.sequence_length,
        record_length,
    )
    validation_data = PackedTokens(
        dataset_dir / "validation.bin",
        args.sequence_length,
        record_length,
    )
    train_model: torch.nn.Module = torch.compile(model) if args.compile else model
    if world_size > 1:
        train_model = DistributedDataParallel(train_model, device_ids=[local_rank])

    tokens_per_step = (
        args.micro_batch_size
        * args.sequence_length
        * args.gradient_accumulation_steps
        * world_size
    )
    if primary:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "tokens_per_optimizer_step": tokens_per_step,
                    "micro_batch_size": args.micro_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "train_tokens": len(train_data.array),
                    "validation_tokens": len(validation_data.array),
                    "sequence_length": args.sequence_length,
                    "model_max_context": args.block_size,
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "tokenizer_sha256": tokenizer_sha256,
                    "bf16": True,
                    "compile": args.compile,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "batch_layout": args.batch_layout,
                    "tokens_processed_at_start": tokens_processed_at_start,
                }
            )
        )

    train_model.train()
    running_loss = 0.0
    interval_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, args.max_steps):
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for micro_step in range(args.gradient_accumulation_steps):
            inputs, targets = train_data.batch(
                args.micro_batch_size,
                batch_generator,
                device,
                args.batch_layout,
            )
            synchronize = micro_step + 1 == args.gradient_accumulation_steps
            sync_context = (
                contextlib.nullcontext()
                if synchronize or not isinstance(train_model, DistributedDataParallel)
                else train_model.no_sync()
            )
            with sync_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = train_model(inputs, targets)
                if loss is None:
                    raise RuntimeError("model did not return training loss")
                scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            running_loss += loss.detach().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1
        tokens_processed = tokens_processed_at_start + (
            completed_step - start_step
        ) * tokens_per_step

        if completed_step % args.log_interval == 0:
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - interval_started
            mean_loss = torch.tensor(
                running_loss / (args.log_interval * args.gradient_accumulation_steps),
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss /= world_size
            if primary:
                print(
                    json.dumps(
                        {
                            "step": completed_step,
                            "tokens_processed": tokens_processed,
                            "train_loss": mean_loss.item(),
                            "lr": lr,
                            "grad_norm": float(grad_norm),
                            "tokens_per_second": args.log_interval * tokens_per_step / elapsed,
                            "gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                        }
                    ),
                    flush=True,
                )
            running_loss = 0.0
            interval_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)

        if completed_step % args.eval_interval == 0 or completed_step == args.max_steps:
            validation_loss = evaluate(
                train_model, validation_data, args, device, rank, world_size
            )
            if primary:
                validation = {"step": completed_step, "validation_loss": validation_loss}
                validation_stats = manifest.get("splits", {}).get("validation", {})
                validation_utf8_bytes = int(validation_stats.get("utf8_bytes", 0))
                validation_tokens = int(validation_stats.get("tokens", 0))
                normalized_loss = nats_per_byte(
                    validation_loss,
                    validation_tokens,
                    validation_utf8_bytes,
                )
                if normalized_loss is not None:
                    validation["validation_nats_per_byte"] = normalized_loss
                print(json.dumps(validation), flush=True)

        should_save = completed_step % args.save_interval == 0 or completed_step == args.max_steps
        if should_save:
            if world_size > 1:
                dist.barrier()
            if primary:
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    config,
                    args,
                    completed_step,
                    batch_generator,
                    tokens_processed,
                    dataset_manifest_sha256,
                    tokenizer_sha256,
                )
                checkpoint_path = out_dir / f"step-{completed_step:08d}.pt"
                atomic_save(payload, checkpoint_path)
                atomic_latest_symlink(checkpoint_path, out_dir / "latest.pt")
                if args.keep_checkpoints > 0:
                    checkpoints = sorted(out_dir.glob("step-*.pt"))
                    for old_checkpoint in checkpoints[: -args.keep_checkpoints]:
                        old_checkpoint.unlink()
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
