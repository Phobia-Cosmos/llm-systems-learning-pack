#!/usr/bin/env python3
"""Verify the main RMSNorm paper ideas in a modern PyTorch environment.

This is not a reproduction of the original WMT/CNN/COCO/CIFAR experiments.
Those code paths depend on Python 2, TensorFlow 1.x, Theano, and Lasagne.
Instead, this script keeps the paper's core intervention:

    replace LayerNorm's mean-centering + variance scaling with RMS scaling.

It reports:
1. Invariance checks: rescaling invariance is preserved, recentering is not.
2. A small training check: RMSNorm and pRMSNorm are compared with LayerNorm
   and no normalization on the same synthetic deep residual MLP task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rmsnorm_torch import RMSNorm  # noqa: E402


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_norm(kind: str, hidden_size: int, eps: float) -> nn.Module:
    if kind == "none":
        return IdentityNorm()
    if kind == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps)
    if kind == "rmsnorm":
        return RMSNorm(hidden_size, eps=eps)
    if kind == "prmsnorm":
        return RMSNorm(hidden_size, p=0.25, eps=eps)
    raise ValueError("unknown norm kind: {}".format(kind))


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, norm_kind: str, eps: float):
        super().__init__()
        self.norm = make_norm(norm_kind, hidden_size, eps)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc2(F.gelu(self.fc1(y)))
        return x + 0.25 * y


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int,
        norm_kind: str,
        eps: float,
    ):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_size, norm_kind, eps) for _ in range(depth)]
        )
        self.final_norm = make_norm(norm_kind, hidden_size, eps)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(self.final_norm(x))


@dataclass
class TrainResult:
    norm: str
    initial_loss: float
    final_loss: float
    final_accuracy: float
    ms_per_step: float


def make_teacher_labels(
    x: torch.Tensor,
    hidden_size: int,
    output_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    w1 = torch.randn(x.size(-1), hidden_size, generator=generator, device=x.device)
    b1 = torch.randn(hidden_size, generator=generator, device=x.device) * 0.1
    w2 = torch.randn(hidden_size, output_size, generator=generator, device=x.device)
    logits = torch.tanh(x @ w1 / (x.size(-1) ** 0.5) + b1) @ w2
    return logits.argmax(dim=-1)


def train_once(
    norm_kind: str,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> TrainResult:
    torch.manual_seed(args.seed)
    model = ResidualMLP(
        input_size=x.size(-1),
        hidden_size=args.hidden_size,
        output_size=args.classes,
        depth=args.depth,
        norm_kind=norm_kind,
        eps=args.eps,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    with torch.no_grad():
        initial_loss = F.cross_entropy(model(x), y).item()

    start = time.perf_counter()
    for step in range(args.steps):
        idx = torch.randint(0, x.size(0), (args.batch_size,), device=args.device)
        logits = model(x[idx])
        loss = F.cross_entropy(logits, y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        logits = model(x)
        final_loss = F.cross_entropy(logits, y).item()
        final_accuracy = (logits.argmax(dim=-1) == y).float().mean().item()

    return TrainResult(
        norm=norm_kind,
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        ms_per_step=elapsed * 1000.0 / args.steps,
    )


def invariance_report(hidden_size: int, eps: float, device: str) -> dict[str, float]:
    torch.manual_seed(123)
    x = torch.randn(32, hidden_size, device=device)
    alpha = 7.0
    shift = 3.0

    layernorm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False).to(device)
    rmsnorm = RMSNorm(hidden_size, eps=eps).to(device)
    prmsnorm = RMSNorm(hidden_size, p=0.25, eps=eps).to(device)
    with torch.no_grad():
        rmsnorm.scale.fill_(1.0)
        prmsnorm.scale.fill_(1.0)

    return {
        "layernorm_rescale_max_abs": (layernorm(alpha * x) - layernorm(x))
        .abs()
        .max()
        .item(),
        "rmsnorm_rescale_max_abs": (rmsnorm(alpha * x) - rmsnorm(x))
        .abs()
        .max()
        .item(),
        "prmsnorm_rescale_max_abs": (prmsnorm(alpha * x) - prmsnorm(x))
        .abs()
        .max()
        .item(),
        "layernorm_shift_mean_abs": (layernorm(x + shift) - layernorm(x))
        .abs()
        .mean()
        .item(),
        "rmsnorm_shift_mean_abs": (rmsnorm(x + shift) - rmsnorm(x))
        .abs()
        .mean()
        .item(),
    }


def print_table(rows: list[TrainResult]) -> None:
    print("\ntraining_check")
    print("norm        initial_loss  final_loss  final_acc  ms_per_step")
    for row in rows:
        print(
            "{:<11} {:>12.4f} {:>10.4f} {:>10.4f} {:>12.3f}".format(
                row.norm,
                row.initial_loss,
                row.final_loss,
                row.final_accuracy,
                row.ms_per_step,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--norms",
        nargs="+",
        default=["none", "layernorm", "rmsnorm", "prmsnorm"],
        choices=["none", "layernorm", "rmsnorm", "prmsnorm"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("requested non-CPU device, but CUDA is not available")

    torch.manual_seed(args.seed)
    x = torch.randn(args.samples, args.input_size, device=args.device)
    y = make_teacher_labels(
        x,
        hidden_size=args.hidden_size,
        output_size=args.classes,
        seed=args.seed + 1,
    )

    report = invariance_report(args.hidden_size, args.eps, args.device)
    print("invariance_check")
    print(json.dumps(report, indent=2, sort_keys=True))

    rows = [train_once(norm, x, y, args) for norm in args.norms]
    print_table(rows)

    print("\nconfig")
    print(json.dumps(vars(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
