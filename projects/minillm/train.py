from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from minillm import CharTokenizer, GPTConfig, MiniGPT
from minillm.data import get_batch, read_text, split_train_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal decoder-only LLM.")
    parser.add_argument("--data", default="data/tiny_corpus.txt")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        result[split] = losses.mean().item()
    model.train()
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    text = read_text(args.data)
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    train_data, val_data = split_train_val(token_ids)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={device}, vocab_size={tokenizer.vocab_size}, parameters={model.parameter_count():,}")
    for step in range(args.max_steps + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.block_size,
                args.batch_size,
                args.eval_iters,
                device,
            )
            print(f"step {step:04d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "minillm.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "tokenizer": tokenizer.to_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
