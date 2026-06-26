from __future__ import annotations

import argparse

import torch

from minillm import CharTokenizer, GPTConfig, MiniGPT
from train import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniGPT checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/minillm.pt")
    parser.add_argument("--prompt", default="LLM")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    token_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
