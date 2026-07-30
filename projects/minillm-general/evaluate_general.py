from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from minillm import GPTConfig, MiniGPT
from train_general import PackedTokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and sample a general-training checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--test-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--export", default=None, help="Optional inference-only .pt output path.")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_loss(
    model: MiniGPT,
    dataset: PackedTokens,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        inputs, targets = dataset.batch(batch_size, generator, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("model did not return test loss")
        total += loss.item()
    return total / batches


@torch.inference_mode()
def sample(
    model: MiniGPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not token_ids:
        raise ValueError(f"prompt encoded to no tokens: {prompt!r}")
    available = model.config.block_size - len(token_ids)
    if available <= 0:
        raise ValueError("prompt is at least as long as the model context window")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model.generate_with_kv_cache(
        inputs,
        max_new_tokens=min(max_new_tokens, available),
        temperature=temperature,
        top_k=None if top_k <= 0 else top_k,
    )[0].tolist()
    return tokenizer.decode(output, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    if args.test_batches <= 0 or args.batch_size <= 0:
        raise ValueError("test batches and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires a CUDA GPU")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    tokenizer_path = (
        Path(args.tokenizer).expanduser().resolve()
        if args.tokenizer
        else checkpoint_path.parent / "tokenizer.json"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    device = torch.device("cuda")
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    test_data = PackedTokens(dataset_dir / "test.bin", config.block_size)
    test_loss = evaluate_loss(model, test_data, args.test_batches, args.batch_size, args.seed, device)
    prompts = args.prompt or ["中国的首都是", "人工智能可以帮助人们", "请介绍一下你自己："]
    torch.manual_seed(args.seed)
    generations = [
        {
            "prompt": prompt,
            "text": sample(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                device,
            ),
        }
        for prompt in prompts
    ]
    result = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint["step"]),
        "parameters": model.parameter_count(),
        "test_loss": test_loss,
        "test_perplexity": math.exp(test_loss),
        "generations": generations,
    }

    if args.export:
        export_path = Path(args.export).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = export_path.with_suffix(export_path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "step": int(checkpoint["step"]),
                "config": checkpoint["config"],
                "model": checkpoint["model"],
                "tokenizer_file": "tokenizer.json",
            },
            temporary,
        )
        temporary.replace(export_path)
        result["export"] = str(export_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
