from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minillm import GPTConfig, MiniGPT  # noqa: E402
from minillm.tokenizer_registry import tokenizer_from_checkpoint  # noqa: E402


MEANINGS = {
    "token_embedding": "把离散 token id 查表成连续向量，表示 token 是什么",
    "position_embedding": "把位置 id 查表成连续向量，表示 token 在第几个位置",
    "drop": "训练时做 dropout；eval 模式下基本等于原样传递",
    "ln_1": "attention 前的 LayerNorm，稳定 attention 输入",
    "attn": "causal self-attention，让当前位置从历史 token 中取信息",
    "ln_2": "MLP 前的 LayerNorm，稳定 MLP 输入",
    "mlp": "逐位置非线性变换，增强每个 token 的特征表达",
    "block": "一个 Transformer block 输出，仍是 [B,T,C]",
    "ln_f": "最后的 LayerNorm，稳定最终 hidden state",
    "lm_head": "把 hidden state 映射到词表维度，得到每个 token 的 logits",
}


def tensor_summary(value: Any) -> str:
    if isinstance(value, tuple):
        value = value[0]
    if not torch.is_tensor(value):
        return type(value).__name__
    data = value.detach().float()
    std = data.std(unbiased=False).item() if data.numel() > 1 else 0.0
    return (
        f"shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"mean={data.mean().item():.4f}, std={std:.4f}"
    )


def display_token(token: str) -> str:
    if token is None:
        return "<none>"
    if token == "\n":
        return "\\n"
    if token == "\t":
        return "\\t"
    if token == " ":
        return "<space>"
    return token


def register_hooks(model: MiniGPT, records: list[tuple[str, str, str]]) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def add(name: str, module: torch.nn.Module, meaning_key: str) -> None:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            records.append((name, tensor_summary(output), MEANINGS[meaning_key]))

        handles.append(module.register_forward_hook(hook))

    add("token_embedding", model.token_embedding, "token_embedding")
    # RoPE models intentionally have no learned absolute-position Embedding.
    if model.position_embedding is not None:
        add("position_embedding", model.position_embedding, "position_embedding")
    add("drop", model.drop, "drop")
    for i, block in enumerate(model.blocks):
        add(f"blocks.{i}.ln_1", block.ln_1, "ln_1")
        add(f"blocks.{i}.attn", block.attn, "attn")
        add(f"blocks.{i}.ln_2", block.ln_2, "ln_2")
        add(f"blocks.{i}.mlp", block.mlp, "mlp")
        add(f"blocks.{i}", block, "block")
    add("ln_f", model.ln_f, "ln_f")
    add("lm_head", model.lm_head, "lm_head")
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one MiniLLM text flow from tokenizer to logits.")
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts" / "checkpoints" / "minillm.pt"))
    parser.add_argument("--prompt", default="用户: 什么是 embedding?\n助手:")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--vocab-limit", type=int, default=80)
    parser.add_argument("--full-vocab", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = args.prompt.replace("\\n", "\n")
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)

    print("== Prompt ==")
    print(repr(prompt))
    print()
    print("== stoi / itos ==")
    print(f"vocab_size={tokenizer.vocab_size}")
    print("词表负责 token <-> id 映射；CharTokenizer 用 stoi/itos，HF tokenizer 用 token_to_id/id_to_token。")
    limit = tokenizer.vocab_size if args.full_vocab else min(args.vocab_limit, tokenizer.vocab_size)
    for i in range(limit):
        if hasattr(tokenizer, "itos"):
            token = tokenizer.itos[i]
            token_id = tokenizer.stoi[token]
        else:
            token = tokenizer.id_to_token(i)
            token_id = tokenizer.token_to_id(token) if token is not None else None
        print(f"id_to_token[{i:03d}]={display_token(token)!r}  token_to_id={token_id}")
    if limit < tokenizer.vocab_size:
        print(f"... 还有 {tokenizer.vocab_size - limit} 个 token；加 --full-vocab 可打印完整词表。")
    print()

    print("== encode ==")
    if hasattr(tokenizer, "itos"):
        for pos, (ch, token_id) in enumerate(zip(prompt, ids)):
            print(f"pos={pos:02d} char={display_token(ch)!r} -> id={token_id}")
    elif hasattr(tokenizer, "encode_with_tokens"):
        piece_ids, pieces = tokenizer.encode_with_tokens(prompt)
        for pos, (piece, token_id) in enumerate(zip(pieces, piece_ids)):
            print(f"pos={pos:02d} token={display_token(piece)!r} -> id={token_id}")
    else:
        for pos, token_id in enumerate(ids):
            print(f"pos={pos:02d} id={token_id}")
    print(f"input_ids shape={tuple(input_ids.shape)} values={ids}")
    print()

    records: list[tuple[str, str, str]] = []
    handles = register_hooks(model, records)
    with torch.no_grad():
        logits, _loss = model(input_ids)
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_probs, top_ids = torch.topk(probs, k=min(args.top_k, tokenizer.vocab_size))
    for handle in handles:
        handle.remove()

    print("== forward layers ==")
    for name, summary, meaning in records:
        print(f"{name:24s} {summary}")
        print(f"{'':24s} -> {meaning}")
    print()

    print("== final logits / next-token candidates ==")
    print(f"logits shape={tuple(logits.shape)}")
    print("只看最后一个位置 logits[0, -1, :]，它表示下一个 token 的词表分数。")
    for rank, (token_id, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), start=1):
        token = tokenizer.itos[token_id] if hasattr(tokenizer, "itos") else tokenizer.id_to_token(token_id)
        logit = logits[0, -1, token_id].item()
        print(
            f"rank={rank:02d} id={token_id:03d} token={display_token(token)!r} "
            f"logit={logit:.4f} prob={prob:.4f}"
        )


if __name__ == "__main__":
    main()
