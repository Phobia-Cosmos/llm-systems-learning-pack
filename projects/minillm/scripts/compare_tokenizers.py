from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minillm import CharTokenizer  # noqa: E402
from minillm.tokenizer_variants import HFByteBPETokenizer  # noqa: E402


DEFAULT_SAMPLES = [
    "用户: 什么是 embedding?\n助手:",
    "Hello, CUDA 13.0 + vLLM?",
    "今天天气怎么样？",
    "🚀 emoji and unseen text",
]


def display(text: str) -> str:
    return text.replace("\n", "\\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MiniLLM CharTokenizer with tokenizer variants.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "minillm.pt"))
    parser.add_argument("--byte-bpe", default=str(ROOT / "tokenizer_variants" / "byte_bpe" / "tokenizer.json"))
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--show-tokens", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    char_tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    byte_bpe = HFByteBPETokenizer.from_file(args.byte_bpe)

    samples = args.sample or DEFAULT_SAMPLES
    print(f"char_vocab_size={char_tokenizer.vocab_size}")
    print(f"byte_bpe_vocab_size={byte_bpe.vocab_size}")
    print()
    print("| sample | char_tokens | char_unk | char_roundtrip | byte_bpe_tokens | byte_bpe_roundtrip |")
    print("| --- | ---: | ---: | --- | ---: | --- |")
    for sample in samples:
        char_ids = char_tokenizer.encode(sample)
        char_decoded = char_tokenizer.decode(char_ids)
        char_unk = sum(1 for token_id in char_ids if token_id == char_tokenizer.stoi[char_tokenizer.unk_token])

        bpe_ids, bpe_tokens = byte_bpe.encode_with_tokens(sample)
        bpe_decoded = byte_bpe.decode(bpe_ids)

        print(
            "| "
            f"`{display(sample)}` | "
            f"{len(char_ids)} | "
            f"{char_unk} | "
            f"{char_decoded == sample} | "
            f"{len(bpe_ids)} | "
            f"{bpe_decoded == sample} |"
        )
        if args.show_tokens:
            char_tokens = [char_tokenizer.itos[token_id] for token_id in char_ids]
            print(f"\nchar ids: {char_ids}")
            print(f"char tokens: {char_tokens}")
            print(f"byte_bpe ids: {bpe_ids}")
            print(f"byte_bpe tokens: {bpe_tokens}\n")


if __name__ == "__main__":
    main()
