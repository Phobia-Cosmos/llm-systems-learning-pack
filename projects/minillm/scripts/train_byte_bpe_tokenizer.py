from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minillm.tokenizer_variants import HFByteBPETokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MiniLLM Byte-level BPE tokenizer variant.")
    parser.add_argument("--input", default=str(ROOT / "data" / "teaching_corpus.txt"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "tokenizers" / "byte_bpe"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--sample", default="用户: 什么是 embedding?\n助手:")
    args = parser.parse_args()

    tokenizer = HFByteBPETokenizer.train(
        [args.input],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(args.output)

    ids, tokens = tokenizer.encode_with_tokens(args.sample)
    print(f"saved_to={args.output}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print("special_tokens:")
    for token in tokenizer.special_tokens:
        print(f"  {token!r}: {tokenizer.token_to_id(token)}")
    print("sample:")
    print(f"  text={args.sample!r}")
    print(f"  ids={ids}")
    print(f"  tokens={tokens}")
    print(f"  decoded={tokenizer.decode(ids)!r}")


if __name__ == "__main__":
    main()
