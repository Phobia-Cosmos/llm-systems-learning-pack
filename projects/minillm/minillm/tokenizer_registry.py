from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .tokenizer import CharTokenizer
from .tokenizer_variants import HFByteBPETokenizer

# TODO:这个文件中还是需要负责训练的是吗？

# TODO:为什么要定义这个函数？是不是我们后续的分词器都要有这些函数？这是标准的HF格式吗？为什么是Protocol？
class MiniTokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def to_dict(self) -> dict: ...


# TODO:def是定义一个函数吗？全局函数？第三个参数*是什么？vocab_size是不变的吗？
def build_tokenizer(
    tokenizer_name: str,
    text: str,
    *,
    training_file: str | None = None,
    tokenizer_path: str | None = None,
    tokenizer_output_dir: str | None = None,
    tokenizer_vocab_size: int = 512,
    retrain_tokenizer: bool = False,
) -> MiniTokenizer:
    if tokenizer_name == "char":
        return CharTokenizer.from_text(text)

    if tokenizer_name == "byte-bpe":
        # TODO:为什么这里是只有一个if能选择？
        if tokenizer_path is not None:
            path = Path(tokenizer_path)
        elif tokenizer_output_dir is not None:
            path = Path(tokenizer_output_dir) / "tokenizer.json"
        else:
            path = Path("tokenizer_variants") / "byte_bpe" / "tokenizer.json"

        if path.exists() and not retrain_tokenizer:
            return HFByteBPETokenizer.from_file(path)

        if training_file is None:
            # TODO:training_file是什么？为什么一定需要这个？这个文件中是什么 和传统的char训练的corpus有和不同吗？
            raise ValueError("training_file is required when training a byte-bpe tokenizer")
        tokenizer = HFByteBPETokenizer.train([training_file], vocab_size=tokenizer_vocab_size)
        if tokenizer_output_dir is not None:
            tokenizer.save(tokenizer_output_dir)
        return tokenizer

    raise ValueError(f"Unsupported tokenizer {tokenizer_name!r}")


def tokenizer_from_checkpoint(checkpoint: dict) -> MiniTokenizer:
    tokenizer_type = checkpoint.get("tokenizer_type")
    payload = checkpoint["tokenizer"]

    if tokenizer_type is None:
        tokenizer_type = payload.get("type", "char") if isinstance(payload, dict) else "char"

    if tokenizer_type == "char":
        return CharTokenizer.from_dict(payload)
    if tokenizer_type == "byte-bpe":
        return HFByteBPETokenizer.from_dict(payload)

    raise ValueError(f"Unsupported tokenizer type in checkpoint: {tokenizer_type!r}")


def tokenizer_to_checkpoint_payload(tokenizer_name: str, tokenizer: MiniTokenizer) -> dict:
    payload = tokenizer.to_dict()
    if tokenizer_name == "char" and isinstance(payload, dict) and "type" not in payload:
        payload = {"type": "char", **payload}
    return payload
