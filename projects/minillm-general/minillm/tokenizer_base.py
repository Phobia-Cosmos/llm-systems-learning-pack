from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class TokenizerBatch:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]

    def to_dict(self) -> dict[str, list[list[int]]]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }


class MiniTokenizer(ABC):
    """Stable tokenizer contract used by MiniLLM training and inference."""

    tokenizer_type: ClassVar[str]

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int | None:
        return None

    @property
    def pad_token_id(self) -> int | None:
        return None

    @property
    def unk_token_id(self) -> int | None:
        return None

    @property
    def padding_side(self) -> str:
        return "right"

    @property
    def truncation_side(self) -> str:
        return "right"

    @abstractmethod
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str: ...

    @abstractmethod
    def token_to_id(self, token: str) -> int | None: ...

    @abstractmethod
    def id_to_token(self, token_id: int) -> str | None: ...

    @abstractmethod
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]: ...

    @abstractmethod
    def save_pretrained(self, directory: str | Path, *, model_max_length: int = 128) -> None: ...

    @abstractmethod
    def to_dict(self) -> dict: ...

    def batch_encode(
        self,
        texts: list[str],
        *,
        padding: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> TokenizerBatch:
        if not texts:
            return TokenizerBatch(input_ids=[], attention_mask=[])

        rows = [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]
        if max_length is not None:
            normalized: list[list[int]] = []
            for row in rows:
                if len(row) > max_length and not truncation:
                    raise ValueError(
                        f"Encoded sequence length {len(row)} exceeds max_length={max_length}; "
                        "enable truncation explicitly"
                    )
                if truncation and self.truncation_side == "left":
                    normalized.append(row[-max_length:])
                else:
                    normalized.append(row[:max_length] if truncation else row)
            rows = normalized

        if not padding:
            return TokenizerBatch(
                input_ids=rows,
                attention_mask=[[1] * len(row) for row in rows],
            )

        target_length = max(len(row) for row in rows) if max_length is None else max_length
        if any(len(row) < target_length for row in rows) and self.pad_token_id is None:
            raise ValueError(f"Tokenizer {self.tokenizer_type!r} has no pad token")

        pad_id = 0 if self.pad_token_id is None else self.pad_token_id
        if self.padding_side == "left":
            input_ids = [[pad_id] * (target_length - len(row)) + row for row in rows]
            attention_mask = [[0] * (target_length - len(row)) + [1] * len(row) for row in rows]
        else:
            input_ids = [row + [pad_id] * (target_length - len(row)) for row in rows]
            attention_mask = [[1] * len(row) + [0] * (target_length - len(row)) for row in rows]
        return TokenizerBatch(input_ids=input_ids, attention_mask=attention_mask)
