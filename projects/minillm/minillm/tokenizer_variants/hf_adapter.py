from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from ..tokenizer_base import MiniTokenizer, TokenizerBatch


@dataclass
class HFTokenizerAdapter(MiniTokenizer):
    """Adapter from a standard Hugging Face fast tokenizer to MiniLLM."""

    tokenizer_type = "hf-auto"

    tokenizer: PreTrainedTokenizerBase
    source: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        trust_remote_code: bool = False,
    ) -> "HFTokenizerAdapter":
        tokenizer = AutoTokenizer.from_pretrained(
            str(source),
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "HFTokenizerAdapter requires a fast tokenizer so its backend can be serialized into checkpoints"
            )
        return cls(tokenizer=tokenizer, source=str(source))

    @classmethod
    def from_dict(cls, payload: dict) -> "HFTokenizerAdapter":
        backend = Tokenizer.from_str(str(payload["tokenizer_json"]))
        special_tokens = dict(payload.get("special_tokens", {}))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            model_max_length=int(payload.get("model_max_length", 128)),
            padding_side=str(payload.get("padding_side", "right")),
            truncation_side=str(payload.get("truncation_side", "right")),
            **special_tokens,
        )
        tokenizer.chat_template = payload.get("chat_template")
        return cls(tokenizer=tokenizer, source=payload.get("source"))

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    @property
    def unk_token_id(self) -> int | None:
        return self.tokenizer.unk_token_id

    @property
    def padding_side(self) -> str:
        return self.tokenizer.padding_side

    @property
    def truncation_side(self) -> str:
        return self.tokenizer.truncation_side

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids)
        if add_bos:
            ids.insert(0, self._required_special_id("bos_token_id"))
        if add_eos:
            ids.append(self._required_special_id("eos_token_id"))
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.get_vocab().get(token)

    def id_to_token(self, token_id: int) -> str | None:
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        return None if token is None else str(token)

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
        # The shared implementation intentionally keeps BOS/EOS behavior identical
        # across Char, Byte-BPE and imported HF tokenizers.
        return super().batch_encode(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        if not self.tokenizer.chat_template:
            raise ValueError(
                "This Hugging Face tokenizer has no chat_template; use a model-specific template before chat inference"
            )
        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return list(result) if tokenize else str(result)

    def save_pretrained(self, directory: str | Path, *, model_max_length: int = 128) -> None:
        self.tokenizer.model_max_length = model_max_length
        self.tokenizer.save_pretrained(str(directory))

    def to_dict(self) -> dict:
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError("Cannot checkpoint a non-fast Hugging Face tokenizer")
        special_tokens: dict[str, str | list[str]] = {}
        for name in ("unk_token", "pad_token", "bos_token", "eos_token", "mask_token", "sep_token", "cls_token"):
            value = getattr(self.tokenizer, name, None)
            if value is not None:
                special_tokens[name] = str(value)
        additional = list(getattr(self.tokenizer, "additional_special_tokens", []) or [])
        if additional:
            special_tokens["additional_special_tokens"] = [str(token) for token in additional]
        return {
            "type": self.tokenizer_type,
            "tokenizer_json": backend.to_str(),
            "special_tokens": special_tokens,
            "chat_template": self.tokenizer.chat_template,
            "model_max_length": int(self.tokenizer.model_max_length),
            "padding_side": self.tokenizer.padding_side,
            "truncation_side": self.tokenizer.truncation_side,
            "source": self.source,
        }

    def _required_special_id(self, attribute: str) -> int:
        token_id = getattr(self, attribute)
        if token_id is None:
            raise ValueError(f"Hugging Face tokenizer has no {attribute}")
        return int(token_id)
