from __future__ import annotations

import base64
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sentencepiece as spm

from ..tokenizer_base import MiniTokenizer


ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


@dataclass
class SentencePieceTokenizer(MiniTokenizer):
    """Standalone SentencePiece BPE/Unigram tokenizer for algorithm experiments."""

    tokenizer_type = "sentencepiece"

    processor: spm.SentencePieceProcessor
    model_proto: bytes
    model_type: str

    @classmethod
    def train(
        cls,
        files: Iterable[str | Path],
        *,
        model_type: str = "unigram",
        vocab_size: int = 512,
        character_coverage: float = 1.0,
    ) -> "SentencePieceTokenizer":
        if model_type not in {"bpe", "unigram"}:
            raise ValueError("SentencePiece model_type must be 'bpe' or 'unigram'")
        file_paths = [str(Path(path)) for path in files]
        # Byte fallback reserves 256 pieces; character_coverage=1.0 also requires
        # every observed Unicode character plus four meta and three role pieces.
        required_chars: set[str] = set()
        for file_path in file_paths:
            with Path(file_path).open(encoding="utf-8") as corpus:
                for line in corpus:
                    required_chars.update(line)
        effective_vocab_size = max(vocab_size, 256 + 7 + len(required_chars))
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir) / "tokenizer")
            spm.SentencePieceTrainer.train(
                input=file_paths,
                model_prefix=prefix,
                model_type=model_type,
                vocab_size=effective_vocab_size,
                character_coverage=character_coverage,
                byte_fallback=True,
                hard_vocab_limit=False,
                unk_id=0,
                pad_id=1,
                bos_id=2,
                eos_id=3,
                user_defined_symbols=list(ROLE_TOKENS.values()),
                normalization_rule_name="identity",
                add_dummy_prefix=False,
                remove_extra_whitespaces=False,
                minloglevel=2,
            )
            model_proto = Path(prefix + ".model").read_bytes()
        return cls._from_proto(model_proto, model_type=model_type)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        model_type: str = "unigram",
    ) -> "SentencePieceTokenizer":
        return cls._from_proto(Path(path).read_bytes(), model_type=model_type)

    @classmethod
    def from_dict(cls, payload: dict) -> "SentencePieceTokenizer":
        model_proto = base64.b64decode(str(payload["model_proto_base64"]))
        return cls._from_proto(model_proto, model_type=str(payload.get("model_type", "unigram")))

    @classmethod
    def _from_proto(cls, model_proto: bytes, *, model_type: str) -> "SentencePieceTokenizer":
        processor = spm.SentencePieceProcessor()
        if not processor.LoadFromSerializedProto(model_proto):
            raise ValueError("Invalid serialized SentencePiece model")
        return cls(processor=processor, model_proto=model_proto, model_type=model_type)

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def bos_token_id(self) -> int | None:
        return self._optional_id(self.processor.bos_id())

    @property
    def eos_token_id(self) -> int | None:
        return self._optional_id(self.processor.eos_id())

    @property
    def pad_token_id(self) -> int | None:
        return self._optional_id(self.processor.pad_id())

    @property
    def unk_token_id(self) -> int | None:
        return self._optional_id(self.processor.unk_id())

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self._required_id(self.bos_token_id, "BOS"))
        if add_eos:
            ids.append(self._required_id(self.eos_token_id, "EOS"))
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            special_ids = {
                token_id
                for token_id in (
                    self.unk_token_id,
                    self.bos_token_id,
                    self.eos_token_id,
                    self.pad_token_id,
                    *(self.token_to_id(token) for token in ROLE_TOKENS.values()),
                )
                if token_id is not None
            }
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self.processor.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        token_id = self.processor.piece_to_id(token)
        if token_id == self.processor.unk_id() and token != self.processor.id_to_piece(token_id):
            return None
        return int(token_id)

    def id_to_token(self, token_id: int) -> str | None:
        if 0 <= token_id < self.vocab_size:
            return str(self.processor.id_to_piece(token_id))
        return None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            if role not in ROLE_TOKENS:
                raise ValueError(f"Unsupported chat role: {role!r}")
            parts.append(f"{ROLE_TOKENS[role]}\n{message.get('content', '').strip()}\n")
        if add_generation_prompt:
            parts.append(f"{ROLE_TOKENS['assistant']}\n")
        prompt = "".join(parts)
        return self.encode(prompt) if tokenize else prompt

    def save_pretrained(self, directory: str | Path, *, model_max_length: int = 128) -> None:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer.model").write_bytes(self.model_proto)
        (out_dir / "tokenizer.vocab").write_text(
            "\n".join(
                f"{self.processor.id_to_piece(token_id)}\t{self.processor.get_score(token_id)}"
                for token_id in range(self.vocab_size)
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "MiniLLMSentencePieceTokenizer",
                    "model_max_length": model_max_length,
                    "model_type": self.model_type,
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "additional_special_tokens": list(ROLE_TOKENS.values()),
                    "note": "Standalone MiniLLM learning format; use HFTokenizerAdapter for production HF tokenizers.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def to_dict(self) -> dict:
        return {
            "type": f"sentencepiece-{self.model_type}",
            "model_type": self.model_type,
            "model_proto_base64": base64.b64encode(self.model_proto).decode("ascii"),
        }

    @staticmethod
    def _optional_id(token_id: int) -> int | None:
        return None if token_id < 0 else int(token_id)

    @staticmethod
    def _required_id(token_id: int | None, name: str) -> int:
        if token_id is None:
            raise ValueError(f"SentencePiece tokenizer has no {name} token")
        return token_id
