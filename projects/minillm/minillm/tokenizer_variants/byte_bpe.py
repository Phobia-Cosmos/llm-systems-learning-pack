from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# TODO:tokenizers是什么 库文件吗？这些作用分别是什么？
from tokenizers import AddedToken, Tokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

# TODO:为什么要加上这个？一开始的分词器中没有这个东西？
ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}

DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{{ '<|' + message['role'] + '|>' }}
{{ message['content'] }}
{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}
{% endif %}"""


@dataclass
class HFByteBPETokenizer:
    """Small HF-compatible Byte-level BPE tokenizer wrapper for MiniLLM experiments."""

    tokenizer: Tokenizer
    special_tokens: list[str]

    @classmethod
    def train(
        cls,
        files: Iterable[str | Path],
        *,
        vocab_size: int = 512,
        min_frequency: int = 1,
    ) -> "HFByteBPETokenizer":
        file_paths = [str(Path(path)) for path in files]
        # TODO:第一个是什么？为什么要有多个size然后选择一个大的？
        min_vocab_size = len(pre_tokenizers.ByteLevel.alphabet()) + len(SPECIAL_TOKENS)
        effective_vocab_size = max(vocab_size, min_vocab_size)

        # TODO:为什么这个不是我们自己实现呢？NFC是什么？NFKC?这里是在做什么？为什么要用normalizer，以及都有哪些normalizer为什么使用Sequnece？为什么还会出现pre_tokenizer？
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        # NFC keeps canonical Unicode forms without folding full-width
        # punctuation such as "？" into ASCII "?"; this preserves LLM text
        # roundtrips better than NFKC for a teaching tokenizer.
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=effective_vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train(files=file_paths, trainer=trainer)
        cls._register_special_tokens(tokenizer)
        return cls(tokenizer=tokenizer, special_tokens=list(SPECIAL_TOKENS))

    @classmethod
    def from_file(cls, path: str | Path) -> "HFByteBPETokenizer":
        tokenizer = Tokenizer.from_file(str(path))
        cls._register_special_tokens(tokenizer)
        return cls(tokenizer=tokenizer, special_tokens=list(SPECIAL_TOKENS))

    @classmethod
    def from_dict(cls, payload: dict) -> "HFByteBPETokenizer":
        tokenizer = Tokenizer.from_str(str(payload["tokenizer_json"]))
        cls._register_special_tokens(tokenizer)
        special_tokens = [str(token) for token in payload.get("special_tokens", SPECIAL_TOKENS)]
        return cls(tokenizer=tokenizer, special_tokens=special_tokens)

    # TODO:类方法和静态方法区别是什么？
    @staticmethod
    def _register_special_tokens(tokenizer: Tokenizer) -> None:
        tokenizer.add_special_tokens(
            [
                AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False)
                for token in SPECIAL_TOKENS
            ]
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.tokenizer.id_to_token(token_id)

    # TODO:这个函数为什么要加入bos和eos？什么叫做_required_token_id？不可以省略的吗是？extend作用是什么？和encode_with_tokens的区别是什么？
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self._required_token_id("<bos>"))
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if add_eos:
            ids.append(self._required_token_id("<eos>"))
        return ids

    def encode_with_tokens(self, text: str) -> tuple[list[int], list[str]]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.tokens

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # TODO:add_generation_prompt作用是什么？不太理解
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            if role not in ROLE_TOKENS:
                raise ValueError(f"Unsupported role {role!r}; expected one of {sorted(ROLE_TOKENS)}")
            content = message.get("content", "").strip()
            parts.append(f"{ROLE_TOKENS[role]}\n{content}\n")
        if add_generation_prompt:
            parts.append(f"{ROLE_TOKENS['assistant']}\n")
        return "".join(parts)

    def save(self, directory: str | Path, *, model_max_length: int = 128) -> None:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(out_dir / "tokenizer.json"))
        (out_dir / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "model_max_length": model_max_length,
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "chat_template": DEFAULT_CHAT_TEMPLATE,
                    "note": "MiniLLM Byte-level BPE tokenizer variant for learning; not tied to existing CharTokenizer checkpoints.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def to_dict(self) -> dict:
        return {
            "type": "byte-bpe",
            "tokenizer_json": self.tokenizer.to_str(),
            "special_tokens": self.special_tokens,
        }

    def _required_token_id(self, token: str) -> int:
        token_id = self.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Missing special token {token!r}")
        return token_id
