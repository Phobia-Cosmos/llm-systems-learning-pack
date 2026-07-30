from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# 问题（已回答）：tokenizers 是什么库，这些模块分别做什么？
# 回答：它是 Hugging Face 的 Rust/Python 高性能分词库。Tokenizer 组织流水线；models 定义 BPE 等算法；
# normalizers 规范文本；pre_tokenizers 初切分；trainers 学词表/merge；decoders 还原文本；AddedToken 注册特殊 token。
from tokenizers import AddedToken, Tokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

from ..tokenizer_base import MiniTokenizer

SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

# 问题（已回答）：为什么增加 ROLE_TOKENS，旧分词器没有？
# 回答：聊天模型需要明确区分 system/user/assistant 边界，role token 是控制标记而非普通文本。
# 旧 CharTokenizer 只演示字符预测，没有标准 chat template，所以不具备这些控制 token。
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
class HFByteBPETokenizer(MiniTokenizer):
    """Small HF-compatible Byte-level BPE tokenizer wrapper for MiniLLM experiments."""

    tokenizer_type = "byte-bpe"

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
        # 问题（已回答）：min_vocab_size 是什么，为什么与请求 vocab_size 取 max？
        # 回答：ByteLevel 基础 alphabet 要覆盖 256 种字节，再加 special tokens；若请求词表更小会丢失完整 byte 覆盖能力。
        # effective_vocab_size 取较大值，保证最低可用词表，同时允许用户请求更多 BPE merge token。
        min_vocab_size = len(pre_tokenizers.ByteLevel.alphabet()) + len(SPECIAL_TOKENS)
        effective_vocab_size = max(vocab_size, min_vocab_size)

        # 问题（已回答）：为什么使用库，NFC/NFKC、normalizer、Sequence 和 pre-tokenizer 是什么？
        # 回答：BPE 训练涉及 byte 映射、offset、特殊 token 和序列化，成熟库更可靠。NFC 只合并规范等价 Unicode；NFKC 还会
        # 折叠兼容字符（如全角问号），可能改变原文。normalizer 统一输入形式；Sequence 便于串联多步；pre-tokenizer 在 BPE 前生成候选片段和边界。
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

    # 问题（已回答）：classmethod 和 staticmethod 有何区别？
    # 回答：classmethod 第一个参数是 cls，适合返回当前类实例并支持子类；staticmethod 不接收 cls/self，
    # 只是放在类命名空间中的辅助函数。注册 special tokens 不依赖实例或具体子类，因此使用 staticmethod。
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

    @property
    def bos_token_id(self) -> int | None:
        return self.token_to_id("<bos>")

    @property
    def eos_token_id(self) -> int | None:
        return self.token_to_id("<eos>")

    @property
    def pad_token_id(self) -> int | None:
        return self.token_to_id("<pad>")

    @property
    def unk_token_id(self) -> int | None:
        return self.token_to_id("<unk>")

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.tokenizer.id_to_token(token_id)

    # 问题（已回答）：BOS/EOS、_required_token_id、extend 和 encode_with_tokens 分别是什么？
    # 回答：BOS/EOS 可选标记序列开始/结束，是否加入取决于模型训练格式；若请求加入却未注册，_required_token_id 会明确报错。
    # list.extend 将正文多个 ids 逐个追加；encode 只返回 ids，encode_with_tokens 额外返回可观察的 token pieces，便于教学调试。
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

    # 问题（已回答）：add_generation_prompt 有什么作用？
    # 回答：True 时在已有对话末尾追加 assistant role token 和换行，提示模型“接下来轮到助手生成”；
    # 训练完整对话或仅序列化历史时可设 False，避免凭空增加一个无内容的 assistant 轮次。
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
                raise ValueError(f"Unsupported role {role!r}; expected one of {sorted(ROLE_TOKENS)}")
            content = message.get("content", "").strip()
            parts.append(f"{ROLE_TOKENS[role]}\n{content}\n")
        if add_generation_prompt:
            parts.append(f"{ROLE_TOKENS['assistant']}\n")
        prompt = "".join(parts)
        return self.encode(prompt) if tokenize else prompt

    def save_pretrained(self, directory: str | Path, *, model_max_length: int = 128) -> None:
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

    def save(self, directory: str | Path, *, model_max_length: int = 128) -> None:
        self.save_pretrained(directory, model_max_length=model_max_length)

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
