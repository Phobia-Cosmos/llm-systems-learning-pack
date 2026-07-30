from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tokenizer_base import MiniTokenizer

# 问题（已回答）:@dataclass作用是什么?cls关键字作用是什么?三个属性分别代表什么?
# 回答：@dataclass 会自动为这个类生成 __init__、__repr__ 等样板代码，
# 所以下面只声明字段，Python 就知道如何构造 CharTokenizer。
# cls 出现在 @classmethod 里，代表“当前类本身”，类似普通方法里的 self 代表“当前对象”。
# stoi 是 string-to-id，把字符/token 映射到整数；itos 是 id-to-string，把整数映射回字符/token；
# unk_token 是 unknown token，遇到词表外字符时用它兜底。
# 问题（已回答）：dataclass 生成的 __init__/__repr__ 有何作用，每个类都需要吗？
# 回答：__init__ 接收并保存字段，__repr__ 提供可读调试字符串；不用 dataclass 就要手写或继承实现。
# 只有“以数据字段为主”的类适合 dataclass，模型 Module、复杂生命周期类通常自己写 __init__。
@dataclass
class CharTokenizer(MiniTokenizer):
    tokenizer_type = "char"

    stoi: dict[str, int]
    itos: list[str]
    unk_token: str = "<unk>"

    @classmethod
    # 问题（已回答）：为什么只有部分函数用 classmethod，set/text/itos 如何工作？
    # 回答：classmethod 用于“从文本构造新实例”；encode/decode 操作已有实例所以用 self。Python 迭代 str 会逐个 Unicode 字符，
    # set 去重，sorted 固定顺序。例如 text="aba" -> chars=["a","b"] -> itos=["<unk>","a","b"]。
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        # 问题（已回答）：chars 是分隔后的字符吗？
        # 回答：是按 Python Unicode code point 得到的不同字符集合，不是单词或 BPE 子词；空格、换行、标点也各算一个字符。
        itos = ["<unk>"] + chars
        # 问题（已回答）:ch: i for i是什么意思?
        # 回答：这是字典推导式，形式是 {key: value for ... in ...}。
        # enumerate(itos) 会依次产生 (0, "<unk>"), (1, 第一个字符) 这样的二元组；
        # for i, ch in enumerate(itos) 把二元组拆成 i 和 ch；ch: i 表示字典里保存“字符 -> 编号”。
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    # 问题（已回答）:返回的是什么?为什么要先得到unk?为什么这样就算encode了？
    # 回答：返回的是 list[int]，也就是把文本中每个字符替换成词表 id 后的序列。
    # 先得到 unk 是为了处理训练语料里没见过的字符：self.stoi.get(ch, unk) 找不到 ch 时返回 unk。
    # encode 的核心含义就是“把人类可读文本变成模型能处理的整数 token id”。
    # 问题（已回答）：字符如何变 id，dict.get 为什么有两个参数？
    # 回答：若 stoi={"<unk>":0,"a":1,"中":2}，encode("a中?") 得 [1,2,0]。unk 是 <unk> 的 id 0；
    # self.stoi.get(ch, unk) 先查 ch，找不到时返回第二个参数 0，而不是同时匹配 ch 和 unk。
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if add_bos or add_eos:
            raise ValueError("Legacy CharTokenizer has no BOS/EOS token")
        unk = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk) for ch in text]

    # 问题（已回答）:原理是什么?为什么传入的是ids?为什么要判断unk token?
    # 回答：decode 是 encode 的反过程：模型生成的是 token id，不是字符串，所以传入 ids。
    # self.itos[int(idx)] 根据 id 找回字符；如果这个 id 对应 <unk>，说明原字符未知，
    # 这里用 "?" 展示，避免把内部占位符 <unk> 直接打印给用户。
    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            pieces.append("?" if token == self.unk_token else token)
        # 问题（已回答）：为什么用 "".join(pieces) 生成字符串？
        # 回答：还可循环拼接或格式化，但 join 一次连接已有字符串列表，语义清晰且避免反复创建中间字符串。
        return "".join(pieces)

    @property
    # 问题（已回答）:这个返回的是什么？
    # 回答：vocab_size 返回词表大小，也就是模型最后要在多少个候选 token 中预测下一个 token。
    # 对字符级 tokenizer 来说，它等于训练文本中不同字符数量 + 1 个 <unk>。
    # 问题（已回答）：字符粒度、property 和候选 token 是什么？
    # 回答：这里 token 是单个 Unicode 字符，不是 word；stoi/itos 是直接存储字段，vocab_size 是由 len(itos) 动态计算的派生值，
    # 所以用 @property 让它像只读属性访问。该模型每步从全部字符 token 中预测一个，因此是 per-token（此处等于 per-char）生成。
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def unk_token_id(self) -> int:
        return self.stoi[self.unk_token]

    def token_to_id(self, token: str) -> int | None:
        return self.stoi.get(token)

    def id_to_token(self, token_id: int) -> str | None:
        if 0 <= token_id < len(self.itos):
            return self.itos[token_id]
        return None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        role_labels = {"system": "系统", "user": "用户", "assistant": "助手"}
        lines: list[str] = []
        for message in messages:
            role = message["role"]
            if role not in role_labels:
                raise ValueError(f"Unsupported chat role: {role!r}")
            lines.append(f"{role_labels[role]}: {message.get('content', '').strip()}")
        if add_generation_prompt:
            lines.append("助手:")
        prompt = "\n".join(lines)
        return self.encode(prompt) if tokenize else prompt

    def save_pretrained(self, directory: str | Path, *, model_max_length: int = 128) -> None:
        import json

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "CharTokenizer",
                    "unk_token": self.unk_token,
                    "model_max_length": model_max_length,
                    "note": "MiniLLM educational character tokenizer; not an HF fast tokenizer.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def to_dict(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "unk_token": self.unk_token,
        }

    @classmethod
    # 问题（已回答）：from_dict 接收什么，payload.get 的两个参数是什么？
    # 回答：传入已解析的 Python dict，可来自 json.loads 或 checkpoint；必须含 stoi/itos。
    # payload.get("unk_token", "<unk>") 是“取该键，否则用默认值”，不是同时匹配两个参数。
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        return cls(
            stoi={str(k): int(v) for k, v in payload["stoi"].items()},
            itos=[str(x) for x in payload["itos"]],
            unk_token=str(payload.get("unk_token", "<unk>")),
        )
