from __future__ import annotations

from dataclasses import dataclass

# 问题（已回答）:@dataclass作用是什么?cls关键字作用是什么?三个属性分别代表什么?
# 回答：@dataclass 会自动为这个类生成 __init__、__repr__ 等样板代码，
# 所以下面只声明字段，Python 就知道如何构造 CharTokenizer。
# cls 出现在 @classmethod 里，代表“当前类本身”，类似普通方法里的 self 代表“当前对象”。
# stoi 是 string-to-id，把字符/token 映射到整数；itos 是 id-to-string，把整数映射回字符/token；
# unk_token 是 unknown token，遇到词表外字符时用它兜底。
# TODO:__init__、__repr__ 等样板代码都是有何作用呢，如果不生成会怎么样？是不是以后每写一个class都要用这个@？
@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    unk_token: str = "<unk>"

    @classmethod
    # TODO:为什么不是每一个函数都使用这个@？set(text)会进行去重吗？是如何将str转换为char的？为什么itos是如此计算？请你给我一个例子来解释
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        # TODO:我们的chars是分隔的字符吗？
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
    # TODO:请你给我一个简单的例子 针对每一个char变为id？为什么get传入两个参数？第一步处理unk我还不太理解 获得的是unk的index吗？为什么get要同时传入ch和unk？
    def encode(self, text: str) -> list[int]:
        unk = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk) for ch in text]

    # 问题（已回答）:原理是什么?为什么传入的是ids?为什么要判断unk token?
    # 回答：decode 是 encode 的反过程：模型生成的是 token id，不是字符串，所以传入 ids。
    # self.itos[int(idx)] 根据 id 找回字符；如果这个 id 对应 <unk>，说明原字符未知，
    # 这里用 "?" 展示，避免把内部占位符 <unk> 直接打印给用户。
    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            pieces.append("?" if token == self.unk_token else token)
        return "".join(pieces)

    @property
    # 问题（已回答）:这个返回的是什么？
    # 回答：vocab_size 返回词表大小，也就是模型最后要在多少个候选 token 中预测下一个 token。
    # 对字符级 tokenizer 来说，它等于训练文本中不同字符数量 + 1 个 <unk>。
    # TODO:这里的字符指的是a-z这种单个 还是word级单词？为什么我们定义的属性就不需要@property？什么叫做在多少个候选 token 中预测下一个 token？per-char生成还是per-word？
    def vocab_size(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "unk_token": self.unk_token,
        }

    @classmethod
    # TODO:如果我们要使用这个函数 传递的参数应该是怎么样的？传入的是一个json吗？get为什么传入两个参数，都匹配吗？
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        return cls(
            stoi={str(k): int(v) for k, v in payload["stoi"].items()},
            itos=[str(x) for x in payload["itos"]],
            unk_token=str(payload.get("unk_token", "<unk>")),
        )
