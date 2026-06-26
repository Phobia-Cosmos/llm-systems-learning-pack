from __future__ import annotations

from dataclasses import dataclass

# TODO:@dataclass作用是什么?cls关键字作用是什么?三个属性分别代表什么?
@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    unk_token: str = "<unk>"

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        itos = ["<unk>"] + chars
        # TODO:ch: i for i是什么意思?
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    # TODO:返回的是什么?为什么要先得到unk?
    def encode(self, text: str) -> list[int]:
        unk = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk) for ch in text]

    # TODO:原理是什么?为什么传入的是ids?为什么要判断unk token?
    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            pieces.append("?" if token == self.unk_token else token)
        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        return cls(
            stoi={str(k): int(v) for k, v in payload["stoi"].items()},
            itos=[str(x) for x in payload["itos"]],
            unk_token=str(payload.get("unk_token", "<unk>")),
        )
