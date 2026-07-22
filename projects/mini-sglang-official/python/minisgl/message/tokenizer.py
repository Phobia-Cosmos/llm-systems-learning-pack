from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    # TODO：BaseTokenizerMsg中是没有属性的吧 那我们这个data存储什么东西呢，存储一些encoder和decoder？
    # 解答：data 存的是多个具体消息对象（如 DetokenizeMsg），不是 encoder/decoder 函数；基类无字段但提供统一协议和类型约束。
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    # TODO：针对List[Dict[str, str]]如何tokenize？
    # 解答：TokenizeManager 先将这种 OpenAI 风格的 role/content 消息列表交给 tokenizer.apply_chat_template 渲染成 prompt 字符串，再像普通文本一样逐条 encode；当前尚未做 batch tokenization。
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
