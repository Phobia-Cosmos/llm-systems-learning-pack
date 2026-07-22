from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type

# TODO：为什么要区分fronted和backend的数据类型？为什么这些前后端都是encoder和decoder？
# 解答：FrontendMsg 用于 API 端回复，BackendMsg 用于 scheduler 命令，分开可防止不同 ZMQ 通道混用消息；encoder/decoder 是队列发送前和接收后的对称转换。
@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    # TODO：这个是如何发挥作用的？用户的历史信息不会被保存是吗？这个增量输出是模型返回给用户的是吗？
    # 解答：detokenizer 只把本轮新解码的文本 delta 放在此字段中，API 再流式发给用户；对话历史由请求方/API 组装输入，不由这个回复 DTO 持久化。
    incremental_output: str
    finished: bool
