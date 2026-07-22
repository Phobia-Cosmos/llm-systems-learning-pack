from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type

# TODO：既然没有属性 那为什么还要定义这么多的数据类型？
# 解答：基类用来统一序列化协议和类型边界，各子类则以具体类名区分命令，并携带各自的 dataclass 字段。
@dataclass
class BaseBackendMsg:
    # TODO：为什么只有decoder是静态方法？为什么这个类没有任何的属性？
    # 解答：encoder 要序列化已存在的 self；decoder 时对象尚未创建，只需输入 json，所以是 staticmethod。基类只承载共同行为，字段在子类中。
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
