from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    # TODO：这个分支帮我举一些例子 返回的是list和tuple吗？
    # 解答：例如 [1, Tensor(...)] 返回递归转换后的 list，(1, msg) 返回 tuple；type(value) 会保留原容器类型。
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        # TODO：什么情况下会出现循环调用？这里会不会导致bug？
        # 解答：这是对嵌套 dataclass、Tensor 等非基本值的递归序列化，正常的无环消息树会终止；若对象图真的循环引用，当前实现会 RecursionError。
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    serialized = {}

    # TODO：为什么要转为numpy？才能存入buffer？
    # 解答：NumPy 提供了便捷的 tobytes 和 dtype 对应，用来生成可经 ZMQ 传输的原始字节；并非 buffer 只能由 NumPy 创建。
    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.numpy().tobytes()
        serialized["dtype"] = str(self.dtype)
        return serialized

    # normal type
    # TODO：这里一般来说都是什么值 这里一般是一个单独的值还是许多值？
    # 解答：__type__ 只存一个类名字符串（如 "UserMsg"）作为反序列化标签，其他多个字段会在后续循环中分别写入。
    serialized["__type__"] = self.__class__.__name__
    # TODO：正常类型都有哪些 为什么会有一个__dict__？这个会返回什么东西？为什么__dict__中还会出现dict吗？
    # 解答：这里通常是项目的消息 dataclass；普通 Python 实例用 __dict__ 保存“属性名 -> 属性值”，某个属性值本身当然也可以是 dict。
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        # TODO：__type__这个类型不是在deserialize_type中就已经被判断过了吗 为什么还要在这里出现？
        # 解答：外层 deserialize_type 只处理了顶层对象，字段中还可能嵌套另一个已序列化的消息或 Tensor，因此递归层仍要检查 __type__。
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        # TODO：这里是在dtype前面加上这个值还是什么意思？
        # 解答：这里是删除而非添加前缀，例如把 "torch.int32" 变成 "int32"，以便 getattr(np, ...) 取得对应的 NumPy dtype。
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        # TODO：为什么这里要返回copy而不是直接返回tensor？
        # 解答：frombuffer 得到的数组是引用 bytes 的只读视图；copy 创建独立且可写的内存，避免 torch.from_numpy 后的生命周期和写入问题。
        return torch.from_numpy(np_tensor.copy())

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
