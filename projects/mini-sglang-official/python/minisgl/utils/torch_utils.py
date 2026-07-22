from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
# TODO：这里是要做什么？为什么要更新default dtype？
# 解答：它在 with 块内临时改变 PyTorch 默认浮点类型，使模型中未显式传 dtype 的 torch.empty 等按配置创建权重，离开时恢复全局值。
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        # TODO：这里yield后会发生什么？
        # 解答：yield 把控制权交给 with 块；块正常结束或抛异常后从此处恢复，finally 都会把原 dtype 设回去。
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    # TODO：这里的fn是哪里传过来的参数？这里是装饰器的标准写法吗？
    # 解答：使用 @nvtx_annotate("...") 时，外层先返回 decorator，Python 再自动把被装饰的方法作为 fn 传入；这是带参装饰器的标准结构。
    def decorator(fn):
        @functools.wraps(fn)
        # TODO：这个wrapper的作用是什么？这个wrapper如何使用以及会在哪里被使用？
        # 解答：wrapper 替代原方法，每次调用时在其外包一层 NVTX 性能标记后再调 fn；模型层、MLP、Attention、LMHead 等 forward 通过装饰器使用它。
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                # TODO：这里format后变成什么？
                # 解答：它把属性值填入 name 的 {} 占位符，例如 name="Layer_{}"、_layer_id=3 时得到 "Layer_3"。
                display_name = name.format(getattr(self, layer_id_field))

            # TODO：这又是什么意思？为什么对name进行range？
            # 解答：nvtx.range 不是对字符串做数值 range，而是在 CUDA profiler 时间线上开启一个名为 display_name 的区间，用来测量该方法耗时。
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
