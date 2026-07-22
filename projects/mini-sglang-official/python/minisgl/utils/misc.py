from __future__ import annotations

# TODO：这个函数如何解释 为什么一定需要一个discard？
# 解答：它是装饰器工厂：传入的 name 等于 "__main__" 时立即调用被装饰函数，否则不调用；通常应写 @call_if_main(__name__)。discard 只决定装饰后是否保留该函数名，并非必须为 True。
def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        # TODO：这个lambda表达式如何解释？
        # 解答：装饰时先执行 f()，(f() or True) 强制中间结果为真，再 and None 使装饰结果恒为 None，即“执行后丢弃函数绑定”。
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        # TODO：为什么这里返回的是1而不是b // a？
        # 解答：b 个 TP rank 多于 a 个 KV head 时，每个 rank 仍只持有 1 个 head；b // a 表示一个 head 被复制到多少个 rank，不是单 rank 的 head 数。
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b

# TODO：这个函数的作用是什么？
# 解答：它对正整数做向上取整除法，例如 div_ceil(5, 2) 返回 3，常用于计算页数或补齐后的分块数。
def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b

# TODO：这个类没有任何作用吧？
# 解答：空类被用作哨兵类型，单例 UNSET 可区分“参数未提供”与“显式传入 None”，调用方通过 isinstance(value, Unset) 判断。
class Unset:
    pass


UNSET = Unset()
