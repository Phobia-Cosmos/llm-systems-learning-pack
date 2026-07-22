from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    # TODO：key_fn和tic作用是什么？为什么孩子节点是Dict以及为什么Dict中还有一个Any类型？为什么初始化要传入这个key_fn？
    # 解答：key_fn 从边标签取首个完整 page 作为哈希键；children 用该键 O(1) 选择唯一下一条边。page_size=1 时键是 int，否则是 tuple，所以标成 Any；tic 可在分裂时继承访问时间。
    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None

        # TODO：ref_count指的是什么？哪些东西会创建这个ref？
        # 解答：ref_count 是有多少个已锁定 Handle 的路径经过该节点；调度器命中/运行请求时 lock_handle 增加，缓存请求完成或释放时 unlock 减少。
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1

        # TODO：为什么节点还需要一个时间属性？
        # 解答：timestamp 记录最近访问时间，evict 把叶节点放入最小堆后优先删除时间最早的节点，即近似 LRU 策略。
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        # TODO：这个self.key_fn(self._key)会返回什么？为什么要使用self._key传入key_fn？
        # 解答：它返回本节点边标签的第一个 token 或第一页 token 元组；父节点正用这个首 page 区分子边，所以把当前 _key 传入并登记 self。
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        # Todo：找到diff就要开始分支存储了是吗？如果使用naive python实现都是如何操作的？
        # 解答：这里只返回从开头连续相等的长度；_tree_walk 遇到边内差异会先在共同前缀处分裂，insert_prefix 再挂上新后缀。朴素实现就是逐 token 比较，直到首个不等位置。
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        # TODO：为什么创建分支时要传入key fn？这里为什么只分叉出一个节点 难道不应该是一个分支节点对应两个子节点吗？
        # 解答：新中间节点也要用同一 page 规则管理 children。split_at 此刻只把“旧后缀”接到共同前缀下；若调用方正在插入新序列，insert_prefix 随后才会添加第二个“新后缀”子节点。
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        # TODO：为什么原节点的父亲节点变成新的节点？请你给我一个具体的例子讲解 真实场景中分叉是如何发生的以及发生后会变成怎么样？
        # 解答：例如已有边 [1,2,3,4]，插入 [1,2,5,6]：先建中间边 [1,2]，旧节点改为其子边 [3,4]，随后再添加兄弟边 [5,6]，从而共享共同前缀且不复制 KV 索引。
        self.set_parent(new_node)

        return new_node

    # TODO：为什么要判断时间？
    # 解答：heapq 需要 __lt__ 比较节点；按 timestamp 排序可让 evict 每次弹出最久未访问的叶节点。
    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    # TODO：这个node代表这个handle处理的node吗？
    # 解答：是，它是本次匹配/插入所得前缀的终止节点；Handle 同时继承 cached_len，二者共同描述这次缓存引用。
    node: RadixTreeNode

    # TODO：返回的是从当前indice对应的node开始的全部prefix？
    # 解答：不是从当前节点向后的内容，而是沿 parent 从终止节点回到根，收集每段 value 后拼成“完整命中前缀”的全部 KV 槽位索引。
    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        # TODO：为什么这里要reverse？
        # 解答：向父节点遍历得到的是“末段到首段”的逆序，拼接前反转才能恢复 token 从前到后的槽位顺序。
        value_list.reverse()
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        # TODO：这里是只有一个元素0的tensor张量吗？
        # 解答：不是；torch.empty(0) 的形状是 (0,)，元素个数为 0。这里主要表示 evict(0) 没有要释放的槽位；零前缀命中由 cached_len=0 的 root handle 表示。
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        # TODO：这个root返回的是这颗tree的root吗？
        # 解答：是，root_node 是整棵 Radix Tree 的哨兵根；它本身不代表 token 边，并用 ref_count=1 保证永不被驱逐。
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        # TODO：这里的unlock处理的是什么？为什么lock handle就是修改evictable_size、protected_size？
        # 解答：unlock 释放该 Handle 对整条祖先路径的引用；节点引用数在 0 与 1 间切换时，其容量才在 evictable/protected 两类间移动，真实 KV 数据不在此处复制。
        if unlock:
            # TODO：这个循环是在做什么？
            # 解答：从 Handle 的终止节点逐级走到根，对命中前缀包含的每一段统一释放引用并更新容量分类。
            while not node.is_root():
                # TODO：为什么要减少ref count？
                # 解答：当前请求不再依赖该节点；减少引用数后，降到 0 的节点便可被后续 LRU 驱逐。
                node.ref_count -= 1
                assert node.ref_count >= 0
                # TODO：这些node都是self包含在内的children是吗？因此可以delete？
                # 解答：这些 node 是终止节点及其祖先，不是它的 children；unlock 不删除节点，只在 ref_count=0 时把其长度记为可驱逐，真正删除发生在 evict。
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        # TODO：这个prefix len会赋值到BaseCacheHandle的cached len是吗？子传父？然后node给RadixCacheHandle？
        # 解答：是；RadixCacheHandle(prefix_len, node) 按 dataclass 继承字段顺序把 prefix_len 赋给基类 cached_len，并把终止节点赋给子类 node。
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    # TODO：请你帮我举一个具体的例子讲解tree walk的过程是什么？为什么要tree walk 返回的是什么？
    # 解答：以 page_size=1 为例，树含边 [1,2]->[3,4]，输入 [1,2,3,9] 时先按首 token 找 [1,2]，再进入 [3,4] 并匹配到 [3]；必要时分裂，返回共同前缀的终止节点和 prefix_len=3，供 match/insert 复用。page_size>1 时只能按完整 page 向下对齐。
    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            # TODO：self.key_fn(input_ids[prefix_len:])返回的是什么？为什么又要使用children.get？筛选了哪些节点？这里返回的是一系列的node还是一个node？
            # 解答：它取剩余输入的首 token/首 page 作为字典键；children.get 只查询当前节点下以该 page 开头的唯一子节点，返回一个 RadixTreeNode 或 None，不是节点列表。
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            # TODO：为什么会返回None？
            # 解答：当前节点没有以该首 page 开头的子边时 get 返回 None，表示从这里起没有更多公共前缀，应返回已匹配的节点和长度。
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            # TODO：为什么不是向上align 如果刚好多出来一些 那多出的部分不算长度？
            # 解答：缓存分配和复用以完整 page 为单位；向上对齐会把尚未验证或不存在的 token 错当命中，所以只能向下截掉不足一页的尾部，由本次计算重新生成。
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            # TODO：为什么要更新时间？
            # 解答：命中即代表最近被使用，刷新时间后 LRU 驱逐会优先保留这条热前缀。
            node.timestamp = tic

        # TODO：如果上述都没有返回最后会返回什么东西？
        # 解答：说明输入已经完整匹配，返回最后到达的节点以及累计 prefix_len；通常 prefix_len 等于可按整页复用的输入长度。
        return node, prefix_len

# TODO：这个返回的KEY_FN就是返回传递的参数的部分区间值？这两个返回的lambda如何解读？
# 解答：是，返回值本身是一个函数：page_size=1 时 x -> 第一个 token 的 Python 标量；否则 x -> 前 page_size 个 token 组成的 tuple，以可哈希值作为 children 的字典键。
def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
