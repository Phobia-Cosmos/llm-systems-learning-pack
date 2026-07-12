from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    # 问题（已回答）：Enum.auto() 的作用是什么？
    # 回答：自动为枚举成员生成唯一值；业务代码只关心状态身份，不依赖具体整数。
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

# 问题（已回答）：Sequence 存储什么，是 KV cache 吗？
# 回答：它保存请求 token、采样参数、状态和 KV block 映射元数据；真正 K/V 张量在 ModelRunner 的 GPU cache 池中。
class Sequence:
    block_size = 256
    # 问题（已回答）：counter 和 next() 在记录什么？
    # 回答：itertools.count() 是递增迭代器；每次 next(Sequence.counter) 取得新整数作为唯一 seq_id。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        # 问题（已回答）：为什么同时记录 num_prompt_tokens 和 num_tokens？
        # 回答：前者初始化后固定，用来分隔 prompt；后者随生成增长，两者之差就是 completion token 数。
        self.num_prompt_tokens = len(token_ids)
        # 问题（已回答）：cached tokens、block_table 和 num_scheduled_tokens 分别是什么？
        # 回答：K/V 存在全局 GPU cache 池；num_cached_tokens 是可复用前缀长度；block_table 映射物理块；
        # num_scheduled_tokens 是 scheduler 本轮准备计算的 token 数。
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 问题（已回答）：为什么按 block 计算，最后一块不满能处理吗？
    # 回答：Paged KV cache 固定大小分配；向上取整使不满的最后一块也占一个物理块，未使用槽位空闲即可。
    # 固定块便于分配、回收、前缀共享并减少显存碎片。
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 问题（已回答）：__getstate__ 是什么？
    # 回答：pickle 序列化时调用它。prefill 传完整 token_ids，decode 只传 last_token 和调度元数据，
    # 可减少跨进程共享内存的数据量；__setstate__ 负责恢复。
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
