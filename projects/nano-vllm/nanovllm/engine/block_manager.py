from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    # TODO：一个block存储多少个token？这个是paged attention使用的存储方式吗？
    def __init__(self, block_id):
        self.block_id = block_id
        # TODO：ref count是什么意思？一个block可以被多个token 引用吗？hash是如何计算的？
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # TODO：reset时ref count为什么不归零？
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # TODO：这个是用来做什么的？为什么block manager也有一个hash？
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    # TODO：这个prefix是什么？prefix从哪里取得？如果有prefix就要连续计算两次是吗？
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
            # TODO：可以根据一个数组进行计算是吗？to_bytes和tobytes区别是什么？h.intdigest()是什么？
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # TODO：和allocate函数的区别是什么？
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # TODO：这个是什么情况？为什么free中的block hash不为一 没有被使用过的？
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)

        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # TODO：为什么内部计算要用cached blocks难道这些block被缓存起来吗？
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            # TODO：循环里每一次h不重置为-1,难道这里会出现prefix吗？
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # TODO：为什么会出现block id内部存储的token和现在的token不一样？为什么会直接break？
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            # TODO:这里在判断是吗？为什么要先加一后面还可能再减一（num_new_blocks）？
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # TODO：为什么这里要判断block table？block table中存储的是什么？要先判断不存在block table采进行分配吧？
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]

            # TODO:你的意思是不同token可能使用同一个block？为什么这个block id会出现在used的blocks中？
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
            # TODO：为什么cached block还需要重新分配block？
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        
        # TODO：这里又是在做什么？为什么需要这样一个range？为什么循环内部需要这样allocate？这里是分配没有出现的block是吗？
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # TODO：一个block会被多个seq使用吗？原因是什么？
    def deallocate(self, seq: Sequence):
        # TODO：为什么deallocate时需要反向？
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # TODO：这个函数在判断什么？append到哪里？
    def can_append(self, seq: Sequence) -> bool:
        # TODO：为什么要判断seq的长度？为什么seq要%block_size？这里是在判断什么东西这个函数？
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # TODO：为什么这里要直接append？
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # TODO：这里是直接返回一个seq对应的hash吗？为什么需要这样一个hash？
    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        # TODO：为什么相等就直接返回？
        if start == end: return

        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
