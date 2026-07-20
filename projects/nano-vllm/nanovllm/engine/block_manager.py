from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    # 问题（已回答）：一个 block 存储多少个 token？这是 PagedAttention 使用的存储方式吗？
    # 回答：容量由 BlockManager.block_size 决定，本项目默认每块 256 个 token 槽位。Block 对象只保存分配和
    # 前缀缓存元数据；真正每层的 K/V 张量位于 ModelRunner 的分页 KV cache 中，并按同一个 block_id 寻址。
    def __init__(self, block_id):
        self.block_id = block_id
        # 问题（已回答）：ref_count 表示什么？一个 block 可以被多个 token 引用吗？hash 如何计算？
        # 回答：ref_count 统计有多少条 Sequence 的 block_table 引用该物理块，而不是统计块内 token 的数量。
        # 多条序列的相同完整前缀可共享一个块。hash 是“前一块的链式 hash 字节 + 本块 token_ids 字节”的 xxHash64，
        # 这样相同块内容处于不同历史前缀时不会被误认为同一份因果 KV。
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 问题（已回答）：reset 时 ref_count 为什么不归零？
    # 回答：reset 只在 _allocate_block 已从 free 队列取出块、并断言旧 ref_count 为 0 后调用；此时该块已经分给
    # 第一位新所有者，所以应初始化为 1。它不是“把块恢复为空闲状态”的操作。
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 问题（已回答）：hash_to_block_id 用来做什么？为什么 BlockManager 也要保存 hash？
        # 回答：每个 Block.hash 记录该块自己的链式前缀键；这个字典则提供 hash 到物理 block_id 的全局索引，
        # 让新序列能按前缀快速定位可复用 KV，而不必遍历所有块。
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    # 问题（已回答）：prefix 是什么、从哪里来？有 prefix 时会连续计算两次吗？
    # 回答：prefix 是前一个完整逻辑块的链式 hash，首次为 -1；can_allocate 和 hash_blocks 会把上一轮结果传给
    # 下一块。函数只创建一次 hash 状态，依次喂入 prefix 的 8 字节和当前 token_ids 的字节，并非把当前块计算两次。
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
            # 问题（已回答）：hash 可以接收数组吗？to_bytes 和 tobytes 有何区别？intdigest() 是什么？
            # 回答：xxhash.update 接收 bytes-like 数据。int.to_bytes 把单个 Python 整数按指定长度和字节序编码；
            # ndarray.tobytes 把整个 NumPy 数组的连续内存编码成 bytes。intdigest() 返回 64 位无符号整数摘要，
            # 与 hexdigest() 返回十六进制字符串不同。
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 问题（已回答）：_allocate_block 和 allocate 有什么区别？
    # 回答：前者是私有原语，只从 free 队列取得一个物理块、清除其旧缓存身份并返回 id；后者面向整条 Sequence，
    # 先挂接可共享的 cached blocks，再为其余逻辑块逐个调用前者，最终构造完整 block_table。
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 问题（已回答）：为什么 free block 的 hash 可能不是 -1？它不是未被使用吗？
        # 回答：free 仅表示当前 ref_count 为 0；为了 prefix cache，释放时会保留 token_ids、hash 和 GPU 中尚未覆盖的
        # K/V。真正把这个物理块改作新用途时，才删除仍指向它的旧索引并 reset，避免后续命中已经被覆盖的缓存。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)

        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 问题（已回答）：can_allocate 为什么计算 cached blocks？这些 block 确实被缓存了吗？
    # 回答：是。完整块释放后，其 hash/token 元数据和未被覆盖的 KV 可以继续作为前缀缓存；仍被其他序列使用的
    # 相同块也可通过引用计数直接共享。函数同时计算可复用前缀长度和仍需占用多少 free blocks。
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            # 问题（已回答）：为什么循环中不把 h 重置为 -1？这里会出现 prefix 吗？
            # 回答：会，h 就是链式 prefix。第 i 块的键必须包含第 0 到 i-1 块的历史；若每次重置，只看当前块的
            # token_ids，就可能错误共享“当前块相同但更早上下文不同”的 KV。
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 问题（已回答）：hash 命中后 token_ids 为什么还可能不同？为何直接 break？是否表示完全没有缓存？
            # 回答：64 位 hash 理论上可能碰撞，索引也可能已过期，因此还要比较原始 token_ids。前缀缓存只能从第 0 块
            # 连续命中；某块缺失或不一致后，后续链式 hash 的前提也不成立，所以停止扫描。此前已命中的块仍然有效，
            # break 只表示缓存前缀在此结束，不表示 num_cached_blocks 必然为 0。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            # 问题（已回答）：为什么 num_cached_blocks 加一后，还可能把 num_new_blocks 减一？
            # 回答：两个计数含义不同：每次命中都增加“无需重算的前缀块数”。若命中块正被使用，当前序列可直接共享，
            # 不消耗 free 队列，所以待取空闲块数减一；若命中块目前 free，重新挂接它仍要从 free 队列移除一个 id，
            # 因而不能减少对 free 块数量的需求。
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # 问题（已回答）：如果空闲块不够分配会怎样？
        # 回答：返回 -1，Scheduler 暂不把该等待请求放入 prefill；它可先运行已有 decode 请求并在必要时抢占序列
        # 来释放 KV。此函数只做可行性检查，不会在失败时部分分配或阻塞等待。
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        # 问题（已回答）：为什么返回已缓存 block 数量？
        # 回答：Scheduler 用它计算本轮还要 prefill 的 token 数，allocate 用它把对应物理块挂到 block_table，
        # 并把 seq.num_cached_tokens 初始化为完整缓存块数乘 block_size。
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 问题（已回答）：为什么断言 block_table 为空？其中存什么？是否应先确认不存在再分配？
        # 回答：block_table 按逻辑块顺序保存该序列对应的物理 KV block_id。这个公开 allocate 正是首次绑定路径，
        # 调用者 Scheduler 已在 not seq.block_table 分支中检查；断言再防止重复分配造成引用计数错误或物理块泄漏。
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]

            # 问题（已回答）：不同 token 会使用同一个 block 吗？为何命中的 block_id 会在 used_block_ids 中？
            # 回答：不是任意不同 token 共享，而是多条序列的相同完整前缀共享整块 K/V。block_id 已在 used 集合中
            # 表示至少一条活跃序列正在引用它；新序列只需增加 ref_count，无需再次占用空闲块。
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                # 问题（已回答）：为什么 cached block 还要“重新分配”？
                # 回答：该块虽然保留有效缓存，但当前 ref_count 为 0，所以仍位于 free 队列。这里不是清空并重算它，
                # 而是把它重新激活：保留 hash/token/KV，设首个引用，并从 free 移到 used。
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        
        # 问题（已回答）：这个 range 在做什么？为何循环 allocate？是在分配未命中的块吗？
        # 回答：是。前 num_cached_blocks 个逻辑块已经挂接；从该下标到 seq.num_blocks 的其余块没有可复用前缀，
        # 每个逻辑块都需要一个新的物理 block_id，逐个追加后 block_table 才覆盖整条当前序列。
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 问题（已回答）：一个 block 会被多个 seq 使用吗？为什么？
    # 回答：会，但只限 token 内容和此前链式前缀都相同的完整块。因果 Transformer 对相同前缀计算出的 K/V 相同，
    # 共享物理块可节省 prefill 计算和显存；ref_count 保证最后一个使用者离开前不会回收它。
    def deallocate(self, seq: Sequence):
        # 问题（已回答）：deallocate 为什么反向遍历？
        # 回答：引用计数正确性并不依赖反向顺序；这里配合 free 队列的 FIFO 策略，先把通常复用价值较低的尾块放入
        # 队列，最后放入更可能被命中的前缀块。之后 _allocate_block 从左侧复用时，前缀块能更久不被覆盖。
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # 问题（已回答）：can_append 判断什么？append 到哪里？
    # 回答：它判断当前序列下一次 decode 写 KV 时是否需要新增物理块，以及 free 队列是否有该块。
    # 真正追加发生在 may_append，目标是 seq.block_table，而不是 token_ids。
    def can_append(self, seq: Sequence) -> bool:
        # 问题（已回答）：为什么检查 seq 长度并对 block_size 取余？seq 长度是什么？为何会余 1？
        # 回答：len(seq) 是 prompt 加已生成 token 的当前总数。postprocess 已先追加了一个尚待下一步计算 KV 的 token；
        # 当余数为 1 时，这个最新 token 正好是新逻辑块的第一个槽位，旧 block_table 尚无对应物理块，因而至少需要
        # 一个 free block。比较中的布尔值会按 0/1 使用：不跨块时要求 0 个，跨块时要求 1 个。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 问题（已回答）：may_append 为什么直接 append block_id？
    # 回答：若最新 token 是新逻辑块的第一个槽位，模型前向前必须先为它建立物理 KV 地址；这里追加的是新分配的
    # block_id。token 本身早已由 Sequence.append_token 加入，不会在此重复追加。
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # 问题（已回答）：hash_blocks 会直接返回一个 seq 的 hash 吗？为什么需要 hash？
    # 回答：不会返回值；它找出本轮新完成的所有完整逻辑块，为每块计算链式前缀 hash，更新 Block 元数据并写入
    # 全局 hash_to_block_id。这样以后相同前缀可定位并复用已算好的 K/V。
    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        # 问题（已回答）：为什么 start == end 时直接返回？
        # 回答：start 和 end 分别是本轮前后“已完整计算的块边界”。二者相等表示本轮没有跨过新的完整块边界，
        # 当前尾块仍未填满，不能作为稳定前缀共享，因此没有任何块需要登记。
        if start == end: return

        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
