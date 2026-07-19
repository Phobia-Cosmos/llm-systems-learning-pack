from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # TODO：返回的第二个参数代表什么意思？
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            # TODO：为什么我们只处理remaining,难道num_batched_tokens已经处理完成了吗？
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # TODO：存在block table就代表有cached tokens？
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # TODO：这个代表没有办法在分配了是吗？
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # TODO：为什么要减缓存的token？num_tokens代表还需处理的新token是吗？
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # TODO：什么是chunked prefill？为什么这里要break？scheduled_seqs一开始是空应该不会执行吧？只要检测到一个不空的scheduled_seqs就开始处理是吗？为什么remaining一定小于num tokens？
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # TODO：num_tokens和remaining的区别是什么？这两者分别代表哪些token？
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # TODO：为什么只有这个条件成立时才开始running？
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # TODO：什么情况下这里不会直接return？为什么会出现这种情况？
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # TODO：为什么会有max_num_seqs限制，一次最多可以处理多少seq？
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # TODO：这个while的条件是什么 为什么要判断是否可以append append到哪里？
            while not self.block_manager.can_append(seq):
                # TODO：这两个分支分别代表什么意思？preempt是什么意思？为什么如果在running就要先pop？为什么else分支直接break？
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs

        # TODO：为什么这里要加入反向后的seqs？目的是什么？
        self.running.extendleft(reversed(scheduled_seqs))
        # TODO：为什么这里返回False？
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        # TODO：为什么要放到waiting队列 不能直接run？
        self.waiting.appendleft(seq)

    # TODO：这个后处理是在处理什么东西？
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # TODO：为什么要这样hash一下？
            self.block_manager.hash_blocks(seq)

            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # TODO：这里是在判断什么东西？直到本seq的所有token全部处理完成才到下一个语句是吗？
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # TODO：这里是什么意思？prefill不是已经处理完成了吗？这里指的是处理到seq的末尾了是吗 需要加eos了？
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
