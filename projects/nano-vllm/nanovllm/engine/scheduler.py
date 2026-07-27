from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduling_policy import BatchPhase, create_scheduling_policy
from nanovllm.engine.sequence import Sequence, SequenceStatus


@dataclass(slots=True)
class _SchedulerCounters:
    total_batches: int = 0
    prefill_batches: int = 0
    decode_batches: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    preemptions: int = 0
    recompute_sequences: int = 0
    recompute_batches: int = 0
    recomputed_tokens: int = 0


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    scheduling_policy: str
    waiting_sequences: int
    running_sequences: int
    free_kv_blocks: int
    used_kv_blocks: int
    total_batches: int
    prefill_batches: int
    decode_batches: int
    prefill_tokens: int
    decode_tokens: int
    preemptions: int
    recompute_sequences: int
    recompute_batches: int
    recomputed_tokens: int


class Scheduler:

    def __init__(self, config: Config):
        # 问题（已回答）：为什么要限制 max_num_seqs？这些 seq 只来自一个用户吗？
        # 回答：它限制一次模型调用同时处理的序列数，从而约束 KV cache、临时张量、kernel 规模和单步延迟。
        # Sequence 表示独立请求，调度器没有“用户”字段；服务层可以把多个用户的请求混入同一批次。
        self.max_num_seqs = config.max_num_seqs

        # 问题（已回答）：一个批次的 token 来自多个 seq，为什么还要限制一次处理的 token 上限？
        # 回答：prefill 的计算量和临时显存主要随本轮所有新 token 的总数增长。总 token 预算可防止长 prompt
        # 独占一次调用或造成显存峰值过高，也让多个请求之间的吞吐、延迟和公平性可控。
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.scheduling_policy = create_scheduling_policy(
            getattr(config, "scheduling_policy", "prefill_first")
        )
        self._counters = _SchedulerCounters()
        self._recomputing_seq_ids: set[int] = set()
        self._recompute_started_seq_ids: set[int] = set()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def snapshot(self) -> SchedulerSnapshot:
        """Return an immutable point-in-time view for metrics and tests."""
        counters = self._counters
        return SchedulerSnapshot(
            scheduling_policy=self.scheduling_policy.name,
            waiting_sequences=len(self.waiting),
            running_sequences=len(self.running),
            free_kv_blocks=len(self.block_manager.free_block_ids),
            used_kv_blocks=len(self.block_manager.used_block_ids),
            total_batches=counters.total_batches,
            prefill_batches=counters.prefill_batches,
            decode_batches=counters.decode_batches,
            prefill_tokens=counters.prefill_tokens,
            decode_tokens=counters.decode_tokens,
            preemptions=counters.preemptions,
            recompute_sequences=counters.recompute_sequences,
            recompute_batches=counters.recompute_batches,
            recomputed_tokens=counters.recomputed_tokens,
        )

    # 问题（已回答）：返回的第二个参数表示什么？为什么要判断是否为 prefill 阶段？
    # 回答：布尔值表示本批次走 prefill 还是 decode。两阶段的输入形状、Attention 读取 KV 的方式、
    # CUDA Graph 路径以及后处理规则不同，所以 ModelRunner 和 postprocess 都必须知道当前阶段。
    def schedule(self) -> tuple[list[Sequence], bool]:
        for phase in self.scheduling_policy.phase_order:
            if phase is BatchPhase.PREFILL:
                scheduled_seqs = self._schedule_prefill()
                is_prefill = True
            else:
                scheduled_seqs = self._schedule_decode()
                is_prefill = False
            if scheduled_seqs:
                self._record_batch(scheduled_seqs, is_prefill)
                return scheduled_seqs, is_prefill

        # 与原实现最后的 assert 相同：队列非空却无法组成任何批次说明容量/状态不一致。
        raise RuntimeError("Scheduler has pending sequences but cannot form a runnable batch")

    def _record_batch(self, seqs: list[Sequence], is_prefill: bool):
        counters = self._counters
        counters.total_batches += 1
        if is_prefill:
            counters.prefill_batches += 1
            counters.prefill_tokens += sum(seq.num_scheduled_tokens for seq in seqs)
        else:
            counters.decode_batches += 1
            counters.decode_tokens += len(seqs)

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        # 问题（已回答）：这一个 batch 会全部发送给模型生成 next token 吗？
        # 回答：会在一次 ModelRunner.run 中处理。decode 时每条序列只计算一个当前 token 并采样下一个 token；
        # prefill 时可能计算多个 prompt token，LM head 只为每条序列本轮最后一个位置产生候选，未完成的分块结果会被丢弃。
        num_batched_tokens = 0

        # prefill
        has_recompute = False
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            # 问题（已回答）：为什么这里只看 remaining，num_batched_tokens 已经处理完成了吗？
            # 回答：此处仍在组装批次，尚未调用模型。num_batched_tokens 是前面序列已经占用的本轮预算，
            # remaining 是当前批次还可继续调度的 token 数，而不是“模型已经算完”的 token 数。
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # 问题（已回答）：存在 block_table 就代表有 cached token 吗？为什么没有 block_table 也可能找到 cached token？
            # 回答：block_table 只表示该序列已经绑定了物理 KV block，不保证其中有可复用前缀；其
            # num_cached_tokens 也可能为 0。首次绑定前，BlockManager 会从全局前缀索引中查找其他序列留下的完整块，
            # 因而序列自己的 block_table 仍为空时，can_allocate 也可能发现 cached blocks。
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # 问题（已回答）：-1 表示无法再分配吗？无法分配会怎样？break 退出哪一层？
                # 回答：-1 表示当前空闲块不足。break 退出最近的 prefill while，不只是退出 if；若还有 running
                # 请求，调度器随后可做 decode 并通过抢占释放块。若没有任何 running 请求且该请求永久无法容纳，
                # 这份精简实现会在两个阶段都无法组批后抛出 RuntimeError；生产服务还应在 admission 时提前返回容量错误。
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 问题（已回答）：为什么要减去缓存 token？num_tokens 表示还需处理的新 token 吗？
                # 回答：是。已有 KV 的前缀不需要重复前向计算；这里的 num_tokens 是该序列当前尚未生成 KV、
                # 仍需在后续一次或多次 prefill 中计算的 token 数。
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 问题（已回答）：什么是 chunked prefill？为什么这里 break？scheduled_seqs 初始为空时条件会执行吗？
            # 回答：chunked prefill 是把长 prompt 拆到多个调度步中计算。本实现只允许本批次的第一条序列被拆分：
            # 第一条序列到来时 scheduled_seqs 为空，因此即使放不下也会继续并取 remaining 个 token；若此前已放入
            # 其他序列，当前序列放不下就 break，把它完整留到下一批。remaining < num_tokens 只在该分支条件成立时为真。
            # 问题（已回答）：remaining < num_tokens 是否表示本轮无法处理？何时 scheduled_seqs 为空但仍有 token？
            # 回答：它只表示本轮无法处理完，不表示完全不能处理；第一条长序列可以先处理 remaining 个 token，
            # 下轮从 num_cached_tokens 处继续。进入循环、尚未加入第一条序列时，scheduled_seqs 正是空的，但该序列仍有工作。
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # 问题（已回答）：num_tokens 和 remaining 有什么区别？分别代表哪些 token？
            # 回答：num_tokens 属于当前序列，表示它尚未计算 KV 的 token 总数；remaining 属于整个批次，表示扣除
            # 先前序列后剩余的调度额度。本轮实际处理二者的较小值。
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.seq_id in self._recomputing_seq_ids:
                has_recompute = True
                self._counters.recomputed_tokens += seq.num_scheduled_tokens
                if seq.seq_id not in self._recompute_started_seq_ids:
                    self._recompute_started_seq_ids.add(seq.seq_id)
                    self._counters.recompute_sequences += 1

            # 问题（已回答）：为什么只有这个条件成立才进入 RUNNING？必须达到该 token 数才能处理吗？
            # 回答：序列在本轮已经会被处理；该条件只决定处理后是否已完成全部 prefill。只有缓存前缀加本轮 token
            # 覆盖当前整条序列时，下一步才可逐 token decode，否则它仍在 WAITING 中等待下一个 prefill chunk。
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                self._recomputing_seq_ids.discard(seq.seq_id)
                self._recompute_started_seq_ids.discard(seq.seq_id)
            scheduled_seqs.append(seq)

        if has_recompute:
            self._counters.recompute_batches += 1
        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        # decode
        # 问题（已回答）：decode 为什么也受 max_num_seqs 限制？一次最多处理多少个 seq？
        # 回答：decode 虽然每条序列只处理一个 token，但 KV 读取、Attention 和临时张量仍随序列数增长。
        # 每步最多处理 config.max_num_seqs 条；实际数量还受 running 队列长度和可用 KV block 限制。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 问题（已回答）：这个 while 判断什么？append 到哪里？为什么不能 append 就要 preempt？
            # 回答：它检查当前待计算 token 是否跨入新逻辑块，以及若跨块是否有空闲物理块可追加到 seq.block_table。
            # 没有页就无法保存该 token 的 K/V；抢占其他序列可释放物理块，让当前较高优先级序列继续。
            while not self.block_manager.can_append(seq):
                # 问题（已回答）：两个分支分别表示什么？preempt 是什么？为什么先 pop，else 又直接 break？
                # 回答：若还有其他 running 序列，就从队尾 pop 一个较低优先级受害者并抢占，释放其 KV 后重试当前序列；
                # pop 是为了避免它同时留在 running 和 waiting。若已无其他受害者，只能抢占当前 seq；break 会跳过
                # while 的 else，因此本轮不再调度它，之后由 waiting/prefill 重建其 KV。
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

        # 问题（已回答）：为什么把反向后的 scheduled_seqs 加回 running？这是抢占 running 队列吗？
        # 回答：decode 前这些序列被 popleft 临时取出，执行后仍未完成，所以要放回队首供下一轮继续。
        # deque.extendleft 会逐项插到左端，先 reversed 才能保持原调度顺序；这里不是抢占。
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    # 问题（已回答）：为什么抢占后设为 WAITING 和 prefill？此前 prefill 不是已结束了吗？为何 deallocate？
    # 回答：抢占会释放该序列的物理 KV 映射以让出显存，token_ids 本身仍保留。没有历史 KV 就不能直接 decode，
    # 所以后续必须像 prefill 一样重新计算已有 token（也可能重新命中完整前缀缓存），并在 waiting 队列等待资源。
    def preempt(self, seq: Sequence):
        self._counters.preemptions += 1
        self._recomputing_seq_ids.add(seq.seq_id)
        self._recompute_started_seq_ids.discard(seq.seq_id)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        # 问题（已回答）：为什么放回 waiting，不能直接 run？
        # 回答：它的 block_table 已清空，直接 decode 无法读取历史 K/V。放回 waiting 后，调度器会重新分配页面并完成
        # prefill/recompute；appendleft 让被抢占请求保持较高优先级，减少饥饿。
        self.waiting.appendleft(seq)

    # 问题（已回答）：postprocess 处理什么？传入的 token_id 是模型新生成的吗？
    # 回答：是，ModelRunner 为每条序列返回一个采样 token。后处理登记新完成的可缓存块、推进 KV 进度，
    # 在 prefill 完成或 decode 后把采样 token 加入序列，并判断 EOS/长度停止条件和释放已完成请求的 KV。
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # 问题（已回答）：为什么这里要 hash blocks？
            # 回答：本轮前向已经把新 K/V 写入 cache；将刚刚完整计算完的逻辑块按前缀链登记后，后续具有相同
            # token 前缀的序列才能通过 hash_to_block_id 找到并共享这些物理块。未填满的块不会登记。
            self.block_manager.hash_blocks(seq)

            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # 问题（已回答）：这里判断什么？未处理完会怎样？会有调度序列在后处理后仍未完成吗？
            # 回答：它判断 chunked prefill 是否只完成了当前 prompt 的一部分。若是，continue 只跳到 zip 循环的
            # 下一条序列，不会退出整个 postprocess；该序列保留在 waiting，记录已完成的 num_cached_tokens，下一调度步继续。
            # 因此后处理后存在尚未完成的序列是正常状态，generate 的外层循环会持续调度。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # 问题（已回答）：prefill 完成后为什么 append token？这是到序列末尾并手工加 EOS 吗？
            # 回答：完整 prefill 已计算到 prompt 最后一个位置，该位置的 logits 正好预测第一个 completion token，
            # 所以应追加采样得到的 token；decode 同理追加下一个 token。它不是手工添加 EOS，只有模型恰好采样到 eos
            # 才触发停止，否则会继续直到 max_tokens。
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
