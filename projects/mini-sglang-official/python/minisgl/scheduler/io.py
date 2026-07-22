from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group

        # TODO：为什么要区分online和offline？为什么offline不实现？offline一般使用场景是什么？
        # 解答：online 通过 ZMQ 连接 API/tokenizer 等独立进程；offline 由 LLM.generate 在同一进程用内存列表收发。这里是通用 mixin，offline_* 由 LLM 子类实现。
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            # TODO：这里都是backend地址吧？
            # 解答：两者都是 Scheduler 后端侧 IPC，但方向不同：backend_addr 接收 tokenizer 的 UserMsg，detokenizer_addr 发送生成出的 DetokenizeMsg。
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        # TODO：self._recv_from_tokenizer、self._send_into_tokenizer和下面的recv、send有何区别？
        # 解答：前两者是实际 ZMQ 队列对象；recv/send 是按单卡、多卡及 rank 选择的策略函数，最后绑定为统一的 receive_msg/send_result 接口。
        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        # TODO：如果有多个GPU 那么rank0负责先把数据全部接收了 然后以pub-sub形式发送给其他rank？还是rank0订阅其他所有rank的数据？
        # 解答：前者；rank0 从 tokenizer 接收并 PUB 原始消息，其他 rank 用 SUB 接收。rank0 本身不订阅，而是在本地解码同一份消息。
        if tp_info.size > 1:
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    # TODO：为什么这几个函数没有实现 他们的作用本来是什么？
    # 解答：它们是模板钩子：Scheduler 实现空闲时的完整性检查，LLM 实现 offline 收发；基类无法假定具体运行模式，所以故意抛 NotImplementedError。
    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    # TODO：这个调用的时候一般是non blocking是吗？run when idle的意思是什么呢？_recv_msg_single_rank和_recv_msg_multi_rank0区别是什么？
    # 解答：有可运行 batch 时通常非阻塞地排空消息，无工作时才阻塞等待；run_when_idle 是阻塞前的后台钩子。single 只本地接收，multi_rank0 还要同步并转发给其余 TP ranks。
    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        # TODO：如果是阻塞式的 该如何回到这些信号的处理上呢？
        # 解答：ZMQ get 会让当前线程睡眠到消息到达；收到第一条后调用返回并继续排空队列，随后主循环处理消息和调度 batch。
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        # TODO：为什么这里只是广播len？为什么这里还要wait？
        # 解答：消息正文走 ZMQ PUB，collective 只广播“随后应收几条”，让所有 rank 执行相同次数的 SUB get；wait 保证长度广播完成后才读取张量并进入下一步。
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        # TODO：这里是在做什么？
        # 解答：rank0 把每条已序列化消息原样发布给其他 ranks，同时在本地解码并加入返回列表，避免额外的反序列化再序列化。
        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        # TODO：为什么要定义一个-1的tensor？作用是什么？为什么要把这个tensor传递出去？
        # 解答：broadcast 需要每个 rank 提供接收缓冲张量；-1 只是尚未接收时的哨兵初值，collective 会原地把它覆盖为 rank0 给出的消息数。
        dst_tensor = torch.tensor(-1)
        # TODO：这个广播是由rank0来负责是吗？root=0代表选择那一个来广播？
        # 解答：是，root=0 指定 rank0 的 src_tensor 为广播源，其余 rank 的 dst_tensor 接收同一个值。
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        # TODO：这个又是在做什么？
        # 解答：把广播后的一元素张量转成 Python 整数，下面据此从 SUB socket 精确接收相同数量的消息。
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    # TODO：为什么对于non primary rank就什么都不做？
    # 解答：所有 TP ranks 共同完成同一批推理，但只有 rank0 持有面向 detokenizer 的输出 socket；其他 rank 再发送会造成重复回复和竞争。
    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
