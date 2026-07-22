from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    # TODO：这个是生成的token数量还是？
    # 解答：不是输出 token 数；它是一次 prefill batch 最多新计算的 prompt/extend token 总数，也是 Engine 估算最大 forward 通信缓冲的长度。
    max_extend_tokens: int = 8192
    # TODO：除了radix还可以是哪些值以及这些cache如何实现？
    # 解答：当前注册了 radix 和 naive；两者共用实际 MHA KV 张量池，radix 用压缩前缀树保留并复用物理槽位，naive 始终匹配长度 0、请求结束后直接回收。
    cache_type: str = "radix"
    # TODO：这个不同会有何区别？online和offline的区别是什么？
    # 解答：offline 让 LLM.generate 在当前进程直接提供 UserMsg 并收集 token；online 使用 ZMQ 连接 API、tokenizer、detokenizer 等进程并支持并发流式服务，模型调度核心相同。
    offline_mode: bool = False

    # networking config
    # TODO：为什么需要独一的suffix？
    # 解答：同一次启动的组件共享该后缀，而不同进程实例用创建配置时的 PID 隔离 IPC 文件名，可避免多服务并存或旧 socket 路径造成冲突。
    _unique_suffix: str = field(default_factory=_get_pid_suffix)


    # TODO：为什么zmq的backend、detokneizer、scheduler要分成三个不同的地址？这里是三个独立的进程吗？
    # 解答：它们是三条方向和 socket 模式不同的通道：tokenizer 到 rank0、rank0 到 detokenizer、rank0 到其他 ranks。在线模式的组件通常是独立进程，但地址表示通道，不等同于恰好三个进程。
    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    # TODO：这个是在判断什么？为什么需要这个函数？
    # 解答：它决定 Scheduler 到 detokenizer 这条 ZMQ 链路由后端 bind 还是 connect；必须明确唯一的 bind 方，ServerArgs 在 tokenizer/detokenizer 共用进程时会覆盖该选择。
    def backend_create_detokenizer_link(self) -> bool:
        return True
