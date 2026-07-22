from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")

# TODO：push、pull、pub三个的区别是什么？
# 解答：PUSH 把消息负载均衡到相连的 PULL 接收者，PULL 是该管线的接收端；PUB 则把每条消息广播给所有匹配订阅条件的 SUB，不负责请求-响应配对。

class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        # TODO：zmq是什么？这几行分别是什么意思？
        # 解答：ZeroMQ 是进程间消息通信库；Context 管理底层 I/O 资源，socket(PUSH) 创建一个只负责向下游推送消息的 socket。
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        # TODO：这个create的主要是创建socket连接？看这个addr是否已经存在是吗？
        # 解答：create 只是项目内“本端是否负责 bind”的标志，不会探测地址；True 在该地址绑定端点，False 连接到已由另一端绑定的端点。
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        # TODO：msgpack是什么？use_bin_type？
        # 解答：MessagePack 是紧凑二进制序列化格式；use_bin_type=True 会把 Python bytes 编成 bin 类型、str 编成字符串类型，便于接收端无歧义还原。
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        # TODO：这个会send到socket的另一端是吗？另一端接收到以后会做什么？copy参数作用是什么？
        # 解答：是，消息会排入 ZMQ 并由相连的 PULL 接收，随后 unpack 和 decoder 还原对象；copy=False 请求尽量零拷贝发送，但 PyZMQ 仍可按消息大小和安全条件选择复制。
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    # TODO：event和raw的区别是什么？还有为什么raw=false？
    # 解答：event 仍是刚收到的 MessagePack 字节，get_raw 则把这些字节直接交给调用者以便原样转发；unpackb(raw=False) 会把 MessagePack 字符串解为 Python str，而二进制字段仍为 bytes。
    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        # TODO：这个poll的作用是什么？为什么要判断poll是否为0？
        # 解答：poll(timeout=0) 非阻塞检查当前是否有可读事件，返回 0 表示此刻没有消息，所以 empty 应返回 True。
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    # TODO:为什么这里不用get raw？
    # 解答：异步队列的当前调用方要直接消费解码后的前端消息，并不需要像多 rank Scheduler 那样转发同一份原始字节，因此这里只提供 get；并非异步 ZMQ 不能接收 raw。
    async def get(self) -> T:
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
