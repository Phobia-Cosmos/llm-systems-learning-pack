import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.registry import create_model
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        # 问题（已回答）：为什么配置 rank，rank 0 为什么重要？
        # 回答：rank 是 tensor-parallel 进程编号，每个 rank 负责一张 GPU 和一片模型参数；rank 0 还是协调者，
        # 负责接收 engine 请求、写共享内存并唤醒其他 rank，最终返回采样结果。
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 问题（已回答）：NCCL、TCP rendezvous、set_device 和 default_dtype 分别是什么？
        # 回答：NCCL 是 NVIDIA GPU 集合通信后端；tcp://localhost:2333 只用于进程会合和交换初始化信息，
        # 真正 GPU 数据通信由 NCCL 完成。set_device(rank) 绑定当前 GPU；default_dtype 是新建浮点参数的默认类型。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()

        # 问题（已回答）：嵌套 getattr 为什么传第三个参数？
        # 回答：getattr(obj, name, default) 的第三项是属性不存在时的默认值；优先读 dtype，缺失时回退到
        # torch_dtype，再缺失就得到 None，从而兼容不同 Transformers config 命名。
        model_dtype = getattr(hf_config, "dtype", getattr(hf_config, "torch_dtype", None))
        if isinstance(model_dtype, str):
            # 问题（已回答）：为什么字符串 dtype 要 getattr(torch, model_dtype)？
            # 回答：config JSON 可能保存 "float16"/"bfloat16" 字符串；这里把它解析成 torch.float16 等 dtype 对象。
            model_dtype = getattr(torch, model_dtype)
        if model_dtype is not None:
            torch.set_default_dtype(model_dtype)

        # 问题（已回答）：rank、CUDA device 和 default device 的关系是什么？
        # 回答：rank 先确定进程负责哪张卡，set_device(rank) 绑定当前 CUDA 卡；set_default_device("cuda")
        # 再让模型构造期间未显式指定 device 的参数直接创建在该卡上，避免先在 CPU 建模再搬运。
        torch.set_default_device("cuda")

        self.model = create_model(hf_config)
        # 问题（已回答）：next(self.model.parameters()).dtype 在取什么？
        # 回答：它读取模型第一个参数的实际 dtype，作为 KV cache dtype 和 cache block 字节数计算依据。
        # 问题（已回答）：一个模型通常有哪些参数？
        # 回答：parameters() 遍历注册在 nn.Module 中的可训练张量；本项目主要包括 token/position embedding、
        # Q/K/V 与输出投影、MLP 投影、可选 bias、Norm scale 和 lm_head。RoPE 表、ALiBi slope、KV cache
        # 属于 buffer 或运行时状态，不是 Parameter。这里的参数刚构造后会由 safetensors checkpoint 覆盖。
        self.model_dtype = next(self.model.parameters()).dtype

        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()

        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        # 问题（已回答）：worker 为什么要 event.wait()？
        # 回答：非零 rank 阻塞等待 rank 0 写完共享内存，避免空轮询占满 CPU 或读到半写数据。
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 问题（已回答）：读取后为什么 clear Event？
        # 回答：Event 是电平触发；clear() 将通知状态复位，下一轮 worker 才会再次等待新命令。
        self.event.clear()
        return method_name, args

    # 问题（已回答）：write_shm 的作用是什么？
    # 回答：rank 0 把“方法名和参数”序列化到共享内存，并通知其他 TP rank 执行同一方法。
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # 问题（已回答）：pickle.dumps 输出什么？
        # 回答：它把 [method_name, *args] Python 对象序列化成 bytes，便于写入 SharedMemory。
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 问题（已回答）：buf 前面保存什么，为什么分两段？
        # 回答：前 4 字节保存 payload 长度 n，后面的 [4:4+n] 保存 pickle 数据；接收端先读固定长度头，
        # 才知道后续读取多少字节。这里不是 5 字节，切片右端 n+4 不包含自身。
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data

        # 问题（已回答）：这里通知什么，worker 会读取刚写入的命令吗？
        # 回答：是。rank 0 已先把完整 payload 写进共享内存，再逐个 set Event；每个非零 rank 从 wait() 醒来，
        # 读取同一条方法名和参数并执行自己的模型分片。Event 只传“数据已就绪”的信号，实际命令仍在共享内存中。
        for event in self.event:
            # 问题（已回答）：event.set() 做什么？
            # 回答：把 Event 置为已通知，使阻塞在 wait() 的 worker 立即醒来读取命令。
            event.set()

    def call(self, method_name, *args):
        # 问题（已回答）：只有一个 GPU 时 call 是否不需要执行？
        # 回答：仍必须在当前 rank 调用目标方法来完成真正的 run/exit；单 GPU 只是不需要 write_shm 广播。
        # 多 GPU 时 rank 0 先唤醒其他 rank，然后自己也执行同一方法，使所有 rank 同步参与 TP 前向和 collective。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        # 问题（已回答）：reset_peak_memory_stats() 的作用是什么？
        # 回答：清零 CUDA allocator 峰值统计；随后 warmup 得到模型执行峰值，用于更准确计算剩余 KV cache 空间。
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        # 问题（已回答）：warmup 为什么构造多个 Sequence？
        # 回答：Sequence 封装一条请求。构造 num_seqs 条最大形状假请求可触发 kernel/JIT/临时显存分配，
        # 避免第一次真实请求承担这些开销。
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # 问题（已回答）：[0] * seq_len 模拟什么，假请求数量怎样受限？
        # 回答：每个 0 都是一个合法的占位 token id，因此一条假 Sequence 含 seq_len 个 token；num_seqs 同时受
        # 总 token 预算 max_num_batched_tokens 和 max_num_seqs 限制。它模拟的是允许预算内的最大 prefill 形状，
        # 不是生成真实文本，也不会写入尚未分配的 KV cache。
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        # 问题（已回答）：warmup 后为什么再次 empty_cache？
        # 回答：warmup 期间编译和算子执行可能让 PyTorch caching allocator 保留不再使用的临时块；empty_cache
        # 把这些空闲 reserved 块还给 CUDA driver，便于随后按真实可用显存分配 KV cache。已加载模型参数不会被清除，
        # 而 reset 后记录的 peak 统计仍可用于估算下一次前向所需的瞬时显存。
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # 问题（已回答）：为什么同时读取 peak 和 current？
        # 回答：current 是 warmup 结束后仍常驻的 PyTorch tensor 显存，peak 是一次最坏前向达到的峰值；
        # 二者之差 peak-current 近似下一次执行还要预留的临时激活/工作区。只看当前空闲量就把它全给 KV cache，
        # 真实运行到峰值时仍会 OOM。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 问题（已回答）：计算一个 KV block 的字节数为什么需要 block_size？
        # 回答：一个物理 block 为连续 block_size 个 token 预留每层的 K 和 V；每个 token 在当前 rank 上各有
        # num_kv_heads * head_dim 个元素。因此 block_size 决定单块容量，block 越大元数据越少但尾块浪费可能越多。
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * self.model_dtype.itemsize
        )
        # 问题（已回答）：KV cache 预算为什么写成 -peak + current？
        # 回答：这两项合起来就是减去 peak-current，即为下一次前向的瞬时峰值增量留出空间。完整预算是
        # “利用率允许的总显存 - 当前设备已用显存 - 运行时额外峰值”，再除以每块字节数得到可分配块数。
        # 这是 warmup 估算而非绝对保证，其他进程占用或新形状产生更大工作区时仍可能改变余量。
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=self.model_dtype,
        )
        layer_id = 0
        for module in self.model.modules():
            # 问题（已回答）：如何判断哪些 module 应绑定这一层的 KV cache？
            # 回答：Attention.__init__ 统一创建 k_cache 和 v_cache 属性，所以递归遍历 model.modules() 并用 hasattr
            # 做结构化识别；普通 Linear/Norm 没有这两个属性会被跳过。每发现一个 attention module，就把全局 cache
            # 对应 layer_id 的 view 赋给它，所有层共享一笔大分配但读写各自的层切片。
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # TODO:这个函数的作用是什么？为什么要全部扩充到最大的长度？
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        # TODO：初始化block节点，对应的是kvcache的映射是吗？
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # TODO：这里是为了让我们的tensor是在GPU上生成的是吗？
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # TODO：这个函数在做什么以及会有什么影响吗？将这次请求的所有seqs全部转换为input和position？
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        # TODO：这里面存储的都是位置数据吧？
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # TODO：为什么seqlen_q和seqlen_k使用的都是cache和schedule token？意思是这些seqlen_q也要存放到和start同一个空间是吗？为什么seqlen_k就是end？
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end

            # TODO:seq[]会调用Sequence的哪一个函数？
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            # TODO：为什么要获得多个seq中调度最多的token数量？
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # TODO：这个是什么意思？block table何时会被赋值？Seq初始化时这个字段为空？为什么不为空就要跳过这个seq的处理？
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # TODO：这里面存储的是什么？
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # TODO：请用画图的方式解释这个end的含义？
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # TODO：这里在判断什么？为什么会出现key的长度大于q？
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # TODO：为什么decode需要准备的参数变少了？
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # TODO：为什么decode需要额外准备block table而prefill不需要？
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        # TODO：pin_memory代表缓存？为什么最后还要加上cuda？non_blocking=True代表什么意思？
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # TODO:为什么需要这个@?这个代表什么意思？
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # TODO：为什么input_ids.size(0) > 512也要直接计算？这个代表什么意思？为什么prefill阶段不需要graph？
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # TODO：这个函数是在哪里定义的？self.model(input_ids, positions)为什么传入的是这个？为什么要用self.model包住？
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # TODO：这些graph字段都是什么意思？为什么需要bs这个字段？
            bs = input_ids.size(0)
            context = get_context()
            # TODO：这个next是什么意思？为什么要判断x是否大于等于bs？这里是从graphs中取出什么东西？
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # TODO：这些一开始都是什么数据类型的？input_ids？为什么需要[:bs]？这两个数组中存储的都是什么？为什么是二维的？
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # TODO：这是在做什么？为什么要先fill -1然后在使用context的slot mapping？
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # TODO：这个数据类型也给我标注出来 这个是在做什么？
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # TODO：这个是什么？做了哪些操作？
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # TODO：为什么只有rank0才需要温度？为什么只有rank0才可以做这个事情？
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # TODO：为什么要reset context 指的是一批请求处理完成吗？
        reset_context()
        return token_ids

    # TODO：这个注释的作用是什么？为什么需要构建一个cuda graph？哪些情况下需要构建这一个cuda 图？
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # TODO：bs和token大小的关系是什么？
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # TODO：为什么需要这样一个graph bs？
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # TODO：一个bs指的是什么？为什么是reversed顺序？
        for bs in reversed(self.graph_bs):
            # TODO：这里一个graph指的是什么？
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # TODO：为什么调用model就会自动输出了呢？他会被转到model的哪一个函数处理呢？这里不是cuda graph？在哪里warm up？CPU？
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup

            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
                # TODO：这里存入的graph指的是什么？
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
