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

    # 问题（已回答）：prepare_block_tables 做什么，为什么补到相同最大长度？
    # 回答：每条 Sequence 的 Python block_table 长度可不同，但 GPU kernel 需要一个矩形 [B,max_blocks_in_batch]
    # int32 Tensor；因此只补到当前 batch 的最大页数，并用 -1 表示无效尾项，不是扩到 max_model_len。
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        # 问题（已回答）：这里初始化的是 block 节点还是 KV cache 映射？
        # 回答：每行复制一条序列的“逻辑 block 序号 -> 物理 KV block_id”映射，再补 -1；不创建 Block 对象，
        # 也不复制 K/V 内容。attention 用这些物理 id 从已分配的 cache Tensor 中寻址。
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 问题（已回答）：torch.tensor(..., pin_memory=True).cuda() 是直接在 GPU 生成吗？
        # 回答：先在页锁定 CPU 内存创建 int32 Tensor，再复制成 CUDA Tensor；pinned host memory 配合 non_blocking=True
        # 才能让 H2D copy 异步排队。若要直接在 GPU 创建需显式 device="cuda"，但这里数据源是 Python lists。
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 问题（已回答）：prepare_prefill 做什么，是把所有 Sequence 都转成 input/position 吗？
    # 回答：它只取 scheduler 本轮选中的 seqs，并且每条只取 [num_cached_tokens:end) 的 scheduled 部分，沿 token 轴
    # 打包为一维 input_ids/positions；同时构造变长 attention、KV 散写和可选 prefix-cache 读取所需的 Context。
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        # 问题（已回答）：cu_seqlens_q/k 中存的都是位置吗？
        # 回答：不是绝对 position id，而是 packed batch 的累计长度，初始 0；相邻两项之差才是某条序列的 Q/K 长度。
        # positions 单独保存每个 scheduled token 的绝对位置，slot_mapping 单独保存其物理 KV 槽位。
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # 问题（已回答）：start、seqlen_q、end、seqlen_k 分别是什么，为什么 K 长度等于 end？
            # 回答：start 是已有 KV 的逻辑 token 数；Q 只包含本轮要算的 seqlen_q 个新 token，覆盖 [start,end)。
            # attention 的 K/V 必须覆盖从位置 0 到本轮末尾的完整上下文，所以长度是 cached prefix + new tokens=end。
            # Q 会打包到独立的 query 累计区间，并不是写回“start 对应的 cu_seqlens 空间”；start 是序列内逻辑位置。
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end

            # 问题（已回答）：seq[start:end] 会调用 Sequence 的哪个函数？
            # 回答：Python 的 [] 语法调用 Sequence.__getitem__(slice(start,end,None))；该方法再转发给 token_ids 列表，
            # 返回一个遵循右端不包含规则的 list[int]，extend 将其逐 token 追加到 packed input_ids。
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            # 问题（已回答）：为什么要记录 batch 中最大的 Q/K 序列长度？
            # 回答：FlashAttention 处理 packed 变长序列时用 cu_seqlens 划分实际边界，但仍需 max_seqlen_q/k 选择 kernel
            # 配置、循环/工作区上限；它不是再次 padding 数据，也不改变每条序列的真实长度。
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # 问题（已回答）：block_table 何时赋值，为什么为空时跳过 slot_mapping？
            # 回答：Sequence 初始化为空，正常请求在 Scheduler.allocate 时分配；唯一预期的空表路径是 allocate_kv_cache 之前的
            # warmup 假请求，此时 Attention 还没有 cache 可写，所以跳过物理槽位计算。非空时不会跳过，反而进入下面的映射循环。
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # 问题（已回答）：slot_start/slot_mapping 存储什么？
                # 回答：把分页 cache 看成 [num_blocks*block_size,...] 后，物理 slot=block_id*block_size+页内偏移；
                # slot_mapping 按 packed input token 顺序记录每个新 K/V 应散写到的扁平物理槽位。
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 问题（已回答）：end 及最后一个 slot_end 的含义是什么？
                    # 回答：end 是本轮区间的序列内 exclusive 右边界。例 block_size=4、block_table=[7,2,9]、
                    # start=5、end=10：逻辑 [5,8) -> 物理页 2 的 slots [9,12)，逻辑 [8,10) -> 物理页 9 的
                    # slots [36,38)。末页有效数量 end-i*block_size=10-2*4=2，所以 slot_end=9*4+2=38；range 不含 38。
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # 问题（已回答）：为什么累计 K 长度可能大于 Q，何时需要 block_tables？
        # 回答：Q 只含本轮 scheduled tokens；命中 prefix cache 或继续 partial prefill 时，K 还必须包含更早的 cached tokens，
        # 因而任一序列有 start>0 就会使总 K>总 Q。历史 K/V 不在本轮 packed k/v 中，只能用页表从 cache 读取。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # 问题（已回答）：为什么 decode 准备的 token/长度参数更少？
    # 回答：decode 的不变量是每条 Sequence 本轮只处理 last_token，所以 input_ids、positions、slot_mapping、context_lens
    # 各一项，无需用 cu_seqlens 划分多 token 变长 Q。完整历史已在 paged KV cache，由 context_lens+block_tables 描述。
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
        # 问题（已回答）：为什么 decode 总需要 block_tables，而普通 prefill 可以不需要？
        # 回答：decode 参数中的 K/V 只包含当前 token，所有历史都必须按页表从 cache 读取；无 prefix 的首次 prefill 则可直接
        # 对本轮 packed K/V 做 attention。prefill 并非永远不需要页表，复用 cached prefix 时上面的分支同样会准备。
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        greedy = [seq.is_greedy for seq in seqs]
        if all(greedy):
            # 整批 greedy 不需要温度 H2D、FP32 softmax 或随机数张量。
            return None, True
        # 问题（已回答）：pin_memory 是 cache 吗，为什么还要 cuda(non_blocking=True)？
        # 回答：pin_memory 创建的是页锁定 CPU staging memory，不是模型/KV cache，也不能被 CUDA kernel 直接当普通 GPU Tensor 使用；
        # .cuda() 才把温度复制到当前 GPU。源内存 pinned 时 non_blocking=True 可让 H2D copy 异步排入当前 CUDA stream。
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        if any(greedy):
            greedy = torch.tensor(greedy, dtype=torch.bool, pin_memory=True).cuda(non_blocking=True)
        else:
            greedy = False
        return temperatures, greedy

    # 问题（已回答）：@torch.inference_mode() 是什么，为什么需要装饰器？
    # 回答：@ 把函数交给 decorator 包装；inference_mode 在调用期间关闭 autograd 记录及额外 view/version tracking，
    # 降低纯推理的内存和调度开销。它不负责选择 GPU，也不是 torch.compile 或 CUDA Graph。
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 问题（已回答）：为什么 prefill、enforce_eager 或 input_ids.size(0)>512 走直接执行？
        # 回答：捕获图只覆盖 batch size 不超过 512 的 decode。decode 中 size(0)=Sequence 数且每条一个 token；prefill 中
        # size(0)=packed scheduled token 总数，cu_seqlens/shape/prefix 状态高度动态，固定图收益和适配都更困难。超过 512 无可选图。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 问题（已回答）：self.model(...) 调到哪里，为什么再交给 compute_logits？
            # 回答：self.model 是 registry 创建的 nn.Module；Module.__call__ 处理 hooks 后分派到具体 Qwen3/MiniGPT forward，
            # input_ids 用于 embedding，positions 用于位置编码，结果是 hidden states。模型类的 compute_logits 再调用 LM head，
            # 只取每条序列本轮最后位置并映射到词表 logits；两层调用不是重复包住同一计算。
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 问题（已回答）：graph 相关字段和 bs 表示什么？
            # 回答：bs 是当前 decode Sequence/token 数；graphs 保存各捕获 batch size 的 CUDAGraph，graph_bs 是可选静态形状，
            # graph_vars 是捕获时固定地址的输入/context/输出 CUDA buffers，graph_pool 是这些图共享的私有显存池。
            bs = input_ids.size(0)
            context = get_context()
            # 问题（已回答）：next(x for x...) 在取什么，为什么 x>=bs？
            # 回答：graph_bs 升序保存已捕获的静态 batch；生成器枚举能容纳实际 bs 的候选，next 取第一个也就是最小 padding 图，
            # 再从字典取其 CUDAGraph。捕获形状不能小于实际 batch，否则静态 buffer 和 kernels 容不下输入。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 问题（已回答）：input_ids/positions 缓冲是什么 dtype/shape，为什么写 [:bs]，它们是二维吗？
            # 回答：两者都是 CUDA int64 一维 Tensor [max_bs]，分别存 token id 和绝对位置，并非二维；只覆盖前 bs 个有效行，
            # 因为被选图可能大于实际 batch。二维缓冲是 int32 block_tables [max_bs,max_num_blocks] 和输出 [max_bs,hidden_size]。
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # 问题（已回答）：为什么 slot_mapping 先全填 -1，再复制实际值？
            # 回答：静态图可能 replay 大于实际 bs 的行，-1 是 KV store kernel 的无效槽位哨兵，阻止 padding 行写坏 cache；
            # 前 bs 项再换成本轮 CUDA int32 context.slot_mapping。context_lens 同理先清零，让 padding 行没有可读历史。
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # 问题（已回答）：block_tables 的 dtype/shape 是什么，这次赋值做什么？
            # 回答：源和目标都是 CUDA int32；目标固定为 [max_bs,max_num_blocks]，源是本轮 [bs,current_max_blocks]。
            # 这里只覆盖有效左上矩形。更右侧的旧 id 不会被读取，因为每行 context_lens 限定实际逻辑长度；padding 行长度为 0。
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 问题（已回答）：graph.replay() 做什么？
            # 回答：它以当前静态 buffers 的内容重新提交捕获时记录的 GPU kernel/collective 序列，不再逐层经过 Python 发起 kernel；
            # 结果写回同一个 outputs 缓冲，随后只取实际 bs 行做 LM head/logits。
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 问题（已回答）：为什么只有 rank 0 准备温度并采样？
        # 回答：所有 rank 都必须执行模型分片和 TP collectives；Vocab Parallel LM head 最后只把各词表 shard gather 到 rank 0，
        # 因而只有它拥有完整 logits 并负责一次采样/返回结果。其他 rank 准备温度或各自随机采样既浪费又可能产生不一致 token。
        temperatures, greedy_mask = self.prepare_sample(seqs) if self.rank == 0 else (None, None)

        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures, greedy_mask).tolist() if self.rank == 0 else None
        # 问题（已回答）：为什么 reset_context，是否表示一批请求全部完成？
        # 回答：只表示当前一次 model invocation 已结束，清除供各 Attention 层共享的临时 batch 元数据，避免下一 step 误读；
        # Sequence 是否完成由 Scheduler.postprocess 的 EOS/max_tokens 判断，许多请求 reset 后仍会继续 decode。
        reset_context()
        return token_ids

    # 问题（已回答）：capture_cudagraph 做什么，为什么/何时构建 CUDA Graph？
    # 回答：它为若干固定 decode batch size 预先 warmup 并捕获完整模型 forward 的 GPU 命令和固定内存地址；反复逐 token
    # decode 时 replay 可减少 Python、dispatcher 和 kernel launch 开销。仅 config.enforce_eager=False 时在初始化期构建。
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # 问题（已回答）：这里 bs 与 token 数是什么关系？
        # 回答：捕获图只服务 decode，而 decode 每条 Sequence 恰好一个输入 token，所以 bs 同时是序列数和本轮 token 数。
        # prefill 的 token 数可远大于 Sequence 数，不使用这些图。实现最多为 512 条 decode Sequence 捕获。
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 问题（已回答）：为什么只捕获这些 graph_bs？
        # 回答：CUDA Graph 要求静态 shape；为每个 1..512 都捕获会增加初始化时间和图/显存开销，因此小 batch 精细覆盖
        # 1/2/4/8，大 batch 每 16 一个档位，实际值向上 padding。默认 max_bs=512 可完整覆盖；当前写法对 max_bs<8
        # 会保留过大的冗余档位，对大于 8 但不是 16 倍数的 max_bs（如 20）还会漏掉尾部 batch，健壮实现应过滤并追加 max_bs。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 问题（已回答）：循环里的 bs 是什么，为什么 reversed？
        # 回答：bs 是当前要捕获的静态 decode batch。先捕获最大图会先建立足够大的 graph memory pool，后续较小图通过
        # self.graph_pool 共享该池，通常可减少重新扩容和碎片；字典仍按各自 bs 保存，运行时按升序 graph_bs 选择。
        for bs in reversed(self.graph_bs):
            # 问题（已回答）：这里的 graph 对象是什么？
            # 回答：torch.cuda.CUDAGraph 是一个可捕获并 replay 当前 CUDA stream 工作的容器；每个对象绑定该 bs 的 kernel 序列、
            # 参数和静态 Tensor 地址，不是模型结构图、autograd graph 或一个普通输出 Tensor。
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 问题（已回答）：self.model 为什么会输出，warmup/capture 在哪里、在哪个设备执行？
            # 回答：nn.Module.__call__ 会分派到具体模型 forward，返回 [bs,hidden_size] hidden states 并写入 outputs view。
            # 第一行在 graph context 外做 eager GPU warmup，使 lazy kernel/torch.compile 分配不进入捕获；下一行位于
            # torch.cuda.graph context 内才记录同一 forward。默认 device 此时仍是当前 rank 的 CUDA，不是在 CPU warmup。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup

            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
                # 问题（已回答）：self.graphs[bs] 保存的是什么？
                # 回答：保存刚捕获完成、可针对该静态 bs replay 的 CUDAGraph 句柄；首次 graph.pool() 另保存共享显存池句柄，
                # 不是把 outputs 数据复制进字典。所有图共用下面 graph_vars 中地址固定的 buffers。
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
